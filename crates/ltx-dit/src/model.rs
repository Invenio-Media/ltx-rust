//! The LTX-2.5 22B video-only DiT (Diffusion Transformer).
//!
//! # Architecture
//!
//! ```text
//! VideoInput { latent (B,T,128), timesteps (B,T), positions (B,3,T,2),
//!              context (B,S,4096), context_mask (B,1,T,S) }
//!         │
//!         ▼
//! patchify_proj(128 → inner_dim)          ← linear with bias
//!         │
//!         ▼
//! adaln_single(timesteps × 1000)          ← per-token AdaLN modulation
//!         │ modulation (B, T_ts, coeff×inner)
//!         │ embedded_timestep (B, T_ts, inner)
//!         ▼
//! precompute_freqs_cis(positions)         ← 3-D RoPE frequencies
//!         │ (cos, sin) each (B, H, T, d_head/2)
//!         ▼
//! [TransformerBlock × num_layers]
//!         │ self-attn + cross-attn + FFN with AdaLN
//!         ▼
//! scale_shift_table + embedded_timestep   ← output modulation
//!         ▼
//! layer_norm_no_affine(inner_dim)
//!         ▼
//! proj_out(inner_dim → 128)               ← predicted velocity / x0
//! ```
//!
//! # Connector / caption projection
//!
//! In LTX-2.5 22B (`caption_proj_before_connector = true`), the text
//! projection from the Gemma encoder to `cross_attention_dim = 4096` is done
//! outside the transformer.  The `context` tensor arrives already projected, so
//! no `caption_projection` module lives inside this crate.
//!
//! # IC-LoRA reference tokens
//!
//! Clean reference tokens (from the IC-LoRA reference segment) are appended to
//! the target tokens before entering this model.  They are distinguished by
//! `timestep = 0` in the `timesteps` field of [`VideoInput`].  No special
//! logic is required in the transformer — the reference simply receives a
//! lower-noise signal through its per-token timestep.

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::adaln::AdaLayerNormSingle;
use crate::block::TransformerBlock;
use crate::config::DiTConfig;
use crate::norm::layer_norm_no_affine;
use crate::offload::OffloadMode;
use crate::rope::precompute_freqs_cis;

/// All inputs to one transformer forward pass (video stream only).
pub struct VideoInput<B: Backend> {
    /// Patchified latent tokens `(B, T, in_channels)`.
    ///
    /// For IC-LoRA, the reference tokens are appended along T after the target
    /// tokens.
    pub latent: Tensor<B, 3>,

    /// Per-token timesteps `(B, T)`.
    ///
    /// Target tokens carry the diffusion noise level; clean reference tokens
    /// carry `0.0`.
    pub timesteps: Tensor<B, 2>,

    /// Patch position bounds `(B, 3, T, 2)`.
    ///
    /// Axis 1 = `(t, h, w)` position dimensions; axis 3 = `[start, end)`.
    /// When `use_middle_indices_grid = true` (default), RoPE uses the midpoint
    /// `(start + end) / 2`.
    pub positions: Tensor<B, 4>,

    /// Text context tokens `(B, S, cross_attention_dim)` from the prompt encoder.
    pub context: Tensor<B, 3>,

    /// Additive log-space cross-attention mask `(B, 1, T, S)`.
    ///
    /// `0.0` = attend, a large negative value = masked out.
    /// `None` = attend to all context tokens.
    pub context_mask: Option<Tensor<B, 4>>,

    /// Optional additive log-space self-attention mask `(B, 1, T, T)`.
    ///
    /// Used when IC-LoRA conditioning sets attention-strength values between
    /// reference and target tokens.  `None` = full self-attention.
    pub self_attn_mask: Option<Tensor<B, 4>>,
}

/// The LTX-2.5 video-only DiT transformer.
#[derive(Module, Debug)]
pub struct VideoTransformer<B: Backend> {
    /// Projects latent patches from `in_channels` to `inner_dim`.
    pub patchify_proj: Linear<B>,
    /// Per-token timestep → AdaLN modulation + embedded timestep.
    pub adaln_single: AdaLayerNormSingle<B>,
    /// Optional prompt-side AdaLN (used when `cross_attention_adaln = true`).
    pub prompt_adaln_single: Option<AdaLayerNormSingle<B>>,
    /// Learnable keyframe absolute-position marker (optional).
    pub keyframes_abs_pos_embedding: Option<Param<Tensor<B, 2>>>,
    /// Static output-norm scale-shift: `(2, inner_dim)`.
    pub scale_shift_table: Param<Tensor<B, 2>>,
    /// Output linear: `inner_dim → out_channels`.
    pub proj_out: Linear<B>,
    /// All N transformer blocks.
    pub blocks: Vec<TransformerBlock<B>>,
    /// Snapshot of the model configuration.
    pub config: DiTConfig,
}

impl<B: Backend> VideoTransformer<B> {
    /// Build a randomly-initialised transformer from a config.
    ///
    /// Weights loaded from a checkpoint should be assigned via the standard
    /// Burn record API (`module.load_record(record)`) or the fixture loader in
    /// the parity test.
    pub fn new(config: &DiTConfig, device: &B::Device) -> Self {
        let inner = config.inner_dim();

        let blocks: Vec<_> = (0..config.num_layers)
            .map(|_| TransformerBlock::new(config, device))
            .collect();

        let prompt_adaln =
            if config.flags.cross_attention_adaln && config.flags.use_prompt_adaln_single {
                Some(AdaLayerNormSingle::new(inner, 2, device))
            } else {
                None
            };

        let keyframes = if config.flags.use_keyframes_abs_pos_embedding {
            Some(Param::from_tensor(Tensor::zeros([1, inner], device)))
        } else {
            None
        };

        Self {
            patchify_proj: LinearConfig::new(config.in_channels, inner).init(device),
            adaln_single: AdaLayerNormSingle::new(inner, config.adaln_coeff(), device),
            prompt_adaln_single: prompt_adaln,
            keyframes_abs_pos_embedding: keyframes,
            scale_shift_table: Param::from_tensor(Tensor::zeros([2, inner], device)),
            proj_out: LinearConfig::new(inner, config.out_channels).init(device),
            blocks,
            config: config.clone(),
        }
    }

    /// Enable chunked self-attention on all blocks.
    ///
    /// Sets `q_chunk_size` tokens per chunk.  `None` reverts to full O(T²)
    /// attention.  Call before the forward pass when the sequence length is known.
    #[must_use]
    pub fn set_q_chunk(mut self, q_chunk: Option<usize>) -> Self {
        self.blocks = self
            .blocks
            .into_iter()
            .map(|blk| blk.with_q_chunk(q_chunk))
            .collect();
        self
    }

    /// Forward pass.
    ///
    /// `mode` controls whether block weights are streamed from host (see
    /// [`crate::offload`]).  On the `ndarray` CPU backend both modes are
    /// numerically identical.
    ///
    /// Returns the predicted velocity (or x0, matching the reference output
    /// convention) of shape `(B, T, out_channels)`.
    pub fn forward(
        &self,
        input: VideoInput<B>,
        _mode: OffloadMode,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let VideoInput {
            latent,
            timesteps,
            positions,
            context,
            context_mask,
            self_attn_mask,
        } = input;

        let [batch, n_tokens, _in_ch] = latent.dims();

        // ── Patch embedding ────────────────────────────────────────────────
        let x = self.patchify_proj.forward(latent); // (B, T, inner)

        // ── Timestep embedding ─────────────────────────────────────────────
        // Scale by timestep_scale_multiplier (default 1000).
        let ts_scale =
            f32::from(u16::try_from(self.config.timestep_scale_multiplier).unwrap_or(u16::MAX));
        let ts_flat = timesteps
            .mul_scalar(ts_scale)
            .reshape([batch.saturating_mul(n_tokens)]);

        let (modulation, embedded_ts) = self.adaln_single.forward(ts_flat, device);

        let coeff_inner = modulation.dims()[1];
        let inner_dim = embedded_ts.dims()[1];
        let modulation = modulation.reshape([batch, n_tokens, coeff_inner]);
        let embedded_ts = embedded_ts.reshape([batch, n_tokens, inner_dim]);

        // ── RoPE frequencies ───────────────────────────────────────────────
        let pe = precompute_freqs_cis(
            positions,
            self.config.inner_dim(),
            self.config.positional_embedding_theta,
            &self.config.positional_embedding_max_pos,
            self.config.num_attention_heads,
            self.config.rope_type,
            device,
        );

        // ── Transformer blocks ─────────────────────────────────────────────
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(
                x,
                context.clone(),
                context_mask.as_ref(),
                &modulation,
                (pe.0.clone(), pe.1.clone()),
                self_attn_mask.as_ref(),
            );
        }

        // ── Output modulation ──────────────────────────────────────────────
        // scale_shift_table: (2, inner_dim)
        // embedded_ts: (B, T, inner_dim)
        // → (B, T, 2, inner_dim) via broadcast sum
        let sst = self.scale_shift_table.val(); // (2, inner_dim)
        let sst_inner = sst.dims()[1];
        let sst_bc = sst.reshape([1, 1, 2, sst_inner]); // (1, 1, 2, inner)
        let emb_bc = embedded_ts.unsqueeze_dim::<4>(2); // (B, T, 1, inner)
        let combined = sst_bc + emb_bc; // (B, T, 2, inner)

        let shift = combined
            .clone()
            .narrow(2, 0, 1)
            .reshape([batch, n_tokens, sst_inner]);
        let scale = combined
            .narrow(2, 1, 1)
            .reshape([batch, n_tokens, sst_inner]);

        // No-affine LayerNorm then modulate.
        let x_norm = layer_norm_no_affine(x, self.config.norm_eps);
        let x_out = x_norm * (scale.add_scalar(1.0_f32)) + shift;
        self.proj_out.forward(x_out)
    }
}
