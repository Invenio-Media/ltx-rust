//! One LTX-2.5 transformer block (video stream only).
//!
//! Each block contains:
//! 1. AdaLN-modulated RMS-norm → self-attention (with 3-D RoPE) → residual.
//! 2. RMS-norm → text cross-attention → residual.
//!    Optionally: AdaLN on the query side and a prompt-side AdaLN on K/V when
//!    `cross_attention_adaln = true`; for the 22B model this is `false`.
//! 3. AdaLN-modulated RMS-norm → feed-forward → residual.
//!
//! The learnable `scale_shift_table` holds the static per-block offset for the
//! AdaLN modulation; the dynamic part is supplied by the timestep embedding.
//!
//! Reference: `BasicAVTransformerBlock.forward` (video branch only) in
//! `transformer.py`.

use burn::module::Param;
use burn::prelude::*;

use crate::attention::Attention;
use crate::config::DiTConfig;
use crate::feed_forward::FeedForward;
use crate::norm::rms_norm;

/// One transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention (no context = self-attention mode).
    pub attn_self: Attention<B>,
    /// Cross-attention to the text context.
    pub attn_cross: Attention<B>,
    /// Feed-forward network.
    pub ff: FeedForward<B>,
    /// Static per-block AdaLN offset: `(adaln_coeff, inner_dim)`.
    /// `adaln_coeff` = 6 (base) or 9 (with `cross_attention_adaln`).
    pub scale_shift_table: Param<Tensor<B, 2>>,
    /// Static scale-shift for cross-attention Q modulation: `(2, inner_dim)`.
    /// Present only when `cross_attention_adaln = true`.
    pub prompt_scale_shift_table: Option<Param<Tensor<B, 2>>>,
    /// Whether per-block cross-attention AdaLN is active.
    pub cross_attention_adaln: bool,
    /// Epsilon for RMS pre-norms.
    pub norm_eps: f32,
}

impl<B: Backend> TransformerBlock<B> {
    /// Build a block from the model config.
    pub fn new(config: &DiTConfig, device: &B::Device) -> Self {
        let inner = config.inner_dim();
        let adaln_coeff = config.adaln_coeff();
        let rope = config.rope_type;
        let norm_eps = config.norm_eps;

        let attn_self = Attention::new(
            inner,
            None, // self-attention
            config.num_attention_heads,
            config.attention_head_dim,
            norm_eps.into(),
            rope,
            config.flags.apply_gated_attention,
            None,
            device,
        );
        let attn_cross = Attention::new(
            inner,
            Some(config.cross_attention_dim),
            config.num_attention_heads,
            config.attention_head_dim,
            norm_eps.into(),
            rope,
            config.flags.apply_gated_attention,
            None, // cross-attention is never chunked (context is short)
            device,
        );
        let ff = FeedForward::new(inner, config.flags.ff_bias, device);

        let sst = Tensor::zeros([adaln_coeff, inner], device);
        let prompt_sst = if config.flags.cross_attention_adaln {
            Some(Param::from_tensor(Tensor::zeros([2, inner], device)))
        } else {
            None
        };

        Self {
            attn_self,
            attn_cross,
            ff,
            scale_shift_table: Param::from_tensor(sst),
            prompt_scale_shift_table: prompt_sst,
            cross_attention_adaln: config.flags.cross_attention_adaln,
            norm_eps,
        }
    }

    /// Set the query-chunking factor for self-attention.
    ///
    /// Used by [`crate::model::VideoTransformer`] to enable chunked attention.
    #[must_use]
    pub const fn with_q_chunk(mut self, q_chunk: Option<usize>) -> Self {
        self.attn_self.q_chunk = q_chunk;
        self
    }

    /// Forward pass for one block.
    ///
    /// - `x`: hidden states `(B, T, inner_dim)`.
    /// - `context`: text context `(B, S, cross_attn_dim)`.
    /// - `context_mask`: additive log-space context mask `(B, 1, T, S)`.
    /// - `timestep`: AdaLN modulation `(B, T_ts, adaln_coeff × inner_dim)`.
    /// - `pe`: RoPE `(cos, sin)` for self-attention, each `(B, H, T, d/2)`.
    /// - `self_attn_mask`: optional self-attention bias `(B, 1, T, T)`.
    ///
    /// Returns updated `x` of the same shape.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<&Tensor<B, 4>>,
        timestep: &Tensor<B, 3>,
        pe: (Tensor<B, 4>, Tensor<B, 4>),
        self_attn_mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch, n_tokens, _inner] = x.dims();

        // ── Self-attention ────────────────────────────────────────────────
        let (shift_msa, scale_msa, gate_msa) = self.ada_values(timestep, batch, n_tokens, 0, 3);
        let norm_x = rms_norm(x.clone(), self.norm_eps);
        let norm_x = norm_x * (scale_msa.add_scalar(1.0_f32)) + shift_msa;
        let attn_out = self
            .attn_self
            .forward(norm_x, None, self_attn_mask, Some(pe), None);
        let x = x + attn_out * gate_msa;

        // ── Cross-attention ───────────────────────────────────────────────
        let norm_x2 = rms_norm(x.clone(), self.norm_eps);
        let ca_out = if self.cross_attention_adaln {
            self.cross_attn_adaln(norm_x2, context, context_mask, timestep, batch, n_tokens)
        } else {
            self.attn_cross
                .forward(norm_x2, Some(context), context_mask, None, None)
        };
        let x = x + ca_out;

        // ── Feed-forward ──────────────────────────────────────────────────
        let (shift_mlp, scale_mlp, gate_mlp) = self.ada_values(timestep, batch, n_tokens, 3, 3);
        let norm_x3 = rms_norm(x.clone(), self.norm_eps);
        let ff_in = norm_x3 * (scale_mlp.add_scalar(1.0_f32)) + shift_mlp;
        x + self.ff.forward(ff_in) * gate_mlp
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Extract `count` (shift, scale, gate) modulation tensors starting at `start`.
    ///
    /// Returns `(v0, v1, v2)` each of shape `(batch, t_ts, inner_dim)`.
    fn ada_values(
        &self,
        timestep: &Tensor<B, 3>, // (batch, t_ts, adaln_coeff * inner_dim)
        batch: usize,
        t_ts: usize,
        start: usize,
        count: usize, // always 3
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let table = self.scale_shift_table.val(); // (adaln_coeff, inner_dim)
        let [adaln_coeff, inner_dim] = table.dims();

        // Static table slice: (count, inner_dim) → (1, 1, count, inner_dim)
        let table_slice = table
            .narrow(0, start, count)
            .reshape([1, 1, count, inner_dim]);

        // Dynamic timestep: (batch, t_ts, adaln_coeff*inner_dim) → (batch, t_ts, adaln_coeff, inner_dim)
        let ts_4d = timestep
            .clone()
            .reshape([batch, t_ts, adaln_coeff, inner_dim]);
        let ts_slice = ts_4d.narrow(2, start, count);

        let combined = table_slice + ts_slice; // (batch, t_ts, count, inner_dim)

        let v0 = combined
            .clone()
            .narrow(2, 0, 1)
            .reshape([batch, t_ts, inner_dim]);
        let v1 = combined
            .clone()
            .narrow(2, 1, 1)
            .reshape([batch, t_ts, inner_dim]);
        let v2 = combined.narrow(2, 2, 1).reshape([batch, t_ts, inner_dim]);
        (v0, v1, v2)
    }

    /// Cross-attention with per-query AdaLN modulation.
    ///
    /// Used only when `cross_attention_adaln = true`.
    /// Slices 6..9 from `timestep` for the Q modulation;
    /// the prompt-side table modulates K/V.
    fn cross_attn_adaln(
        &self,
        x_normed: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<&Tensor<B, 4>>,
        timestep: &Tensor<B, 3>,
        batch: usize,
        t_ts: usize,
    ) -> Tensor<B, 3> {
        let (shift_q, scale_q, gate) = self.ada_values(timestep, batch, t_ts, 6, 3);
        let x_mod = x_normed * (scale_q.add_scalar(1.0_f32)) + shift_q;

        // K/V modulation from the static prompt-side table.
        let ctx_mod = if let Some(prompt_table) = &self.prompt_scale_shift_table {
            let pt = prompt_table.val(); // (2, inner_dim)
            let [_, inner_dim] = pt.dims();
            let pt4 = pt.reshape([1, 1, 2, inner_dim]);
            let shift_kv = pt4.clone().narrow(2, 0, 1).reshape([1, 1, inner_dim]);
            let scale_kv = pt4.narrow(2, 1, 1).reshape([1, 1, inner_dim]);
            context * (scale_kv.add_scalar(1.0_f32)) + shift_kv
        } else {
            context
        };

        self.attn_cross
            .forward(x_mod, Some(ctx_mod), context_mask, None, None)
            * gate
    }
}
