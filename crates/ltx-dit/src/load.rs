//! Checkpoint loading for the LTX-2.5 video `DiT`.
//!
//! [`VideoTransformer::load`] reads tensors by the reference module's
//! `state_dict()` names, relative to the caller's [`Scope`].  When opened
//! with [`ltx_weights::KeyMap::identity`] (fixtures), keys are used as-is.
//! When opened with [`ltx_weights::KeyMap::transformer`] (real checkpoints),
//! `model.diffusion_model.` is stripped first.
//!
//! ## Key layout (`state_dict` relative to module root)
//!
//! ```text
//! patchify_proj.{weight,bias}
//! adaln_single.emb.timestep_embedder.linear_1.{weight,bias}
//! adaln_single.emb.timestep_embedder.linear_2.{weight,bias}
//! adaln_single.linear.{weight,bias}
//! scale_shift_table
//! proj_out.{weight,bias}
//! transformer_blocks.{i}.attn1.{to_q,to_k,to_v}.{weight,bias}
//! transformer_blocks.{i}.attn1.{q_norm,k_norm}.weight
//! transformer_blocks.{i}.attn1.to_out.0.{weight,bias}
//! transformer_blocks.{i}.attn2.{to_q,to_k,to_v}.{weight,bias}
//! transformer_blocks.{i}.attn2.{q_norm,k_norm}.weight
//! transformer_blocks.{i}.attn2.to_out.0.{weight,bias}
//! transformer_blocks.{i}.ff.net.0.proj.{weight,bias}
//! transformer_blocks.{i}.ff.net.2.{weight,bias}
//! transformer_blocks.{i}.scale_shift_table
//! ```
//!
//! `nn.Linear` weight is transposed from `PyTorch` `[out, in]` to Burn `[in, out]`.
//! The `RmsNorm` `gamma` weight maps to the reference `weight` field.

use burn::module::Param;
use burn::nn::{Gelu, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use ltx_weights::Scope;

use crate::{
    adaln::{AdaLayerNormSingle, TimestepEmbedding},
    attention::Attention,
    block::TransformerBlock,
    config::{DiTConfig, RopeType},
    error::DitError,
    feed_forward::FeedForward,
    model::VideoTransformer,
};

impl<B: Backend> VideoTransformer<B> {
    /// Load a [`VideoTransformer`] from checkpoint weights.
    ///
    /// `scope` must be rooted at the transformer module root.  For a real
    /// checkpoint opened with [`ltx_weights::KeyMap::transformer`], pass
    /// `store.scope("")` (the root scope after prefix stripping).  For a
    /// fixture opened with [`ltx_weights::KeyMap::identity`], pass the root
    /// scope directly.
    ///
    /// # Errors
    ///
    /// - [`DitError::Config`] — invalid config (same as [`VideoTransformer::new`]).
    /// - [`DitError::Weight`] — a required tensor is absent or has the wrong rank.
    pub fn load(scope: &Scope, config: &DiTConfig, device: &B::Device) -> Result<Self, DitError> {
        config.validate()?;

        let patchify_proj = load_linear(scope, "patchify_proj", device)?;
        let adaln_single = load_adaln_single(scope, config, device)?;

        let scale_shift_table: Tensor<B, 2> = scope
            .tensor("scale_shift_table", device)
            .map_err(DitError::Weight)?;
        let scale_shift_table = Param::from_tensor(scale_shift_table);

        let proj_out = load_linear(scope, "proj_out", device)?;

        let blocks: Vec<TransformerBlock<B>> = (0..config.num_layers)
            .map(|i| {
                let block_scope = scope.scope(&format!("transformer_blocks.{i}"));
                load_block(&block_scope, config, device)
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            patchify_proj,
            adaln_single,
            scale_shift_table,
            proj_out,
            blocks,
            config: config.clone(),
        })
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Load a `Linear` module, transposing the weight from `PyTorch` `[out, in]` to
/// Burn `[in, out]`.
///
/// Bias is optional: if the key `{prefix}.bias` is absent the returned module
/// has no bias.
fn load_linear<B: Backend>(
    scope: &Scope,
    prefix: &str,
    device: &B::Device,
) -> Result<Linear<B>, DitError> {
    let w: Tensor<B, 2> = scope
        .tensor(&format!("{prefix}.weight"), device)
        .map_err(DitError::Weight)?;
    // PyTorch nn.Linear weight: [out_features, in_features].
    // Burn Linear weight: [in_features, out_features].  Transpose.
    let w = w.transpose();
    let [in_dim, out_dim] = w.dims();

    let bias: Option<Tensor<B, 1>> = scope
        .optional(&format!("{prefix}.bias"), device)
        .map_err(DitError::Weight)?;

    let mut lin = LinearConfig::new(in_dim, out_dim)
        .with_bias(bias.is_some())
        .init(device);
    lin.weight = Param::from_tensor(w);
    if let Some(b) = bias {
        lin.bias = Some(Param::from_tensor(b));
    }
    Ok(lin)
}

/// Load a `RmsNorm` from `{prefix}.weight` into the `gamma` field.
///
/// `norm_eps` must match the config to keep the epsilon consistent.
fn load_rms_norm<B: Backend>(
    scope: &Scope,
    prefix: &str,
    norm_eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>, DitError> {
    let gamma: Tensor<B, 1> = scope
        .tensor(&format!("{prefix}.weight"), device)
        .map_err(DitError::Weight)?;
    let [d_model] = gamma.dims();
    let mut norm = RmsNormConfig::new(d_model)
        .with_epsilon(norm_eps)
        .init(device);
    norm.gamma = Param::from_tensor(gamma);
    Ok(norm)
}

/// Load `AdaLayerNormSingle` from `adaln_single.*` keys.
fn load_adaln_single<B: Backend>(
    scope: &Scope,
    config: &DiTConfig,
    device: &B::Device,
) -> Result<AdaLayerNormSingle<B>, DitError> {
    let inner = config.inner_dim();
    let coeff = config.adaln_coeff();

    // Reference path: `adaln_single.emb.timestep_embedder.{linear_1,linear_2}`
    let ts_scope = scope.scope("adaln_single.emb.timestep_embedder");
    let linear_1 = load_linear(&ts_scope, "linear_1", device)?;
    let linear_2 = load_linear(&ts_scope, "linear_2", device)?;

    // Verify expected shapes after loading (sanity, not load failure).
    debug_assert_eq!(linear_1.weight.dims(), [256, inner]);
    debug_assert_eq!(linear_2.weight.dims(), [inner, inner]);

    // `adaln_single.linear` — the output projection
    let linear = load_linear(&scope.scope("adaln_single"), "linear", device)?;
    debug_assert_eq!(linear.weight.dims(), [inner, inner.saturating_mul(coeff)]);

    Ok(AdaLayerNormSingle {
        timestep_embedder: TimestepEmbedding { linear_1, linear_2 },
        linear,
    })
}

/// Load an `Attention` module from `{block_scope}.{ref_prefix}.*`.
///
/// `ref_prefix` is `"attn1"` for self-attention or `"attn2"` for cross-attention.
#[allow(clippy::too_many_arguments)]
fn load_attention<B: Backend>(
    block_scope: &Scope,
    ref_prefix: &str,
    query_dim: usize,
    context_dim: Option<usize>,
    heads: usize,
    d_head: usize,
    norm_eps: f64,
    gated: bool,
    rope_type: RopeType,
    device: &B::Device,
) -> Result<Attention<B>, DitError> {
    let inner = heads.saturating_mul(d_head);
    let attn_scope = block_scope.scope(ref_prefix);

    let to_q = load_linear(&attn_scope, "to_q", device)?;
    let to_k = load_linear(&attn_scope, "to_k", device)?;
    let to_v = load_linear(&attn_scope, "to_v", device)?;

    let q_norm = load_rms_norm(&attn_scope, "q_norm", norm_eps, device)?;
    let k_norm = load_rms_norm(&attn_scope, "k_norm", norm_eps, device)?;

    // Reference uses `torch.nn.Sequential`; key is `to_out.0.{weight,bias}`.
    let to_out = load_linear(&attn_scope.scope("to_out"), "0", device)?;

    let to_gate_logits = if gated {
        Some(load_linear(&attn_scope, "to_gate_logits", device)?)
    } else {
        None
    };

    // Sanity: verify shapes match expected dims (context_dim for k/v).
    let ctx = context_dim.unwrap_or(query_dim);
    debug_assert_eq!(to_q.weight.dims(), [query_dim, inner]);
    debug_assert_eq!(to_k.weight.dims(), [ctx, inner]);
    debug_assert_eq!(to_v.weight.dims(), [ctx, inner]);
    debug_assert_eq!(to_out.weight.dims(), [inner, query_dim]);

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "d_head ≤ 128 in all supported configs; lossless f32 conversion"
    )]
    let attn_scale = 1.0_f32 / (d_head as f32).sqrt();

    Ok(Attention {
        to_q,
        to_k,
        to_v,
        q_norm,
        k_norm,
        to_out,
        to_gate_logits,
        heads,
        d_head,
        attn_scale,
        rope_type,
        q_chunk: None,
    })
}

/// Load a `FeedForward` module.
///
/// Reference layout:
/// - `ff.net.0.proj.{weight,bias}` → `linear_in`
/// - `ff.net.2.{weight,bias}` → `linear_out`
fn load_feed_forward<B: Backend>(
    block_scope: &Scope,
    device: &B::Device,
) -> Result<FeedForward<B>, DitError> {
    let ff_scope = block_scope.scope("ff");
    let linear_in = load_linear(&ff_scope.scope("net.0"), "proj", device)?;
    let linear_out = load_linear(&ff_scope.scope("net"), "2", device)?;
    Ok(FeedForward {
        linear_in,
        act: Gelu::new_approximate(),
        linear_out,
    })
}

/// Load a `TransformerBlock` from its per-block scope.
fn load_block<B: Backend>(
    block_scope: &Scope,
    config: &DiTConfig,
    device: &B::Device,
) -> Result<TransformerBlock<B>, DitError> {
    let inner = config.inner_dim();
    let norm_eps = f64::from(config.norm_eps);
    let gated = config.flags.apply_gated_attention;

    let attn_self = load_attention(
        block_scope,
        "attn1",
        inner,
        None,
        config.num_attention_heads,
        config.attention_head_dim,
        norm_eps,
        gated,
        config.rope_type,
        device,
    )?;

    let attn_cross = load_attention(
        block_scope,
        "attn2",
        inner,
        Some(config.cross_attention_dim),
        config.num_attention_heads,
        config.attention_head_dim,
        norm_eps,
        gated,
        config.rope_type,
        device,
    )?;

    let ff = load_feed_forward(block_scope, device)?;

    let scale_shift_table: Tensor<B, 2> = block_scope
        .tensor("scale_shift_table", device)
        .map_err(DitError::Weight)?;
    let scale_shift_table = Param::from_tensor(scale_shift_table);

    let prompt_scale_shift_table = if config.flags.cross_attention_adaln {
        let t: Tensor<B, 2> = block_scope
            .tensor("prompt_scale_shift_table", device)
            .map_err(DitError::Weight)?;
        Some(Param::from_tensor(t))
    } else {
        None
    };

    Ok(TransformerBlock {
        attn_self,
        attn_cross,
        ff,
        scale_shift_table,
        prompt_scale_shift_table,
        cross_attention_adaln: config.flags.cross_attention_adaln,
        norm_eps: config.norm_eps,
    })
}
