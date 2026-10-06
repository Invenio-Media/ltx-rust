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
use ltx_weights::{Scope, WeightError};

use crate::{
    adaln::{AdaLayerNormSingle, SINUSOIDAL_HALF, TimestepEmbedding},
    attention::Attention,
    block::TransformerBlock,
    config::DiTConfig,
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
    /// - [`DitError::Weight`] — a required tensor is absent, has the wrong
    ///   rank, or has a shape that does not match the config.
    pub fn load(scope: &Scope, config: &DiTConfig, device: &B::Device) -> Result<Self, DitError> {
        config.validate()?;

        let inner = config.inner_dim();

        let patchify_proj = load_linear(scope, "patchify_proj", device)?;
        check_2d(
            &patchify_proj.weight.val(),
            [config.in_channels, inner],
            "patchify_proj.weight",
        )?;
        if let Some(b) = &patchify_proj.bias {
            check_1d(&b.val(), inner, "patchify_proj.bias")?;
        }

        let adaln_single = load_adaln_single(scope, config, device)?;

        let scale_shift_table: Tensor<B, 2> = scope.tensor("scale_shift_table", device)?;
        check_2d(&scale_shift_table, [2, inner], "scale_shift_table")?;
        let scale_shift_table = Param::from_tensor(scale_shift_table);

        let proj_out = load_linear(scope, "proj_out", device)?;
        check_2d(
            &proj_out.weight.val(),
            [inner, config.out_channels],
            "proj_out.weight",
        )?;
        if let Some(b) = &proj_out.bias {
            check_1d(&b.val(), config.out_channels, "proj_out.bias")?;
        }

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

// ── shape helpers ─────────────────────────────────────────────────────────────

fn shape_err(key: &str, expected: &[usize], got: &[usize]) -> DitError {
    DitError::Weight(WeightError::InvalidTensorData {
        key: key.to_owned(),
        message: format!("shape mismatch: expected {expected:?}, got {got:?}"),
    })
}

/// Assert that a 2-D weight tensor (post-transpose) has the expected shape.
fn check_2d<B: Backend>(w: &Tensor<B, 2>, expected: [usize; 2], key: &str) -> Result<(), DitError> {
    let got = w.dims();
    if got != expected {
        return Err(shape_err(key, &expected, &got));
    }
    Ok(())
}

/// Assert that a 1-D tensor has the expected length.
fn check_1d<B: Backend>(t: &Tensor<B, 1>, expected: usize, key: &str) -> Result<(), DitError> {
    let [got] = t.dims();
    if got != expected {
        return Err(shape_err(key, &[expected], &[got]));
    }
    Ok(())
}

// ── tensor loaders ────────────────────────────────────────────────────────────

/// Load a `Linear` module, transposing the weight from `PyTorch` `[out, in]` to
/// Burn `[in, out]`.
///
/// Bias is optional: if the key `{prefix}.bias` is absent the returned module
/// has no bias.
///
/// Note: `LinearConfig::init` allocates random weights before they are
/// overwritten by the checkpoint tensors.  On large models this doubles the
/// per-layer peak memory transiently.  Burn does not currently expose a
/// direct struct constructor that avoids this allocation.
fn load_linear<B: Backend>(
    scope: &Scope,
    prefix: &str,
    device: &B::Device,
) -> Result<Linear<B>, DitError> {
    let w: Tensor<B, 2> = scope.tensor(&format!("{prefix}.weight"), device)?;
    // `PyTorch` nn.Linear weight: `[out_features, in_features]`.
    // Burn Linear weight: `[in_features, out_features]`.  Transpose.
    let w = w.transpose();
    let [in_dim, out_dim] = w.dims();

    let bias: Option<Tensor<B, 1>> = scope.optional(&format!("{prefix}.bias"), device)?;

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
fn load_rms_norm<B: Backend>(
    scope: &Scope,
    prefix: &str,
    norm_eps: f64,
    d_model: usize,
    device: &B::Device,
) -> Result<RmsNorm<B>, DitError> {
    let gamma: Tensor<B, 1> = scope.tensor(&format!("{prefix}.weight"), device)?;
    check_1d(&gamma, d_model, &format!("{prefix}.weight"))?;
    let mut norm = RmsNormConfig::new(d_model)
        .with_epsilon(norm_eps)
        .init(device);
    norm.gamma = Param::from_tensor(gamma);
    Ok(norm)
}

// ── module loaders ────────────────────────────────────────────────────────────

/// Load `AdaLayerNormSingle` from `adaln_single.*` keys.
fn load_adaln_single<B: Backend>(
    scope: &Scope,
    config: &DiTConfig,
    device: &B::Device,
) -> Result<AdaLayerNormSingle<B>, DitError> {
    let inner = config.inner_dim();
    let coeff = config.adaln_coeff();
    let sinusoidal_dim = SINUSOIDAL_HALF.saturating_mul(2);

    let ts_scope = scope.scope("adaln_single.emb.timestep_embedder");
    let linear_1 = load_linear(&ts_scope, "linear_1", device)?;
    check_2d(
        &linear_1.weight.val(),
        [sinusoidal_dim, inner],
        "adaln_single.emb.timestep_embedder.linear_1.weight",
    )?;

    let linear_2 = load_linear(&ts_scope, "linear_2", device)?;
    check_2d(
        &linear_2.weight.val(),
        [inner, inner],
        "adaln_single.emb.timestep_embedder.linear_2.weight",
    )?;

    let adaln_scope = scope.scope("adaln_single");
    let linear = load_linear(&adaln_scope, "linear", device)?;
    check_2d(
        &linear.weight.val(),
        [inner, inner.saturating_mul(coeff)],
        "adaln_single.linear.weight",
    )?;

    Ok(AdaLayerNormSingle {
        timestep_embedder: TimestepEmbedding { linear_1, linear_2 },
        linear,
    })
}

/// Load an `Attention` module from `{block_scope}.{ref_prefix}.*`.
///
/// `context_dim` is `None` for self-attention (`attn1`) and
/// `Some(cross_attention_dim)` for cross-attention (`attn2`).
fn load_attention<B: Backend>(
    block_scope: &Scope,
    ref_prefix: &str,
    context_dim: Option<usize>,
    config: &DiTConfig,
    device: &B::Device,
) -> Result<Attention<B>, DitError> {
    let inner = config.inner_dim();
    let ctx = context_dim.unwrap_or(inner);
    let norm_eps = f64::from(config.norm_eps);
    let attn_scope = block_scope.scope(ref_prefix);

    let to_q = load_linear(&attn_scope, "to_q", device)?;
    let to_k = load_linear(&attn_scope, "to_k", device)?;
    let to_v = load_linear(&attn_scope, "to_v", device)?;

    // q_norm and k_norm gamma shapes: both are [inner_dim] regardless of context_dim.
    let q_norm = load_rms_norm(&attn_scope, "q_norm", norm_eps, inner, device)?;
    let k_norm = load_rms_norm(&attn_scope, "k_norm", norm_eps, inner, device)?;

    // `to_out` is `torch.nn.Sequential`; checkpoint key is `to_out.0.*`.
    let to_out = load_linear(&attn_scope.scope("to_out"), "0", device)?;

    let to_gate_logits = if config.flags.apply_gated_attention {
        Some(load_linear(&attn_scope, "to_gate_logits", device)?)
    } else {
        None
    };

    // Shape validation — wrong config silently produces mismatched matmul
    // shapes inside `forward`, which is hard to diagnose.  We check here in
    // both debug and release.
    check_2d(
        &to_q.weight.val(),
        [inner, inner],
        &format!("{ref_prefix}.to_q.weight"),
    )?;
    check_2d(
        &to_k.weight.val(),
        [ctx, inner],
        &format!("{ref_prefix}.to_k.weight"),
    )?;
    check_2d(
        &to_v.weight.val(),
        [ctx, inner],
        &format!("{ref_prefix}.to_v.weight"),
    )?;
    check_2d(
        &to_out.weight.val(),
        [inner, inner],
        &format!("{ref_prefix}.to_out.0.weight"),
    )?;

    if let Some(g) = &to_gate_logits {
        check_2d(
            &g.weight.val(),
            [inner, config.num_attention_heads],
            &format!("{ref_prefix}.to_gate_logits.weight"),
        )?;
    }

    let d_head = config.attention_head_dim;
    // `d_head` is validated as even and fits in `usize` by `DiTConfig::validate`.
    // All supported configs have `d_head ≤ 65535`, so `u16` conversion is safe.
    // `f32::from(u16)` is lossless for any value ≤ 65535.
    let d_head_u16 = u16::try_from(d_head)
        .map_err(|_| DitError::Config(format!("attention_head_dim {d_head} exceeds u16::MAX")))?;
    let attn_scale = 1.0_f32 / f32::from(d_head_u16).sqrt();

    Ok(Attention {
        to_q,
        to_k,
        to_v,
        q_norm,
        k_norm,
        to_out,
        to_gate_logits,
        heads: config.num_attention_heads,
        d_head,
        attn_scale,
        rope_type: config.rope_type,
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
    inner: usize,
    device: &B::Device,
) -> Result<FeedForward<B>, DitError> {
    let ff_inner = inner.saturating_mul(4);
    let ff_scope = block_scope.scope("ff");
    let linear_in = load_linear(&ff_scope.scope("net.0"), "proj", device)?;
    check_2d(
        &linear_in.weight.val(),
        [inner, ff_inner],
        "ff.net.0.proj.weight",
    )?;

    let linear_out = load_linear(&ff_scope.scope("net"), "2", device)?;
    check_2d(
        &linear_out.weight.val(),
        [ff_inner, inner],
        "ff.net.2.weight",
    )?;

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

    let attn_self = load_attention(block_scope, "attn1", None, config, device)?;
    let attn_cross = load_attention(
        block_scope,
        "attn2",
        Some(config.cross_attention_dim),
        config,
        device,
    )?;

    let ff = load_feed_forward(block_scope, inner, device)?;

    let scale_shift_table: Tensor<B, 2> = block_scope.tensor("scale_shift_table", device)?;
    check_2d(
        &scale_shift_table,
        [config.adaln_coeff(), inner],
        "scale_shift_table",
    )?;
    let scale_shift_table = Param::from_tensor(scale_shift_table);

    let prompt_scale_shift_table = if config.flags.cross_attention_adaln {
        let t: Tensor<B, 2> = block_scope.tensor("prompt_scale_shift_table", device)?;
        check_2d(&t, [2, inner], "prompt_scale_shift_table")?;
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
