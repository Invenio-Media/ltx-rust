//! Block-offloading strategy for the video DiT.
//!
//! The reference Python implementation (`OffloadMode`) can keep all N transformer
//! blocks resident on the GPU or upload only N blocks at a time, releasing the
//! others to host memory between blocks.
//!
//! # Current implementation
//!
//! On the **`ndarray` CPU backend** there is no host/device distinction, so
//! [`OffloadMode::Stream`] and [`OffloadMode::Resident`] produce the same peak
//! footprint and **identical numerical outputs** — verified by the parity test.
//!
//! On GPU backends (wgpu, cuda, metal) a future pass would implement actual
//! block migration by serialising each block to a host `Vec<f32>` record before
//! the forward step of that group and loading it back to the device.  The
//! interface is already designed for this: the model stores all blocks as
//! `Vec<TransformerBlock<B>>` and the forward loop receives the [`OffloadMode`].
//!
//! # Resident-bytes formula
//!
//! One `TransformerBlock` for the 22B config (`inner_dim = 4096`,
//! `cross_attention_dim = 4096`, `d_head = 128`, `H = 32`, no FFN bias):
//!
//! ```text
//! Self-attn:  to_q + to_k + to_v = 3 × (inner × inner) weights
//!             + 3 × inner biases
//!             q_norm + k_norm = 2 × inner weights
//!             to_out = inner × inner + inner
//! Cross-attn: to_q = inner²,  to_k + to_v = 2 × (cross × inner)
//!             + same norms and to_out
//! FFN:        linear_in = 4 × inner² (no bias), linear_out = 4 × inner²
//! scale_shift_table: adaln_coeff × inner (6 or 9)
//! ```
//!
//! The helper [`block_param_count`] computes this for any config.
//! Multiply by 4 bytes (f32) for the resident byte count per block.

use crate::config::DiTConfig;

/// Specifies how transformer block weights are managed during a forward pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OffloadMode {
    /// All blocks remain on device throughout the forward pass.
    #[default]
    Resident,
    /// At most `n_blocks` blocks are on the device at any time; the rest live
    /// in host memory and are uploaded just before they are needed.
    ///
    /// On the CPU (`ndarray`) backend this is a no-op; the blocks stay in RAM
    /// in both modes.  On GPU backends this trades latency for device memory.
    Stream {
        /// Number of blocks to keep resident simultaneously.
        n_blocks: usize,
    },
}

/// Convert `usize` to `u64`, saturating at `u64::MAX` on overflow.
///
/// `usize` can be 32 or 64 bits depending on the target.  This conversion is
/// always defined and never truncates.
fn u64_of(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// Number of f32 parameters in one transformer block for the given config.
///
/// Multiply by `4` for bytes.  The formula accounts for all weight matrices,
/// norms, bias vectors, and the `scale_shift_table`.
#[must_use]
pub fn block_param_count(c: &DiTConfig) -> u64 {
    let inner = u64_of(c.inner_dim());
    let cross = u64_of(c.cross_attention_dim);
    let adaln = u64_of(c.adaln_coeff());
    let heads = u64_of(c.num_attention_heads);

    // Self-attention
    let sa_qkv_w = 3u64.saturating_mul(inner.saturating_mul(inner));
    let sa_qkv_b = 3u64.saturating_mul(inner);
    let sa_qk_norm = 2u64.saturating_mul(inner);
    let sa_out = inner.saturating_mul(inner).saturating_add(inner);

    // Cross-attention (Q from inner, K/V from cross_attention_dim)
    let ca_q = inner.saturating_mul(inner).saturating_add(inner);
    let ca_kv_w = 2u64.saturating_mul(cross.saturating_mul(inner));
    let ca_kv_b = 2u64.saturating_mul(inner);
    let ca_qk_norm = 2u64.saturating_mul(inner);
    let ca_out = inner.saturating_mul(inner).saturating_add(inner);

    // Optional per-head gating (gate_proj weight + bias for both self and cross)
    let gate = if c.flags.apply_gated_attention {
        4u64.saturating_mul(inner.saturating_mul(heads))
            .saturating_add(4u64.saturating_mul(heads))
    } else {
        0
    };

    // FFN: inner → 4×inner → inner
    let ff_w = 8u64.saturating_mul(inner.saturating_mul(inner));
    let ff_b = if c.flags.ff_bias {
        5u64.saturating_mul(inner)
    } else {
        0
    };

    // AdaLN scale-shift table
    let sst = adaln.saturating_mul(inner);

    // Optional prompt AdaLN table (2 × inner)
    let prompt_sst = if c.flags.cross_attention_adaln {
        2u64.saturating_mul(inner)
    } else {
        0
    };

    sa_qkv_w
        .saturating_add(sa_qkv_b)
        .saturating_add(sa_qk_norm)
        .saturating_add(sa_out)
        .saturating_add(ca_q)
        .saturating_add(ca_kv_w)
        .saturating_add(ca_kv_b)
        .saturating_add(ca_qk_norm)
        .saturating_add(ca_out)
        .saturating_add(gate)
        .saturating_add(ff_w)
        .saturating_add(ff_b)
        .saturating_add(sst)
        .saturating_add(prompt_sst)
}

/// Number of f32 parameters in the model's non-block components
/// (patch embedding, AdaLN single, output norm and projection).
#[must_use]
pub fn head_tail_param_count(c: &DiTConfig) -> u64 {
    let inner = u64_of(c.inner_dim());
    let in_ch = u64_of(c.in_channels);
    let out_ch = u64_of(c.out_channels);
    let coeff = u64_of(c.adaln_coeff());

    // patchify_proj: in_channels → inner  (weight + bias)
    let patch = in_ch.saturating_mul(inner).saturating_add(inner);

    // adaln_single: linear_1(256→inner) + linear_2(inner→inner) + linear(inner→coeff×inner)
    let adaln_l1 = 256u64.saturating_mul(inner).saturating_add(inner);
    let adaln_l2 = inner.saturating_mul(inner).saturating_add(inner);
    let adaln_proj = coeff
        .saturating_mul(inner.saturating_mul(inner))
        .saturating_add(coeff.saturating_mul(inner));

    // prompt_adaln_single (optional)
    let prompt_adaln = if c.flags.cross_attention_adaln && c.flags.use_prompt_adaln_single {
        // same l1 + l2 + proj(inner→2×inner)
        adaln_l1
            .saturating_add(adaln_l2)
            .saturating_add(2u64.saturating_mul(inner.saturating_mul(inner)))
            .saturating_add(2u64.saturating_mul(inner))
    } else {
        0
    };

    // output: scale_shift_table(2×inner) + proj_out(inner→out_channels, bias)
    let out_sst = 2u64.saturating_mul(inner);
    let proj_out = inner.saturating_mul(out_ch).saturating_add(out_ch);

    // Optional keyframe embedding (1 × inner)
    let kfe = if c.flags.use_keyframes_abs_pos_embedding {
        inner
    } else {
        0
    };

    patch
        .saturating_add(adaln_l1)
        .saturating_add(adaln_l2)
        .saturating_add(adaln_proj)
        .saturating_add(prompt_adaln)
        .saturating_add(out_sst)
        .saturating_add(proj_out)
        .saturating_add(kfe)
}

/// Estimated peak device-resident bytes for f32 weights under a given mode.
///
/// This is a weight-only estimate; activations during the forward pass are
/// additional.
///
/// # Resident mode
///
/// All `num_layers` blocks + head/tail components.
///
/// # Stream mode
///
/// At most `n_blocks` transformer blocks + head/tail components.
#[must_use]
pub fn resident_bytes(c: &DiTConfig, mode: OffloadMode) -> u64 {
    let block_f32 = block_param_count(c);
    let ht_f32 = head_tail_param_count(c);

    let n_blocks_on_device = match mode {
        OffloadMode::Resident => u64_of(c.num_layers),
        OffloadMode::Stream { n_blocks } => u64_of(n_blocks).min(u64_of(c.num_layers)),
    };

    block_f32
        .saturating_mul(n_blocks_on_device)
        .saturating_add(ht_f32)
        // 4 bytes per f32
        .saturating_mul(4)
}
