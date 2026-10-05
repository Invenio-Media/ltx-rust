//! 3-D Rotary Position Embeddings for the LTX-2.5 video DiT.
//!
//! Two variants match the reference (`rope.py`):
//! - **Split** ([`RopeType::Split`], default): each head's `d_head`-dimensional
//!   vector is treated as two halves; the first half and second half rotate
//!   together as a complex pair.  This is the current production variant.
//! - **Interleaved** ([`RopeType::Interleaved`], legacy): adjacent pairs
//!   `(x[0], x[1])`, `(x[2], x[3])`, … each rotate as a complex pair.
//!
//! The frequency grid follows `generate_freq_grid_pytorch` in the reference:
//! `freq[i] = theta^(i / (grid_size - 1)) × π/2`, which is a geometric series
//! from `π/2` (at `i=0`) to `theta × π/2` (at `i=grid_size-1`).
//!
//! Positions arrive as `(B, n_pos_dims, T, 2)` with `use_middle_indices_grid=true`
//! (the default); the last axis holds `[start, end)` index bounds and RoPE is
//! evaluated at the midpoint `(start + end) / 2`.

use std::f64::consts::PI;

use burn::prelude::*;

use crate::config::RopeType;

/// Precompute RoPE cosine / sine tensors for one forward pass.
///
/// # Arguments
/// - `positions`: patch position bounds `(B, n_pos_dims, T, 2)` where the
///   last axis is `[start, end)`.  `n_pos_dims = 3` for video (t, h, w).
/// - `inner_dim`: model's hidden dimension (`H × d_head`).
/// - `theta`: RoPE base period (default 10 000).
/// - `max_pos`: maximum position in each dimension `[t_max, h_max, w_max]`.
/// - `num_heads`: number of attention heads.
/// - `rope_type`: `Split` or `Interleaved`.
/// - `device`: target device.
///
/// # Returns
///
/// `(cos, sin)` each of shape:
/// - **Split**: `(B, H, T, d_head/2)`
/// - **Interleaved**: `(B, 1, T, inner_dim)` (head dim is 1 for broadcasting)
#[allow(clippy::too_many_arguments)]
pub fn precompute_freqs_cis<B: Backend>(
    positions: Tensor<B, 4>,
    inner_dim: usize,
    theta: f64,
    max_pos: &[usize; 3],
    num_heads: usize,
    rope_type: RopeType,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, n_pos, n_tokens, _two] = positions.dims();

    // --- frequency grid -------------------------------------------------------
    // n_elem = 2 × n_pos_dims.
    let n_elem = n_pos.saturating_mul(2);
    let grid_size = inner_dim.saturating_div(n_elem);

    let freq_vec = make_freq_grid(theta, grid_size);
    let freq_1d = Tensor::<B, 1>::from_floats(freq_vec.as_slice(), device); // (grid_size,)

    // --- midpoint positions ---------------------------------------------------
    let pos_start = positions
        .clone()
        .narrow(3, 0, 1)
        .reshape([batch, n_pos, n_tokens]);
    let pos_end = positions.narrow(3, 1, 1).reshape([batch, n_pos, n_tokens]);

    // max_pos tensor (1, n_pos, 1) for broadcasting.
    let max_pos_vec: Vec<f32> = max_pos
        .iter()
        .map(|&m| f32::from(u16::try_from(m).unwrap_or(u16::MAX)))
        .collect();
    let max_pos_t =
        Tensor::<B, 1>::from_floats(max_pos_vec.as_slice(), device).reshape([1, n_pos, 1]);

    // Fractional midpoint in [0, 1].
    let frac = (pos_start + pos_end).div_scalar(2.0_f32) / max_pos_t;

    // Map to [-1, 1].
    let frac_mapped = frac.mul_scalar(2.0_f32).add_scalar(-1.0_f32);

    // --- multiply by frequency grid ------------------------------------------
    // (B, n_pos, T) → (B, T, n_pos, 1)
    let frac_t = frac_mapped
        .swap_dims(1, 2)
        .reshape([batch, n_tokens, n_pos, 1]);

    // freq_1d: (grid_size,) → (1, 1, 1, grid_size)
    let freq_4d = freq_1d
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);

    // (B, T, n_pos, 1) × (1, 1, 1, grid_size) → (B, T, n_pos, grid_size)
    let freqs = frac_t * freq_4d;

    // (B, T, n_pos, grid_size) → (B, T, grid_size, n_pos) → (B, T, grid*n_pos)
    let freqs = freqs.swap_dims(2, 3).flatten::<3>(2, 3);

    match rope_type {
        RopeType::Split => split_freqs_cis(freqs, inner_dim, num_heads, batch, n_tokens, device),
        RopeType::Interleaved => interleaved_freqs_cis(freqs, inner_dim, batch, n_tokens, device),
    }
}

/// Build `(cos, sin)` for the **Split** RoPE variant.
///
/// Returns `(cos, sin)` each of shape `(B, H, T, d_head/2)`.
fn split_freqs_cis<B: Backend>(
    freqs: Tensor<B, 3>, // (B, T, grid_size*n_pos)
    inner_dim: usize,
    num_heads: usize,
    batch: usize,
    n_tokens: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let current_freqs = freqs.dims()[2];
    let d_half = inner_dim.saturating_div(2);
    let pad_size = d_half.saturating_sub(current_freqs);

    let cos_raw = freqs.clone().cos();
    let sin_raw = freqs.sin();

    // Pad at the start with 1s (cos) and 0s (sin) if needed.
    let (cos_padded, sin_padded) = if pad_size > 0 {
        let cos_pad = Tensor::<B, 3>::ones([batch, n_tokens, pad_size], device);
        let sin_pad = Tensor::<B, 3>::zeros([batch, n_tokens, pad_size], device);
        (
            Tensor::cat(vec![cos_pad, cos_raw], 2),
            Tensor::cat(vec![sin_pad, sin_raw], 2),
        )
    } else {
        (cos_raw, sin_raw)
    };

    // (B, T, d_half) → (B, T, H, d_head/2) → swap → (B, H, T, d_head/2)
    let d_head_half = d_half.saturating_div(num_heads);
    let cos_4d = cos_padded
        .reshape([batch, n_tokens, num_heads, d_head_half])
        .swap_dims(1, 2);
    let sin_4d = sin_padded
        .reshape([batch, n_tokens, num_heads, d_head_half])
        .swap_dims(1, 2);
    (cos_4d, sin_4d)
}

/// Build `(cos, sin)` for the **Interleaved** RoPE variant (legacy).
///
/// Returns `(cos, sin)` each of shape `(B, 1, T, inner_dim)`.
fn interleaved_freqs_cis<B: Backend>(
    freqs: Tensor<B, 3>,
    inner_dim: usize,
    batch: usize,
    n_tokens: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let cos_base = freqs.clone().cos();
    let sin_base = freqs.sin();
    let base_dim = cos_base.dims()[2];

    // Expand each angle to the adjacent feature pair: [a, b] -> [a, a, b, b].
    let cos_rep = Tensor::cat(
        vec![
            cos_base.clone().reshape([batch, n_tokens, base_dim, 1]),
            cos_base.reshape([batch, n_tokens, base_dim, 1]),
        ],
        3,
    )
    .reshape([batch, n_tokens, base_dim.saturating_mul(2)]);
    let sin_rep = Tensor::cat(
        vec![
            sin_base.clone().reshape([batch, n_tokens, base_dim, 1]),
            sin_base.reshape([batch, n_tokens, base_dim, 1]),
        ],
        3,
    )
    .reshape([batch, n_tokens, base_dim.saturating_mul(2)]);

    let current = cos_rep.dims()[2];
    let pad_size = inner_dim.saturating_sub(current);

    let (cos_full, sin_full) = if pad_size > 0 {
        let cos_pad = Tensor::<B, 3>::ones([batch, n_tokens, pad_size], device);
        let sin_pad = Tensor::<B, 3>::zeros([batch, n_tokens, pad_size], device);
        (
            Tensor::cat(vec![cos_pad, cos_rep], 2),
            Tensor::cat(vec![sin_pad, sin_rep], 2),
        )
    } else {
        (cos_rep, sin_rep)
    };

    (
        cos_full.reshape([batch, 1, n_tokens, inner_dim]),
        sin_full.reshape([batch, 1, n_tokens, inner_dim]),
    )
}

/// Apply the **Split** RoPE rotation to query or key.
///
/// - `x`: `(B, T, H × d_head)` — the raw Q or K before splitting into heads.
/// - `cos` / `sin`: `(B, H, T, d_head/2)` from [`precompute_freqs_cis`].
///
/// Returns the rotated tensor with the same shape as `x`.
pub fn apply_split_rope<B: Backend>(
    x: Tensor<B, 3>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    heads: usize,
) -> Tensor<B, 3> {
    let [batch, n_tokens, inner] = x.dims();
    let d_head = inner.saturating_div(heads);
    let half = d_head.saturating_div(2);

    // Reshape to (B, H, T, d_head).
    let x4 = x.reshape([batch, n_tokens, heads, d_head]).swap_dims(1, 2);

    // Split last dim into two halves.
    let x_first = x4.clone().narrow(3, 0, half);
    let x_second = x4.narrow(3, half, half);

    // Rotate: complex multiplication in split-half form.
    // out_first  = cos * x_first  - sin * x_second
    // out_second = sin * x_first  + cos * x_second
    let out_first = cos.clone() * x_first.clone() - sin.clone() * x_second.clone();
    let out_second = sin * x_first + cos * x_second;

    Tensor::cat(vec![out_first, out_second], 3)
        .swap_dims(1, 2)
        .reshape([batch, n_tokens, inner])
}

/// Apply the **Interleaved** RoPE rotation to query or key (legacy).
///
/// - `x`: `(B, T, H × d_head)`.
/// - `cos` / `sin`: `(B, 1, T, inner_dim)` from [`precompute_freqs_cis`].
///
/// Returns the rotated tensor with the same shape as `x`.
pub fn apply_interleaved_rope<B: Backend>(
    x: Tensor<B, 3>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 3> {
    let [batch, n_tokens, inner] = x.dims();
    let cos_3d = cos.reshape([batch, n_tokens, inner]);
    let sin_3d = sin.reshape([batch, n_tokens, inner]);

    // Interleaved rotation: pair (x[2i], x[2i+1]) rotates together.
    let half = inner.saturating_div(2);
    let x_pairs = x.clone().reshape([batch, n_tokens, half, 2]);
    let x0 = x_pairs
        .clone()
        .narrow(3, 0, 1)
        .reshape([batch, n_tokens, half]);
    let x1 = x_pairs.narrow(3, 1, 1).reshape([batch, n_tokens, half]);

    // Negated rotation pair: [-x1, x0] interleaved → (B, T, inner)
    let neg_x1 = x1.neg().reshape([batch, n_tokens, half, 1]);
    let x0_r = x0.reshape([batch, n_tokens, half, 1]);
    let x_rot = Tensor::cat(vec![neg_x1, x0_r], 3).reshape([batch, n_tokens, inner]);

    x * cos_3d + x_rot * sin_3d
}

// ---------------------------------------------------------------------------
// Frequency grid (pure Rust, no Burn tensor)
// ---------------------------------------------------------------------------

/// Compute the frequency grid: `freq[i] = theta^(i/(n-1)) × π/2`.
///
/// This matches `generate_freq_grid_pytorch` in the reference.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects,
    reason = "loop index i in 0..grid_size ≤ 682; grid_size-1 ≤ 681; \
              f64→f32 for the grid output is intentional (sufficient precision); \
              usize→f64 is lossless for these small values"
)]
fn make_freq_grid(theta: f64, grid_size: usize) -> Vec<f32> {
    if grid_size == 0 {
        return vec![];
    }
    if grid_size == 1 {
        return vec![(theta * PI / 2.0) as f32];
    }
    let denom = (grid_size - 1) as f64;
    (0..grid_size)
        .map(|i| (theta.powf(i as f64 / denom) * PI / 2.0) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Device;

    type B = NdArray<f32>;

    #[test]
    fn interleaved_freqs_repeat_each_angle_for_adjacent_pair() {
        let device = Device::<B>::default();
        let freqs =
            Tensor::<B, 1>::from_floats([0.0_f32, 1.0_f32].as_slice(), &device).reshape([1, 1, 2]);
        let (cos, sin) = interleaved_freqs_cis(freqs, 4, 1, 1, &device);

        let cos_vals: Vec<f32> = cos.into_data().to_vec().unwrap();
        let sin_vals: Vec<f32> = sin.into_data().to_vec().unwrap();
        let expected_cos = [1.0_f32, 1.0, 1.0_f32.cos(), 1.0_f32.cos()];
        let expected_sin = [0.0_f32, 0.0, 1.0_f32.sin(), 1.0_f32.sin()];

        for (got, expected) in cos_vals.iter().zip(expected_cos) {
            assert!((*got - expected).abs() < 1e-6_f32);
        }
        for (got, expected) in sin_vals.iter().zip(expected_sin) {
            assert!((*got - expected).abs() < 1e-6_f32);
        }
    }
}
