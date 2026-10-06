//! Absolute `RoPE` utilities for the diffusion VAE NA transformer.
//!
//! The reference applies separate per-axis rotations to Q and K:
//! - Axis T: positions `0, 1, …, T-1`, applied to the first `d_t` head dims.
//! - Axis H: positions `0, 1, …, H-1`, applied to the next `d_h` dims.
//! - Axis W: positions `0, 1, …, W-1`, applied to the last `d_w` dims.
//!
//! Standard `RoPE` rotation formula per pair `(x_even, x_odd)`:
//! `x_even' = x_even * cos(pos * inv_freq) - x_odd * sin(pos * inv_freq)`,
//! `x_odd'  = x_even * sin(pos * inv_freq) + x_odd * cos(pos * inv_freq)`.

use burn::tensor::{Tensor, backend::Backend};

/// Pre-computed inverse frequencies for one `RoPE` axis.
///
/// `inv_f[i] = base^(−2i/dim)` for `i` in `0..dim/2`.
#[must_use]
pub fn inv_freqs(dim: usize, base: f64) -> Vec<f32> {
    let half = dim.checked_div(2).unwrap_or(0);
    (0..half)
        .map(|i| {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "freq index is small (< head_dim/2 ≤ 32); precision loss is negligible"
            )]
            let exp = (i as f64) * 2.0 / (dim as f64);
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "RoPE inverse frequencies fit f32 for model head dimensions"
            )]
            {
                base.powf(-exp) as f32
            }
        })
        .collect()
}

/// Rotate a chunk of head dimensions along one spatial axis.
///
/// `x` shape: `[B, T, H, W, NH, d_axis]` (channels-last Q or K tensor).
/// `pos_axis` is `1`=T, `2`=H, `3`=W.
/// `inv_f`: `Vec<f32>` with length `d_axis / 2`.
///
/// Returns a new tensor of the same shape with rotations applied.
#[expect(
    clippy::indexing_slicing,
    reason = "bounds are verified by construction"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
)]
pub fn rot_abs_axis<B: Backend>(
    x: Tensor<B, 6>,
    pos_axis: usize,
    inv_f: &[f32],
    device: &B::Device,
) -> Tensor<B, 6> {
    let all_dims = x.dims();
    let axis_len = all_dims[pos_axis];
    let head_size = all_dims[5];
    let d_half = head_size.checked_div(2).unwrap_or(0);
    let [d0, d1, d2, d3, d4, _d5] = all_dims;
    // n = total elements across all non-head dims
    let n = d0
        .saturating_mul(d1)
        .saturating_mul(d2)
        .saturating_mul(d3)
        .saturating_mul(d4);

    // positions: [axis_len]
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "spatial dim index; values are ≤ spatial resolution"
    )]
    let pos: Vec<f32> = (0..axis_len).map(|i| i as f32).collect();
    let pos_t = Tensor::<B, 1>::from_floats(pos.as_slice(), device);
    let inv_t = Tensor::<B, 1>::from_floats(inv_f, device);

    // angles: [axis_len, d_half]
    let ang = pos_t
        .unsqueeze_dim::<2>(1)
        .matmul(inv_t.unsqueeze_dim::<2>(0));

    // Broadcast shape: axis_len on `pos_axis`, d_half on dim 5.
    let mut bcast_shape = [1usize; 6];
    bcast_shape[pos_axis] = axis_len;
    bcast_shape[5] = d_half;

    let cos_ang = ang.clone().reshape(bcast_shape).cos();
    let sin_ang = ang.reshape(bcast_shape).sin();

    // Extract even/odd pairs via 3-D reshape (avoids NdArray's 6-D limit).
    // x: [d0,d1,d2,d3,d4,head_size] → flat → [n, d_half, 2] (3-D)
    let x_pairs: Tensor<B, 3> = x.reshape([n, d_half, 2]);
    let xe: Tensor<B, 6> = x_pairs
        .clone()
        .slice([0..n, 0..d_half, 0..1])
        .reshape([d0, d1, d2, d3, d4, d_half]);
    let xo: Tensor<B, 6> = x_pairs
        .slice([0..n, 0..d_half, 1..2])
        .reshape([d0, d1, d2, d3, d4, d_half]);

    // Apply rotation (6-D broadcast; cos/sin broadcast over spatial dims).
    let re: Tensor<B, 6> = xe.clone() * cos_ang.clone() - xo.clone() * sin_ang.clone();
    let ro: Tensor<B, 6> = xe * sin_ang + xo * cos_ang;

    // Interleave via 3-D cat (avoids Tensor::stack::<7> which exceeds NdArray 6-D limit).
    // re, ro: [d0,d1,d2,d3,d4,d_half] → each → [n, d_half, 1] → cat → [n, d_half, 2]
    let re_3: Tensor<B, 3> = re.reshape([n, d_half, 1]);
    let ro_3: Tensor<B, 3> = ro.reshape([n, d_half, 1]);
    Tensor::cat(vec![re_3, ro_3], 2).reshape([d0, d1, d2, d3, d4, head_size])
}

/// Apply full-volume absolute `RoPE` to a `[B, T, H, W, NH, HD]` tensor.
///
/// `rope_dim_split = [d_t, d_h, d_w]` must sum to `HD`.
/// Returns the rotated tensor with the same shape.
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
#[expect(
    clippy::similar_names,
    reason = "d_t, d_h, d_w are canonical RoPE dim names"
)]
pub fn apply_rope<B: Backend>(
    x: Tensor<B, 6>,
    rope_dim_split: [usize; 3],
    inv_t: &[f32],
    inv_h: &[f32],
    inv_w: &[f32],
    device: &B::Device,
) -> Tensor<B, 6> {
    let [b, t, h, w, nh, hd] = x.dims();
    let [d_t, d_h, _d_w] = rope_dim_split;
    let d_t_end = d_t;
    let d_h_end = d_t.saturating_add(d_h);

    // Split HD into three axis-chunks.
    let xt = x.clone().slice([0..b, 0..t, 0..h, 0..w, 0..nh, 0..d_t_end]);
    let xh = x
        .clone()
        .slice([0..b, 0..t, 0..h, 0..w, 0..nh, d_t_end..d_h_end]);
    let xw = x.slice([0..b, 0..t, 0..h, 0..w, 0..nh, d_h_end..hd]);

    // Rotate each chunk on its own axis.
    let xt = rot_abs_axis(xt, 1, inv_t, device);
    let xh = rot_abs_axis(xh, 2, inv_h, device);
    let xw = rot_abs_axis(xw, 3, inv_w, device);

    Tensor::cat(vec![xt, xh, xw], 5)
}
