//! Eager 3-D neighbourhood attention (NA3D) in pure Burn.
//!
//! ## Semantics
//!
//! Mirrors NATTEN's `na3d` window-shift behaviour: for each query position
//! `(ti, hi, wi)` the attention window is a `(kt × kh × kw)` block that
//! slides inward at grid boundaries, so every position has exactly
//! `kt × kh × kw` keys (no masking, no zero-padding).
//!
//! ## Memory
//!
//! This implementation materialises the full `[N, N]` score matrix where
//! `N = T × H × W`.  For the production decoder (N ≈ 70 k) that is
//! prohibitive; NATTEN / Triton kernels compute only the `N × K` window.
//! For CPU parity tests with N ≲ 100 the naive O(N²) approach is fine.

use burn::tensor::{Tensor, activation::softmax, backend::Backend};

/// One-axis window start with NATTEN inward-shift semantics.
///
/// Returns the start index of the `kernel`-wide window for query at `pos`
/// in a dimension of `length`.
#[must_use]
pub const fn window_start(pos: usize, length: usize, kernel: usize) -> usize {
    let kern = if kernel > length { length } else { kernel };
    let half = match kern.checked_div(2) {
        Some(v) => v,
        None => 0,
    };
    let raw = pos.saturating_sub(half);
    let max_start = length.saturating_sub(kern);
    if raw > max_start { max_start } else { raw }
}

/// Build an additive NA mask `[N, N]` where `N = time × height × width`.
///
/// `mask[i, j] = 0.0` when `j` is in the window of `i`, else `−∞`.
fn build_na_mask<B: Backend>(
    time: usize,
    height: usize,
    width: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let n = time.saturating_mul(height).saturating_mul(width);
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![neg_inf; n.saturating_mul(n)];

    for ti in 0..time {
        for hi in 0..height {
            for wi in 0..width {
                let i = ti
                    .saturating_mul(height)
                    .saturating_mul(width)
                    .saturating_add(hi.saturating_mul(width))
                    .saturating_add(wi);

                let t_start = window_start(ti, time, kt);
                let h_start = window_start(hi, height, kh);
                let w_start = window_start(wi, width, kw);
                let kern_t = kt.min(time);
                let kern_h = kh.min(height);
                let kern_w = kw.min(width);

                for dt in 0..kern_t {
                    for dh in 0..kern_h {
                        for dw in 0..kern_w {
                            let tj = t_start.saturating_add(dt);
                            let hj = h_start.saturating_add(dh);
                            let wj = w_start.saturating_add(dw);
                            let j = tj
                                .saturating_mul(height)
                                .saturating_mul(width)
                                .saturating_add(hj.saturating_mul(width))
                                .saturating_add(wj);
                            if i < n && j < n {
                                let idx = i.saturating_mul(n).saturating_add(j);
                                if let Some(slot) = data.get_mut(idx) {
                                    *slot = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::<B, 2>::from_floats(data.as_slice(), device).reshape([n, n])
}

/// Eager 3-D neighbourhood attention.
///
/// `q`, `k`, `v` have shape `[B, T, H, W, NH, HD]`.
/// Returns the weighted sum with the same shape.
///
/// ## Memory note
///
/// Materialises a full `[B·NH, N, N]` score matrix.  Use NATTEN for
/// production (`N ≈ 70 k`); this is only suitable for parity tests.
#[expect(
    clippy::many_single_char_names,
    reason = "q/k/v follow standard attention notation; b/t/h/w/n/nh/hd are standard dimension names"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
)]
pub fn na3d<B: Backend>(
    q: Tensor<B, 6>,
    k: Tensor<B, 6>,
    v: Tensor<B, 6>,
    kernel_size: [usize; 3],
    device: &B::Device,
) -> Tensor<B, 6> {
    let [b, t, h, w, nh, hd] = q.dims();
    let [kt, kh, kw] = kernel_size;
    let n = t.saturating_mul(h).saturating_mul(w);
    let bnh = b.saturating_mul(nh);

    // Build the additive NA mask `[N, N]`.
    let mask = build_na_mask::<B>(t, h, w, kt, kh, kw, device);

    // Permute to [B, NH, T, H, W, HD] then flatten to [B*NH, N, HD].
    let q_flat = q.permute([0, 4, 1, 2, 3, 5]).reshape([bnh, n, hd]);
    let k_flat = k.permute([0, 4, 1, 2, 3, 5]).reshape([bnh, n, hd]);
    let v_flat = v.permute([0, 4, 1, 2, 3, 5]).reshape([bnh, n, hd]);

    // Scores [B*NH, N, N] with additive NA mask.
    let scores = q_flat.matmul(k_flat.swap_dims(1, 2));
    let mask_broadcast = mask.unsqueeze_dim::<3>(0).expand([bnh, n, n]);
    let weights = softmax(scores + mask_broadcast, 2);

    // Weighted sum [B*NH, N, HD] → [B, NH, T, H, W, HD] → [B, T, H, W, NH, HD].
    let out = weights.matmul(v_flat);
    out.reshape([b, nh, t, h, w, hd])
        .permute([0, 2, 3, 4, 1, 5])
}
