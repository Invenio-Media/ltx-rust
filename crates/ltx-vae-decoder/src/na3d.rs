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
//! Each query attends only to its `NK = kt × kh × kw` neighbours.
//! Peak device memory is `O(B · NH · q_chunk · NK)` where `q_chunk`
//! is the largest group of queries sharing the same window start along W
//! (at most `kw / 2 + 1`).  Queries are processed row-by-row in `T × H`
//! host iterations; no `N × N` score matrix is materialised.

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

/// For W-axis: find the end of the contiguous run of `wi` values that share
/// the same `window_start`.  Returns `wi_end` (exclusive).
#[must_use]
const fn w_group_end(wi_start: usize, w: usize, kw: usize) -> usize {
    let ws = window_start(wi_start, w, kw);
    let mut end = wi_start.saturating_add(1);
    while end < w && window_start(end, w, kw) == ws {
        end = end.saturating_add(1);
    }
    end
}

/// Eager 3-D neighbourhood attention.
///
/// `q`, `k`, `v` have shape `[B, T, H, W, NH, HD]`.
/// Returns the weighted sum with the same shape.
///
/// ## Algorithm
///
/// Queries are processed in `T × H` row iterations.  Within each row all
/// `W` queries are grouped by their W-axis window start (at most `kw` distinct
/// groups).  Each group shares the same `NK = kt × kh × kw` key/value
/// window, so a single batched matrix multiply covers the group.  The output
/// is assembled by `Tensor::cat` along the W and then T/H axes without
/// materialising any `N × N` matrix.
#[expect(
    clippy::many_single_char_names,
    reason = "q/k/v/b/t/h/w/nh/hd are standard attention/tensor-dim names"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "q/k/v are consumed via repeated .clone() + .slice() throughout the loop"
)]
pub fn na3d<B: Backend>(
    q: Tensor<B, 6>,
    k: Tensor<B, 6>,
    v: Tensor<B, 6>,
    kernel_size: [usize; 3],
    device: &B::Device,
) -> Tensor<B, 6> {
    // device is kept for API stability; no host tensor construction needed.
    let _ = device;

    let [b, t, h, w, nh, hd] = q.dims();
    let [kt, kh, kw] = kernel_size;
    let eff_kt = kt.min(t);
    // Rename to avoid "similar binding" lint with kh/kw.
    let eff_kernel_h = kh.min(h);
    let eff_kernel_w = kw.min(w);

    // Outer loop: one T-slice per iteration, catted along T at the end.
    let mut t_slices: Vec<Tensor<B, 6>> = Vec::with_capacity(t);

    for ti in 0..t {
        let ts = window_start(ti, t, kt);

        // Inner loop: one H-row per iteration, catted along H within this T-slice.
        let mut h_rows: Vec<Tensor<B, 5>> = Vec::with_capacity(h);

        for hi in 0..h {
            let hs = window_start(hi, h, kh);

            // Q for this (ti, hi) row: [B, W, NH, HD].
            let q_row: Tensor<B, 4> = q
                .clone()
                .slice([
                    0..b,
                    ti..ti.saturating_add(1),
                    hi..hi.saturating_add(1),
                    0..w,
                    0..nh,
                    0..hd,
                ])
                .reshape([b, w, nh, hd]);

            // K/V TH block: [B, eff_kt, eff_kernel_h, W, NH, HD].
            let k_th: Tensor<B, 6> = k.clone().slice([
                0..b,
                ts..ts.saturating_add(eff_kt),
                hs..hs.saturating_add(eff_kernel_h),
                0..w,
                0..nh,
                0..hd,
            ]);
            let v_th: Tensor<B, 6> = v.clone().slice([
                0..b,
                ts..ts.saturating_add(eff_kt),
                hs..hs.saturating_add(eff_kernel_h),
                0..w,
                0..nh,
                0..hd,
            ]);

            // Process W queries in window-groups (contiguous runs sharing the same ws).
            let mut w_groups: Vec<Tensor<B, 4>> = Vec::new();
            let mut wi = 0_usize;
            while wi < w {
                let ws = window_start(wi, w, kw);
                let wi_end = w_group_end(wi, w, kw);
                let n_wi = wi_end.saturating_sub(wi);

                // Q group: [B, n_wi, NH, HD] → permute → [B, NH, n_wi, HD].
                let q_g: Tensor<B, 3> = q_row
                    .clone()
                    .slice([0..b, wi..wi_end, 0..nh, 0..hd])
                    .reshape([b, n_wi, nh.saturating_mul(hd)]);
                // Permute to [B, NH, n_wi, HD] by going via flat dims.
                // q_g is [B, n_wi, NH*HD]; reshape to [B, n_wi, NH, HD] then permute.
                let q_g: Tensor<B, 4> = q_g.reshape([b, n_wi, nh, hd]).swap_dims(1, 2); // [B, NH, n_wi, HD]

                // K/V window: [B, eff_kt, eff_kernel_h, eff_kernel_w, NH, HD].
                let nk = eff_kt
                    .saturating_mul(eff_kernel_h)
                    .saturating_mul(eff_kernel_w);
                let k_win: Tensor<B, 6> = k_th.clone().slice([
                    0..b,
                    0..eff_kt,
                    0..eff_kernel_h,
                    ws..ws.saturating_add(eff_kernel_w),
                    0..nh,
                    0..hd,
                ]);
                let v_win: Tensor<B, 6> = v_th.clone().slice([
                    0..b,
                    0..eff_kt,
                    0..eff_kernel_h,
                    ws..ws.saturating_add(eff_kernel_w),
                    0..nh,
                    0..hd,
                ]);

                // Flatten to [B, NH, NK, HD].
                // k_win is [B, kt, kh, kw, NH, HD]; permute to [B, NH, kt, kh, kw, HD] → [B, NH, NK, HD].
                let k_flat: Tensor<B, 4> =
                    k_win.permute([0, 4, 1, 2, 3, 5]).reshape([b, nh, nk, hd]);
                let v_flat: Tensor<B, 4> =
                    v_win.permute([0, 4, 1, 2, 3, 5]).reshape([b, nh, nk, hd]);

                // Scores [B, NH, n_wi, NK] = Q @ K^T.
                let scores: Tensor<B, 4> = q_g.matmul(k_flat.swap_dims(2, 3));
                let weights: Tensor<B, 4> = softmax(scores, 3);

                // Output [B, NH, n_wi, HD] → swap → [B, n_wi, NH, HD].
                let out: Tensor<B, 4> = weights.matmul(v_flat).swap_dims(1, 2);

                w_groups.push(out); // [B, n_wi, NH, HD]
                wi = wi_end;
            }

            // Cat W groups → [B, W, NH, HD], then unsqueeze H dim.
            let row_out: Tensor<B, 4> = Tensor::cat(w_groups, 1); // [B, W, NH, HD]
            // Insert H dim: [B, 1, W, NH, HD]
            h_rows.push(row_out.unsqueeze_dim(1));
        }

        // Cat H rows → [B, H, W, NH, HD], then unsqueeze T dim.
        let t_row: Tensor<B, 5> = Tensor::cat(h_rows, 1); // [B, H, W, NH, HD]
        // Insert T dim: [B, 1, H, W, NH, HD]
        t_slices.push(t_row.unsqueeze_dim(1));
    }

    // Cat T slices → [B, T, H, W, NH, HD].
    Tensor::cat(t_slices, 1)
}
