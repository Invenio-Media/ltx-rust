//! Decode-window geometry for the diffusion VAE decoder.
//!
//! Provides:
//! - [`stage_min_tile_size`]: per-axis latent floor so every NA kernel fits.
//! - [`pad_trailing_latent`]: NATTEN last-frame border ghost-pad.
//! - [`crop_trailing_context`]: remove ghost-pad appendix before stage 5.
//! - [`ensure_min_latent`]: symmetric padding to reach the floor size.
//! - [`decode_window_pixels`]: report the decode-tile pixel extent for budget.
//! - [`stage4_thw`]: compute stage-4 shape from a latent shape.

use crate::config::UpsampleSpec;
use crate::error::VaeDecoderError;
use burn::tensor::{Tensor, backend::Backend};

/// Per-axis latent-grid floor `[t, h, w]` so each NA stage sees dims ≥ kernel.
///
/// Mirrors `diffusion_tiling.all_stages_min_tile_size`.
#[expect(
    clippy::indexing_slicing,
    reason = "bounds are verified by construction"
)]
#[must_use]
pub fn stage_min_tile_size(
    stage_kernels: &[[usize; 3]],
    upsamples: &[UpsampleSpec],
    stage5_kernel: [usize; 3],
) -> [usize; 3] {
    // Cumulative upsample strides: `strides[i]` is the product after i hops.
    let mut strides: Vec<[usize; 3]> = vec![[1usize; 3]];
    let mut cur = [1usize, 1, 1];
    for up in upsamples {
        cur[0] = cur[0].saturating_mul(up.stride[0]);
        cur[1] = cur[1].saturating_mul(up.stride[1]);
        cur[2] = cur[2].saturating_mul(up.stride[2]);
        strides.push(cur);
    }

    let mut mins = [1usize; 3];
    for (stage_i, kern) in stage_kernels.iter().take(upsamples.len()).enumerate() {
        let s = strides.get(stage_i).copied().unwrap_or([1; 3]);
        for axis in 0..3 {
            let needed = ceil_div(kern[axis], s[axis]);
            if needed > mins[axis] {
                mins[axis] = needed;
            }
        }
    }
    let s5 = strides.get(upsamples.len()).copied().unwrap_or([1; 3]);
    for axis in 0..3 {
        let needed = ceil_div(stage5_kernel[axis], s5[axis]);
        if needed > mins[axis] {
            mins[axis] = needed;
        }
    }
    mins
}

const fn ceil_div(a: usize, b: usize) -> usize {
    if b == 0 {
        return a;
    }
    let numerator = a.saturating_add(b).saturating_sub(1);
    match numerator.checked_div(b) {
        Some(v) => v,
        None => 0,
    }
}

/// Replicate the last latent frame `n_frames` times for NATTEN ghost-pad.
///
/// Input is channels-first `[B, C, T, H, W]`.
#[must_use]
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
pub fn pad_trailing_latent<B: Backend>(latent: Tensor<B, 5>, n_frames: usize) -> Tensor<B, 5> {
    if n_frames == 0 {
        return latent;
    }
    let [b, c, t, h, w] = latent.dims();
    let last = latent
        .clone()
        .slice([0..b, 0..c, t.saturating_sub(1)..t, 0..h, 0..w]);
    let tail = last.expand([b, c, n_frames, h, w]);
    Tensor::cat(vec![latent, tail], 2)
}

/// Crop the ghosting appendix from a channels-last `[B, T, H, W, C]` context.
///
/// Mirrors `diffusion_tiling.crop_trailing_context_natten_pad`.
#[must_use]
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
pub fn crop_trailing_context<B: Backend>(
    context: Tensor<B, 5>,
    n_latent_frames: usize,
    time_scale: usize,
    stage5_kernel_t: usize,
) -> Tensor<B, 5> {
    if n_latent_frames == 0 {
        return context;
    }
    let [b, t, h, w, c] = context.dims();
    let ghost = n_latent_frames.saturating_mul(time_scale);
    let content_t = t.saturating_sub(ghost).max(1);
    let keep = t.min(content_t.max(stage5_kernel_t));
    if keep >= t {
        return context;
    }
    context.slice([0..b, 0..keep, 0..h, 0..w, 0..c])
}

/// Pad a channels-first `[B, C, T, H, W]` latent symmetrically to `min_sizes`.
///
/// Temporal: trailing `repeat_last` pad.
/// Spatial: symmetric edge-replicate pad (odd remainder goes to the end).
///
/// Returns `(padded_latent, [[t_before, t_after], [h_before, h_after], [w_before, w_after]])`.
#[must_use]
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
pub fn ensure_min_latent<B: Backend>(
    latent: Tensor<B, 5>,
    min_sizes: [usize; 3],
) -> (Tensor<B, 5>, [[usize; 2]; 3]) {
    let [b, c, t, h, w] = latent.dims();
    let mut out = latent;
    let mut pads = [[0usize; 2]; 3];

    if t < min_sizes[0] {
        let need = min_sizes[0].saturating_sub(t);
        let [bo, co, to, ho, wo] = out.dims();
        let last = out
            .clone()
            .slice([0..bo, 0..co, to.saturating_sub(1)..to, 0..ho, 0..wo]);
        let tail = last.expand([bo, co, need, ho, wo]);
        out = Tensor::cat(vec![out, tail], 2);
        pads[0] = [0, need];
    }
    if h < min_sizes[1] {
        let need = min_sizes[1].saturating_sub(h);
        out = sym_pad_height(out, h, min_sizes[1]);
        pads[1] = [
            need.checked_div(2).unwrap_or(0),
            need.saturating_sub(need.checked_div(2).unwrap_or(0)),
        ];
    }
    if w < min_sizes[2] {
        let need = min_sizes[2].saturating_sub(w);
        out = sym_pad_width(out, w, min_sizes[2]);
        pads[2] = [
            need.checked_div(2).unwrap_or(0),
            need.saturating_sub(need.checked_div(2).unwrap_or(0)),
        ];
    }

    let _ = (b, c, t, h, w);
    (out, pads)
}

#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
fn sym_pad_height<B: Backend>(x: Tensor<B, 5>, cur_h: usize, target_h: usize) -> Tensor<B, 5> {
    let need = target_h.saturating_sub(cur_h);
    if need == 0 {
        return x;
    }
    let before = need.checked_div(2).unwrap_or(0);
    let after = need.saturating_sub(before);
    let [b, c, t, h, w] = x.dims();
    let first = x.clone().slice([0..b, 0..c, 0..t, 0..1, 0..w]);
    let last = x
        .clone()
        .slice([0..b, 0..c, 0..t, h.saturating_sub(1)..h, 0..w]);
    let mut parts = Vec::with_capacity(3);
    if before > 0 {
        parts.push(first.expand([b, c, t, before, w]));
    }
    parts.push(x);
    if after > 0 {
        parts.push(last.expand([b, c, t, after, w]));
    }
    Tensor::cat(parts, 3)
}

#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
fn sym_pad_width<B: Backend>(x: Tensor<B, 5>, cur_w: usize, target_w: usize) -> Tensor<B, 5> {
    let need = target_w.saturating_sub(cur_w);
    if need == 0 {
        return x;
    }
    let before = need.checked_div(2).unwrap_or(0);
    let after = need.saturating_sub(before);
    let [b, c, t, h, w] = x.dims();
    let first = x.clone().slice([0..b, 0..c, 0..t, 0..h, 0..1]);
    let last = x
        .clone()
        .slice([0..b, 0..c, 0..t, 0..h, w.saturating_sub(1)..w]);
    let mut parts = Vec::with_capacity(3);
    if before > 0 {
        parts.push(first.expand([b, c, t, h, before]));
    }
    parts.push(x);
    if after > 0 {
        parts.push(last.expand([b, c, t, h, after]));
    }
    Tensor::cat(parts, 4)
}

/// Crop channels-first pixels `[B, C, F, H, W]` back to content dimensions.
///
/// Temporal crop removes trailing frames; spatial crop removes symmetric pad.
#[must_use]
pub fn crop_pixels_to_content<B: Backend>(
    pixels: Tensor<B, 5>,
    frames: usize,
    height: usize,
    width: usize,
    h_pad: [usize; 2],
    w_pad: [usize; 2],
) -> Tensor<B, 5> {
    let [b, c, _f, h_in, w_in] = pixels.dims();
    let px = pixels.slice([0..b, 0..c, 0..frames, 0..h_in, 0..w_in]);
    let h_start = h_pad[0];
    let h_keep = height.min(h_in.saturating_sub(h_start));
    let px = px.slice([
        0..b,
        0..c,
        0..frames,
        h_start..h_start.saturating_add(h_keep),
        0..w_in,
    ]);
    let w_start = w_pad[0];
    let w_keep = width.min(w_in.saturating_sub(w_start));
    px.slice([
        0..b,
        0..c,
        0..frames,
        0..h_keep,
        w_start..w_start.saturating_add(w_keep),
    ])
}

/// Stage-4 `[T, H, W]` after the first three upsample hops from a latent.
///
/// Mirrors `diffusion_tiling.stage4_thw_from_latent`.
#[must_use]
pub fn stage4_thw(
    upsamples: &[UpsampleSpec],
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
) -> [usize; 3] {
    let mut t = latent_t;
    let mut h = latent_h;
    let mut w = latent_w;
    for up in upsamples.iter().take(3) {
        t = t.saturating_mul(up.stride[0]);
        h = h.saturating_mul(up.stride[1]);
        w = w.saturating_mul(up.stride[2]);
        if up.stride[0] == 2 {
            t = t.saturating_sub(1);
        }
    }
    [t, h, w]
}

/// Report the pixel canvas `[frames, height, width]` of one decode tile.
///
/// This is consumed by `ltx-budget` for memory estimation.
///
/// # Errors
///
/// Returns [`VaeDecoderError::InvalidArgument`] when any dimension is zero.
pub fn decode_window_pixels(
    stage4_t: usize,
    stage4_h: usize,
    stage4_w: usize,
    upsample4_stride: [usize; 3],
    patch_size: usize,
    drop_leading_frame: bool,
) -> Result<[usize; 3], VaeDecoderError> {
    let [st, sh, sw] = upsample4_stride;
    let frames_raw = stage4_t.saturating_mul(st);
    let frames = if st == 2 && drop_leading_frame {
        frames_raw.saturating_sub(1)
    } else {
        frames_raw
    };
    let height = stage4_h.saturating_mul(sh).saturating_mul(patch_size);
    let width = stage4_w.saturating_mul(sw).saturating_mul(patch_size);
    if frames == 0 || height == 0 || width == 0 {
        return Err(VaeDecoderError::InvalidArgument {
            detail: format!(
                "decode window is degenerate: frames={frames} height={height} width={width}"
            ),
        });
    }
    Ok([frames, height, width])
}
