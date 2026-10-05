//! Video latent patchifier (patch size 1) and pixel-coordinate grid.
//!
//! Ports `VideoLatentPatchifier` and `get_pixel_coords` from
//! `ltx_core.components.patchifiers`.  The patch size is always 1 (the only
//! size used by the LTX-2 transformer), so patchify/unpatchify reduce to
//! reshape + permute.
//!
//! # Position convention
//! Matches `VideoLatentTools.create_initial_state` (causal-fix always on):
//! - `positions[b, 0, t, :]` = `[time_start_s, time_end_s]`
//! - `positions[b, 1, t, :]` = `[h_start_px, h_end_px]`
//! - `positions[b, 2, t, :]` = `[w_start_px, w_end_px]`
//!
//! Token order: `t = f * H * W + h * W + w`.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use ltx_shape::ScaleFactors;

use crate::SamplerError;

// ---------------------------------------------------------------------------
// u32 → f32 without the `as_conversions`-denied cast.
// ---------------------------------------------------------------------------

/// Convert `u32` to `f32`.
///
/// For latent dimensions (always ≪ 2²⁴ = 16 777 216) the cast is exact.
/// For larger values the result is the nearest representable `f32`.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::missing_const_for_fn,
    reason = "u32→f32 via `as`: lossless for latent dims (≤ thousands); may round for values > 2^24 but that cannot happen in practice"
)]
fn u32_to_f32(n: u32) -> f32 {
    n as f32
}

// ---------------------------------------------------------------------------
// Patchify / Unpatchify
// ---------------------------------------------------------------------------

/// Flatten `[B, C, F, H, W]` into tokens `[B, F·H·W, C]`.
///
/// Equivalent to `VideoLatentPatchifier(patch_size=1).patchify(latent)`.
///
/// # Errors
/// [`SamplerError::Overflow`] when the token count overflows `usize`.
pub fn patchify<B: Backend>(latent: Tensor<B, 5>) -> Result<Tensor<B, 3>, SamplerError> {
    let [batch, channels, frames, height, width] = latent.dims();
    let tokens = frames
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .ok_or(SamplerError::Overflow)?;
    // b c f h w  →  b f h w c  →  b (f h w) c
    Ok(latent
        .permute([0, 2, 3, 4, 1])
        .reshape([batch, tokens, channels]))
}

/// Restore tokens `[B, T, C]` to a 5-D latent `[B, C, F, H, W]`.
///
/// Equivalent to `VideoLatentPatchifier(patch_size=1).unpatchify(tokens, shape)`.
///
/// # Errors
/// [`SamplerError::Shape`] when `T != F * H * W`.
pub fn unpatchify<B: Backend>(
    tokens: Tensor<B, 3>,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<B, 5>, SamplerError> {
    let [batch, token_count, channels] = tokens.dims();
    let expected = frames
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .ok_or(SamplerError::Overflow)?;
    if token_count != expected {
        return Err(SamplerError::Shape {
            context: "token count does not equal frames * height * width",
        });
    }
    // b (f h w) c  →  b f h w c  →  b c f h w
    Ok(tokens
        .reshape([batch, frames, height, width, channels])
        .permute([0, 4, 1, 2, 3]))
}

// ---------------------------------------------------------------------------
// Position grid
// ---------------------------------------------------------------------------

/// Build target-video positions `[B, 3, F·H·W, 2]`.
///
/// Pixel coords come from latent indices scaled by `scale`, causal-fix
/// applied, then temporal axis divided by `fps` to give seconds.
///
/// # Errors
/// [`SamplerError::Overflow`] for shape arithmetic overflows.
pub fn make_target_positions<B: Backend>(
    frames: u32,
    height: u32,
    width: u32,
    batch: u32,
    scale: ScaleFactors,
    fps: f32,
    device: &B::Device,
) -> Result<Tensor<B, 4>, SamplerError> {
    let raw = target_positions_raw(frames, height, width, batch, scale, fps)?;
    let tokens = tokens_u32(frames, height, width)?;
    let batch_s = to_usize(batch)?;
    let tokens_s = to_usize(tokens)?;
    Ok(Tensor::<B, 4>::from_data(
        TensorData::new(raw, [batch_s, 3, tokens_s, 2]),
        device,
    ))
}

/// Build reference-video positions `[B, 3, T_ref, 2]`.
///
/// Mirrors the position calculation in `VideoConditionByReferenceLatent.apply_to`:
/// 1. Compute pixel coords with VAE scale and causal-fix.
/// 2. Normalise temporal axis by `fps / temporal_scale_factor` (ref runs slower).
/// 3. Shift temporal left by `(temporal_scale_factor − 1) / fps`, clamp to 0.
/// 4. Scale spatial axes by `downscale_factor` (ref is lower-res than target).
///
/// # Errors
/// [`SamplerError::Overflow`] for shape arithmetic overflows.
#[allow(clippy::too_many_arguments)]
pub fn make_reference_positions<B: Backend>(
    ref_frames: u32,
    ref_height: u32,
    ref_width: u32,
    batch: u32,
    first_latent_frame: u32,
    scale: ScaleFactors,
    fps: f32,
    temporal_scale_factor: u32,
    downscale_factor: u32,
    device: &B::Device,
) -> Result<Tensor<B, 4>, SamplerError> {
    let raw = reference_positions_raw(
        ref_frames,
        ref_height,
        ref_width,
        batch,
        first_latent_frame,
        scale,
        fps,
        temporal_scale_factor,
        downscale_factor,
    )?;
    let tokens = tokens_u32(ref_frames, ref_height, ref_width)?;
    let batch_s = to_usize(batch)?;
    let tokens_s = to_usize(tokens)?;
    Ok(Tensor::<B, 4>::from_data(
        TensorData::new(raw, [batch_s, 3, tokens_s, 2]),
        device,
    ))
}

// ---------------------------------------------------------------------------
// CPU-side raw computation → Vec<f32> in [B, 3, T, 2] row-major order.
// ---------------------------------------------------------------------------

fn tokens_u32(frames: u32, height: u32, width: u32) -> Result<u32, SamplerError> {
    frames
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .ok_or(SamplerError::Overflow)
}

fn to_usize(n: u32) -> Result<usize, SamplerError> {
    usize::try_from(n).map_err(|_| SamplerError::Overflow)
}

/// Causal-fixed temporal coordinates in **seconds** for latent frame `frame_idx`.
///
/// Matches `get_pixel_coords(causal_fix=True)` then `/ fps`:
/// `t_start = clamp(f * time_scale + 1 - time_scale, 0) / fps`
/// `t_end   = clamp((f+1) * time_scale + 1 - time_scale, 0) / fps`
fn time_coords_s(frame_idx: u32, time_scale: u32, fps: f32) -> (f32, f32) {
    let start = frame_idx
        .saturating_mul(time_scale)
        .saturating_add(1)
        .saturating_sub(time_scale);
    let end = frame_idx
        .saturating_add(1)
        .saturating_mul(time_scale)
        .saturating_add(1)
        .saturating_sub(time_scale);
    (u32_to_f32(start) / fps, u32_to_f32(end) / fps)
}

/// Decode token index `t_idx` into `(frame, height, width)` indices.
///
/// # Safety annotation
/// The `as usize` casts are guarded by `to_usize` checks at function entry,
/// which reject u32 values that cannot fit in usize (16-bit platforms).
#[expect(
    clippy::as_conversions,
    clippy::arithmetic_side_effects,
    reason = "integer index math is checked by shape validation and `try_from`; `as` only converts small u32 dimensions back to usize"
)]
fn decode_token_idx(
    token_idx: usize,
    hw: usize,
    width: usize,
) -> Result<(u32, u32, u32), SamplerError> {
    let frame = u32::try_from(token_idx / hw).map_err(|_| SamplerError::Overflow)?;
    let rem = token_idx.saturating_sub(frame as usize * hw);
    let h = u32::try_from(rem / width).map_err(|_| SamplerError::Overflow)?;
    let w = u32::try_from(rem.saturating_sub(h as usize * width))
        .map_err(|_| SamplerError::Overflow)?;
    Ok((frame, h, w))
}

fn target_positions_raw(
    frames: u32,
    height: u32,
    width: u32,
    batch: u32,
    scale: ScaleFactors,
    fps: f32,
) -> Result<Vec<f32>, SamplerError> {
    let hw = height.checked_mul(width).ok_or(SamplerError::Overflow)?;
    let tokens = frames.checked_mul(hw).ok_or(SamplerError::Overflow)?;
    let tokens_s = to_usize(tokens)?;
    let batch_s = to_usize(batch)?;
    let hw_s = to_usize(hw)?;
    let width_s = to_usize(width)?;

    let time_scale = scale.time.get();
    let height_scale = scale.height.get();
    let width_scale = scale.width.get();

    let mut time_d: Vec<f32> = Vec::with_capacity(tokens_s.saturating_mul(2));
    let mut h_d: Vec<f32> = Vec::with_capacity(tokens_s.saturating_mul(2));
    let mut w_d: Vec<f32> = Vec::with_capacity(tokens_s.saturating_mul(2));

    for t_idx in 0..tokens_s {
        let (frame, h_idx, w_idx) = decode_token_idx(t_idx, hw_s, width_s)?;

        let (t_start, t_end) = time_coords_s(frame, time_scale, fps);
        time_d.push(t_start);
        time_d.push(t_end);

        let h_start = u32_to_f32(h_idx.saturating_mul(height_scale));
        let h_end = u32_to_f32(h_idx.saturating_add(1).saturating_mul(height_scale));
        h_d.push(h_start);
        h_d.push(h_end);

        let w_start = u32_to_f32(w_idx.saturating_mul(width_scale));
        let w_end = u32_to_f32(w_idx.saturating_add(1).saturating_mul(width_scale));
        w_d.push(w_start);
        w_d.push(w_end);
    }

    let per = tokens_s.saturating_mul(2).saturating_mul(3);
    let mut out = Vec::with_capacity(per.saturating_mul(batch_s));
    for _ in 0..batch_s {
        out.extend(time_d.iter().copied());
        out.extend(h_d.iter().copied());
        out.extend(w_d.iter().copied());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn reference_positions_raw(
    ref_frames: u32,
    ref_height: u32,
    ref_width: u32,
    batch: u32,
    first_latent_frame: u32,
    scale: ScaleFactors,
    fps: f32,
    temporal_scale_factor: u32,
    downscale_factor: u32,
) -> Result<Vec<f32>, SamplerError> {
    let hw = ref_height
        .checked_mul(ref_width)
        .ok_or(SamplerError::Overflow)?;
    let tokens = ref_frames.checked_mul(hw).ok_or(SamplerError::Overflow)?;
    let tokens_s = to_usize(tokens)?;
    let batch_s = to_usize(batch)?;
    let hw_s = to_usize(hw)?;
    let ref_width_s = to_usize(ref_width)?;

    let time_scale = scale.time.get();
    let height_scale = scale.height.get();
    let width_scale = scale.width.get();

    let ref_fps = fps / u32_to_f32(temporal_scale_factor);
    let shift_s = u32_to_f32(temporal_scale_factor.saturating_sub(1)) / fps;
    let downscale_f = u32_to_f32(downscale_factor);

    let mut time_d: Vec<f32> = Vec::with_capacity(tokens_s.saturating_mul(2));
    let mut h_d: Vec<f32> = Vec::with_capacity(tokens_s.saturating_mul(2));
    let mut w_d: Vec<f32> = Vec::with_capacity(tokens_s.saturating_mul(2));

    for t_idx in 0..tokens_s {
        let (frame_ref, h_idx, w_idx) = decode_token_idx(t_idx, hw_s, ref_width_s)?;
        let frame_adj = frame_ref
            .checked_add(first_latent_frame)
            .ok_or(SamplerError::Overflow)?;

        let (t_start_raw, t_end_raw) = time_coords_s(frame_adj, time_scale, ref_fps);
        time_d.extend([
            (t_start_raw - shift_s).max(0.0),
            (t_end_raw - shift_s).max(0.0),
        ]);

        let h_start = u32_to_f32(h_idx.saturating_mul(height_scale)) * downscale_f;
        let h_end = u32_to_f32(h_idx.saturating_add(1).saturating_mul(height_scale)) * downscale_f;
        h_d.extend([h_start, h_end]);

        let w_start = u32_to_f32(w_idx.saturating_mul(width_scale)) * downscale_f;
        let w_end = u32_to_f32(w_idx.saturating_add(1).saturating_mul(width_scale)) * downscale_f;
        w_d.extend([w_start, w_end]);
    }

    let per = tokens_s.saturating_mul(2).saturating_mul(3);
    let mut out = Vec::with_capacity(per.saturating_mul(batch_s));
    for _ in 0..batch_s {
        out.extend(time_d.iter().copied());
        out.extend(h_d.iter().copied());
        out.extend(w_d.iter().copied());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Device;

    type B = NdArray<f32>;

    fn dev() -> Device<B> {
        Device::<B>::default()
    }

    #[test]
    fn patchify_shape() {
        let latent = Tensor::<B, 5>::ones([2, 128, 5, 4, 4], &dev());
        let tokens = patchify(latent).unwrap();
        assert_eq!(tokens.dims(), [2, 80, 128]); // T = 5*4*4 = 80
    }

    #[test]
    fn unpatchify_shape() {
        let tokens = Tensor::<B, 3>::ones([1, 20, 128], &dev());
        let lat = unpatchify(tokens, 5, 2, 2).unwrap();
        assert_eq!(lat.dims(), [1, 128, 5, 2, 2]);
    }

    #[test]
    fn patchify_unpatchify_roundtrip() {
        let device = dev();
        let original = Tensor::<B, 5>::random(
            [1, 4, 3, 2, 2],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let tokens = patchify(original.clone()).unwrap();
        let restored = unpatchify(tokens, 3, 2, 2).unwrap();
        let diff_data: Vec<f32> = (original - restored).abs().into_data().to_vec().unwrap();
        let max_diff = diff_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_diff < 1e-6, "roundtrip max diff {max_diff}");
    }

    #[test]
    fn target_position_shape() {
        let pos = make_target_positions::<B>(3, 2, 2, 1, ScaleFactors::LTX2, 24.0, &dev()).unwrap();
        assert_eq!(pos.dims(), [1, 3, 12, 2]);
    }

    #[test]
    fn first_token_time_starts_at_zero() {
        let pos = make_target_positions::<B>(3, 1, 1, 1, ScaleFactors::LTX2, 24.0, &dev()).unwrap();
        let data: Vec<f32> = pos.into_data().to_vec().unwrap();
        let first = data.first().copied().unwrap();
        assert!(first.abs() < 1e-6, "first token time_start = {first}");
    }

    #[test]
    fn second_token_time_start_equals_first_end() {
        let pos = make_target_positions::<B>(3, 1, 1, 1, ScaleFactors::LTX2, 24.0, &dev()).unwrap();
        let data: Vec<f32> = pos.into_data().to_vec().unwrap();
        let tok0_end = data.get(1).copied().unwrap();
        let tok1_start = data.get(2).copied().unwrap();
        assert!(
            (tok0_end - tok1_start).abs() < 1e-6,
            "gap: token[0].end={tok0_end} token[1].start={tok1_start}"
        );
    }

    #[test]
    fn reference_position_spatial_downscale() {
        let pos_ref =
            make_reference_positions::<B>(1, 1, 1, 1, 0, ScaleFactors::LTX2, 24.0, 1, 2, &dev())
                .unwrap();
        let pos_tgt =
            make_reference_positions::<B>(1, 1, 1, 1, 0, ScaleFactors::LTX2, 24.0, 1, 1, &dev())
                .unwrap();
        let ref_d: Vec<f32> = pos_ref.into_data().to_vec().unwrap();
        let tgt_d: Vec<f32> = pos_tgt.into_data().to_vec().unwrap();
        let h_start_ref = ref_d.get(2).copied().unwrap();
        let h_start_tgt = tgt_d.get(2).copied().unwrap();
        assert!(
            2.0_f32.mul_add(-h_start_tgt, h_start_ref).abs() < 1e-4,
            "downscale=2 h_start: ref={h_start_ref} tgt={h_start_tgt}"
        );
        let h_end_ref = ref_d.get(3).copied().unwrap();
        let h_end_tgt = tgt_d.get(3).copied().unwrap();
        assert!(
            2.0_f32.mul_add(-h_end_tgt, h_end_ref).abs() < 1e-4,
            "downscale=2 h_end: ref={h_end_ref} tgt={h_end_tgt}"
        );
    }
}
