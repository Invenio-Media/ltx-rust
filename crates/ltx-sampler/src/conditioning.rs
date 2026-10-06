//! IC-LoRA reference-video conditioning.
//!
//! Ports `VideoConditionByReferenceLatent.apply_to` from
//! `ltx_core.conditioning.types.reference_video_cond`.
//!
//! Appending a reference video to the denoising sequence:
//! - Reference tokens are inserted into `clean_latent`.
//! - The noisy `latent` receives zeros at those positions.
//! - `denoise_mask` is set to `1 − strength` (0 for a fully-fixed reference).
//! - Positions are scaled to match the target frame.
//!
//! Also contains [`ImageKeyframeCondition`] for seam-keyframe conditioning via
//! `combined_image_conditionings` (latent replacement at a single frame).

use burn::prelude::Backend;
use burn::tensor::Tensor;
use ltx_shape::ScaleFactors;

use crate::SamplerError;
use crate::patchifier::{make_reference_positions, patchify};
use crate::state::LatentState;

// ---------------------------------------------------------------------------
// Reference-video conditioning
// ---------------------------------------------------------------------------

/// Appends an IC-LoRA reference video to the token sequence.
///
/// Mirrors `VideoConditionByReferenceLatent` in the reference Python.
///
/// # Fields
/// - `latent`: VAE-encoded reference video `[B, C, F, H, W]`.
/// - `downscale_factor`: Spatial ratio target / reference (e.g. 2 for half-res
///   reference). Must match the `LoRA`'s `reference_downscale_factor`.
/// - `temporal_scale_factor`: Temporal ratio (e.g. 4 for 1/4-fps reference).
///   Must match `reference_temporal_scale_factor` in the `LoRA` metadata.
/// - `strength`: `1.0` = reference held clean, `0.0` = reference fully denoised.
/// - `first_latent_frame`: temporal grid offset (for prefix-stripped references).
pub struct VideoReferenceCondition<B: Backend> {
    /// VAE-encoded reference latent `[B, C, F, H, W]`.
    pub latent: Tensor<B, 5>,
    /// Spatial ratio target / reference (≥ 1).
    pub downscale_factor: u32,
    /// Temporal subsampling ratio (≥ 1).
    pub temporal_scale_factor: u32,
    /// Conditioning strength: `1.0` = fully clean reference.
    pub strength: f32,
    /// Temporal-grid index of the first frame of `latent`.
    pub first_latent_frame: u32,
}

impl<B: Backend> VideoReferenceCondition<B> {
    /// Append reference tokens to `state`.
    ///
    /// # Parameters
    /// - `state`: Current target latent state (patchified and optionally noised).
    /// - `scale`: VAE spatio-temporal scale factors.
    /// - `fps`: Target video frame rate.
    /// - `device`: Burn device for new tensors.
    ///
    /// # Errors
    /// [`SamplerError::Overflow`] for shape arithmetic overflows.
    pub fn apply_to(
        self,
        state: LatentState<B>,
        scale: ScaleFactors,
        fps: f32,
        device: &B::Device,
    ) -> Result<LatentState<B>, SamplerError> {
        let [batch, _c, ref_f, ref_h, ref_w] = self.latent.dims();

        let ref_tokens = patchify(self.latent)?;
        let ref_t = ref_tokens.dims()[1];

        let ref_pos = make_reference_positions::<B>(
            u32::try_from(ref_f).map_err(|_| SamplerError::Overflow)?,
            u32::try_from(ref_h).map_err(|_| SamplerError::Overflow)?,
            u32::try_from(ref_w).map_err(|_| SamplerError::Overflow)?,
            u32::try_from(batch).map_err(|_| SamplerError::Overflow)?,
            self.first_latent_frame,
            scale,
            fps,
            self.temporal_scale_factor,
            self.downscale_factor,
            device,
        )?;

        let channels = ref_tokens.dims()[2];
        let mask_val = 1.0_f32 - self.strength;
        let ref_mask = Tensor::<B, 3>::full([batch, ref_t, 1], mask_val, device);
        let ref_latent_zeros = Tensor::<B, 3>::zeros([batch, ref_t, channels], device);

        let new_kf_mask = state.keyframes_mask.map(|km| {
            let ref_km = Tensor::<B, 3>::zeros([batch, ref_t, 1], device);
            Tensor::cat(vec![km, ref_km], 1)
        });

        Ok(LatentState {
            latent: Tensor::cat(vec![state.latent, ref_latent_zeros], 1),
            denoise_mask: Tensor::cat(vec![state.denoise_mask, ref_mask], 1),
            positions: Tensor::cat(vec![state.positions, ref_pos], 2),
            clean_latent: Tensor::cat(vec![state.clean_latent, ref_tokens], 1),
            attention_mask: state.attention_mask,
            keyframes_mask: new_kf_mask,
        })
    }
}

// ---------------------------------------------------------------------------
// Image/keyframe conditioning
// ---------------------------------------------------------------------------

/// Replace target latent tokens at a specific frame with a pre-encoded image.
///
/// Mirrors `VideoConditionByLatentIndex` (`_apply_condition_by_latent_index` in
/// `ltx_core.conditioning.types.latent_cond`), which `combined_image_conditionings`
/// uses for an image at frame 0 — the seam keyframe.
///
/// The frame's tokens are written into `clean_latent`, and `denoise_mask` on
/// those tokens is set to `1 − strength`. Tokens outside the frame are unchanged.
///
/// # Fields
/// - `image_latent`: encoded image `[B, C, 1, H, W]` (one frame).
/// - `latent_frame_index`: which latent frame to replace (0-based).
/// - `tokens_per_frame`: number of tokens one latent frame occupies (= H × W).
/// - `strength`: `1.0` holds the frame clean, `0.0` leaves it fully denoised.
pub struct ImageKeyframeCondition<B: Backend> {
    /// Encoded image latent `[B, C, 1, H, W]`.
    pub image_latent: Tensor<B, 5>,
    /// Latent frame index to overwrite (0-based).
    pub latent_frame_index: usize,
    /// Tokens per latent frame (= H × W).
    pub tokens_per_frame: usize,
    /// Conditioning strength; the frame's denoise mask becomes `1 − strength`.
    pub strength: f32,
}

impl<B: Backend> ImageKeyframeCondition<B> {
    /// Apply the image conditioning to `state`.
    ///
    /// # Errors
    /// [`SamplerError::Overflow`] for shape arithmetic overflows;
    /// [`SamplerError::Shape`] for dimension mismatches.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor arithmetic is device math; it cannot overflow Rust integers"
    )]
    pub fn apply_to(
        self,
        mut state: LatentState<B>,
        device: &B::Device,
    ) -> Result<LatentState<B>, SamplerError> {
        let [batch, _c, _f_one, height, width] = self.image_latent.dims();

        let expected = height.checked_mul(width).ok_or(SamplerError::Overflow)?;
        if expected != self.tokens_per_frame {
            return Err(SamplerError::Shape {
                context: "image_latent H*W does not match tokens_per_frame",
            });
        }

        let tpf = self.tokens_per_frame;
        let img_tokens = patchify(self.image_latent)?;
        let channels = img_tokens.dims()[2];

        let start = self
            .latent_frame_index
            .checked_mul(tpf)
            .ok_or(SamplerError::Overflow)?;
        let [_b, total_t, _c2] = state.latent.dims();
        let end = start.checked_add(tpf).ok_or(SamplerError::Overflow)?;
        if end > total_t {
            return Err(SamplerError::Shape {
                context: "image keyframe index exceeds latent frame count",
            });
        }

        let after = total_t.saturating_sub(end);

        // Build replacement: zeros before, image tokens at frame, zeros after.
        let before_zeros = Tensor::<B, 3>::zeros([batch, start, channels], device);
        let after_zeros = Tensor::<B, 3>::zeros([batch, after, channels], device);
        let img_clean = Tensor::cat(vec![before_zeros, img_tokens, after_zeros], 1);

        // 1.0 on the image frame's tokens, 0.0 elsewhere.
        let before_1 = Tensor::<B, 3>::zeros([batch, start, 1], device);
        let frame_1 = Tensor::<B, 3>::ones([batch, tpf, 1], device);
        let after_1 = Tensor::<B, 3>::zeros([batch, after, 1], device);
        let frame_mask = Tensor::cat(vec![before_1, frame_1, after_1], 1);
        let keep = Tensor::ones_like(&frame_mask) - frame_mask.clone();

        // clean_latent[span] = image tokens; denoise_mask[span] = 1 − strength.
        state.clean_latent = state.clean_latent * keep.clone() + img_clean * frame_mask.clone();
        state.denoise_mask = state.denoise_mask * keep + frame_mask * (1.0 - self.strength);

        Ok(state)
    }
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
    fn reference_condition_extends_sequence() {
        let device = dev();
        let scale = ScaleFactors::LTX2;
        let target_state = LatentState::<B> {
            latent: Tensor::zeros([1, 1, 128], &device),
            denoise_mask: Tensor::ones([1, 1, 1], &device),
            positions: Tensor::zeros([1, 3, 1, 2], &device),
            clean_latent: Tensor::zeros([1, 1, 128], &device),
            attention_mask: None,
            keyframes_mask: Some(Tensor::ones([1, 1, 1], &device)),
        };

        let ref_latent = Tensor::<B, 5>::ones([1, 128, 1, 1, 1], &device);
        let cond = VideoReferenceCondition {
            latent: ref_latent,
            downscale_factor: 1,
            temporal_scale_factor: 1,
            strength: 1.0,
            first_latent_frame: 0,
        };

        let new_state = cond.apply_to(target_state, scale, 24.0, &device).unwrap();

        assert_eq!(new_state.latent.dims(), [1, 2, 128]);
        assert_eq!(new_state.denoise_mask.dims(), [1, 2, 1]);
        assert_eq!(new_state.positions.dims(), [1, 3, 2, 2]);

        let latent_vals: Vec<f32> = new_state.latent.into_data().to_vec().unwrap();
        assert!(
            latent_vals.iter().skip(128).all(|&v| v.abs() < 1e-6),
            "reference token in noisy latent should be zero"
        );

        let mask_vals: Vec<f32> = new_state.denoise_mask.into_data().to_vec().unwrap();
        let ref_mask = mask_vals.get(1).copied().unwrap();
        assert!(
            ref_mask.abs() < 1e-6,
            "reference mask should be 0, got {ref_mask}"
        );
    }

    #[test]
    fn keyframe_condition_writes_only_its_frame() {
        let device = dev();
        // 3 latent frames x 2 tokens per frame, 2 channels.
        let state = LatentState::<B> {
            latent: Tensor::zeros([1, 6, 2], &device),
            denoise_mask: Tensor::ones([1, 6, 1], &device),
            positions: Tensor::zeros([1, 3, 6, 2], &device),
            clean_latent: Tensor::full([1, 6, 2], -1.0, &device),
            attention_mask: None,
            keyframes_mask: None,
        };
        // Image latent [B=1, C=2, F=1, H=1, W=2]: tokens (1,2) and (3,4).
        let image_latent =
            Tensor::<B, 1>::from_floats([1.0, 3.0, 2.0, 4.0], &device).reshape([1, 2, 1, 1, 2]);
        let cond = ImageKeyframeCondition {
            image_latent,
            latent_frame_index: 1,
            tokens_per_frame: 2,
            strength: 0.95,
        };

        let new_state = cond.apply_to(state, &device).unwrap();

        let clean: Vec<f32> = new_state.clean_latent.into_data().to_vec().unwrap();
        assert_eq!(
            clean,
            vec![
                -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0
            ]
        );
        let mask: Vec<f32> = new_state.denoise_mask.into_data().to_vec().unwrap();
        let expected_mask = [1.0, 1.0, 0.05, 0.05, 1.0, 1.0];
        for (got, want) in mask.iter().zip(expected_mask) {
            assert!(
                (got - want).abs() < 1e-6,
                "mask {mask:?} != {expected_mask:?}"
            );
        }
    }
}
