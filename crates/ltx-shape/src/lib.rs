//! Shape rules for the LTX-2 video VAE and the IC-LoRA token layout.
//!
//! The VAE compresses 8x in time and 32x in each spatial axis. A clip must have
//! `8k + 1` frames: the first frame maps to its own latent frame, and every
//! following group of 8 frames maps to one more latent frame. Each latent cell is
//! one transformer token (the video patchifier uses a patch size of 1).
//!
//! `IC-LoRA` appends the VAE-encoded reference clip to the target tokens. The
//! reference can be smaller than the target: the adapter metadata keys
//! `reference_downscale_factor` and `reference_temporal_scale_factor` set how much.

use std::num::NonZeroU32;

use thiserror::Error;

/// Errors for shapes that the VAE cannot encode.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    #[error("frame count {0} is not 8k + 1")]
    FrameCount(u32),
    #[error("{axis} {value} is not a positive multiple of {multiple}")]
    Spatial {
        axis: &'static str,
        value: u32,
        multiple: u32,
    },
    #[error("reference scale factor must be at least 1, got {0}")]
    ReferenceScale(u32),
    #[error("{dimension} {value} is not divisible by the reference downscale factor {factor}")]
    ReferenceDivisibility {
        dimension: &'static str,
        value: u32,
        factor: u32,
    },
    #[error("reference {height}x{width} (target / reference downscale factor) is off the VAE grid")]
    ReferenceOffGrid { height: u32, width: u32 },
    #[error("shape arithmetic overflowed")]
    Overflow,
}

/// Compression between pixel space and the latent grid, as (time, height, width).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleFactors {
    pub time: NonZeroU32,
    pub height: NonZeroU32,
    pub width: NonZeroU32,
}

impl ScaleFactors {
    /// The LTX-2 video VAE: 8x in time, 32x in each spatial axis.
    pub const LTX2: Self = Self {
        time: NonZeroU32::new(8).unwrap(),
        height: NonZeroU32::new(32).unwrap(),
        width: NonZeroU32::new(32).unwrap(),
    };
}

impl Default for ScaleFactors {
    fn default() -> Self {
        Self::LTX2
    }
}

/// A clip shape in pixel space that the VAE can encode. It keeps the scale
/// factors it was validated against, so its latent grid cannot disagree with them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelShape {
    frames: u32,
    height: u32,
    width: u32,
    scale: ScaleFactors,
}

impl PixelShape {
    /// Validates `frames = time * k + 1` and spatial multiples of the scale factors.
    ///
    /// # Errors
    /// Returns [`ShapeError`] when a dimension does not fit the VAE grid.
    pub fn new(
        frames: u32,
        height: u32,
        width: u32,
        scale: ScaleFactors,
    ) -> Result<Self, ShapeError> {
        if !on_frame_grid(frames, scale.time) {
            return Err(ShapeError::FrameCount(frames));
        }
        check_spatial("height", height, scale.height)?;
        check_spatial("width", width, scale.width)?;
        Ok(Self {
            frames,
            height,
            width,
            scale,
        })
    }

    #[must_use]
    pub const fn frames(self) -> u32 {
        self.frames
    }

    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn scale(self) -> ScaleFactors {
        self.scale
    }

    /// The latent grid that the VAE encoder produces for this clip.
    #[must_use]
    pub fn latent(self) -> LatentShape {
        let scale = self.scale;
        // `frames >= 1`, so the quotient is below `frames` and the `+ 1` cannot overflow.
        LatentShape {
            frames: (self.frames.saturating_sub(1) / scale.time).saturating_add(1),
            height: self.height / scale.height,
            width: self.width / scale.width,
        }
    }
}

/// True when `frames = time * k + 1` for some `k >= 0`.
const fn on_frame_grid(frames: u32, time: NonZeroU32) -> bool {
    match frames.checked_sub(1) {
        Some(rest) => rest.is_multiple_of(time.get()),
        None => false,
    }
}

const fn check_spatial(
    axis: &'static str,
    value: u32,
    multiple: NonZeroU32,
) -> Result<(), ShapeError> {
    if value == 0 || !value.is_multiple_of(multiple.get()) {
        return Err(ShapeError::Spatial {
            axis,
            value,
            multiple: multiple.get(),
        });
    }
    Ok(())
}

/// A latent grid as (frames, height, width). One cell is one token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentShape {
    pub frames: u32,
    pub height: u32,
    pub width: u32,
}

impl LatentShape {
    /// Tokens in this grid.
    ///
    /// # Errors
    /// Returns [`ShapeError::Overflow`] if the count does not fit in `u64`.
    pub fn tokens(self) -> Result<u64, ShapeError> {
        u64::from(self.frames)
            .checked_mul(u64::from(self.height))
            .and_then(|n| n.checked_mul(u64::from(self.width)))
            .ok_or(ShapeError::Overflow)
    }

    /// Tokens in one latent frame.
    ///
    /// # Errors
    /// Returns [`ShapeError::Overflow`] if the count does not fit in `u64`.
    pub fn tokens_per_frame(self) -> Result<u64, ShapeError> {
        u64::from(self.height)
            .checked_mul(u64::from(self.width))
            .ok_or(ShapeError::Overflow)
    }
}

/// The largest valid frame count (`time * k + 1`) that is not above `frames`.
#[must_use]
pub fn floor_frames(frames: u32, scale: ScaleFactors) -> Option<u32> {
    // The remainder is at most `frames - 1`, so the subtraction stays at or above 1.
    frames
        .checked_sub(1)
        .map(|rest| frames.saturating_sub(rest % scale.time))
}

/// The smallest valid frame count (`time * k + 1`) that is not below `frames`.
#[must_use]
pub const fn ceil_frames(frames: u32, scale: ScaleFactors) -> Option<u32> {
    let Some(rest) = frames.checked_sub(1) else {
        return Some(1);
    };
    match rest.checked_next_multiple_of(scale.time.get()) {
        Some(grid) => grid.checked_add(1),
        None => None,
    }
}

/// Rounds a spatial dimension up to the next multiple of `multiple`.
#[must_use]
pub const fn ceil_spatial(value: u32, multiple: NonZeroU32) -> Option<u32> {
    value.checked_next_multiple_of(multiple.get())
}

/// How an IC-LoRA lays out its reference clip next to the target clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcLoraLayout {
    reference_downscale: NonZeroU32,
    reference_temporal: NonZeroU32,
}

impl IcLoraLayout {
    /// A reference clip at full target size, which doubles the sequence.
    pub const FULL: Self = Self {
        reference_downscale: NonZeroU32::MIN,
        reference_temporal: NonZeroU32::MIN,
    };

    /// # Errors
    /// Returns [`ShapeError::ReferenceScale`] when a factor is zero.
    pub const fn new(
        reference_downscale: u32,
        reference_temporal: u32,
    ) -> Result<Self, ShapeError> {
        let Some(reference_downscale) = NonZeroU32::new(reference_downscale) else {
            return Err(ShapeError::ReferenceScale(reference_downscale));
        };
        let Some(reference_temporal) = NonZeroU32::new(reference_temporal) else {
            return Err(ShapeError::ReferenceScale(reference_temporal));
        };
        Ok(Self {
            reference_downscale,
            reference_temporal,
        })
    }

    #[must_use]
    pub const fn reference_downscale(self) -> u32 {
        self.reference_downscale.get()
    }

    #[must_use]
    pub const fn reference_temporal(self) -> u32 {
        self.reference_temporal.get()
    }

    /// The reference clip in pixel space, after the spatial downscale and the
    /// VAE-aligned temporal subsample (keep frame 0, then every Nth frame).
    ///
    /// # Errors
    /// Returns [`ShapeError`] when the target does not divide by the downscale
    /// factor, or when the reference does not fit the VAE grid.
    pub fn reference_pixels(self, target: PixelShape) -> Result<PixelShape, ShapeError> {
        let scale = target.scale;
        let factor = self.reference_downscale;
        if !target.height.is_multiple_of(factor.get()) {
            return Err(ShapeError::ReferenceDivisibility {
                dimension: "height",
                value: target.height,
                factor: factor.get(),
            });
        }
        if !target.width.is_multiple_of(factor.get()) {
            return Err(ShapeError::ReferenceDivisibility {
                dimension: "width",
                value: target.width,
                factor: factor.get(),
            });
        }
        // Frame 0, then every Nth frame of the rest. The count is at most `frames`.
        let kept = target
            .frames
            .saturating_sub(1)
            .div_ceil(self.reference_temporal.get())
            .saturating_add(1);
        // The VAE encoder drops trailing frames that do not fit `time * k + 1`.
        let frames = floor_frames(kept, scale).ok_or(ShapeError::FrameCount(kept))?;
        let height = target.height / factor;
        let width = target.width / factor;
        PixelShape::new(frames, height, width, scale).map_err(|error| match error {
            ShapeError::Spatial { .. } => ShapeError::ReferenceOffGrid { height, width },
            other => other,
        })
    }

    /// Tokens in the transformer sequence: target tokens plus reference tokens.
    ///
    /// # Errors
    /// Returns [`ShapeError`] for an invalid reference shape or an overflow.
    pub fn sequence_tokens(self, target: PixelShape) -> Result<SequenceTokens, ShapeError> {
        let target_tokens = target.latent().tokens()?;
        let reference_tokens = self.reference_pixels(target)?.latent().tokens()?;
        Ok(SequenceTokens {
            target: target_tokens,
            reference: reference_tokens,
        })
    }
}

impl Default for IcLoraLayout {
    fn default() -> Self {
        Self::FULL
    }
}

/// Token counts in one IC-LoRA transformer pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceTokens {
    /// Tokens that the sampler denoises.
    pub target: u64,
    /// Clean reference tokens, appended after the target tokens.
    pub reference: u64,
}

impl SequenceTokens {
    /// All tokens in the sequence.
    ///
    /// # Errors
    /// Returns [`ShapeError::Overflow`] if the sum does not fit in `u64`.
    pub fn total(self) -> Result<u64, ShapeError> {
        self.target
            .checked_add(self.reference)
            .ok_or(ShapeError::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALE: ScaleFactors = ScaleFactors::LTX2;

    #[test]
    fn rejects_frame_counts_off_the_grid() {
        assert_eq!(
            PixelShape::new(0, 32, 32, SCALE),
            Err(ShapeError::FrameCount(0))
        );
        assert_eq!(
            PixelShape::new(120, 32, 32, SCALE),
            Err(ShapeError::FrameCount(120))
        );
        assert!(PixelShape::new(1, 32, 32, SCALE).is_ok());
        assert!(PixelShape::new(121, 32, 32, SCALE).is_ok());
    }

    #[test]
    fn rejects_spatial_sizes_off_the_grid() {
        assert!(matches!(
            PixelShape::new(9, 1080, 1920, SCALE),
            Err(ShapeError::Spatial { axis: "height", .. })
        ));
        assert!(matches!(
            PixelShape::new(9, 32, 0, SCALE),
            Err(ShapeError::Spatial { axis: "width", .. })
        ));
    }

    #[test]
    fn full_hd_token_count_matches_the_vae_grid() {
        let height = ceil_spatial(1080, SCALE.height).unwrap();
        assert_eq!(height, 1088);
        let shape = PixelShape::new(121, height, 1920, SCALE).unwrap();
        let latent = shape.latent();
        assert_eq!(
            latent,
            LatentShape {
                frames: 16,
                height: 34,
                width: 60
            }
        );
        assert_eq!(latent.tokens_per_frame().unwrap(), 2040);
        let seq = IcLoraLayout::FULL.sequence_tokens(shape).unwrap();
        assert_eq!(seq.target, 32_640);
        assert_eq!(seq.total().unwrap(), 65_280);
    }

    #[test]
    fn frame_rounding_lands_on_the_grid() {
        assert_eq!(floor_frames(0, SCALE), None);
        assert_eq!(floor_frames(1, SCALE), Some(1));
        assert_eq!(floor_frames(8, SCALE), Some(1));
        assert_eq!(floor_frames(9, SCALE), Some(9));
        assert_eq!(floor_frames(130, SCALE), Some(129));
        assert_eq!(ceil_frames(0, SCALE), Some(1));
        assert_eq!(ceil_frames(2, SCALE), Some(9));
        assert_eq!(ceil_frames(9, SCALE), Some(9));
        assert_eq!(ceil_frames(u32::MAX, SCALE), None);
    }

    #[test]
    fn downscaled_reference_shrinks_the_sequence() {
        let layout = IcLoraLayout::new(2, 1).unwrap();
        let target = PixelShape::new(33, 512, 768, SCALE).unwrap();
        let reference = layout.reference_pixels(target).unwrap();
        assert_eq!(
            (reference.height(), reference.width(), reference.frames()),
            (256, 384, 33)
        );
        let seq = layout.sequence_tokens(target).unwrap();
        assert_eq!(seq.target, 5 * 16 * 24);
        assert_eq!(seq.reference, 5 * 8 * 12);
    }

    #[test]
    fn temporal_subsample_keeps_the_first_frame_and_every_nth() {
        // 121 frames, every 2nd after frame 0: 1 + 60 = 61 frames = 8 * 7 + 5, so the
        // encoder crops to 57 frames, which is 8 latent frames.
        let layout = IcLoraLayout::new(1, 2).unwrap();
        let target = PixelShape::new(121, 64, 64, SCALE).unwrap();
        let reference = layout.reference_pixels(target).unwrap();
        assert_eq!(reference.frames(), 57);
        assert_eq!(reference.latent().frames, 8);
    }

    #[test]
    fn reference_frames_match_the_python_reference() {
        // (target frames, temporal factor, encoded reference frames). Generated with
        // `ltx_pipelines.iclora_utils.temporal_subsample` followed by the encoder's
        // 8k + 1 crop, at LTX-2 commit 9ec55f9.
        let cases = [
            (33, 3, 9),
            (121, 7, 17),
            (97, 5, 17),
            (49, 4, 9),
            (9, 2, 1),
            (17, 16, 1),
        ];
        for (frames, temporal, expected) in cases {
            let layout = IcLoraLayout::new(1, temporal).unwrap();
            let target = PixelShape::new(frames, 32, 32, SCALE).unwrap();
            let reference = layout.reference_pixels(target).unwrap();
            assert_eq!(reference.frames(), expected, "F={frames} N={temporal}");
        }
    }

    #[test]
    fn reference_downscale_must_divide_the_target() {
        let layout = IcLoraLayout::new(3, 1).unwrap();
        let target = PixelShape::new(9, 64, 96, SCALE).unwrap();
        assert!(matches!(
            layout.reference_pixels(target),
            Err(ShapeError::ReferenceDivisibility {
                dimension: "height",
                ..
            })
        ));
        assert_eq!(IcLoraLayout::new(0, 1), Err(ShapeError::ReferenceScale(0)));
    }

    #[test]
    fn reference_off_the_vae_grid_names_the_reference() {
        // 96 / 2 = 48 divides cleanly but is not a multiple of 32.
        let layout = IcLoraLayout::new(2, 1).unwrap();
        let target = PixelShape::new(9, 96, 64, SCALE).unwrap();
        assert_eq!(
            layout.reference_pixels(target),
            Err(ShapeError::ReferenceOffGrid {
                height: 48,
                width: 32
            })
        );
    }
}
