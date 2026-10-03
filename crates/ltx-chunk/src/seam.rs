//! Seam conditioning plan for IC-LoRA chunk transitions.
//!
//! When consecutive chunks overlap, the last rendered chunk's output at the
//! seam provides clean "keyframe" conditionings that guide the next chunk's
//! generation.  This mirrors the reference pipeline's DFR seam-keyframe logic
//! in `ltx_pipelines/hdr_ic_lora.py`.
//!
//! ## Reference (commit `9ec55f9`, `hdr_ic_lora.py`)
//! - Line 68–69: `DEFAULT_KEYFRAME_STRENGTH = 0.95`
//! - Lines 95–101 (`_seam_roles`): every `resolve_canvas` seam position
//!   receives **one** 1-frame SDR keyframe conditioning.  The function
//!   returns `(positions, positions)` – each seam gets both a generated HDR
//!   slot and a 1-frame SDR guide.
//! - Lines 109–143 (`_keyframe_conditionings_from_pixel_frames`): each
//!   guide is VAE-encoded as a 1-frame `VideoConditionByKeyframeIndex` with
//!   `num_pixel_frames = 1` and `frame_idx = seam_position`.
//!
//! For our chunking scheme the analogue is:
//! - Seam = start of chunk n (chunk-local index 0, the first frame in the
//!   overlap).
//! - 1 conditioning frame from chunk n−1 per seam, at chunk-n-local index 0.
//! - Strength: 0.95 (`DEFAULT_KEYFRAME_STRENGTH` from the reference).
//! - Blend tail: the remaining `overlap − 1` frames are crossfaded by the
//!   pixel-level [`Blender`]; the model sees the conditioned seam frame and
//!   generates the rest of the chunk normally.
//!
//! The conditioning frames are at **latent-aligned** chunk-local indices
//! (multiples of 8), matching the VAE temporal compression factor.  With one
//! keyframe at index 0 the `8k+1` requirement for the 1-frame keyframe latent
//! is trivially satisfied (`1 = 8×0 + 1`).
//!
//! ## Commands used to verify the reference values
//! ```sh
//! python -W ignore -c "
//! from ltx_pipelines.hdr_ic_lora import DEFAULT_KEYFRAME_STRENGTH
//! print(DEFAULT_KEYFRAME_STRENGTH)
//! # Output: 0.95
//! "
//!
//! python -W ignore -c "
//! from ltx_pipelines.dfr_helpers.layout import resolve_canvas
//! for n in [33, 49, 97]:
//!     c, s, p = resolve_canvas(n)
//!     print(f'{n}: canvas={c} segment={s} seams={p}')
//! # 33: canvas=33 segment=32 seams=[32]
//! # 49: canvas=49 segment=24 seams=[24, 48]
//! # 97: canvas=97 segment=32 seams=[32, 64, 96]
//! "
//! ```
//!
//! [`Blender`]: crate::blend::Blender

use crate::error::ChunkError;

/// Keyframe conditioning strength, matching `DEFAULT_KEYFRAME_STRENGTH` in
/// `ltx_pipelines/hdr_ic_lora.py` line 69.
pub const DEFAULT_KEYFRAME_STRENGTH: f32 = 0.95;

/// One conditioning frame supplied from chunk n−1 to guide chunk n.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CondFrame {
    /// Frame index within chunk n (0-based, chunk-local).
    ///
    /// Always a multiple of 8 (latent-aligned), matching the VAE temporal
    /// compression factor.  Currently always 0: the seam frame.
    pub chunk_local_idx: u32,

    /// VAE conditioning strength in `(0, 1]`.
    ///
    /// 0.95 by default, matching `DEFAULT_KEYFRAME_STRENGTH` in the
    /// reference.
    pub strength: f32,
}

/// Seam conditioning plan for chunk n > 0.
///
/// Describes which frames of chunk n−1's decoded output become keyframe
/// conditionings injected into chunk n, and how many overlap frames remain
/// as a pure pixel-level blend tail after the last conditioning.
#[derive(Debug, Clone, PartialEq)]
pub struct SeamCondPlan {
    /// Chunk-n-local frame indices (multiples of 8) that should be supplied
    /// as 1-frame `VideoConditionByKeyframeIndex` conditionings from chunk
    /// n−1's output.
    ///
    /// Currently one entry at index 0 (the seam itself), matching the
    /// reference's one-per-seam rule.
    pub conditioning_frames: Vec<CondFrame>,

    /// Number of frames remaining in the overlap after the last conditioning
    /// frame.  These are blended at the pixel level by [`Blender`] and do not
    /// receive an explicit conditioning.
    ///
    /// `blend_tail_len = overlap − (last_cond_idx + 1)`.  For a single
    /// conditioning at index 0 and overlap O: `blend_tail_len = O − 1`.
    ///
    /// [`Blender`]: crate::blend::Blender
    pub blend_tail_len: u32,
}

/// Returns the seam conditioning plan for chunk n > 0 given an `overlap`
/// that is a multiple of 8 and ≥ 1.
///
/// The plan places one conditioning frame at chunk-local index 0 (the seam)
/// with `strength = 0.95`.  The remaining `overlap − 1` frames form the
/// blend tail.
///
/// # Errors
/// Returns [`ChunkError::ZeroOverlapForSeam`] when `overlap == 0`.
pub fn seam_cond_plan(overlap: u32) -> Result<SeamCondPlan, ChunkError> {
    if overlap == 0 {
        return Err(ChunkError::ZeroOverlapForSeam);
    }

    Ok(SeamCondPlan {
        conditioning_frames: vec![CondFrame {
            chunk_local_idx: 0,
            strength: DEFAULT_KEYFRAME_STRENGTH,
        }],
        blend_tail_len: overlap.saturating_sub(1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── seam_cond_plan ─────────────────────────────────────────────────────

    #[test]
    fn zero_overlap_is_error() {
        assert_eq!(seam_cond_plan(0), Err(ChunkError::ZeroOverlapForSeam));
    }

    #[test]
    fn single_conditioning_at_seam() {
        let plan = seam_cond_plan(8).unwrap();
        assert_eq!(plan.conditioning_frames.len(), 1);
        assert_eq!(plan.conditioning_frames[0].chunk_local_idx, 0);
    }

    #[test]
    fn strength_matches_reference() {
        // python -W ignore -c "from ltx_pipelines.hdr_ic_lora import DEFAULT_KEYFRAME_STRENGTH; print(DEFAULT_KEYFRAME_STRENGTH)"
        // Output: 0.95
        let plan = seam_cond_plan(8).unwrap();
        assert!(
            (plan.conditioning_frames[0].strength - 0.95_f32).abs() < 1e-6_f32,
            "strength should be 0.95 (DEFAULT_KEYFRAME_STRENGTH from reference)"
        );
    }

    #[test]
    fn blend_tail_is_overlap_minus_one() {
        for overlap in [8_u32, 16, 24, 32] {
            let plan = seam_cond_plan(overlap).unwrap();
            assert_eq!(
                plan.blend_tail_len,
                overlap.saturating_sub(1),
                "overlap={overlap}"
            );
        }
    }

    #[test]
    fn conditioning_index_is_latent_aligned() {
        // chunk_local_idx must be a multiple of 8 (VAE temporal alignment).
        for overlap in [8_u32, 16, 32] {
            let plan = seam_cond_plan(overlap).unwrap();
            for cf in &plan.conditioning_frames {
                assert_eq!(
                    cf.chunk_local_idx % 8,
                    0,
                    "conditioning index {} is not latent-aligned",
                    cf.chunk_local_idx
                );
            }
        }
    }

    #[test]
    fn conditioning_plus_tail_eq_overlap() {
        // n_cond_frames + blend_tail_len == overlap.
        for overlap in [8_u32, 16, 24, 32] {
            let plan = seam_cond_plan(overlap).unwrap();
            let n_cond = u32::try_from(plan.conditioning_frames.len()).unwrap();
            assert_eq!(
                n_cond.saturating_add(plan.blend_tail_len),
                overlap,
                "overlap={overlap}"
            );
        }
    }

    #[test]
    fn reference_segment_sizes_match_strength() {
        // The reference produces seams at multiples of the segment length
        // (24 or 32 frames).  For our chunking with those same overlaps,
        // the conditioning strength must equal DEFAULT_KEYFRAME_STRENGTH.
        //
        // Reference output (from resolve_canvas):
        //   num_frames=33: segment=32, seams=[32]
        //   num_frames=49: segment=24, seams=[24, 48]
        //   num_frames=97: segment=32, seams=[32, 64, 96]
        for overlap in [24_u32, 32] {
            let plan = seam_cond_plan(overlap).unwrap();
            assert!(
                (plan.conditioning_frames[0].strength - DEFAULT_KEYFRAME_STRENGTH).abs() < 1e-6,
                "overlap={overlap}: strength mismatch"
            );
        }
    }
}
