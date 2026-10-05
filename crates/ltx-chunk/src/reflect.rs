//! Reflection padding for short clips.
//!
//! When a clip has fewer frames than `chunk_len` (or its frame count is not
//! `8k + 1`), the VAE cannot process it directly. [`pad_to_grid`] computes
//! the smallest valid `8k + 1` length that is not below the actual length.
//! Frames beyond the real content are supplied by mirror-without-edge
//! reflection: for a clip `[0, 1, 2, 3, 4]`, frame 5 maps to 3, frame 6 to
//! 2, frame 7 to 1, frame 8 back to 0, frame 9 to 1, etc.
//!
//! This is the same mode as `torch.nn.ReflectionPad1d` / `PyTorch` `"reflect"`
//! padding (the reference uses `ResizeMode.REFLECT_PAD` in `media_io` for
//! spatial padding; we apply the same rule temporally).

use crate::error::ChunkError;
use ltx_shape::ScaleFactors;

/// Padding plan for a clip that is shorter than or not on the `8k + 1` grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadPlan {
    /// The `8k + 1` frame count the VAE will see (≥ `keep_len`).
    pub padded_len: u32,
    /// Number of leading frames that are real content.
    ///
    /// Frames at indices `[keep_len, padded_len)` in the padded sequence are
    /// reflected copies; trim the output back to `keep_len` after decoding.
    pub keep_len: u32,
}

impl PadPlan {
    /// Maps a padded-sequence index to the real source frame index.
    ///
    /// Indices `< keep_len` pass through unchanged. Indices `≥ keep_len` are
    /// reflected by [`reflect_index`] around the boundary of the real content.
    ///
    /// Returns 0 when `keep_len` is 0 (degenerate clip).
    #[must_use]
    pub fn source_frame(self, padded_idx: u32) -> u32 {
        reflect_index(padded_idx, self.keep_len)
    }

    /// True when no reflection is needed (`padded_len == keep_len`).
    #[must_use]
    pub const fn is_noop(self) -> bool {
        self.padded_len == self.keep_len
    }
}

/// Returns the padding plan that rounds `actual_len` up to the smallest
/// `8k + 1` value not below it, using the given [`ScaleFactors`].
///
/// If `actual_len` is already on the grid, [`PadPlan::is_noop`] returns
/// `true` (no reflection frames need to be generated).
///
/// Returns `None` when the arithmetic overflows (clip too large).
#[must_use]
pub fn pad_to_grid(actual_len: u32, scale: ScaleFactors) -> Option<PadPlan> {
    let padded_len = ltx_shape::ceil_frames(actual_len, scale)?;
    Some(PadPlan {
        padded_len,
        keep_len: actual_len,
    })
}

/// Maps a virtual frame index `idx` into the real range `[0, len)` using
/// mirror-without-edge-repeat reflection (`PyTorch` `"reflect"` mode).
///
/// The reflection period is `2 × (len − 1)`:
/// ```text
/// len = 5 → indices: 0 1 2 3 4 | 3 2 1 0 | 1 2 3 4 | …
/// ```
///
/// - `idx < len` → returns `idx` unchanged.
/// - `len == 0` or `len == 1` → returns 0 (degenerate).
///
/// **Note:** The edge frames (0 and `len − 1`) are NOT repeated at the fold;
/// this is the "reflect" convention, not "replicate".
#[must_use]
pub fn reflect_index(idx: u32, len: u32) -> u32 {
    // Degenerate cases.
    let Some(content) = len.checked_sub(1) else {
        return 0;
    };
    if content == 0 {
        return 0;
    }

    // Period = 2 × (len − 1).  Use u64 to avoid overflow for large `len`.
    let period = u64::from(content).saturating_mul(2);
    // checked_rem returns None only when divisor is 0; period ≥ 2 here.
    let r = u64::from(idx).checked_rem(period).unwrap_or(0);
    let len64 = u64::from(len);

    if r < len64 {
        // r fits in u32 because r < len ≤ u32::MAX.
        u32::try_from(r).unwrap_or(u32::MAX)
    } else {
        // period − r is in [1, len − 1], which fits in u32.
        u32::try_from(period.saturating_sub(r)).unwrap_or(u32::MAX)
    }
}

/// Produces the full padded index sequence `[0, padded_len)` as source
/// frame indices, using [`PadPlan::source_frame`].
///
/// # Errors
/// Returns [`ChunkError::Overflow`] when `padded_len` overflows `usize`.
pub fn padded_frame_map(plan: PadPlan) -> Result<Vec<u32>, ChunkError> {
    let len = usize::try_from(plan.padded_len).map_err(|_| ChunkError::Overflow)?;
    (0..len)
        .map(|i| {
            u32::try_from(i)
                .map(|idx| plan.source_frame(idx))
                .map_err(|_| ChunkError::Overflow)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ltx_shape::ScaleFactors;

    const SCALE: ScaleFactors = ScaleFactors::LTX2;

    // ── reflect_index ──────────────────────────────────────────────────────

    #[test]
    fn reflect_passthrough() {
        for i in 0..5_u32 {
            assert_eq!(reflect_index(i, 5), i, "index {i}");
        }
    }

    #[test]
    fn reflect_beyond_end() {
        // len=5, period=8 → [0,1,2,3,4,3,2,1, 0,1,2,3,4,3,2,1, …]
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(7, 5), 1);
        assert_eq!(reflect_index(8, 5), 0);
        assert_eq!(reflect_index(9, 5), 1);
        assert_eq!(reflect_index(12, 5), 4);
    }

    #[test]
    fn reflect_degenerate_len_zero() {
        assert_eq!(reflect_index(0, 0), 0);
        assert_eq!(reflect_index(99, 0), 0);
    }

    #[test]
    fn reflect_degenerate_len_one() {
        assert_eq!(reflect_index(0, 1), 0);
        assert_eq!(reflect_index(42, 1), 0);
    }

    #[test]
    fn reflect_len_two() {
        // len=2, period=2 → [0,1, 0,1, …]
        assert_eq!(reflect_index(0, 2), 0);
        assert_eq!(reflect_index(1, 2), 1);
        assert_eq!(reflect_index(2, 2), 0);
        assert_eq!(reflect_index(3, 2), 1);
    }

    #[test]
    fn reflect_no_edge_repetition() {
        // "reflect" mode: the folded edge is NOT repeated.
        // len=5: indices 4,3,2,1 after the last real frame (4), not 4,4,3,2.
        let seq: Vec<u32> = (0..13).map(|i| reflect_index(i, 5)).collect();
        assert_eq!(seq, [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4]);
    }

    // ── pad_to_grid ────────────────────────────────────────────────────────

    #[test]
    fn pad_already_on_grid() {
        let p = pad_to_grid(9, SCALE).unwrap();
        assert_eq!(p.padded_len, 9);
        assert_eq!(p.keep_len, 9);
        assert!(p.is_noop());
    }

    #[test]
    fn pad_off_grid() {
        // Next 8k+1 above 10 is 17.
        let p = pad_to_grid(10, SCALE).unwrap();
        assert_eq!(p.padded_len, 17);
        assert_eq!(p.keep_len, 10);
        assert!(!p.is_noop());
    }

    #[test]
    fn pad_single_frame() {
        let p = pad_to_grid(1, SCALE).unwrap();
        assert_eq!(p.padded_len, 1);
        assert!(p.is_noop());
    }

    #[test]
    fn padded_frame_map_roundtrip() {
        let p = pad_to_grid(5, SCALE).unwrap(); // padded_len = 9
        assert_eq!(p.padded_len, 9);
        let map = padded_frame_map(p).unwrap();
        // First 5 are identity; frames 5..9 are reflections.
        assert_eq!(&map[..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&map[5..], &[3, 2, 1, 0]); // reflect_index(5..8, 5)
    }

    #[test]
    fn source_frame_identity_for_real_indices() {
        let p = PadPlan {
            padded_len: 17,
            keep_len: 5,
        };
        for i in 0..5_u32 {
            assert_eq!(p.source_frame(i), i);
        }
    }

    #[test]
    fn source_frame_reflects_virtual_indices() {
        let p = PadPlan {
            padded_len: 9,
            keep_len: 5,
        };
        assert_eq!(p.source_frame(5), 3);
        assert_eq!(p.source_frame(6), 2);
        assert_eq!(p.source_frame(7), 1);
        assert_eq!(p.source_frame(8), 0);
    }
}
