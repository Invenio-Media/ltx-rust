//! Temporal chunk planner.
//!
//! Splits a clip of `total` frames into overlapping windows of `chunk_len`
//! (which must be `8k + 1`) with an `overlap` that is a multiple of 8.
//!
//! The stride `S = chunk_len − overlap`. Chunks start at `0, S, 2S, …`; the
//! last chunk is anchored to `total − chunk_len` so no padding is needed at
//! the right edge. When `total ≤ chunk_len` a single chunk `0..total` is
//! returned (the caller is responsible for reflection-padding if `total` is
//! not `8k + 1`).
//!
//! **Overlap constraint** – To keep blend weights summing to 1, the stride
//! must be at least as long as the overlap (`stride ≥ overlap`, equivalently
//! `chunk_len ≥ 2 × overlap`). This ensures every pixel frame belongs to at
//! most two chunks.

use crate::error::ChunkError;
use ltx_shape::ScaleFactors;

/// One window in the temporal chunk plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    /// First frame in this chunk (inclusive, global index).
    pub start: u32,
    /// One past the last frame in this chunk (exclusive, global index).
    pub end: u32,
    /// Overlap with the previous chunk (0 for the first chunk).
    ///
    /// For interior chunks this equals the `overlap` argument of [`plan`].
    /// For the anchored last chunk it may be larger when `total` is close to
    /// a stride boundary.
    pub real_overlap: u32,
}

impl Chunk {
    /// Number of frames in this chunk.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// True when the chunk contains no frames.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// Plans overlapping temporal chunks.
///
/// # Parameters
/// - `total` – total frame count; must be ≥ 1.
/// - `chunk_len` – frames per chunk; must be `8k + 1` for some `k ≥ 0`.
/// - `overlap` – frames shared between consecutive chunks; must be a multiple
///   of 8, less than `chunk_len`, and at most `chunk_len / 2` (so the stride
///   is at least the overlap length).
///
/// # Errors
/// Returns [`ChunkError`] when any constraint is violated.
pub fn plan(total: u32, chunk_len: u32, overlap: u32) -> Result<Vec<Chunk>, ChunkError> {
    if total == 0 {
        return Err(ChunkError::ZeroTotal);
    }

    // chunk_len must be 8k+1.
    if !on_frame_grid(chunk_len) {
        return Err(ChunkError::ChunkLenNotOnGrid(chunk_len));
    }

    // overlap < chunk_len.
    if overlap >= chunk_len {
        return Err(ChunkError::OverlapNotSmallerThanChunkLen { overlap, chunk_len });
    }

    // overlap must be a multiple of 8.
    if !overlap.is_multiple_of(ScaleFactors::LTX2.time.get()) {
        return Err(ChunkError::OverlapNotMultipleOf8(overlap));
    }

    // stride >= overlap (ensures ≤ 2 chunks per frame, blend weights sum to 1).
    let stride = chunk_len.saturating_sub(overlap);
    if stride < overlap {
        return Err(ChunkError::StrideSmallerThanOverlap { stride, overlap });
    }

    // Single-chunk path.
    if total <= chunk_len {
        return Ok(vec![Chunk {
            start: 0,
            end: total,
            real_overlap: 0,
        }]);
    }

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut start = 0_u32;

    // Push all non-last chunks: each covers [start, start + chunk_len).
    loop {
        let end = start.saturating_add(chunk_len);
        if end >= total {
            break;
        }
        let real_overlap = if chunks.is_empty() { 0 } else { overlap };
        chunks.push(Chunk {
            start,
            end,
            real_overlap,
        });
        start = start.saturating_add(stride);
    }

    // Anchored last chunk: always ends exactly at `total`.
    let last_start = total.saturating_sub(chunk_len);
    let real_overlap = chunks
        .last()
        .map_or(0, |c| c.end.saturating_sub(last_start));
    chunks.push(Chunk {
        start: last_start,
        end: total,
        real_overlap,
    });

    Ok(chunks)
}

/// Returns `true` when `frames` is of the form `8k + 1` (the LTX-2 VAE
/// temporal grid). Delegates the check to the `ltx-shape` scale factors.
pub(crate) fn on_frame_grid(frames: u32) -> bool {
    frames
        .checked_sub(1)
        .is_some_and(|rest| rest.is_multiple_of(ScaleFactors::LTX2.time.get()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────

    /// Every frame 0..total appears in at least one chunk.
    fn assert_all_frames_covered(chunks: &[Chunk], total: u32) {
        for f in 0..total {
            assert!(
                chunks.iter().any(|c| c.start <= f && f < c.end),
                "frame {f} not covered"
            );
        }
    }

    /// All chunks (except a short single chunk) have length == `chunk_len`.
    fn assert_all_lengths(chunks: &[Chunk], chunk_len: u32, total: u32) {
        for c in chunks {
            if total <= chunk_len {
                // Single-chunk: length == total.
                assert_eq!(c.len(), total);
            } else {
                assert_eq!(c.len(), chunk_len);
            }
        }
    }

    /// Chunk starts are strictly increasing.
    fn assert_starts_monotonic(chunks: &[Chunk]) {
        for pair in chunks.windows(2) {
            assert!(pair[0].start < pair[1].start, "starts not monotonic");
        }
    }

    // ── error cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_total_is_error() {
        assert_eq!(plan(0, 9, 0), Err(ChunkError::ZeroTotal));
    }

    #[test]
    fn chunk_len_not_on_grid_is_error() {
        assert_eq!(plan(10, 10, 0), Err(ChunkError::ChunkLenNotOnGrid(10)));
        assert_eq!(plan(10, 2, 0), Err(ChunkError::ChunkLenNotOnGrid(2)));
    }

    #[test]
    fn overlap_not_multiple_of_8_is_error() {
        assert_eq!(plan(50, 17, 4), Err(ChunkError::OverlapNotMultipleOf8(4)));
    }

    #[test]
    fn overlap_ge_chunk_len_is_error() {
        assert_eq!(
            plan(50, 17, 17),
            Err(ChunkError::OverlapNotSmallerThanChunkLen {
                overlap: 17,
                chunk_len: 17
            })
        );
    }

    #[test]
    fn stride_smaller_than_overlap_is_error() {
        // chunk_len=17, overlap=16 → stride=1 < 16
        assert_eq!(
            plan(50, 17, 16),
            Err(ChunkError::StrideSmallerThanOverlap {
                stride: 1,
                overlap: 16
            })
        );
    }

    // ── single-chunk cases ─────────────────────────────────────────────────

    #[test]
    fn single_chunk_total_eq_chunk_len() {
        // total == chunk_len: one chunk, no overlap
        let chunks = plan(17, 17, 8).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 17);
        assert_eq!(chunks[0].real_overlap, 0);
        assert_all_frames_covered(&chunks, 17);
        assert_all_lengths(&chunks, 17, 17);
    }

    #[test]
    fn single_chunk_total_less_than_chunk_len() {
        // total < chunk_len: one short chunk
        let chunks = plan(5, 17, 8).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].end, 5);
        assert_eq!(chunks[0].real_overlap, 0);
        assert_all_frames_covered(&chunks, 5);
    }

    #[test]
    fn single_chunk_zero_overlap() {
        let chunks = plan(17, 17, 0).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_all_frames_covered(&chunks, 17);
    }

    // ── multi-chunk: coverage + shape ─────────────────────────────────────

    #[test]
    fn two_chunks_last_anchored() {
        // total=18, chunk_len=17, overlap=8, stride=9
        // Chunk 0: 0..17; anchored last: 1..18 (real_overlap = 17−1 = 16)
        let chunks = plan(18, 17, 8).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 17);
        assert_eq!(chunks[0].real_overlap, 0);
        assert_eq!(chunks[1].start, 1);
        assert_eq!(chunks[1].end, 18);
        assert_eq!(chunks[1].real_overlap, 16); // anchored overlap > nominal
        assert_all_frames_covered(&chunks, 18);
        assert_all_lengths(&chunks, 17, 18);
        assert_starts_monotonic(&chunks);
    }

    #[test]
    fn three_chunks_exact_fit() {
        // total=35, chunk_len=17, overlap=8, stride=9
        // 0..17, 9..26, 18..35 (last anchored, real_overlap=26−18=8)
        let chunks = plan(35, 17, 8).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[1].start, 9);
        assert_eq!(chunks[2].start, 18);
        assert_eq!(chunks[2].end, 35);
        assert_eq!(chunks[2].real_overlap, 8);
        assert_all_frames_covered(&chunks, 35);
        assert_all_lengths(&chunks, 17, 35);
        assert_starts_monotonic(&chunks);
    }

    #[test]
    fn many_chunks_full_coverage() {
        // chunk_len=33, overlap=8, stride=25
        let (total, chunk_len, overlap) = (200_u32, 33_u32, 8_u32);
        let chunks = plan(total, chunk_len, overlap).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks.last().unwrap().end, total);
        assert_all_frames_covered(&chunks, total);
        assert_all_lengths(&chunks, chunk_len, total);
        assert_starts_monotonic(&chunks);
        // First chunk has no overlap
        assert_eq!(chunks[0].real_overlap, 0);
        // Interior chunks have exactly the nominal overlap
        for c in chunks.iter().take(chunks.len().saturating_sub(1)).skip(1) {
            assert_eq!(c.real_overlap, overlap);
        }
    }

    #[test]
    fn zero_overlap_no_blend_zones() {
        let (total, chunk_len) = (100_u32, 17_u32);
        let chunks = plan(total, chunk_len, 0).unwrap();
        assert_all_frames_covered(&chunks, total);
        for c in chunks.iter().take(chunks.len().saturating_sub(1)) {
            assert_eq!(c.real_overlap, 0);
        }
    }

    #[test]
    fn total_eq_chunk_len_plus_one() {
        // Boundary: exactly one anchored chunk beside the first.
        let (chunk_len, overlap) = (17_u32, 8_u32);
        let total = chunk_len.saturating_add(1); // 18
        let chunks = plan(total, chunk_len, overlap).unwrap();
        assert_eq!(chunks.last().unwrap().end, total);
        assert_all_frames_covered(&chunks, total);
        assert_starts_monotonic(&chunks);
    }

    #[test]
    fn first_chunk_zero_overlap_second_chunk_nominal() {
        let chunks = plan(60, 33, 8).unwrap();
        assert_eq!(chunks[0].real_overlap, 0);
        assert_eq!(chunks[1].real_overlap, 8);
    }

    #[test]
    fn on_frame_grid_correct() {
        assert!(on_frame_grid(1));
        assert!(on_frame_grid(9));
        assert!(on_frame_grid(17));
        assert!(on_frame_grid(33));
        assert!(!on_frame_grid(0));
        assert!(!on_frame_grid(8));
        assert!(!on_frame_grid(10));
    }
}
