//! Smoothstep crossfade blending across temporal chunk seams.
//!
//! Each seam has `overlap` frames where consecutive chunks contribute. The
//! weight for the incoming chunk at offset `i` within the overlap is
//! `smoothstep(i / overlap)` and for the outgoing chunk it is
//! `1 − smoothstep(i / overlap)`.  By construction these sum to 1 at every
//! frame.  Because the stride ≥ overlap constraint is enforced by [`plan`],
//! no frame belongs to more than two chunks, so the weights tile perfectly.
//!
//! ## Fixed-seed note
//! The random seed used to generate each chunk's latent noise is the caller's
//! responsibility. The blender receives decoded pixel frames and blends their
//! f32 values; it does not touch any random state.
//!
//! ## Streaming, bounded memory
//! The [`Blender`] buffers exactly `overlap × frame_size` f32 values: the
//! tail of the previous chunk that will be crossfaded with the start of the
//! next chunk. Frames that no future chunk can modify are moved to an internal
//! ready queue and emitted via [`Blender::drain`]. After the last chunk is
//! pushed, call [`Blender::flush`] to emit the tail.
//!
//! [`plan`]: crate::plan::plan

use std::collections::VecDeque;

use crate::error::ChunkError;

// ── math ──────────────────────────────────────────────────────────────────────

/// Cubic smoothstep on t ∈ [0, 1].  Values outside [0, 1] are clamped.
///
/// `smoothstep(t) = t² (3 − 2t)`
#[must_use]
pub fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0_f64, 1.0_f64);
    t * t * 2.0_f64.mul_add(-t, 3.0_f64)
}

/// Blend weight for the **incoming** chunk at frame `offset` within an overlap
/// of `overlap_len` frames.
///
/// Returns `smoothstep(offset / overlap_len)` in `[0.0, 1.0]` (f32).
/// The outgoing-chunk weight is `1.0 − crossfade_weight(offset, overlap_len)`.
///
/// # Panics (never)
/// `overlap_len` is taken as `u32`; the function returns 1.0 when called with
/// `offset ≥ overlap_len` (clamped by smoothstep).
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "smoothstep output ∈ [0.0, 1.0]; narrowing f64 to f32 loses ≤ 1 ULP; safe"
)]
pub fn crossfade_weight(offset: u32, overlap_len: u32) -> f32 {
    if overlap_len == 0 {
        return 1.0_f32;
    }
    let t = f64::from(offset) / f64::from(overlap_len);
    smoothstep(t) as f32
}

// ── blender ───────────────────────────────────────────────────────────────────

/// Streaming crossfade blender for overlapping temporal chunks.
///
/// Push each chunk's frames in order via [`push_frame`][`Blender::push_frame`],
/// call [`finish_chunk`][`Blender::finish_chunk`] after the last frame of each
/// chunk, drain emitted frames with [`drain`][`Blender::drain`], and call
/// [`flush`][`Blender::flush`] after the last chunk to release the tail.
///
/// Memory: `overlap × frame_pixels × channel_count` f32 values.
pub struct Blender {
    chunk_len: u32,
    overlap: u32,
    /// `stride` = `chunk_len` − `overlap`.
    stride: u32,
    /// Number of f32 values per frame.
    frame_size: usize,
    /// Tail frames from the previous chunk waiting to be crossfaded.
    /// `pending[i * frame_size .. (i+1) * frame_size]` is pending frame i.
    pending: Vec<f32>,
    /// Frames ready to emit, in global order.
    ready: VecDeque<Vec<f32>>,
    /// How many frames have been pushed to the current chunk (resets each chunk).
    chunk_cursor: u32,
    /// True until the first chunk is complete.
    is_first_chunk: bool,
}

impl Blender {
    /// Creates a new blender.
    ///
    /// - `chunk_len` must be `8k + 1`.
    /// - `overlap` must be a multiple of 8 and satisfy `stride ≥ overlap`.
    /// - `frame_pixels` = `width × height`.
    /// - `channel_count` ≥ 1.
    ///
    /// # Errors
    /// Returns [`ChunkError`] when any argument is inconsistent.
    pub fn new(
        chunk_len: u32,
        overlap: u32,
        frame_pixels: u32,
        channel_count: usize,
    ) -> Result<Self, ChunkError> {
        use crate::plan::on_frame_grid;
        use ltx_shape::ScaleFactors;

        if !on_frame_grid(chunk_len) {
            return Err(ChunkError::ChunkLenNotOnGrid(chunk_len));
        }
        if !overlap.is_multiple_of(ScaleFactors::LTX2.time.get()) {
            return Err(ChunkError::OverlapNotMultipleOf8(overlap));
        }
        if overlap >= chunk_len {
            return Err(ChunkError::OverlapNotSmallerThanChunkLen { overlap, chunk_len });
        }
        let stride = chunk_len.saturating_sub(overlap);
        if stride < overlap {
            return Err(ChunkError::StrideSmallerThanOverlap { stride, overlap });
        }

        let frame_size = usize::try_from(frame_pixels)
            .map_err(|_| ChunkError::Overflow)?
            .checked_mul(channel_count)
            .ok_or(ChunkError::Overflow)?;

        let pending_len = usize::try_from(overlap)
            .map_err(|_| ChunkError::Overflow)?
            .checked_mul(frame_size)
            .ok_or(ChunkError::Overflow)?;

        Ok(Self {
            chunk_len,
            overlap,
            stride,
            frame_size,
            pending: vec![0.0_f32; pending_len],
            ready: VecDeque::new(),
            chunk_cursor: 0,
            is_first_chunk: true,
        })
    }

    /// Pushes the next frame of the current chunk.
    ///
    /// Frames must be pushed in order 0, 1, …, `chunk_len − 1` for each
    /// chunk. The chunk-local index is tracked internally; callers do not
    /// supply it explicitly, but [`push_frame`][`Blender::push_frame`] returns
    /// an error if the frame count exceeds `chunk_len` without an intervening
    /// [`finish_chunk`][`Blender::finish_chunk`] call.
    ///
    /// `pixels` must have length `frame_pixels × channel_count`.
    ///
    /// # Errors
    /// Returns [`ChunkError`] on length mismatch or counter overflow.
    pub fn push_frame(&mut self, pixels: &[f32]) -> Result<(), ChunkError> {
        if pixels.len() != self.frame_size {
            return Err(ChunkError::PixelsWrongLen {
                got: pixels.len(),
                expected: self.frame_size,
            });
        }
        if self.chunk_cursor >= self.chunk_len {
            return Err(ChunkError::FrameOutOfOrder {
                idx: self.chunk_cursor,
                expected: 0,
            });
        }

        let i = self.chunk_cursor;

        if self.is_first_chunk {
            // First chunk has no previous pending.
            if i < self.stride {
                // Unique zone: emit directly.
                self.ready.push_back(pixels.to_vec());
            } else {
                // Tail zone: store for the next chunk.
                let pending_i = i.saturating_sub(self.stride);
                self.write_pending(pending_i, pixels)?;
            }
        } else if i < self.overlap {
            // Blend zone: crossfade pending with this frame.
            let w_new = crossfade_weight(i, self.overlap);
            let blended = self.blend_with_pending(i, pixels, w_new)?;
            self.ready.push_back(blended);
        } else if i < self.stride {
            // Middle zone: emit directly.
            self.ready.push_back(pixels.to_vec());
        } else {
            // Tail zone: store for the next chunk.
            let pending_i = i.saturating_sub(self.stride);
            self.write_pending(pending_i, pixels)?;
        }

        self.chunk_cursor = self.chunk_cursor.saturating_add(1);
        Ok(())
    }

    /// Signals that the current chunk is complete.
    ///
    /// Resets the frame counter and clears the first-chunk flag. Call this
    /// after pushing all `chunk_len` frames for each chunk.
    pub const fn finish_chunk(&mut self) {
        self.chunk_cursor = 0;
        self.is_first_chunk = false;
    }

    /// Returns an iterator over frames that are ready to emit, draining them
    /// from the internal queue.
    ///
    /// Safe to call at any time; yields nothing when no frames are ready.
    pub fn drain(&mut self) -> impl Iterator<Item = Vec<f32>> + '_ {
        self.ready.drain(..)
    }

    /// Emits the tail frames buffered from the last chunk, then drains them.
    ///
    /// Call this once after pushing and finishing the final chunk. Subsequent
    /// calls yield nothing (the buffer is cleared).
    pub fn flush(&mut self) -> impl Iterator<Item = Vec<f32>> + '_ {
        let overlap_usize = usize::try_from(self.overlap).unwrap_or(0);
        for i in 0..overlap_usize {
            let offset = i.saturating_mul(self.frame_size);
            let end = offset.saturating_add(self.frame_size);
            if let Some(slice) = self.pending.get(offset..end) {
                self.ready.push_back(slice.to_vec());
            }
        }
        self.ready.drain(..)
    }

    // ── private helpers ───────────────────────────────────────────────────

    fn pending_slice(&self, pending_i: u32) -> Result<&[f32], ChunkError> {
        let i = usize::try_from(pending_i).map_err(|_| ChunkError::Overflow)?;
        let offset = i.saturating_mul(self.frame_size);
        let end = offset.saturating_add(self.frame_size);
        self.pending.get(offset..end).ok_or(ChunkError::Overflow)
    }

    fn write_pending(&mut self, pending_i: u32, pixels: &[f32]) -> Result<(), ChunkError> {
        let i = usize::try_from(pending_i).map_err(|_| ChunkError::Overflow)?;
        let offset = i.saturating_mul(self.frame_size);
        let end = offset.saturating_add(self.frame_size);
        let slot = self
            .pending
            .get_mut(offset..end)
            .ok_or(ChunkError::Overflow)?;
        slot.copy_from_slice(pixels);
        Ok(())
    }

    fn blend_with_pending(
        &self,
        pending_i: u32,
        new_pixels: &[f32],
        w_new: f32,
    ) -> Result<Vec<f32>, ChunkError> {
        let old = self.pending_slice(pending_i)?;
        let w_old = 1.0_f32 - w_new;
        let blended: Vec<f32> = old
            .iter()
            .zip(new_pixels.iter())
            .map(|(a, b)| w_new.mul_add(*b, w_old * a))
            .collect();
        Ok(blended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_PIX: u32 = 4; // 2×2 pixels
    const CHANNELS: usize = 1;
    const FSIZE: usize = 4; // FRAME_PIX * CHANNELS

    fn make_frame(val: f32) -> Vec<f32> {
        vec![val; FSIZE]
    }

    // ── smoothstep & crossfade_weight ──────────────────────────────────────

    #[test]
    fn smoothstep_endpoints() {
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn smoothstep_midpoint() {
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn smoothstep_clamps_outside() {
        assert!((smoothstep(-0.5) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn crossfade_weight_endpoints() {
        // At offset 0: weight for incoming = 0 (seam: still fully old chunk).
        assert!((crossfade_weight(0, 8) - 0.0).abs() < 1e-6_f32);
        // At offset == overlap: weight would be 1 (clamped past end).
        assert!((crossfade_weight(8, 8) - 1.0).abs() < 1e-6_f32);
    }

    #[test]
    fn crossfade_weights_sum_to_one() {
        let overlap: u32 = 16;
        for i in 0..overlap {
            let w_new = crossfade_weight(i, overlap);
            let w_old = 1.0_f32 - w_new;
            assert!(
                (w_new + w_old - 1.0_f32).abs() < 1e-6_f32,
                "weights at i={i} do not sum to 1: w_new={w_new}, w_old={w_old}"
            );
        }
    }

    // ── blender ────────────────────────────────────────────────────────────

    #[test]
    fn constant_signal_reconstructs_exactly() {
        // Two chunks of constant value 1.0; blend should emit all 1.0.
        let chunk_len: u32 = 17;
        let overlap: u32 = 8;
        let total: u32 = 26; // stride=9, two chunks covering 26 frames
        let mut blender = Blender::new(chunk_len, overlap, FRAME_PIX, CHANNELS).unwrap();

        let mut output: Vec<f32> = Vec::new();

        // Chunk 0
        for _ in 0..chunk_len {
            blender.push_frame(&make_frame(1.0)).unwrap();
        }
        blender.finish_chunk();
        for f in blender.drain() {
            output.extend(f);
        }

        // Chunk 1
        for _ in 0..chunk_len {
            blender.push_frame(&make_frame(1.0)).unwrap();
        }
        blender.finish_chunk();
        for f in blender.drain() {
            output.extend(f);
        }
        for f in blender.flush() {
            output.extend(f);
        }

        // Every pixel should be 1.0.
        for (idx, &val) in output.iter().enumerate() {
            assert!(
                (val - 1.0_f32).abs() < 1e-6_f32,
                "pixel {idx}: expected 1.0, got {val}"
            );
        }
        let expected_pixels = usize::try_from(total).unwrap().saturating_mul(FSIZE);
        assert_eq!(output.len(), expected_pixels);
    }

    #[test]
    fn ramp_signal_reconstructs_exactly() {
        // Chunk 0 (starts at 0): frame i → value i.
        // Chunk 1 (starts at stride=9): frame i → value (9 + i).
        // Blend at seam offset k:
        //   w_new = smoothstep(k/8), output = w_new*(9+k) + (1-w_new)*k = k + 9*w_new.
        // This IS (9+k) when the chunks carry the correct global value.
        let chunk_len: u32 = 17;
        let overlap: u32 = 8;
        let stride: u32 = 9;
        let mut blender = Blender::new(chunk_len, overlap, FRAME_PIX, CHANNELS).unwrap();
        let mut output: Vec<f32> = Vec::new();

        // Chunk 0: frame i has value i.
        for i in 0..chunk_len {
            let v = f32::from(u16::try_from(i).unwrap());
            blender.push_frame(&[v; FSIZE]).unwrap();
        }
        blender.finish_chunk();
        for f in blender.drain() {
            output.extend(f);
        }

        // Chunk 1: frame i has value (stride + i).
        for i in 0..chunk_len {
            let v = f32::from(u16::try_from(stride.saturating_add(i)).unwrap());
            blender.push_frame(&[v; FSIZE]).unwrap();
        }
        blender.finish_chunk();
        for f in blender.drain() {
            output.extend(f);
        }
        for f in blender.flush() {
            output.extend(f);
        }

        // output.len() = (stride + chunk_len) * FSIZE = 26 * 4 = 104
        let n_frames = output.len().checked_div(FSIZE).unwrap_or(0);
        for g in 0..n_frames {
            let expected = f32::from(u16::try_from(g).unwrap());
            for ch in 0..FSIZE {
                let idx = g.saturating_mul(FSIZE).saturating_add(ch);
                let got = output[idx];
                assert!(
                    (got - expected).abs() < 1e-4_f32,
                    "frame {g} ch {ch}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn frames_emitted_in_order() {
        // Use chunk 0 → all -1.0, chunk 1 → all +1.0.
        // Unique-zone frames from chunk 0 must be -1.0.
        // Blend frames must be non-decreasing (smoothstep rises from -1 toward +1).
        // Frames from chunk 1's non-blend zone must be +1.0.
        let chunk_len: u32 = 17;
        let overlap: u32 = 8;
        let stride: u32 = 9;
        let mut blender = Blender::new(chunk_len, overlap, 1, 1).unwrap();
        let mut output: Vec<f32> = Vec::new();

        for _ in 0..chunk_len {
            blender.push_frame(&[-1.0_f32]).unwrap();
        }
        blender.finish_chunk();
        for f in blender.drain() {
            if let Some(&v) = f.first() {
                output.push(v);
            }
        }

        for _ in 0..chunk_len {
            blender.push_frame(&[1.0_f32]).unwrap();
        }
        blender.finish_chunk();
        for f in blender.drain() {
            if let Some(&v) = f.first() {
                output.push(v);
            }
        }
        for f in blender.flush() {
            if let Some(&v) = f.first() {
                output.push(v);
            }
        }

        let total = usize::try_from(stride.saturating_add(chunk_len)).unwrap();
        assert_eq!(output.len(), total, "expected {total} frames");

        // Unique zone of chunk 0: first stride frames are -1.0.
        let unique0 = usize::try_from(stride).unwrap();
        for v in output.iter().take(unique0) {
            assert!(
                (*v - (-1.0_f32)).abs() < 1e-6_f32,
                "unique-zone frame should be -1.0, got {v}"
            );
        }

        // Blend frames: monotonically non-decreasing.
        let blend_len = usize::try_from(overlap).unwrap();
        for i in unique0..unique0.saturating_add(blend_len).saturating_sub(1) {
            assert!(
                output[i] <= output[i.saturating_add(1)],
                "blend frames not monotonic at {i}"
            );
        }

        // Non-blend frames of chunk 1: all +1.0.
        let rest = unique0.saturating_add(blend_len);
        for v in output.iter().skip(rest) {
            assert!(
                (*v - 1.0_f32).abs() < 1e-6_f32,
                "chunk-1 non-blend frame should be +1.0, got {v}"
            );
        }
    }

    #[test]
    fn single_chunk_emits_all_via_flush() {
        let chunk_len: u32 = 9;
        let mut blender = Blender::new(chunk_len, 0, FRAME_PIX, CHANNELS).unwrap();
        for _ in 0..chunk_len {
            blender.push_frame(&make_frame(7.0)).unwrap();
        }
        blender.finish_chunk();
        let n_drain = blender.drain().count();
        let n_flush = blender.flush().count();
        let total_frames = n_drain.saturating_add(n_flush);
        assert_eq!(total_frames, usize::try_from(chunk_len).unwrap());
    }

    #[test]
    fn blender_wrong_pixel_len_is_error() {
        let mut b = Blender::new(9, 0, FRAME_PIX, CHANNELS).unwrap();
        assert!(matches!(
            b.push_frame(&[1.0_f32, 2.0]),
            Err(ChunkError::PixelsWrongLen { .. })
        ));
    }

    #[test]
    fn memory_is_bounded_by_overlap() {
        // Blender with overlap=8 and large chunk should buffer at most 8 frames.
        let overlap: u32 = 8;
        let chunk_len: u32 = 33;
        let b = Blender::new(chunk_len, overlap, 1, 1).unwrap();
        // pending vec length == overlap * frame_size
        assert_eq!(b.pending.len(), usize::try_from(overlap).unwrap());
    }
}
