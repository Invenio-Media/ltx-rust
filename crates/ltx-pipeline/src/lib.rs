//! End-to-end orchestration for LTX alpha matte generation.
//!
//! The pipeline probes an input clip, reflects it onto the LTX temporal grid,
//! splits it into overlapping chunks, calls an [`AlphaBackend`] for each chunk,
//! blends chunk seams, and writes one alpha EXR per real input frame.
//!
//! [`AlphaBackend`]: ltx_backend::AlphaBackend

use std::path::Path;

use ltx_backend::{AlphaBackend, AlphaChunk, BackendError, Keyframes, VideoChunk};
use ltx_chunk::{Blender, Chunk, ChunkError, PadPlan, pad_to_grid, plan, seam_cond_plan};
use ltx_io::{FrameStream, IoError, MatteWriter, Metadata};
use ltx_shape::ScaleFactors;
use thiserror::Error;

/// User-tunable pipeline settings.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum frames sent to the backend in one chunk. Must be `8k + 1`.
    pub chunk_len: u32,
    /// Temporal overlap between chunks. Must be a multiple of 8.
    pub overlap: u32,
    /// Base seed passed to the backend. Chunk index is added to this value.
    pub seed: u64,
    /// EXR filename prefix.
    pub matte_prefix: String,
    /// Supply seam keyframes from the previous generated matte.
    pub seam_keyframes: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            chunk_len: 49,
            overlap: 8,
            seed: 0,
            matte_prefix: "alpha".to_owned(),
            seam_keyframes: true,
        }
    }
}

impl PipelineConfig {
    /// Check the chunk length and overlap rules that the chunk planner and
    /// blender enforce, without touching any input.
    ///
    /// # Errors
    /// Returns [`ChunkError`] when `chunk_len` is not `8k + 1`, `overlap` is
    /// not a multiple of 8, or the stride is smaller than the overlap.
    pub fn validate(&self) -> Result<(), ChunkError> {
        Blender::new(self.chunk_len, self.overlap, 1, 1).map(drop)
    }
}

/// Summary of a completed pipeline run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineReport {
    /// Real input frames reported by `ffprobe`.
    pub input_frames: u32,
    /// Reflected frame count passed through chunk planning.
    pub padded_frames: u32,
    /// Number of backend chunks run.
    pub chunk_count: u32,
    /// Number of real alpha frames written.
    pub written_frames: u32,
}

/// Pipeline failures.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Video I/O failed.
    #[error("I/O: {0}")]
    Io(#[from] IoError),
    /// Backend failed.
    #[error("backend: {0}")]
    Backend(#[from] BackendError),
    /// Chunk math failed.
    #[error("chunk: {0}")]
    Chunk(#[from] ChunkError),
    /// Temporal grid padding overflowed.
    #[error("frame count is too large to pad onto the LTX grid")]
    PadOverflow,
    /// Source clip contains no frames.
    #[error("input has zero frames")]
    EmptyInput,
    /// Dimension arithmetic overflowed.
    #[error("dimension overflow")]
    Overflow,
    /// Backend input chunk validation failed.
    #[error("invalid video chunk: {0}")]
    InvalidVideoChunk(String),
    /// Backend output has wrong dimensions or length.
    #[error("invalid alpha chunk: {0}")]
    InvalidAlphaChunk(String),
    /// Blending did not emit all real frames.
    #[error("pipeline wrote {written} frames, emitted {emitted}, expected {expected}")]
    OutputFrameCount {
        written: u32,
        emitted: u32,
        expected: u32,
    },
}

/// Run the alpha pipeline and write one EXR per real input frame.
///
/// # Errors
/// Returns [`PipelineError`] when probing, decoding, backend generation,
/// blending, or EXR writing fails.
pub fn run_video<B: AlphaBackend>(
    backend: &B,
    input: &Path,
    output_dir: &Path,
    config: &PipelineConfig,
) -> Result<PipelineReport, PipelineError> {
    config.validate()?;
    let metadata = ltx_io::probe(input)?;
    if metadata.frame_count == 0 {
        return Err(PipelineError::EmptyInput);
    }
    let writer = MatteWriter::new(output_dir.to_path_buf(), config.matte_prefix.clone())?;
    let mut source = FfmpegSource { input, metadata };
    run_with_source(
        backend,
        &mut source,
        config,
        |frame_index, width, height, alpha| {
            writer.write_frame(frame_index, width, height, alpha)?;
            Ok(())
        },
    )
}

trait ChunkSource {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn real_frame_count(&self) -> u32;
    fn read_chunk(&mut self, chunk: Chunk, pad: PadPlan) -> Result<VideoChunk, PipelineError>;
}

struct FfmpegSource<'a> {
    input: &'a Path,
    metadata: Metadata,
}

impl ChunkSource for FfmpegSource<'_> {
    fn width(&self) -> u32 {
        self.metadata.width
    }

    fn height(&self) -> u32 {
        self.metadata.height
    }

    fn real_frame_count(&self) -> u32 {
        self.metadata.frame_count
    }

    fn read_chunk(&mut self, chunk: Chunk, pad: PadPlan) -> Result<VideoChunk, PipelineError> {
        let source_indices = reflected_indices(chunk, pad)?;
        let first = source_indices
            .iter()
            .copied()
            .min()
            .ok_or(PipelineError::EmptyInput)?;
        let last = source_indices
            .iter()
            .copied()
            .max()
            .ok_or(PipelineError::EmptyInput)?;
        let end = last.checked_add(1).ok_or(PipelineError::Overflow)?;

        let frame_values = frame_pixels(self.metadata.width, self.metadata.height)?
            .checked_mul(3)
            .ok_or(PipelineError::Overflow)?;
        let chunk_frames = usize::try_from(chunk.len()).map_err(|_| PipelineError::Overflow)?;
        let mut data = Vec::with_capacity(
            chunk_frames
                .checked_mul(frame_values)
                .ok_or(PipelineError::Overflow)?,
        );

        let mut stream = FrameStream::open_with_size(
            self.input,
            first,
            end,
            self.metadata.width,
            self.metadata.height,
        )?;

        if source_indices_are_contiguous(&source_indices, first) {
            append_contiguous_stream(&mut stream, chunk.len(), &mut data)?;
        } else {
            let frames = stream.collect_frames()?;
            append_mapped_frames(&frames, &source_indices, first, &mut data)?;
        }

        VideoChunk::new(
            chunk.start,
            self.metadata.width,
            self.metadata.height,
            chunk.len(),
            data,
        )
        .ok_or_else(|| PipelineError::InvalidVideoChunk("VideoChunk::new rejected data".to_owned()))
    }
}

#[derive(Debug, Clone)]
struct PreviousAlphaTail {
    width: u32,
    height: u32,
    data: Vec<f32>,
    frame_count: u32,
}

fn source_indices_are_contiguous(indices: &[u32], first: u32) -> bool {
    indices.iter().copied().enumerate().all(|(offset, value)| {
        u32::try_from(offset)
            .ok()
            .and_then(|offset_u32| first.checked_add(offset_u32))
            == Some(value)
    })
}

fn append_contiguous_stream(
    stream: &mut FrameStream,
    expected_frames: u32,
    data: &mut Vec<f32>,
) -> Result<(), PipelineError> {
    let mut read_frames = 0_u32;
    while let Some(frame) = stream.next_frame() {
        let frame = frame?;
        data.extend_from_slice(&frame.data);
        read_frames = read_frames.saturating_add(1);
    }
    if read_frames != expected_frames {
        return Err(PipelineError::InvalidVideoChunk(format!(
            "ffmpeg returned {read_frames} frames, expected {expected_frames}"
        )));
    }
    Ok(())
}

fn append_mapped_frames(
    frames: &[ltx_io::RgbF32Frame],
    source_indices: &[u32],
    first: u32,
    data: &mut Vec<f32>,
) -> Result<(), PipelineError> {
    for source_idx in source_indices {
        let rel = source_idx
            .checked_sub(first)
            .ok_or(PipelineError::Overflow)?;
        let rel_usize = usize::try_from(rel).map_err(|_| PipelineError::Overflow)?;
        let frame = frames.get(rel_usize).ok_or_else(|| {
            PipelineError::InvalidVideoChunk(format!("source frame {source_idx} missing"))
        })?;
        if frame.index != *source_idx {
            return Err(PipelineError::InvalidVideoChunk(format!(
                "source frame {} decoded as {}",
                source_idx, frame.index
            )));
        }
        data.extend_from_slice(&frame.data);
    }
    Ok(())
}

fn run_with_source<B, S, W>(
    backend: &B,
    source: &mut S,
    config: &PipelineConfig,
    mut write_frame: W,
) -> Result<PipelineReport, PipelineError>
where
    B: AlphaBackend,
    S: ChunkSource,
    W: FnMut(u32, u32, u32, &[f32]) -> Result<(), PipelineError>,
{
    let input_frames = source.real_frame_count();
    if input_frames == 0 {
        return Err(PipelineError::EmptyInput);
    }

    let pad = pad_to_grid(input_frames, ScaleFactors::LTX2).ok_or(PipelineError::PadOverflow)?;
    let chunks = plan(pad.padded_len, config.chunk_len, config.overlap)?;
    let chunk_count = u32::try_from(chunks.len()).map_err(|_| PipelineError::Overflow)?;
    let frame_pixels_u32 = source
        .width()
        .checked_mul(source.height())
        .ok_or(PipelineError::Overflow)?;
    let frame_values = frame_pixels(source.width(), source.height())?;
    let mut blender = Blender::new(config.chunk_len, config.overlap, frame_pixels_u32, 1)?;

    let mut emitted = 0_u32;
    let mut written = 0_u32;
    let mut previous_tail: Option<PreviousAlphaTail> = None;

    for (chunk_idx, chunk) in chunks.iter().copied().enumerate() {
        let video = source.read_chunk(chunk, pad)?;
        let cond = if config.seam_keyframes {
            match previous_tail.as_ref() {
                Some(prev) => keyframes_from_previous(prev, chunk.real_overlap)?,
                None => None,
            }
        } else {
            None
        };
        let seed = seed_for_chunk(config.seed, chunk_idx)?;
        let mut alpha = backend.run_chunk(&video, seed, cond.as_ref())?;
        validate_alpha_chunk(&alpha, &video, frame_values)?;
        sanitize_alpha_chunk(&mut alpha)?;
        push_alpha_to_blender(&mut blender, &alpha, frame_values)?;
        let next_overlap = chunks
            .get(chunk_idx.saturating_add(1))
            .map_or(0, |next| next.real_overlap);
        blender.finish_chunk(chunk.len(), next_overlap)?;
        drain_ready(
            &mut blender,
            &mut emitted,
            &mut written,
            input_frames,
            source.width(),
            source.height(),
            &mut write_frame,
        )?;
        previous_tail = alpha_tail_for_next(&alpha, next_overlap, frame_values)?;
    }

    for frame in blender.flush() {
        if emitted < input_frames {
            write_frame(emitted, source.width(), source.height(), &frame)?;
            written = written.saturating_add(1);
        }
        emitted = emitted.saturating_add(1);
    }

    if written != input_frames {
        return Err(PipelineError::OutputFrameCount {
            written,
            emitted,
            expected: input_frames,
        });
    }

    Ok(PipelineReport {
        input_frames,
        padded_frames: pad.padded_len,
        chunk_count,
        written_frames: written,
    })
}

fn reflected_indices(chunk: Chunk, pad: PadPlan) -> Result<Vec<u32>, PipelineError> {
    let len = usize::try_from(chunk.len()).map_err(|_| PipelineError::Overflow)?;
    let mut indices = Vec::with_capacity(len);
    for offset in 0..chunk.len() {
        let padded_idx = chunk
            .start
            .checked_add(offset)
            .ok_or(PipelineError::Overflow)?;
        indices.push(pad.source_frame(padded_idx));
    }
    Ok(indices)
}

fn frame_pixels(width: u32, height: u32) -> Result<usize, PipelineError> {
    let w = usize::try_from(width).map_err(|_| PipelineError::Overflow)?;
    let h = usize::try_from(height).map_err(|_| PipelineError::Overflow)?;
    w.checked_mul(h).ok_or(PipelineError::Overflow)
}

fn seed_for_chunk(seed: u64, chunk_idx: usize) -> Result<u64, PipelineError> {
    let offset = u64::try_from(chunk_idx).map_err(|_| PipelineError::Overflow)?;
    seed.checked_add(offset).ok_or(PipelineError::Overflow)
}

fn validate_alpha_chunk(
    alpha: &AlphaChunk,
    video: &VideoChunk,
    frame_values: usize,
) -> Result<(), PipelineError> {
    if alpha.start_frame != video.start_frame
        || alpha.width != video.width
        || alpha.height != video.height
        || alpha.frame_count != video.frame_count
    {
        return Err(PipelineError::InvalidAlphaChunk(
            "metadata does not match input chunk".to_owned(),
        ));
    }
    let frames = usize::try_from(alpha.frame_count).map_err(|_| PipelineError::Overflow)?;
    let expected = frames
        .checked_mul(frame_values)
        .ok_or(PipelineError::Overflow)?;
    if alpha.data.len() != expected {
        return Err(PipelineError::InvalidAlphaChunk(format!(
            "data length {} != expected {expected}",
            alpha.data.len()
        )));
    }
    Ok(())
}

fn sanitize_alpha_chunk(alpha: &mut AlphaChunk) -> Result<(), PipelineError> {
    for value in &mut alpha.data {
        if !value.is_finite() {
            return Err(PipelineError::InvalidAlphaChunk(
                "alpha contains NaN or Inf".to_owned(),
            ));
        }
        *value = value.clamp(0.0_f32, 1.0_f32);
    }
    Ok(())
}

fn push_alpha_to_blender(
    blender: &mut Blender,
    alpha: &AlphaChunk,
    frame_values: usize,
) -> Result<(), PipelineError> {
    for frame_idx in 0..alpha.frame_count {
        let frame = frame_slice(
            &alpha.data,
            frame_idx,
            alpha.frame_count,
            frame_values,
            "frame slice missing",
        )?;
        blender.push_frame(frame)?;
    }
    Ok(())
}

fn drain_ready<W>(
    blender: &mut Blender,
    emitted: &mut u32,
    written: &mut u32,
    input_frames: u32,
    width: u32,
    height: u32,
    write_frame: &mut W,
) -> Result<(), PipelineError>
where
    W: FnMut(u32, u32, u32, &[f32]) -> Result<(), PipelineError>,
{
    for frame in blender.drain() {
        if *emitted < input_frames {
            write_frame(*emitted, width, height, &frame)?;
            *written = written.saturating_add(1);
        }
        *emitted = emitted.saturating_add(1);
    }
    Ok(())
}

fn alpha_tail_for_next(
    alpha: &AlphaChunk,
    next_overlap: u32,
    frame_values: usize,
) -> Result<Option<PreviousAlphaTail>, PipelineError> {
    if next_overlap == 0 {
        return Ok(None);
    }
    if next_overlap > alpha.frame_count {
        return Err(PipelineError::InvalidAlphaChunk(
            "next overlap exceeds alpha chunk length".to_owned(),
        ));
    }
    let start_frame = alpha.frame_count.saturating_sub(next_overlap);
    let start = usize::try_from(start_frame)
        .map_err(|_| PipelineError::Overflow)?
        .checked_mul(frame_values)
        .ok_or(PipelineError::Overflow)?;
    let end = usize::try_from(alpha.frame_count)
        .map_err(|_| PipelineError::Overflow)?
        .checked_mul(frame_values)
        .ok_or(PipelineError::Overflow)?;
    let data = alpha
        .data
        .get(start..end)
        .ok_or_else(|| PipelineError::InvalidAlphaChunk("tail slice missing".to_owned()))?
        .to_vec();
    Ok(Some(PreviousAlphaTail {
        width: alpha.width,
        height: alpha.height,
        data,
        frame_count: next_overlap,
    }))
}

fn keyframes_from_previous(
    previous: &PreviousAlphaTail,
    overlap: u32,
) -> Result<Option<Keyframes>, PipelineError> {
    if overlap == 0 {
        return Ok(None);
    }
    if overlap > previous.frame_count {
        return Err(PipelineError::InvalidAlphaChunk(
            "requested overlap exceeds saved previous tail".to_owned(),
        ));
    }

    let plan = seam_cond_plan(overlap)?;
    let frame_values = frame_pixels(previous.width, previous.height)?;
    let rgb_values = frame_values.checked_mul(3).ok_or(PipelineError::Overflow)?;
    let cond_count = plan.conditioning_frames.len();
    let mut frames = Vec::with_capacity(
        cond_count
            .checked_mul(rgb_values)
            .ok_or(PipelineError::Overflow)?,
    );
    let mut indices = Vec::with_capacity(cond_count);
    let mut strength: Option<f32> = None;

    for cond_frame in plan.conditioning_frames {
        let alpha = frame_slice(
            &previous.data,
            cond_frame.chunk_local_idx,
            previous.frame_count,
            frame_values,
            "keyframe slice missing",
        )?;
        append_gray_rgb(alpha, &mut frames);
        indices.push(cond_frame.chunk_local_idx);
        match strength {
            Some(prev) if prev.to_bits() != cond_frame.strength.to_bits() => {
                return Err(PipelineError::InvalidAlphaChunk(
                    "seam plan returned mixed keyframe strengths".to_owned(),
                ));
            }
            Some(_) => {}
            None => strength = Some(cond_frame.strength),
        }
    }

    let frame_count = u32::try_from(indices.len()).map_err(|_| PipelineError::Overflow)?;
    Ok(Some(Keyframes {
        frames,
        indices,
        strength: strength.unwrap_or(ltx_chunk::DEFAULT_KEYFRAME_STRENGTH),
        width: previous.width,
        height: previous.height,
        frame_count,
    }))
}

fn frame_slice<'a>(
    data: &'a [f32],
    frame_idx: u32,
    frame_count: u32,
    frame_values: usize,
    missing: &str,
) -> Result<&'a [f32], PipelineError> {
    if frame_idx >= frame_count {
        return Err(PipelineError::InvalidAlphaChunk(format!(
            "frame index {frame_idx} outside frame count {frame_count}"
        )));
    }
    let start = usize::try_from(frame_idx)
        .map_err(|_| PipelineError::Overflow)?
        .checked_mul(frame_values)
        .ok_or(PipelineError::Overflow)?;
    let end = start
        .checked_add(frame_values)
        .ok_or(PipelineError::Overflow)?;
    data.get(start..end)
        .ok_or_else(|| PipelineError::InvalidAlphaChunk(missing.to_owned()))
}

fn append_gray_rgb(alpha: &[f32], out: &mut Vec<f32>) {
    for value in alpha {
        out.push(*value);
        out.push(*value);
        out.push(*value);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use ltx_backend::{BackendError, MemSample};

    struct MemorySource {
        width: u32,
        height: u32,
        frames: Vec<Vec<f32>>,
    }

    impl MemorySource {
        fn new(width: u32, height: u32, frame_count: u32) -> Self {
            let pixels = usize::try_from(width)
                .unwrap()
                .checked_mul(usize::try_from(height).unwrap())
                .unwrap();
            let mut frames = Vec::new();
            for frame_idx in 0..frame_count {
                let value = frame_idx.to_string().parse::<f32>().unwrap() / 100.0_f32;
                let mut frame = Vec::with_capacity(pixels.checked_mul(3).unwrap());
                for _ in 0..pixels {
                    frame.push(value);
                    frame.push(value);
                    frame.push(value);
                }
                frames.push(frame);
            }
            Self {
                width,
                height,
                frames,
            }
        }
    }

    impl ChunkSource for MemorySource {
        fn width(&self) -> u32 {
            self.width
        }

        fn height(&self) -> u32 {
            self.height
        }

        fn real_frame_count(&self) -> u32 {
            u32::try_from(self.frames.len()).unwrap()
        }

        fn read_chunk(&mut self, chunk: Chunk, pad: PadPlan) -> Result<VideoChunk, PipelineError> {
            let mut frames = Vec::new();
            for padded_idx in chunk.start..chunk.end {
                let src = pad.source_frame(padded_idx);
                let src_usize = usize::try_from(src).unwrap();
                frames.extend_from_slice(self.frames.get(src_usize).unwrap());
            }
            VideoChunk::new(chunk.start, self.width, self.height, chunk.len(), frames).ok_or_else(
                || PipelineError::InvalidVideoChunk("memory source built bad chunk".to_owned()),
            )
        }
    }

    #[derive(Default)]
    struct RecordingBackend {
        keyframe_counts: RefCell<Vec<u32>>,
        seeds: RefCell<Vec<u64>>,
        keyframes: RefCell<Vec<Option<Keyframes>>>,
    }

    impl AlphaBackend for RecordingBackend {
        fn probe(&self, _shape: ltx_shape::PixelShape) -> Result<MemSample, BackendError> {
            Ok(MemSample { peak_bytes: 1 })
        }

        fn run_chunk(
            &self,
            rgb: &VideoChunk,
            seed: u64,
            cond: Option<&Keyframes>,
        ) -> Result<AlphaChunk, BackendError> {
            self.seeds.borrow_mut().push(seed);
            self.keyframe_counts
                .borrow_mut()
                .push(cond.map_or(0, |kf| kf.frame_count));
            self.keyframes.borrow_mut().push(cond.cloned());

            let pixels = usize::try_from(rgb.width)
                .unwrap()
                .checked_mul(usize::try_from(rgb.height).unwrap())
                .unwrap();
            let frames = usize::try_from(rgb.frame_count).unwrap();
            let mut data = Vec::with_capacity(frames.checked_mul(pixels).unwrap());
            for frame_idx in 0..frames {
                let frame_start = frame_idx
                    .checked_mul(pixels)
                    .unwrap()
                    .checked_mul(3)
                    .unwrap();
                let value = *rgb.frames.get(frame_start).unwrap();
                for _ in 0..pixels {
                    data.push(value);
                }
            }

            Ok(AlphaChunk {
                start_frame: rgb.start_frame,
                width: rgb.width,
                height: rgb.height,
                data,
                frame_count: rgb.frame_count,
            })
        }
    }

    #[test]
    fn pipeline_blends_chunks_and_trims_reflection_padding() {
        let backend = RecordingBackend::default();
        let mut source = MemorySource::new(32, 32, 25);
        let config = PipelineConfig {
            chunk_len: 17,
            overlap: 8,
            seed: 10,
            matte_prefix: "alpha".to_owned(),
            seam_keyframes: true,
        };
        let mut written: Vec<(u32, Vec<f32>)> = Vec::new();

        let report = run_with_source(
            &backend,
            &mut source,
            &config,
            |frame_idx, _width, _height, alpha| {
                written.push((frame_idx, alpha.to_vec()));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(report.input_frames, 25);
        assert_eq!(report.padded_frames, 25);
        assert_eq!(report.written_frames, 25);
        assert_eq!(written.len(), 25);
        assert!(report.chunk_count >= 2);

        for (frame_idx, alpha) in written {
            let expected = frame_idx.to_string().parse::<f32>().unwrap() / 100.0_f32;
            assert!(
                alpha
                    .iter()
                    .all(|value| (*value - expected).abs() < 0.000_001)
            );
        }

        assert_eq!(*backend.seeds.borrow(), vec![10, 11]);
        assert_eq!(*backend.keyframe_counts.borrow(), vec![0, 1]);
        let keyframes = backend.keyframes.borrow();
        let seam_frame = ltx_chunk::plan(25, 17, 8).unwrap().get(1).unwrap().start;
        let expected_keyframe = seam_frame.to_string().parse::<f32>().unwrap() / 100.0_f32;
        let keyframe = keyframes.get(1).unwrap().as_ref().unwrap();
        assert_eq!(keyframe.indices, vec![0]);
        assert_eq!(keyframe.frame_count, 1);
        assert_eq!(keyframe.width, 32);
        assert_eq!(keyframe.height, 32);
        assert_eq!(
            keyframe.strength.to_bits(),
            ltx_chunk::DEFAULT_KEYFRAME_STRENGTH.to_bits()
        );
        assert!(
            keyframe
                .frames
                .iter()
                .all(|value| (*value - expected_keyframe).abs() < 0.000_001)
        );
    }

    #[test]
    fn pipeline_writes_only_real_frames_after_padding() {
        let backend = RecordingBackend::default();
        let mut source = MemorySource::new(32, 32, 10);
        let config = PipelineConfig {
            chunk_len: 17,
            overlap: 8,
            seed: 0,
            matte_prefix: "alpha".to_owned(),
            seam_keyframes: false,
        };
        let mut written = Vec::new();

        let report = run_with_source(
            &backend,
            &mut source,
            &config,
            |frame_idx, _width, _height, _alpha| {
                written.push(frame_idx);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(report.input_frames, 10);
        assert_eq!(report.padded_frames, 17);
        assert_eq!(report.written_frames, 10);
        assert_eq!(written, (0..10).collect::<Vec<u32>>());
    }
}
