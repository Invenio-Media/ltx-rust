//! Data types shared between the backend trait and its implementations.

/// A chunk of decoded RGB frames ready for alpha generation.
#[derive(Debug, Clone)]
pub struct VideoChunk {
    /// Clip-global index of the first frame.
    pub start_frame: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Interleaved RGB f32 values in `[0, 1]`.
    /// Layout: frame × row × col × channel (RGBRGB…), length
    /// `frame_count * width * height * 3`.
    pub frames: Vec<f32>,
    /// Number of frames in this chunk.
    pub frame_count: u32,
}

impl VideoChunk {
    /// Create a chunk, validating that `frames.len() == frame_count * width * height * 3`.
    ///
    /// # Errors
    /// Returns `None` when any dimension is zero or `frames.len()` does not
    /// match the expected size.
    #[must_use]
    pub fn new(
        start_frame: u32,
        width: u32,
        height: u32,
        frame_count: u32,
        frames: Vec<f32>,
    ) -> Option<Self> {
        let w = usize::try_from(width).ok()?;
        let h = usize::try_from(height).ok()?;
        let f = usize::try_from(frame_count).ok()?;
        let expected = f.checked_mul(h)?.checked_mul(w)?.checked_mul(3)?;
        if frames.len() == expected {
            Some(Self {
                start_frame,
                width,
                height,
                frames,
                frame_count,
            })
        } else {
            None
        }
    }
}

/// Seam-conditioning keyframes passed to a chunk's backend call.
///
/// Keyframes are previously generated frames (RGB or matte) placed at specific
/// chunk-local frame indices.  The runner feeds them as
/// `ImageConditioningInput` entries to the alpha-gen pipeline, conditioning the
/// new chunk's generation to be consistent at the seam.
///
/// See: `hdr_ic_lora.py`, `_ANCHOR_KEYFRAME_STRENGTH = 0.95` and
/// `VideoConditionByKeyframeIndex` in the reference.
#[derive(Debug, Clone)]
pub struct Keyframes {
    /// RGB f32 frame data in `[0, 1]`, interleaved RGBRGB…
    /// One frame per entry in `indices`.
    /// Length `indices.len() * width * height * 3`.
    pub frames: Vec<f32>,
    /// Chunk-local frame indices these keyframes are applied to.
    pub indices: Vec<u32>,
    /// IC-LoRA conditioning strength (typically 0.95 per the reference).
    pub strength: f32,
    /// Frame width (must match the chunk being generated).
    pub width: u32,
    /// Frame height (must match the chunk being generated).
    pub height: u32,
    /// Number of keyframes (`indices.len()`).
    pub frame_count: u32,
}

/// Per-frame f32 alpha matte output from one chunk.
#[derive(Debug, Clone)]
pub struct AlphaChunk {
    /// Clip-global index of the first frame.
    pub start_frame: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Alpha values in `[0, 1]`, one per pixel per frame.
    /// Length `frame_count * width * height`.
    pub data: Vec<f32>,
    /// Number of frames.
    pub frame_count: u32,
}

/// Peak GPU memory bytes for one probe call.
#[derive(Debug, Clone, Copy)]
pub struct MemSample {
    /// Peak allocated bytes on the device during the synthetic forward pass.
    ///
    /// On CUDA: `torch.cuda.max_memory_allocated()` after reset.
    /// On MPS: total memory allocated by `PyTorch` at peak, approximated via
    /// `torch.mps.current_allocated_memory()` (does not include Metal driver
    /// overhead; actual VRAM use is higher).
    pub peak_bytes: u64,
}
