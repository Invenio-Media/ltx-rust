//! `ffmpeg`-backed RGB f32 frame streaming.
//!
//! [`FrameStream`] pipes raw `rgb24` bytes from `ffmpeg` and converts each
//! frame to f32 values in `[0, 1]`.  Frame selection uses `ffmpeg`'s `select`
//! filter so the frame indices are exact — no reliance on container timestamps.

use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};

use crate::IoError;

/// A single RGB f32 frame.
#[derive(Debug, Clone)]
pub struct RgbF32Frame {
    /// Clip-global frame index (matches the index passed to [`FrameStream::open`]).
    pub index: u32,
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB values in `[0, 1]`, length `width * height * 3`.
    pub data: Vec<f32>,
}

/// Streams RGB f32 frames from a video file via `ffmpeg`.
///
/// Frames are selected with `ffmpeg`'s `select` filter using 0-based frame
/// indices, so seeking is exact regardless of container timestamps.
///
/// # Example
/// ```no_run
/// use ltx_io::FrameStream;
/// use std::path::Path;
///
/// let mut stream = FrameStream::open_with_size(Path::new("clip.mp4"), 0, 25, 320, 240).unwrap();
/// while let Some(frame) = stream.next_frame() {
///     let frame = frame.unwrap();
///     println!("frame {}: {}×{}", frame.index, frame.width, frame.height);
/// }
/// ```
#[derive(Debug)]
pub struct FrameStream {
    child: Child,
    reader: BufReader<ChildStdout>,
    width: u32,
    height: u32,
    /// Next expected clip-global frame index.
    next_index: u32,
    end_frame: u32,
}

impl FrameStream {
    /// Open a frame range `[start_frame, end_frame)` from `path`.
    ///
    /// Both indices are 0-based.  `end_frame` is exclusive.
    /// Width and height must be supplied; call [`crate::probe`] first.
    ///
    /// # Errors
    /// - [`IoError::ToolNotFound`] when `ffmpeg` is absent.
    /// - [`IoError::EmptyRange`] when `start_frame >= end_frame`.
    /// - [`IoError::ZeroDimension`] when `width` or `height` is zero.
    pub fn open_with_size(
        path: &Path,
        start_frame: u32,
        end_frame: u32,
        width: u32,
        height: u32,
    ) -> Result<Self, IoError> {
        if start_frame >= end_frame {
            return Err(IoError::EmptyRange {
                start: start_frame,
                end: end_frame,
            });
        }
        if width == 0 || height == 0 {
            return Err(IoError::ZeroDimension);
        }

        let path_str = path
            .to_str()
            .ok_or_else(|| IoError::ParseError("path contains non-UTF-8 bytes".into()))?;

        // end_frame is exclusive; last accepted index is end_frame - 1.
        // Safe: end_frame > start_frame >= 0, so end_frame >= 1.
        let last = end_frame.saturating_sub(1);

        // Use ffmpeg `select` filter with `between(n, start, last)` for exact frame
        // index selection. The `n` variable in the select filter counts decoded
        // frames from 0 for each input, independent of container timestamps.
        let select_expr = format!("between(n\\,{start_frame}\\,{last})");

        let mut child = Command::new("ffmpeg")
            .args([
                "-v",
                "quiet",
                "-i",
                path_str,
                "-vf",
                &format!("select={select_expr}"),
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| IoError::ToolNotFound(e.to_string()))?;

        let stdout = child.stdout.take().ok_or_else(|| {
            IoError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "ffmpeg stdout unavailable",
            ))
        })?;

        Ok(Self {
            child,
            reader: BufReader::new(stdout),
            width,
            height,
            next_index: start_frame,
            end_frame,
        })
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Read the next frame from `ffmpeg`'s stdout.
    ///
    /// Returns `None` when all frames in the range have been consumed.
    /// Early `ffmpeg` EOF before `end_frame` is reported as an error.
    ///
    /// # Errors
    /// [`IoError::Io`] on unexpected I/O failure; [`IoError::DimensionOverflow`]
    /// when `width × height × 3` overflows `usize`.
    pub fn next_frame(&mut self) -> Option<Result<RgbF32Frame, IoError>> {
        if self.next_index >= self.end_frame {
            return None;
        }

        let Ok(frame_w) = usize::try_from(self.width) else {
            return Some(Err(IoError::DimensionOverflow));
        };
        let Ok(frame_h) = usize::try_from(self.height) else {
            return Some(Err(IoError::DimensionOverflow));
        };

        let Some(pixel_count) = frame_w.checked_mul(frame_h) else {
            return Some(Err(IoError::DimensionOverflow));
        };
        let Some(byte_count) = pixel_count.checked_mul(3) else {
            return Some(Err(IoError::DimensionOverflow));
        };

        let mut buf = vec![0u8; byte_count];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                let index = self.next_index;
                self.next_index = self.end_frame;
                return Some(Err(IoError::ShortFrameStream {
                    index,
                    end_frame: self.end_frame,
                }));
            }
            Err(e) => return Some(Err(IoError::Io(e))),
        }

        // u8 → f32 is a lossless widening (u8 has 8 bits, f32 has 24-bit mantissa).
        let data: Vec<f32> = buf.iter().map(|&b| f32::from(b) / 255.0_f32).collect();

        let index = self.next_index;
        self.next_index = self.next_index.saturating_add(1);

        Some(Ok(RgbF32Frame {
            index,
            width: self.width,
            height: self.height,
            data,
        }))
    }

    /// Collect all remaining frames into a `Vec`.
    ///
    /// # Errors
    /// First error encountered while streaming.
    pub fn collect_frames(mut self) -> Result<Vec<RgbF32Frame>, IoError> {
        let mut frames = Vec::new();
        while let Some(result) = self.next_frame() {
            frames.push(result?);
        }
        Ok(frames)
    }
}

impl Drop for FrameStream {
    fn drop(&mut self) {
        // Best-effort: terminate the ffmpeg process on drop to avoid leaking it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
