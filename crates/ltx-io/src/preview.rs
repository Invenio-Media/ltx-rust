//! Optional H.264 preview encode via `ffmpeg`.
//!
//! Encodes a sequence of RGB f32 frames to an H.264 MP4 for visual inspection.
//! This is a convenience tool; production output uses [`crate::MatteWriter`].

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::IoError;

/// Encode RGB f32 frames to an H.264 MP4.
///
/// `frames` is an iterator of interleaved RGB f32 slices in `[0, 1]`, one per
/// frame, each of length `width * height * 3`.
///
/// # Parameters
/// - `frames`: iterator of `(frame_data, width, height)` tuples.
/// - `fps_num` / `fps_den`: frame rate numerator and denominator.
/// - `output`: path for the output `.mp4`.
/// - `crf`: H.264 constant-rate factor (`0`=lossless, `23`=default, `51`=worst).
///
/// # Errors
/// [`IoError::ToolNotFound`] when `ffmpeg` is absent.  [`IoError::ProcessFailed`]
/// when `ffmpeg` exits non-zero.  [`IoError::DimensionOverflow`] on overflow.
pub fn encode_preview<I>(
    frames: I,
    fps_num: u32,
    fps_den: u32,
    width: u32,
    height: u32,
    output: &Path,
    crf: u8,
) -> Result<(), IoError>
where
    I: IntoIterator<Item = Vec<f32>>,
{
    let output_str = output
        .to_str()
        .ok_or_else(|| IoError::ParseError("output path contains non-UTF-8 bytes".into()))?;

    let fps_str = format!("{fps_num}/{fps_den}");
    let w_str = width.to_string();
    let h_str = height.to_string();
    let crf_str = crf.to_string();

    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "quiet",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{w_str}x{h_str}"),
            "-r",
            &fps_str,
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            &crf_str,
            "-y",
            output_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| IoError::ToolNotFound(e.to_string()))?;

    let mut stdin = child.stdin.take().ok_or_else(|| {
        IoError::Io(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "ffmpeg stdin unavailable",
        ))
    })?;

    let w = usize::try_from(width).map_err(|_| IoError::DimensionOverflow)?;
    let h = usize::try_from(height).map_err(|_| IoError::DimensionOverflow)?;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or(IoError::DimensionOverflow)?;

    for frame in frames {
        if frame.len() != expected {
            return Err(IoError::DataLengthMismatch {
                expected,
                got: frame.len(),
            });
        }
        // Convert f32 [0,1] → u8 [0,255].
        let bytes: Vec<u8> = frame
            .iter()
            .map(|&v| {
                let clamped = v.clamp(0.0_f32, 1.0_f32);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::as_conversions,
                    reason = "f32 clamped to [0,1], multiplied by 255 and rounded; \
                              result is in [0.0, 255.0] so u8 cast is exact"
                )]
                let byte = (clamped * 255.0_f32).round() as u8;
                byte
            })
            .collect();
        stdin.write_all(&bytes)?;
    }

    // Close stdin so ffmpeg knows we're done.
    drop(stdin);

    let status = child.wait()?;
    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(IoError::ProcessFailed {
            tool: "ffmpeg",
            code,
            stderr: String::new(),
        });
    }

    Ok(())
}
