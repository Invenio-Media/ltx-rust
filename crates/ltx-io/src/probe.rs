//! `ffprobe` metadata: width, height, FPS rational, frame count.

use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use crate::IoError;

/// A rational number `num / den`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub num: u32,
    pub den: u32,
}

impl Rational {
    /// Convert to `f64`.  Returns `None` when `den` is zero.
    #[must_use]
    pub fn to_f64(self) -> Option<f64> {
        if self.den == 0 {
            None
        } else {
            Some(f64::from(self.num) / f64::from(self.den))
        }
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

/// Clip metadata returned by [`probe`].
#[derive(Debug, Clone)]
pub struct Metadata {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second as a rational (e.g. 24000/1001 for 23.976).
    pub fps: Rational,
    /// Total frame count.  May be estimated from duration when `nb_frames` is absent.
    pub frame_count: u32,
}

// ── ffprobe JSON shapes ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    width: Option<u64>,
    height: Option<u64>,
    r_frame_rate: Option<String>,
    nb_frames: Option<String>,
    duration: Option<String>,
}

// ── public API ─────────────────────────────────────────────────────────────────

/// Read clip metadata without decoding frames.
///
/// Calls `ffprobe -v quiet -print_format json -show_streams <path>` and
/// returns the first video stream's attributes.
///
/// # Errors
/// Returns [`IoError::ToolNotFound`] when `ffprobe` is not on PATH.
pub fn probe(path: &Path) -> Result<Metadata, IoError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| IoError::ParseError("path contains non-UTF-8 bytes".into()))?;

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            path_str,
        ])
        .output()
        .map_err(|e| IoError::ToolNotFound(e.to_string()))?;

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(IoError::ProcessFailed {
            tool: "ffprobe",
            code,
            stderr,
        });
    }

    let probe: ProbeOutput =
        serde_json::from_slice(&output.stdout).map_err(|e| IoError::ParseError(e.to_string()))?;

    let stream = probe
        .streams
        .into_iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or(IoError::MissingField("video stream"))?;

    let width = u32::try_from(stream.width.ok_or(IoError::MissingField("width"))?)
        .map_err(|_| IoError::DimensionOverflow)?;
    let height = u32::try_from(stream.height.ok_or(IoError::MissingField("height"))?)
        .map_err(|_| IoError::DimensionOverflow)?;

    if width == 0 || height == 0 {
        return Err(IoError::ZeroDimension);
    }

    let fps = parse_rational(
        stream
            .r_frame_rate
            .as_deref()
            .ok_or(IoError::MissingField("r_frame_rate"))?,
    )?;

    let frame_count = if let Some(nb) = stream.nb_frames.filter(|s| !s.is_empty()) {
        nb.parse::<u32>()
            .map_err(|_| IoError::ParseError(format!("nb_frames is not a u32: {nb}")))?
    } else {
        // Estimate from duration × fps.
        let dur_str = stream
            .duration
            .as_deref()
            .ok_or(IoError::MissingField("duration or nb_frames"))?;
        let dur: f64 = dur_str
            .parse()
            .map_err(|_| IoError::ParseError(format!("duration is not f64: {dur_str}")))?;
        let fps_f = fps
            .to_f64()
            .ok_or_else(|| IoError::ParseError("fps denominator is zero".into()))?;
        // dur and fps_f are both non-negative (validated values from ffprobe).
        // Clamp to u32 range before converting.
        let count_f = (dur * fps_f).floor();
        if count_f < 0.0 || count_f > f64::from(u32::MAX) {
            0u32
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::as_conversions,
                reason = "value checked to be in [0, u32::MAX] and floored before cast"
            )]
            let count = count_f as u32;
            count
        }
    };

    Ok(Metadata {
        width,
        height,
        fps,
        frame_count,
    })
}

// ── helpers ────────────────────────────────────────────────────────────────────

/// Parse "num/den" into a [`Rational`].
fn parse_rational(s: &str) -> Result<Rational, IoError> {
    let (num_s, den_s) = s
        .split_once('/')
        .ok_or_else(|| IoError::ParseError(format!("r_frame_rate not 'num/den': {s}")))?;
    let num = num_s
        .parse::<u32>()
        .map_err(|_| IoError::ParseError(format!("fps numerator: {num_s}")))?;
    let den = den_s
        .parse::<u32>()
        .map_err(|_| IoError::ParseError(format!("fps denominator: {den_s}")))?;
    Ok(Rational { num, den })
}
