//! Video I/O helpers for ltx-rust.
//!
//! - [`probe`] reads clip metadata via `ffprobe`.
//! - [`FrameStream`] streams RGB f32 frames via `ffmpeg`, seeking by frame index exactly.
//! - [`MatteWriter`] writes per-frame single-channel (`A`) EXR files for VFX pipelines.
//! - [`encode_preview`] encodes a frame sequence to H.264 via `ffmpeg` (optional, feature-free).
//!
//! Tests that require `ffprobe`/`ffmpeg` skip automatically when the tools are absent.

pub mod frame_stream;
pub mod matte;
pub mod preview;
pub mod probe;

pub use frame_stream::{FrameStream, RgbF32Frame};
pub use matte::MatteWriter;
pub use preview::encode_preview;
pub use probe::{Metadata, Rational, probe};

use thiserror::Error;

/// Errors from `ltx-io` operations.
#[derive(Debug, Error)]
pub enum IoError {
    /// `ffprobe` or `ffmpeg` is not on PATH.
    #[error("ffprobe/ffmpeg not found on PATH: {0}")]
    ToolNotFound(String),

    /// A process returned a non-zero exit code.
    #[error("{tool} exited with code {code}: {stderr}")]
    ProcessFailed {
        tool: &'static str,
        code: i32,
        stderr: String,
    },

    /// JSON from `ffprobe` could not be parsed.
    #[error("ffprobe JSON: {0}")]
    ParseError(String),

    /// Video metadata is missing a required field.
    #[error("missing metadata field: {0}")]
    MissingField(&'static str),

    /// Width or height is zero.
    #[error("video dimensions are zero")]
    ZeroDimension,

    /// Arithmetic overflow converting dimension to `usize`.
    #[error("dimension overflow")]
    DimensionOverflow,

    /// Frame index is out of range.
    #[error("frame index {index} out of range 0..{count}")]
    FrameOutOfRange { index: u32, count: u32 },

    /// `start_frame >= end_frame`.
    #[error("empty frame range {start}..{end}")]
    EmptyRange { start: u32, end: u32 },

    /// Data length does not match expected dimensions.
    #[error("data length {got} != expected {expected}")]
    DataLengthMismatch { expected: usize, got: usize },

    /// EXR library error.
    #[error("EXR: {0}")]
    Exr(#[from] exr::error::Error),

    /// I/O error from the standard library.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}
