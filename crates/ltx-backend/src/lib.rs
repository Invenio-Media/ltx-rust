//! `AlphaBackend` trait and Python backend for the LTX-2.5 alpha-gen pipeline.
//!
//! # Backend trait
//! [`AlphaBackend`] has two methods:
//! - [`AlphaBackend::probe`] — peak GPU bytes for a given [`PixelShape`].
//! - [`AlphaBackend::run_chunk`] — generate alpha mattes for one video chunk.
//!
//! # Python backend
//! [`PythonBackend`] spawns `python/alphagen_runner.py` and communicates
//! over a JSON-lines protocol on stdin/stdout (see [`python`] module).
//!
//! [`PixelShape`]: ltx_shape::PixelShape

pub mod python;
pub mod types;

pub use python::PythonBackend;
pub use types::{AlphaChunk, Keyframes, MemSample, VideoChunk};

use ltx_shape::PixelShape;
use thiserror::Error;

/// Errors from backend operations.
#[derive(Debug, Error)]
pub enum BackendError {
    /// The Python runner process could not be started.
    #[error("Python runner spawn failed: {0}")]
    SpawnFailed(String),

    /// A runner request failed.  Includes the last lines of stderr.
    #[error("runner error: {msg}\nstderr tail:\n{stderr}")]
    RunnerError { msg: String, stderr: String },

    /// JSON serialisation or deserialisation failed.
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O error communicating with the runner.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    /// The probe or run output data has the wrong length.
    #[error("data length mismatch: expected {expected}, got {got}")]
    DataLengthMismatch { expected: usize, got: usize },

    /// A dimension value overflows `usize`.
    #[error("dimension overflow")]
    DimensionOverflow,

    /// The response from the runner is missing a required field.
    #[error("missing response field: {0}")]
    MissingField(&'static str),

    /// An I/O error from the `ltx-io` crate.
    #[error("ltx-io: {0}")]
    LtxIo(#[from] ltx_io::IoError),

    /// An operation is not supported by this backend.
    ///
    /// Used by `BurnBackend::probe`: Burn does not expose device-allocator
    /// statistics, so peak-memory estimation is unavailable.
    #[error("operation not supported: {0}")]
    Unsupported(String),
}

/// A backend that generates alpha mattes from RGB video chunks.
///
/// # Memory probe
/// Call [`AlphaBackend::probe`] first with the target [`PixelShape`] to get
/// peak GPU byte usage; budget math (in `ltx-budget`) uses this to decide chunk
/// length.
///
/// # Chunk generation
/// Call [`AlphaBackend::run_chunk`] for each chunk.  Pass seam frames in
/// `cond` to condition the first frame of each new chunk on the last frame of
/// the previous chunk (preventing seam discontinuities).
pub trait AlphaBackend {
    /// Return peak GPU memory bytes for a synthetic forward pass at `shape`.
    ///
    /// # Errors
    /// Backend-specific error.
    fn probe(&self, shape: PixelShape) -> Result<MemSample, BackendError>;

    /// Generate per-frame alpha mattes for one video chunk.
    ///
    /// # Errors
    /// Backend-specific error.
    fn run_chunk(
        &self,
        rgb: &VideoChunk,
        seed: u64,
        cond: Option<&Keyframes>,
    ) -> Result<AlphaChunk, BackendError>;
}
