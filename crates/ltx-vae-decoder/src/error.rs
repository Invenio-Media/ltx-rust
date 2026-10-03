//! Error types for the diffusion VAE decoder.

use thiserror::Error;

/// Errors produced by decoder construction, weight loading, and decode.
#[derive(Debug, Error)]
pub enum VaeDecoderError {
    /// A weight key was not found in the safetensors file.
    #[error("weight key not found: {key}")]
    KeyNotFound { key: String },

    /// The tensor shape from safetensors does not match the expected shape.
    #[error("shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// A numeric conversion overflowed.
    #[error("numeric overflow in shape computation: {detail}")]
    NumericOverflow { detail: String },

    /// An argument violates an invariant.
    #[error("invalid argument: {detail}")]
    InvalidArgument { detail: String },

    /// A decoder config field has an invalid value.
    #[error("invalid config: {detail}")]
    InvalidConfig { detail: String },

    /// safetensors I/O error.
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// JSON config parse error.
    #[error("config JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Tensor dimension too small for neighborhood attention.
    #[error("tensor too small for NA: axis {axis} has size {size} < kernel {kernel}")]
    TensorTooSmall {
        axis: &'static str,
        size: usize,
        kernel: usize,
    },
}
