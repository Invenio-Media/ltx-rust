//! Error types for the diffusion VAE decoder.

use thiserror::Error;

/// Errors produced by decoder construction, weight loading, and decode.
#[derive(Debug, Error)]
pub enum VaeDecoderError {
    /// A weight key was not found in the safetensors file.
    #[error("weight key not found: {key}")]
    KeyNotFound {
        /// Missing safetensors key.
        key: String,
    },

    /// The tensor shape from safetensors does not match the expected shape.
    #[error("shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Safetensors key with the wrong shape.
        key: String,
        /// Expected tensor shape.
        expected: Vec<usize>,
        /// Actual tensor shape.
        actual: Vec<usize>,
    },

    /// A numeric conversion overflowed.
    #[error("numeric overflow in shape computation: {detail}")]
    NumericOverflow {
        /// Operation that overflowed.
        detail: String,
    },

    /// An argument violates an invariant.
    #[error("invalid argument: {detail}")]
    InvalidArgument {
        /// Invalid argument detail.
        detail: String,
    },

    /// A decoder config field has an invalid value.
    #[error("invalid config: {detail}")]
    InvalidConfig {
        /// Invalid configuration detail.
        detail: String,
    },

    /// Weight loading error.
    #[error("weight error: {0}")]
    Weight(#[from] ltx_weights::WeightError),

    /// JSON config parse error.
    #[error("config JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Tensor dimension too small for neighborhood attention.
    #[error("tensor too small for NA: axis {axis} has size {size} < kernel {kernel}")]
    TensorTooSmall {
        /// Axis name.
        axis: &'static str,
        /// Axis size.
        size: usize,
        /// Required neighborhood kernel.
        kernel: usize,
    },
}
