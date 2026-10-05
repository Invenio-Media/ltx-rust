use thiserror::Error;

/// Errors from the ltx-dit crate.
#[derive(Debug, Error)]
pub enum DitError {
    /// A required config field is missing or has an unexpected value.
    #[error("config error: {0}")]
    Config(String),

    /// A weight key is missing from the weight store.
    #[error("missing weight key: {key}")]
    MissingKey { key: String },

    /// Shape mismatch when loading a weight tensor.
    #[error("shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// A safetensors I/O error.
    #[error("safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// A JSON decode error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
