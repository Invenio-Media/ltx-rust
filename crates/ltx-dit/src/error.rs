use thiserror::Error;

/// Errors from the ltx-dit crate.
#[derive(Debug, Error)]
pub enum DitError {
    /// A required config field is missing or has an unexpected value.
    #[error("config error: {0}")]
    Config(String),

    /// A safetensors I/O error (used by fixture loading in tests).
    #[error("safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// A JSON decode error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// A weight-loading error from `ltx-weights`.
    #[error("weight error: {0}")]
    Weight(#[from] ltx_weights::WeightError),
}
