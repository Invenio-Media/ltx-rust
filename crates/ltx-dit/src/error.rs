use thiserror::Error;

/// Errors from the ltx-dit crate.
#[derive(Debug, Error)]
pub enum DitError {
    /// A required config field is missing or has an unexpected value.
    #[error("config error: {0}")]
    Config(String),

    /// A JSON decode error (from [`DiTConfig::from_json`]).
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// A weight-loading error from `ltx-weights`.
    #[error("weight error: {0}")]
    Weight(#[from] ltx_weights::WeightError),
}
