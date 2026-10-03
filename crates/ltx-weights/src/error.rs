//! Error type for all weight-store operations.

use thiserror::Error;

/// All errors produced by `ltx-weights`.
#[derive(Debug, Error)]
pub enum WeightError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid or corrupt safetensors data.
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// A requested key is not present in the store.
    #[error("missing key: {0:?}")]
    MissingKey(String),

    /// The tensor rank does not match the generic const `D`.
    #[error("rank mismatch for {key:?}: tensor has rank {got}, caller wants rank {expected}")]
    RankMismatch {
        /// Requested key.
        key: String,
        /// Actual rank.
        got: usize,
        /// Rank the caller requested.
        expected: usize,
    },

    /// The tensor dtype is not supported for dequantization.
    #[error("unsupported dtype {dtype:?} for key {key:?}")]
    UnsupportedDtype {
        /// The key whose dtype is unsupported.
        key: String,
        /// The dtype string from the safetensors header.
        dtype: String,
    },

    /// The tensor's byte range or byte length does not match its header.
    #[error("invalid tensor data for key {key:?}: {message}")]
    InvalidTensorData {
        /// The key whose data is invalid.
        key: String,
        /// What was wrong with the bytes.
        message: String,
    },

    /// Two safetensors files both define the same post-rename key.
    #[error("duplicate key {0:?} across checkpoint files")]
    DuplicateKey(String),

    /// The `LoRA` `A`/`B` shape is inconsistent with the base weight.
    #[error("LoRA shape mismatch for key {0:?}")]
    LoraShapeMismatch(String),

    /// `LoRA` `A` and `B` inner dimensions differ (rank mismatch within the `LoRA` itself).
    #[error("LoRA inner rank mismatch for key {key:?}: A inner {a_rank}, B inner {b_rank}")]
    LoraRankMismatch {
        /// The weight key.
        key: String,
        /// Inner dim of A (must equal rank of B).
        a_rank: usize,
        /// Inner dim of B.
        b_rank: usize,
    },

    /// The `__metadata__["config"]` key is absent.
    #[error("checkpoint metadata has no \"config\" field")]
    MissingConfig,

    /// The config string is not valid JSON.
    #[error("config JSON parse error: {0}")]
    ConfigParse(#[from] serde_json::Error),

    /// A shape rule from `ltx-shape` was violated.
    #[error("shape error: {0}")]
    Shape(#[from] ltx_shape::ShapeError),

    /// QKV split: the leading dimension is not divisible by 3.
    #[error("QKV split: leading dim {dim} for key {key:?} is not divisible by 3")]
    QkvSplitDim {
        /// The weight key.
        key: String,
        /// The leading dimension.
        dim: usize,
    },

    /// Gate fold: a gate scalar has an unexpected rank.
    #[error("gate fold: expected scalar gate for key {key:?}, got shape {shape:?}")]
    GateNotScalar {
        /// The gate key.
        key: String,
        /// The actual shape.
        shape: Vec<usize>,
    },
}
