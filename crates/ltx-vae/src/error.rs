//! Error types for the VAE encoder.

use thiserror::Error;

/// All errors the encoder can return.
#[derive(Debug, Error)]
pub enum VaeError {
    /// Shape arithmetic overflowed `usize`.
    #[error("shape dimension arithmetic overflowed")]
    DimOverflow,

    /// A tensor dimension was zero when a positive value was required.
    #[error("tensor dimension {dim} is zero")]
    ZeroDim {
        /// Name of the dimension.
        dim: &'static str,
    },

    /// The frame count does not satisfy `1 + k * temporal_factor`.
    #[error("frame count {frames} is not 1 + k*{temporal_factor} (cropped to {cropped})")]
    InvalidFrameCount {
        /// Requested frame count.
        frames: usize,
        /// Required temporal factor.
        temporal_factor: usize,
        /// Closest valid count after cropping.
        cropped: usize,
    },

    /// A block type string in the config is not recognized.
    #[error("unknown encoder block type: {0:?}")]
    UnknownBlockType(String),

    /// Spatial or temporal dimension is not divisible by the patch size.
    #[error("dimension {value} is not divisible by patch_size {patch_size}")]
    PatchifyDim {
        /// Actual dimension.
        value: usize,
        /// Required divisor.
        patch_size: usize,
    },

    /// A config field had an unexpected value.
    #[error("config error: {0}")]
    Config(String),

    /// An unsupported spatial padding mode was requested.
    #[error("spatial padding mode {0:?} is not supported (only \"zeros\")")]
    UnsupportedPaddingMode(String),

    /// Convolution dimensions value is not supported.
    #[error("convolution dimensions {0} not supported (only 3)")]
    UnsupportedDims(usize),

    /// A `serde_json` parse error.
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),
}
