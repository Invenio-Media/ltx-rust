//! Error type for the ltx-sampler crate.

use thiserror::Error;

/// All errors this crate can produce.
#[derive(Debug, Error)]
pub enum SamplerError {
    /// Integer arithmetic overflowed while computing a shape or index.
    #[error("integer overflow in shape or index computation")]
    Overflow,

    /// The sigma schedule is empty (needs at least two values).
    #[error("sigma schedule must have at least 2 elements (got {0})")]
    EmptySchedule(usize),

    /// Requested a sigma at a step index that is out of bounds.
    #[error("step index {step} is out of bounds for schedule length {len}")]
    ScheduleIndexOutOfBounds { step: usize, len: usize },

    /// A sigma value is invalid for an Euler step.
    #[error("sigma at step {step} must be finite and positive for current step, got {sigma}")]
    InvalidSigma { step: usize, sigma: f32 },

    /// The caller requested more target tokens than the state contains.
    #[error("target token count {target} exceeds available tokens {available}")]
    TargetTokensTooLarge { target: usize, available: usize },

    /// CFG scale != 1 but no negative context was provided.
    #[error("cfg_scale != 1.0 requires a negative context tensor")]
    MissingNegativeContext,

    /// A tensor dimension was zero where a positive value is required.
    #[error("tensor dimension {dim} is zero")]
    ZeroDimension { dim: &'static str },

    /// The reference latent has the wrong number of spatial dimensions.
    #[error("latent must be 5-D [B, C, F, H, W], got {ndim} dimensions")]
    LatentRank { ndim: usize },

    /// Patchify/unpatchify shape mismatch.
    #[error("shape mismatch: {context}")]
    Shape { context: &'static str },

    /// The `first_latent_frame` value is negative (validated elsewhere, here as sentinel).
    #[error("first_latent_frame must be non-negative")]
    NegativeFirstFrame,

    /// A u32→f32 conversion could not be represented exactly (value out of f32 range).
    #[error("value {0} cannot be converted to f32")]
    FloatConversion(u64),

    /// A conditioning strength is outside `[0, 1]` (or not a number).
    #[error("conditioning strength {0} must be in [0, 1]")]
    InvalidStrength(f32),
}
