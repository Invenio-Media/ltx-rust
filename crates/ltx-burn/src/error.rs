//! Error type for the `ltx-burn` crate.

use thiserror::Error;

/// All errors this crate can produce.
#[derive(Debug, Error)]
pub enum BurnError {
    /// Weight loading or `LoRA` merge failed.
    #[error("weights: {0}")]
    Weights(#[from] ltx_weights::WeightError),

    /// Video VAE encoder error.
    #[error("encoder: {0}")]
    Encoder(#[from] ltx_vae::VaeError),

    /// Diffusion video VAE decoder error.
    #[error("decoder: {0}")]
    Decoder(#[from] ltx_vae_decoder::VaeDecoderError),

    /// Transformer (`DiT`) error.
    #[error("transformer: {0}")]
    Transformer(#[from] ltx_dit::DitError),

    /// Sampler (scheduler, noiser, loop) error.
    #[error("sampler: {0}")]
    Sampler(#[from] ltx_sampler::SamplerError),

    /// Shape mismatch or out-of-bounds.
    #[error("shape: {0}")]
    Shape(#[from] ltx_shape::ShapeError),

    /// Prompt-context file has the wrong `format` metadata value.
    #[error("prompt context format {found:?} is not \"ltx-prompt-context/1\"")]
    PromptContextFormat { found: String },

    /// `IC-LoRA` spatial reference scale factors from different `LoRA` files disagree.
    #[error(
        "LoRA reference_downscale_factor values disagree: {values:?}; all set values must match"
    )]
    LoraScaleDisagreement { values: Vec<u32> },

    /// `IC-LoRA` temporal reference scale factors from different `LoRA` files disagree.
    #[error(
        "LoRA reference_temporal_scale_factor values disagree: {values:?}; all set values must match"
    )]
    LoraTemporalDisagreement { values: Vec<u32> },

    /// Integer arithmetic overflowed while computing a shape or index.
    #[error("dimension overflow")]
    Overflow,

    /// An input argument is out of range or malformed.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Peak-memory probe is not supported for this backend.
    ///
    /// Burn does not expose allocator or device-memory statistics.
    #[error(
        "memory probe is not supported for Burn backends: \
         Burn does not expose device-allocator statistics"
    )]
    ProbeNotSupported,

    /// `cfg_scale != 1.0` requires a negative context in the prompt-context
    /// file, but `negative.video_encoding` is absent.
    #[error(
        "cfg_scale {cfg_scale} requires a negative context but \
         negative.video_encoding is absent from the prompt-context file"
    )]
    MissingNegativeContext { cfg_scale: f32 },
}
