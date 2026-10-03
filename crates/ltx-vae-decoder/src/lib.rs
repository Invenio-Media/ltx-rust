//! LTX-2.5 diffusion video VAE decoder in Burn.
//!
//! # Public API
//!
//! ```rust,ignore
//! use ltx_vae_decoder::{DecoderConfig, DiffusionVideoDecoder};
//! use burn::tensor::backend::Backend;
//!
//! // Build from a parity fixture / checkpoint.
//! let decoder = DiffusionVideoDecoder::<B>::load(
//!     &path_to_safetensors,
//!     &config,
//!     "",          // key prefix
//!     &device,
//! )?;
//!
//! // Decode a latent (with explicit noise for determinism in tests).
//! let pixels = decoder.decode(latent, Some(noise), &device)?;
//! // pixels: [B, 3, F, H, W] in [-1, 1]
//! ```
//!
//! # Tiling
//!
//! `DiffusionVideoDecoder::decode_window_pixels(t, h, w)` reports the pixel
//! extent of the decode canvas for a given latent shape.  The value is fed
//! to `ltx-budget` for memory estimation.
//!
//! # Memory behaviour
//!
//! The eager 3-D NA implementation (`na3d`) materialises a full `[N, N]`
//! score matrix where `N = T × H × W`.  For the production decoder this
//! is prohibitive; production uses NATTEN or Triton kernels.  For CPU
//! parity fixtures with N ≲ 100 the naive implementation is correct and
//! tractable.

#![warn(missing_docs)]

pub mod config;
pub mod decoder;
pub mod error;
pub mod load;
pub mod na3d;
pub mod nn;
pub mod ops;
pub mod rope;
pub mod tiling;

// Re-export the most-used types at the crate root.
pub use config::{DecoderConfig, ModelOutputType, UpsampleSpec};
pub use decoder::DiffusionVideoDecoder;
pub use error::VaeDecoderError;
pub use tiling::decode_window_pixels;
