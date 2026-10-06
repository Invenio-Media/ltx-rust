//! LTX-2.5 diffusion video VAE decoder in Burn.
//!
//! # Public API
//!
//! ```rust,ignore
//! use ltx_vae_decoder::{DecoderConfig, DiffusionVideoDecoder, DecodeTileConfig, TileDim};
//! use ltx_weights::{KeyMap, WeightStore};
//! use burn::backend::NdArray;
//!
//! // Open checkpoint with decoder key-map and build a scope.
//! let store = WeightStore::open(&["checkpoint.safetensors"], &KeyMap::video_decoder())?;
//! let scope = store.scope("");
//!
//! // Build decoder from a parity fixture (scope at empty prefix).
//! let decoder = DiffusionVideoDecoder::<NdArray>::load(&scope, &config, &device)?;
//!
//! // Untiled decode (suitable for small clips or parity tests).
//! let pixels = decoder.decode(latent, Some(noise), &device)?;
//! // pixels: [B, 3, F, H, W] in [-1, 1]
//!
//! // Tiled decode for production-scale clips (1280×720×49 frames).
//! let tile_cfg = decoder.recommend_tile_config();
//! let pixels = decoder.decode_with_tiling(latent, None, Some(&tile_cfg), &device)?;
//! ```
//!
//! # Tiling
//!
//! `DiffusionVideoDecoder::decode_with_tiling` tiles the stage-4 context and
//! blends outputs with trapezoidal masks.  See [`DecodeTileConfig`] for the
//! config API.  `recommend_tile_config` returns conservative defaults.
//!
//! # Neighborhood attention memory
//!
//! `na3d` processes queries in `T × H` row iterations.  Each row groups W
//! queries by their shared window start and runs one batched matmul per group.
//! Peak device memory is `O(B · NH · kw · NK)` — no `N × N` score matrix.

#![warn(missing_docs)]

pub mod config;
pub mod decoder;
pub mod error;
pub mod na3d;
pub mod nn;
pub mod ops;
pub mod rope;
pub mod tiling;

// Re-export the most-used types at the crate root.
pub use config::{DecoderConfig, ModelOutputType, UpsampleSpec};
pub use decoder::DiffusionVideoDecoder;
pub use error::VaeDecoderError;
pub use tiling::{DecodeTileConfig, TileDim, decode_window_pixels};
