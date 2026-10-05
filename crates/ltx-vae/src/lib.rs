//! LTX-2.5 video VAE encoder in Burn.
//!
//! This crate ports the **encoder half** of the LTX-2.5 diffusion video VAE
//! (`CausalDiffusionVAE`) to Rust on Burn 0.21.  The decoder lives in
//! `ltx-vae-decoder`.
//!
//! # Quick start
//! ```rust,ignore
//! use ltx_vae::{VaeEncoderConfig, VideoEncoder};
//! use burn::backend::NdArray;
//!
//! let device = Default::default();
//! let cfg = VaeEncoderConfig::from_vae_json(&vae_json)?;
//! let encoder: VideoEncoder<NdArray> = VideoEncoder::new(&cfg, &device)?;
//! let latents = encoder.encode(video_tensor)?;
//! ```
//!
//! # Reference
//! `ltx_core/model/video_vae/video_vae.py`,
//! `ltx_core/model/video_vae/convolution.py`,
//! `ltx_core/model/video_vae/resnet.py`,
//! `ltx_core/model/video_vae/ops.py`,
//! `ltx_core/model/video_vae/sampling.py`,
//! `ltx_core/model/video_vae/attention.py`,
//! `ltx_core/model/video_vae/model_configurator.py`
//! at commit 9ec55f9.

pub mod attention;
pub mod config;
pub mod conv;
pub mod encoder;
pub mod error;
pub mod norm;
pub mod patchify;
pub mod resnet;
pub mod sampling;

// ── top-level re-exports ──────────────────────────────────────────────────────

pub use config::VaeEncoderConfig;
pub use encoder::{EncoderBlock, VideoEncoder};
pub use error::VaeError;
pub use norm::{NormLayer, PerChannelStatistics, PixelNorm};
pub use patchify::{patchify, unpatchify};
