//! Pure Burn backend for the LTX-2.5 Alpha Gen `IC-LoRA` pipeline.
//!
//! # Architecture
//!
//! [`BurnBackend`] implements [`ltx_backend::AlphaBackend`].  It loads the
//! transformer, video VAE encoder, video VAE decoder, `IC-LoRA` weights, and
//! a pre-computed prompt context file once at construction, then exposes
//! [`run_chunk`][BurnBackend::run_chunk] for per-chunk generation.
//!
//! The pipeline inside `run_chunk` mirrors `AlphaGenPipeline.__call__`
//! from the reference Python at commit 9ec55f9:
//!
//! 1. VAE-encode the input RGB as the `IC-LoRA` reference video.
//! 2. Build a zero target latent and pixel-coordinate positions.
//! 3. Append reference tokens via [`VideoReferenceCondition`].
//! 4. Optionally apply seam keyframe conditioning via [`ImageKeyframeCondition`].
//! 5. Seed a [`GaussianNoiser`] from the chunk seed and add noise.
//! 6. Run [`euler_denoising_loop`] with a [`GuidedDenoiser`] adapter over the `DiT`.
//! 7. Unpatchify the denoised latent and decode via [`DiffusionVideoDecoder`].
//! 8. Convert decoded RGB to an alpha matte via Rec.709 luminance.
//!
//! # Memory probe
//! [`BurnBackend::probe`] returns [`BurnError::ProbeNotSupported`] because Burn
//! does not expose device-allocator statistics.  Use the Python backend for
//! memory calibration.
//!
//! # Backends
//! | Cargo feature | Backend |
//! |---|---|
//! | `ndarray` (default) | CPU `NdArray<f32>` |
//! | `metal` | Apple Metal `Metal<half::bf16>` |
//! | `cuda` | NVIDIA CUDA `Cuda<half::bf16>` |
//!
//! [`VideoReferenceCondition`]: ltx_sampler::VideoReferenceCondition
//! [`ImageKeyframeCondition`]: ltx_sampler::ImageKeyframeCondition
//! [`GaussianNoiser`]: ltx_sampler::GaussianNoiser
//! [`GuidedDenoiser`]: ltx_sampler::GuidedDenoiser
//! [`euler_denoising_loop`]: ltx_sampler::euler_denoising_loop
//! [`DiffusionVideoDecoder`]: ltx_vae_decoder::DiffusionVideoDecoder

pub mod backend;
pub mod error;
pub mod settings;

pub use backend::BurnBackend;
pub use error::BurnError;
pub use settings::{GenerationSettings, ModelFiles};
