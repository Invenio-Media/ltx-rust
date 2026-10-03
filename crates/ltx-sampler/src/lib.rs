//! LTX-2.5 `alpha_gen` diffusion sampler.
//!
//! Implements the pipeline segment between `create_initial_video_latent` and
//! the VAE decoder:
//!
//! 1. **Scheduler** — [`LTX2Scheduler::execute`] builds the sigma schedule
//!    (token-count-dependent shift; default 40 steps for LTX-2.0, 30 for
//!    LTX-2.3+).
//! 2. **Patchifier** — [`patchify`] / [`unpatchify`] fold a `[B, C, F, H, W]`
//!    latent into `[B, T, C]` tokens and back; [`make_target_positions`] /
//!    [`make_reference_positions`] build the pixel-space coordinate grid.
//! 3. **[`GaussianNoiser`]** — seeds a `StdRng` and samples Gaussian noise
//!    according to the reference formula.  Seeds do **not** match `PyTorch`.
//! 4. **[`VideoReferenceCondition`]** — appends IC-LoRA reference tokens.
//! 5. **[`GuidedDenoiser`]** — wraps a [`VideoDenoiserModel`] with CFG
//!    (`cfg_scale`, rescale 0.7) matching `alpha_gen_guider_params`.  When
//!    `cfg_scale == 1` the negative pass is skipped.
//! 6. **[`euler_denoising_loop`]** — first-order Euler loop; returns target
//!    tokens after stripping the conditioning suffix.
//! 7. **[`VideoDenoiserModel`]** — trait the sampler calls; mirrors the
//!    reference transformer's `Modality`-based interface.
//!
//! # Generic backend
//! All computation is generic over `B: burn::tensor::backend::Backend`.
//! Tests run on `burn::backend::NdArray<f32>` (CI, no GPU required).
//! Enable the `metal` or `cuda` cargo features for GPU inference.

pub mod conditioning;
pub mod error;
pub mod guidance;
pub mod loop_;
pub mod model;
pub mod noiser;
pub mod patchifier;
pub mod scheduler;
pub mod state;

pub use conditioning::{ImageKeyframeCondition, VideoReferenceCondition};
pub use error::SamplerError;
pub use guidance::{GuidedDenoiser, GuiderParams};
pub use loop_::{clear_conditioning, euler_denoising_loop};
pub use model::{VideoDenoiserModel, VideoModelInput};
pub use noiser::GaussianNoiser;
pub use patchifier::{make_reference_positions, make_target_positions, patchify, unpatchify};
pub use scheduler::{LTX2Scheduler, SchedulerConfig};
pub use state::LatentState;
