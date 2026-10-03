//! Trait that the sampler calls into the video diffusion model.
//!
//! `VideoDenoiserModel` mirrors the reference transformer call-site in
//! `ltx_pipelines.utils.denoisers._guided_denoise` for the **video modality**
//! only (`alpha_gen` has no audio branch). Field names match the reference
//! `Modality` dataclass so a future adapter can wrap `ltx-dit` with no math
//! changes.
//!
//! The trait is intentionally **not** aware of IC-LoRA or conditioning layout;
//! it receives the full patchified sequence (target tokens + appended reference
//! tokens, if any) and returns the predicted x₀ for the same sequence.

use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::SamplerError;

/// All inputs the video transformer needs for one denoising pass.
///
/// Field names match the reference `Modality` dataclass.
pub struct VideoModelInput<B: Backend> {
    /// Patchified latent tokens `[batch, tokens, channels]`.
    pub latent: Tensor<B, 3>,
    /// Scalar noise level per sample `[batch]`.
    pub sigma: Tensor<B, 1>,
    /// Per-token timestep = `denoise_mask × σ`, shape `[batch, tokens, 1]`.
    pub timesteps: Tensor<B, 3>,
    /// Pixel-space position bounds `[batch, 3, tokens, 2]`.
    pub positions: Tensor<B, 4>,
    /// Text (or other) context `[batch, context_len, d_model]`.
    pub context: Tensor<B, 3>,
    /// Optional text-padding mask `[batch, context_len]`.
    pub context_mask: Option<Tensor<B, 2>>,
    /// Optional self-attention mask `[batch, tokens, tokens]`.
    pub attention_mask: Option<Tensor<B, 3>>,
    /// Optional first-frame / keyframe marker `[batch, tokens, 1]`.
    pub keyframes_mask: Option<Tensor<B, 3>>,
}

/// A video diffusion model that predicts x₀ from a noisy latent.
///
/// The output tensor is the predicted clean latent `[batch, tokens, channels]`,
/// matching the shape of `VideoModelInput::latent`.
pub trait VideoDenoiserModel<B: Backend>: Send + Sync {
    /// Run one denoising forward pass.
    ///
    /// # Errors
    /// Returns [`SamplerError`] if the model encounters a shape mismatch or
    /// any backend error expressible as a `SamplerError`.
    fn forward(&self, input: VideoModelInput<B>) -> Result<Tensor<B, 3>, SamplerError>;
}
