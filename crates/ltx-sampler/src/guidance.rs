//! Classifier-free guidance for the alpha\_gen pipeline.
//!
//! Ports the CFG path of `MultiModalGuider.calculate` and the per-step logic
//! in `FactoryGuidedDenoiser` for the video-only, no-STG, no-modality-guidance
//! variant that alpha\_gen uses:
//!
//! ```text
//! pred = cond + (cfg_scale − 1) × (cond − uncond)
//! if rescale_scale != 0:
//!     factor = std(cond) / std(pred)
//!     factor = rescale_scale × factor + (1 − rescale_scale)
//!     pred   = pred × factor
//! ```
//!
//! When `cfg_scale == 1.0` the unconditioned pass is skipped entirely; the
//! sampler only calls the model once.

use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::SamplerError;
use crate::model::{VideoDenoiserModel, VideoModelInput};
use crate::state::LatentState;

// ---------------------------------------------------------------------------
// Guider parameters
// ---------------------------------------------------------------------------

/// CFG + rescale parameters for one denoising pass.
///
/// Mirrors the relevant fields of `MultiModalGuiderParams`.
#[derive(Debug, Clone, Copy)]
pub struct GuiderParams {
    /// Classifier-free guidance scale.  `1.0` = conditioned pass only.
    pub cfg_scale: f32,
    /// Rescale coefficient applied after CFG.  Reference default: `0.7` for alpha\_gen.
    pub rescale_scale: f32,
}

impl GuiderParams {
    /// Parameters for the alpha\_gen pipeline (STG off, modality guidance off).
    ///
    /// Reference: `alpha_gen_guider_params(base, cfg_scale=cfg_scale)`.
    /// Default `cfg_scale` is **`1.0`** (conditioned pass only).
    #[must_use]
    pub const fn alpha_gen(cfg_scale: f32) -> Self {
        Self {
            cfg_scale,
            rescale_scale: 0.7,
        }
    }

    /// Whether the unconditioned pass is needed.
    ///
    /// Matches `MultiModalGuider.do_unconditional_generation`:
    /// `not math.isclose(cfg_scale, 1.0)`.
    #[must_use]
    pub fn needs_uncond(&self) -> bool {
        (self.cfg_scale - 1.0_f32).abs() > 1e-5
    }
}

impl Default for GuiderParams {
    /// Default: `cfg_scale = 1.0` (conditioned pass only), `rescale_scale = 0.7`.
    fn default() -> Self {
        Self::alpha_gen(1.0)
    }
}

// ---------------------------------------------------------------------------
// Guided denoiser
// ---------------------------------------------------------------------------

/// Wraps a [`VideoDenoiserModel`] with CFG guidance and optional rescaling.
///
/// Equivalent to the video-only path of `FactoryGuidedDenoiser` with constant
/// params (alpha\_gen uses the same guider for every step).
pub struct GuidedDenoiser<B: Backend> {
    /// Positive (conditioned) text context `[batch, context_len, d_model]`.
    pub context: Tensor<B, 3>,
    /// Negative context for CFG; required when `params.needs_uncond()`.
    pub negative_context: Option<Tensor<B, 3>>,
    /// Guidance parameters.
    pub params: GuiderParams,
}

impl<B: Backend> GuidedDenoiser<B> {
    /// Run the denoiser for one sigma step and return the predicted x₀.
    ///
    /// When `cfg_scale == 1.0` only the conditioned pass executes.
    /// When `cfg_scale != 1.0` the model runs twice (cond, then uncond) and
    /// CFG + rescaling are applied.
    ///
    /// # Errors
    /// - [`SamplerError::MissingNegativeContext`] when `cfg_scale != 1.0` but
    ///   `negative_context` is `None`.
    /// - Propagates errors from the model's `forward`.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor arithmetic is device math; it cannot overflow Rust integers"
    )]
    pub fn apply<M>(
        &self,
        model: &M,
        state: &LatentState<B>,
        sigma: f32,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>, SamplerError>
    where
        M: VideoDenoiserModel<B>,
    {
        let [batch, tokens, _channels] = state.latent.dims();
        let sigma_t = Tensor::<B, 1>::full([batch], sigma, device);
        let timesteps = state.denoise_mask.clone().mul_scalar(sigma);

        let cond_input = VideoModelInput {
            latent: state.latent.clone(),
            sigma: sigma_t.clone(),
            timesteps: timesteps.clone(),
            positions: state.positions.clone(),
            context: self.context.clone(),
            context_mask: None,
            attention_mask: state.attention_mask.clone(),
            keyframes_mask: state.keyframes_mask.clone(),
        };
        let cond = model.forward(cond_input)?;

        if !self.params.needs_uncond() {
            return Ok(cond);
        }

        let neg_ctx = self
            .negative_context
            .clone()
            .ok_or(SamplerError::MissingNegativeContext)?;

        let uncond_input = VideoModelInput {
            latent: state.latent.clone(),
            sigma: sigma_t,
            timesteps,
            positions: state.positions.clone(),
            context: neg_ctx,
            context_mask: None,
            attention_mask: state.attention_mask.clone(),
            keyframes_mask: state.keyframes_mask.clone(),
        };
        let uncond = model.forward(uncond_input)?;

        // pred = cond + (cfg_scale − 1) × (cond − uncond)
        let cfg_delta = self.params.cfg_scale - 1.0_f32;
        let diff = cond.clone() - uncond;
        let pred = cond.clone() + diff.mul_scalar(cfg_delta);

        if self.params.rescale_scale.abs() < 1e-6 {
            return Ok(pred);
        }

        // Rescale: adj = rescale × (std(cond) / std(pred)) + (1 − rescale)
        let cond_std = tensor3_std::<B>(cond, batch, tokens, device)?;
        let pred_std = tensor3_std::<B>(pred.clone(), batch, tokens, device)?;
        let ratio = cond_std / pred_std;
        let adj = ratio
            .mul_scalar(self.params.rescale_scale)
            .add_scalar(1.0_f32 - self.params.rescale_scale);
        Ok(pred * adj.reshape([1, 1, 1]))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Standard deviation of all elements in a `[B, T, C]` tensor.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor arithmetic is device math; it cannot overflow Rust integers"
)]
fn tensor3_std<B: Backend>(
    tensor: Tensor<B, 3>,
    batch: usize,
    tokens: usize,
    device: &B::Device,
) -> Result<Tensor<B, 1>, SamplerError> {
    let [_, _, channels] = tensor.dims();
    let n_elems = batch
        .checked_mul(tokens)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(SamplerError::Overflow)?;
    let flat = tensor.reshape([n_elems]);
    let mean = flat.clone().mean();
    let diff = flat - mean.expand([n_elems]);
    let variance = (diff.clone() * diff).mean();
    let eps = Tensor::<B, 1>::full([1], 1e-8_f32, device);
    Ok((variance + eps).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Device;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    fn dev() -> Device<B> {
        Device::<B>::default()
    }

    #[test]
    fn alpha_gen_default_no_uncond() {
        let p = GuiderParams::alpha_gen(1.0);
        assert!(!p.needs_uncond());
    }

    #[test]
    fn alpha_gen_cfg3_needs_uncond() {
        let p = GuiderParams::alpha_gen(3.0);
        assert!(p.needs_uncond());
    }

    /// Toy model: x₀ = latent + context\_mean × 0.1.
    struct ToyModel;
    impl VideoDenoiserModel<B> for ToyModel {
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "Burn tensor arithmetic in a test model cannot overflow Rust integers"
        )]
        fn forward(&self, input: VideoModelInput<B>) -> Result<Tensor<B, 3>, SamplerError> {
            let ctx_mean = input.context.mean().reshape([1, 1, 1]);
            Ok(input.latent + ctx_mean.mul_scalar(0.1_f32))
        }
    }

    #[test]
    fn cfg_1_single_pass() {
        let device = dev();
        let ctx = Tensor::<B, 3>::ones([1, 4, 16], &device);
        let denoiser = GuidedDenoiser::<B> {
            context: ctx,
            negative_context: None,
            params: GuiderParams::alpha_gen(1.0),
        };
        let state = LatentState {
            latent: Tensor::zeros([1, 8, 4], &device),
            denoise_mask: Tensor::ones([1, 8, 1], &device),
            positions: Tensor::zeros([1, 3, 8, 2], &device),
            clean_latent: Tensor::zeros([1, 8, 4], &device),
            attention_mask: None,
            keyframes_mask: None,
        };
        let x0 = denoiser.apply(&ToyModel, &state, 0.5, &device).unwrap();
        // x0 = 0 + 1.0 * 0.1 = 0.1
        let vals: Vec<f32> = x0.into_data().to_vec().unwrap();
        for val in &vals {
            assert!((val - 0.1).abs() < 1e-5, "expected 0.1, got {val}");
        }
    }

    #[test]
    fn cfg_3_two_passes_and_rescale() {
        let device = dev();
        let ctx = Tensor::<B, 3>::ones([1, 2, 8], &device);
        let neg = Tensor::<B, 3>::zeros([1, 2, 8], &device);
        let denoiser = GuidedDenoiser::<B> {
            context: ctx,
            negative_context: Some(neg),
            params: GuiderParams::alpha_gen(3.0),
        };
        let latent = Tensor::<B, 3>::random([1, 6, 4], Distribution::Uniform(0.0, 1.0), &device);
        let state = crate::state::LatentState {
            latent: latent.clone(),
            denoise_mask: Tensor::ones([1, 6, 1], &device),
            positions: Tensor::zeros([1, 3, 6, 2], &device),
            clean_latent: latent,
            attention_mask: None,
            keyframes_mask: None,
        };
        let x0 = denoiser.apply(&ToyModel, &state, 0.5, &device).unwrap();
        assert_eq!(x0.dims(), [1, 6, 4]);
    }
}
