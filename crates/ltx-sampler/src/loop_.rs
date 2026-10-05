//! Euler denoising loop.
//!
//! Ports `euler_denoising_loop` from `ltx_pipelines.utils.samplers` and the
//! supporting helpers `post_process_latent` and `EulerDiffusionStep.step`.
//!
//! # Loop summary
//! ```text
//! for step in 0..steps:
//!     x0 = denoiser.apply(model, state, σ[step])
//!     x0 = x0 * mask + clean_latent * (1 - mask)    -- post-process
//!     v  = (latent - x0) / σ[step]                  -- velocity
//!     latent = latent + v * (σ[step+1] - σ[step])   -- Euler step
//! ```
//! The returned state has conditioning tokens stripped; `latent` contains only
//! the target frames.

use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::SamplerError;
use crate::guidance::GuidedDenoiser;
use crate::model::VideoDenoiserModel;
use crate::state::LatentState;

// ---------------------------------------------------------------------------
// Post-processing and Euler step
// ---------------------------------------------------------------------------

/// Blend predicted x₀ with frozen reference values at masked positions.
///
/// Matches `post_process_latent` in `ltx_pipelines.utils.helpers`:
/// `result = x0 * mask + clean * (1 - mask)`
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor arithmetic is device math; it cannot overflow Rust integers"
)]
fn post_process<B: Backend>(
    x0: Tensor<B, 3>,
    denoise_mask: &Tensor<B, 3>,
    clean_latent: &Tensor<B, 3>,
) -> Tensor<B, 3> {
    let one_minus = Tensor::ones_like(denoise_mask) - denoise_mask.clone();
    x0 * denoise_mask.clone() + clean_latent.clone() * one_minus
}

/// Advance the latent by one Euler step from σ to σ\_next.
///
/// ```text
/// velocity = (latent - x0_adj) / σ
/// latent_next = latent + velocity * (σ_next - σ)
/// ```
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor arithmetic is device math; it cannot overflow Rust integers"
)]
fn euler_step<B: Backend>(
    latent: Tensor<B, 3>,
    x0_adj: Tensor<B, 3>,
    sigma: f32,
    sigma_next: f32,
) -> Tensor<B, 3> {
    let dt = sigma_next - sigma;
    let velocity = (latent.clone() - x0_adj).div_scalar(sigma);
    latent + velocity.mul_scalar(dt)
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Run the full Euler denoising loop and return the denoised target latent.
///
/// After the loop the reference/conditioning tokens are stripped so the
/// returned [`LatentState::latent`] has exactly `target_token_count` tokens.
///
/// # Parameters
/// - `sigmas`: Schedule produced by [`LTX2Scheduler`][crate::LTX2Scheduler].
///   Must have at least 2 elements.
/// - `state`: Initial noised state (target + reference tokens concatenated).
/// - `target_token_count`: Target tokens to keep after the loop.
/// - `model`: The video diffusion model.
/// - `denoiser`: [`GuidedDenoiser`] providing CFG.
/// - `device`: Burn device.
///
/// # Errors
/// - [`SamplerError::EmptySchedule`] when `sigmas.len() < 2`.
/// - [`SamplerError::InvalidSigma`] when a current step sigma is not finite or
///   is not positive, or when the next sigma is not finite or is negative.
/// - [`SamplerError::TargetTokensTooLarge`] when `target_token_count` exceeds
///   the state's token count.
/// - [`SamplerError::Overflow`] for index arithmetic overflows.
/// - Propagates model errors.
pub fn euler_denoising_loop<B, M>(
    sigmas: &[f32],
    mut state: LatentState<B>,
    target_token_count: usize,
    model: &M,
    denoiser: &GuidedDenoiser<B>,
    device: &B::Device,
) -> Result<LatentState<B>, SamplerError>
where
    B: Backend,
    M: VideoDenoiserModel<B>,
{
    let n_steps = sigmas
        .len()
        .checked_sub(1)
        .filter(|&n| n > 0)
        .ok_or(SamplerError::EmptySchedule(sigmas.len()))?;

    for step_idx in 0..n_steps {
        let sigma = *sigmas
            .get(step_idx)
            .ok_or(SamplerError::ScheduleIndexOutOfBounds {
                step: step_idx,
                len: sigmas.len(),
            })?;
        let next_idx = step_idx.checked_add(1).ok_or(SamplerError::Overflow)?;
        let sigma_next = *sigmas
            .get(next_idx)
            .ok_or(SamplerError::ScheduleIndexOutOfBounds {
                step: next_idx,
                len: sigmas.len(),
            })?;

        if !sigma.is_finite() || sigma <= 0.0_f32 {
            return Err(SamplerError::InvalidSigma {
                step: step_idx,
                sigma,
            });
        }
        if !sigma_next.is_finite() || sigma_next < 0.0_f32 {
            return Err(SamplerError::InvalidSigma {
                step: next_idx,
                sigma: sigma_next,
            });
        }

        let x0 = denoiser.apply(model, &state, sigma, device)?;
        let x0_adj = post_process(x0, &state.denoise_mask, &state.clean_latent);
        let new_latent = euler_step(state.latent, x0_adj, sigma, sigma_next);
        state = LatentState {
            latent: new_latent,
            ..state
        };
    }

    clear_conditioning(state, target_token_count)
}

// ---------------------------------------------------------------------------
// Strip conditioning tokens after the loop
// ---------------------------------------------------------------------------

/// Keep only the first `target_token_count` tokens, dropping conditioning.
///
/// Mirrors `VideoLatentTools.clear_conditioning` in the reference.
///
/// # Errors
/// Returns [`SamplerError::TargetTokensTooLarge`] when `target_token_count`
/// exceeds the state's token count.
pub fn clear_conditioning<B: Backend>(
    state: LatentState<B>,
    target_token_count: usize,
) -> Result<LatentState<B>, SamplerError> {
    let [_batch, available_tokens, _channels] = state.latent.dims();
    if target_token_count > available_tokens {
        return Err(SamplerError::TargetTokensTooLarge {
            target: target_token_count,
            available: available_tokens,
        });
    }

    let tgt = target_token_count;
    Ok(LatentState {
        latent: state.latent.narrow(1, 0, tgt),
        denoise_mask: state.denoise_mask.narrow(1, 0, tgt),
        positions: state.positions.narrow(2, 0, tgt),
        clean_latent: state.clean_latent.narrow(1, 0, tgt),
        attention_mask: state
            .attention_mask
            .map(|mask| mask.narrow(1, 0, tgt).narrow(2, 0, tgt)),
        keyframes_mask: state.keyframes_mask.map(|km| km.narrow(1, 0, tgt)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Device;

    use crate::guidance::{GuidedDenoiser, GuiderParams};
    use crate::model::{VideoDenoiserModel, VideoModelInput};

    type B = NdArray<f32>;

    fn dev() -> Device<B> {
        Device::<B>::default()
    }

    /// Toy model: x₀ = 0 (pure noise → denoises to zero).
    struct ZeroModel;
    impl VideoDenoiserModel<B> for ZeroModel {
        fn forward(&self, input: VideoModelInput<B>) -> Result<Tensor<B, 3>, SamplerError> {
            Ok(Tensor::zeros_like(&input.latent))
        }
    }

    /// Toy model: x₀ = latent + context\_mean × 0.1.
    struct ContextModel;
    impl VideoDenoiserModel<B> for ContextModel {
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "Burn tensor arithmetic in a test model cannot overflow Rust integers"
        )]
        fn forward(&self, input: VideoModelInput<B>) -> Result<Tensor<B, 3>, SamplerError> {
            let mean = input.context.mean().reshape([1, 1, 1]);
            Ok(input.latent + mean.mul_scalar(0.1_f32))
        }
    }

    fn simple_state(device: Device<B>) -> LatentState<B> {
        LatentState {
            latent: Tensor::full([1, 4, 2], 0.5_f32, &device),
            denoise_mask: Tensor::ones([1, 4, 1], &device),
            positions: Tensor::zeros([1, 3, 4, 2], &device),
            clean_latent: Tensor::zeros([1, 4, 2], &device),
            attention_mask: None,
            keyframes_mask: None,
        }
    }

    #[test]
    fn zero_model_converges() {
        let device = dev();
        let sigmas = [0.9_f32, 0.5, 0.1, 0.0];
        let ctx = Tensor::<B, 3>::ones([1, 1, 4], &device);
        let denoiser = GuidedDenoiser::<B> {
            context: ctx,
            negative_context: None,
            params: GuiderParams::alpha_gen(1.0),
        };
        let result = euler_denoising_loop(
            &sigmas,
            simple_state(device),
            4,
            &ZeroModel,
            &denoiser,
            &device,
        )
        .unwrap();
        let vals: Vec<f32> = result.latent.into_data().to_vec().unwrap();
        for val in &vals {
            assert!(val.abs() < 1e-5, "should converge to 0, got {val}");
        }
    }

    #[test]
    fn conditioning_tokens_stripped() {
        let device = dev();
        let state = LatentState::<B> {
            latent: Tensor::zeros([1, 6, 2], &device),
            denoise_mask: Tensor::ones([1, 6, 1], &device),
            positions: Tensor::zeros([1, 3, 6, 2], &device),
            clean_latent: Tensor::zeros([1, 6, 2], &device),
            attention_mask: None,
            keyframes_mask: None,
        };
        let stripped = clear_conditioning(state, 4).unwrap();
        assert_eq!(stripped.latent.dims(), [1, 4, 2]);
        assert_eq!(stripped.positions.dims(), [1, 3, 4, 2]);
    }

    #[test]
    fn schedule_too_short_errors() {
        let device = dev();
        let sigmas = [0.5_f32];
        let ctx = Tensor::<B, 3>::zeros([1, 1, 4], &device);
        let denoiser = GuidedDenoiser::<B> {
            context: ctx,
            negative_context: None,
            params: GuiderParams::alpha_gen(1.0),
        };
        let result = euler_denoising_loop(
            &sigmas,
            simple_state(device),
            4,
            &ZeroModel,
            &denoiser,
            &device,
        );
        assert!(matches!(result, Err(SamplerError::EmptySchedule(_))));
    }

    #[test]
    fn zero_sigma_before_final_step_errors() {
        let device = dev();
        let sigmas = [0.5_f32, 0.0, 0.0];
        let ctx = Tensor::<B, 3>::zeros([1, 1, 4], &device);
        let denoiser = GuidedDenoiser::<B> {
            context: ctx,
            negative_context: None,
            params: GuiderParams::alpha_gen(1.0),
        };
        let result = euler_denoising_loop(
            &sigmas,
            simple_state(device),
            4,
            &ZeroModel,
            &denoiser,
            &device,
        );
        assert!(matches!(
            result,
            Err(SamplerError::InvalidSigma {
                step: 1,
                sigma: 0.0
            })
        ));
    }

    #[test]
    fn clear_conditioning_rejects_too_many_tokens() {
        let device = dev();
        let result = clear_conditioning(simple_state(device), 5);
        assert!(matches!(
            result,
            Err(SamplerError::TargetTokensTooLarge {
                target: 5,
                available: 4
            })
        ));
    }

    #[test]
    fn cfg3_loop_runs() {
        let device = dev();
        let sigmas = [0.8_f32, 0.3, 0.0];
        let ctx = Tensor::<B, 3>::ones([1, 2, 8], &device);
        let neg = Tensor::<B, 3>::zeros([1, 2, 8], &device);
        let denoiser = GuidedDenoiser::<B> {
            context: ctx,
            negative_context: Some(neg),
            params: GuiderParams::alpha_gen(3.0),
        };
        let result = euler_denoising_loop(
            &sigmas,
            simple_state(device),
            4,
            &ContextModel,
            &denoiser,
            &device,
        )
        .unwrap();
        assert_eq!(result.latent.dims(), [1, 4, 2]);
    }
}
