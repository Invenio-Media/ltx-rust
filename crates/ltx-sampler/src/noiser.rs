//! Gaussian noise injection for the initial latent state.
//!
//! Ports `GaussianNoiser.__call__` from `ltx_core.components.noisers`.
//!
//! # RNG note
//! This crate uses `rand::rngs::StdRng` (seeded via [`SeedableRng`]) for
//! Gaussian sampling.  Seeds do **not** produce the same noise as `PyTorch`'s
//! `torch.Generator` because the two libraries use different algorithms.
//! Use [`GaussianNoiser::apply_with_noise`] when bit-exact comparison with
//! Python is required.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::SamplerError;
use crate::state::LatentState;

/// Adds Gaussian noise to a patchified latent state.
pub struct GaussianNoiser {
    rng: StdRng,
    normal: Normal<f32>,
}

impl GaussianNoiser {
    /// Create a noiser seeded with `seed`.
    ///
    /// Seeds are not compatible with `PyTorch`'s `torch.Generator`; see the
    /// module documentation.
    ///
    /// # Errors
    /// Returns [`SamplerError::Overflow`] if the `Normal` distribution cannot
    /// be constructed (which cannot happen for μ=0, σ=1 — error path is
    /// unreachable but required by the API).
    pub fn new(seed: u64) -> Result<Self, SamplerError> {
        let rng = StdRng::seed_from_u64(seed);
        let normal = Normal::<f32>::new(0.0, 1.0).map_err(|_| SamplerError::Overflow)?;
        Ok(Self { rng, normal })
    }

    /// Add noise to `state` using the internal RNG.
    ///
    /// Follows the reference formula:
    /// ```text
    /// noise  = N(0, 1) with shape [B, T, C]
    /// noised = lerp(latent, noise, noise_scale)
    /// result = lerp(clean_latent, noised, denoise_mask)
    /// ```
    ///
    /// # Errors
    /// [`SamplerError::Overflow`] when the token count overflows `usize`.
    pub fn apply<B: Backend>(
        &mut self,
        state: LatentState<B>,
        noise_scale: f32,
        device: &B::Device,
    ) -> Result<LatentState<B>, SamplerError> {
        let [batch, tokens, channels] = state.latent.dims();
        let n_elems = batch
            .checked_mul(tokens)
            .and_then(|n| n.checked_mul(channels))
            .ok_or(SamplerError::Overflow)?;
        let noise_vals: Vec<f32> = (0..n_elems)
            .map(|_| self.normal.sample(&mut self.rng))
            .collect();
        let noise = Tensor::<B, 3>::from_data(
            TensorData::new(noise_vals, [batch, tokens, channels]),
            device,
        );
        Ok(apply_noise(state, noise, noise_scale))
    }

    /// Add noise using a caller-supplied tensor (for parity fixtures).
    ///
    /// The noise tensor must have the same shape as `state.latent`.
    pub fn apply_with_noise<B: Backend>(
        state: LatentState<B>,
        noise: Tensor<B, 3>,
        noise_scale: f32,
    ) -> LatentState<B> {
        apply_noise(state, noise, noise_scale)
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor arithmetic is device math; it cannot overflow Rust integers"
)]
fn apply_noise<B: Backend>(
    state: LatentState<B>,
    noise: Tensor<B, 3>,
    noise_scale: f32,
) -> LatentState<B> {
    // lerp(latent, noise, noise_scale)
    let noised =
        state.latent.clone().mul_scalar(1.0_f32 - noise_scale) + noise.mul_scalar(noise_scale);
    // lerp(clean_latent, noised, denoise_mask)
    let one_minus_mask = Tensor::ones_like(&state.denoise_mask) - state.denoise_mask.clone();
    let result = state.clean_latent.clone() * one_minus_mask + noised * state.denoise_mask.clone();
    LatentState {
        latent: result,
        ..state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Device;

    type B = NdArray<f32>;

    fn dev() -> Device<B> {
        Device::<B>::default()
    }

    fn make_state(device: Device<B>) -> LatentState<B> {
        LatentState {
            latent: Tensor::zeros([1, 6, 4], &device),
            denoise_mask: Tensor::ones([1, 6, 1], &device),
            positions: Tensor::zeros([1, 3, 6, 2], &device),
            clean_latent: Tensor::zeros([1, 6, 4], &device),
            attention_mask: None,
            keyframes_mask: None,
        }
    }

    #[test]
    fn noise_shape_preserved() {
        let device = dev();
        let state = make_state(device);
        let mut noiser = GaussianNoiser::new(42).unwrap();
        let result = noiser.apply(state, 1.0, &device).unwrap();
        assert_eq!(result.latent.dims(), [1, 6, 4]);
    }

    #[test]
    fn frozen_tokens_stay_clean() {
        let device = dev();
        let mask_data: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let clean_val = 99.0_f32;
        let state = LatentState::<B> {
            latent: Tensor::zeros([1, 6, 1], &device),
            denoise_mask: Tensor::from_data(TensorData::new(mask_data, [1, 6, 1]), &device),
            positions: Tensor::zeros([1, 3, 6, 2], &device),
            clean_latent: Tensor::full([1, 6, 1], clean_val, &device),
            attention_mask: None,
            keyframes_mask: None,
        };
        let mut noiser = GaussianNoiser::new(7).unwrap();
        let result = noiser.apply(state, 1.0, &device).unwrap();
        let vals: Vec<f32> = result.latent.into_data().to_vec().unwrap();
        let first = vals.first().copied().unwrap();
        assert!(
            (first - clean_val).abs() < 1e-5,
            "frozen token should equal {clean_val}, got {first}"
        );
    }

    #[test]
    fn deterministic_seed() {
        let device = dev();
        let mut n1 = GaussianNoiser::new(123).unwrap();
        let mut n2 = GaussianNoiser::new(123).unwrap();
        let result1: Vec<f32> = n1
            .apply(make_state(device), 1.0, &device)
            .unwrap()
            .latent
            .into_data()
            .to_vec()
            .unwrap();
        let result2: Vec<f32> = n2
            .apply(make_state(device), 1.0, &device)
            .unwrap()
            .latent
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(result1, result2, "same seed must produce same noise");
    }
}
