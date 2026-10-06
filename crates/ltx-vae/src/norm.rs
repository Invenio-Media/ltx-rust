//! Normalisation layers used by the VAE encoder.
//!
//! - [`PixelNorm`]: per-location RMS normalisation over the channel axis
//!   (port of `PixelNorm` in `ltx_core/model/common/normalization.py`).
//! - [`PerChannelStatistics`]: dataset-level per-channel mean/std buffers
//!   used to normalise latents
//!   (port of `PerChannelStatistics` in `ltx_core/model/video_vae/ops.py`).
//! - [`NormLayer`]: enum that selects `GroupNorm` or `PixelNorm` at build time.
//!
//! Reference commit: 9ec55f9.

use burn::{
    module::{Module, Param},
    nn::GroupNorm,
    tensor::{Tensor, backend::Backend},
};

// ── PixelNorm ─────────────────────────────────────────────────────────────────

/// Per-location RMS normalisation over the channel axis (dim 1).
///
/// `y = x / sqrt(mean(x², dim=1, keepdim=True) + eps)`
///
/// No learnable parameters.
#[derive(Module, Debug, Clone)]
pub struct PixelNorm {}

impl PixelNorm {
    /// Create a new `PixelNorm`.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }

    /// Normalise `x` (any rank ≥ 2, channels on dim 1).
    ///
    /// Uses method-form tensor ops to satisfy `arithmetic_side_effects`.
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // mean(x², dim=1, keepdim=True) + 1e-8
        let mean_sq = x.clone().powf_scalar(2.0_f32).mean_dim(1);
        // Use add_scalar / div to avoid arithmetic operator lints.
        let rms = mean_sq.add_scalar(1e-8_f32).sqrt();
        x.div(rms)
    }
}

impl Default for PixelNorm {
    fn default() -> Self {
        Self::new()
    }
}

// ── NormLayer ─────────────────────────────────────────────────────────────────

/// Selects between `GroupNorm` and `PixelNorm` for the encoder.
#[expect(
    clippy::large_enum_variant,
    reason = "boxing the Burn module would add avoidable allocation and indirection"
)]
#[derive(Module, Debug)]
pub enum NormLayer<B: Backend> {
    /// Group normalisation with learnable affine transform.
    Group(GroupNorm<B>),
    /// Per-location RMS normalisation (no parameters).
    Pixel(PixelNorm),
}

impl<B: Backend> NormLayer<B> {
    /// Normalise a 5-D tensor `(B, C, F, H, W)`.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        match self {
            Self::Group(g) => g.forward(x),
            Self::Pixel(p) => p.forward(x),
        }
    }
}

impl<B: Backend> NormLayer<B> {
    /// Load affine parameters from a [`ltx_weights::Scope`] (no-op for [`PixelNorm`]).
    ///
    /// For `GroupNorm`, reads `weight` → `gamma` and `bias` → `beta` relative
    /// to `scope` (matching `nn.GroupNorm.state_dict()` key names).
    ///
    /// # Errors
    /// Returns [`crate::VaeError::Load`] if any tensor is missing or wrong rank.
    pub(crate) fn load_weights_from_scope(
        &mut self,
        scope: &ltx_weights::Scope<'_>,
        device: &B::Device,
    ) -> Result<(), crate::error::VaeError> {
        match self {
            Self::Pixel(_) => {}
            Self::Group(gn) => {
                gn.gamma = Some(Param::from_tensor(scope.tensor("weight", device)?));
                gn.beta = Some(Param::from_tensor(scope.tensor("bias", device)?));
            }
        }
        Ok(())
    }
}

// ── PerChannelStatistics ──────────────────────────────────────────────────────

/// Dataset-level per-channel mean and standard deviation for latent
/// normalisation.
///
/// Both buffers default to identity (std = 1, mean = 0) so a random-init
/// encoder returns unnormalised latents rather than garbage.
///
/// Checkpoint keys use hyphens: `"std-of-means"` and `"mean-of-means"`.
#[derive(Module, Debug)]
pub struct PerChannelStatistics<B: Backend> {
    /// Per-channel standard deviation of latent means.
    ///
    /// Checkpoint key: `per_channel_statistics.std-of-means`
    pub std_of_means: Tensor<B, 1>,
    /// Per-channel mean of latent means.
    ///
    /// Checkpoint key: `per_channel_statistics.mean-of-means`
    pub mean_of_means: Tensor<B, 1>,
}

impl<B: Backend> PerChannelStatistics<B> {
    /// Construct with identity statistics (std = 1, mean = 0).
    pub fn new(latent_channels: usize, device: &B::Device) -> Self {
        Self {
            std_of_means: Tensor::ones([latent_channels], device),
            mean_of_means: Tensor::zeros([latent_channels], device),
        }
    }

    /// `(x - mean) / std` applied per channel over `(B, C, F, H, W)`.
    pub fn normalize(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_b, nc, _f, _h, _w] = x.dims();
        let mean = self.mean_of_means.clone().reshape([1, nc, 1, 1, 1]);
        let std = self.std_of_means.clone().reshape([1, nc, 1, 1, 1]);
        x.sub(mean).div(std)
    }

    /// `x * std + mean` (inverse of [`normalize`](Self::normalize)).
    pub fn unnormalize(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_b, nc, _f, _h, _w] = x.dims();
        let mean = self.mean_of_means.clone().reshape([1, nc, 1, 1, 1]);
        let std = self.std_of_means.clone().reshape([1, nc, 1, 1, 1]);
        x.mul(std).add(mean)
    }

    /// Load statistics from a [`ltx_weights::Scope`].
    ///
    /// Reads `std-of-means` and `mean-of-means` relative to `scope` (hyphens
    /// preserved, matching the reference `PerChannelStatistics.state_dict()` keys).
    ///
    /// # Errors
    /// Returns [`crate::VaeError::Load`] if any tensor is missing or wrong rank.
    pub(crate) fn load_from_scope(
        scope: &ltx_weights::Scope<'_>,
        device: &B::Device,
    ) -> Result<Self, crate::error::VaeError> {
        let std_of_means = scope.tensor("std-of-means", device)?;
        let mean_of_means = scope.tensor("mean-of-means", device)?;
        Ok(Self {
            std_of_means,
            mean_of_means,
        })
    }
}
