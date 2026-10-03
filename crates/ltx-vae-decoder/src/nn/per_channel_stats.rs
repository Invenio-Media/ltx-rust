//! Per-channel latent (de)normalisation.
//!
//! The encoder outputs are normalised per channel; the decoder undoes this
//! before `conv_in`.  Statistics are stored in the checkpoint under
//! `per_channel_statistics.std-of-means` and `per_channel_statistics.mean-of-means`.

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

/// Per-channel normalisation statistics.
///
/// Defaults to identity (std = 1, mean = 0) so a model without a checkpoint
/// is a valid no-op.
#[allow(clippy::module_name_repetitions)]
#[derive(Module, Debug)]
pub struct PerChannelStatistics<B: Backend> {
    /// Standard deviation of channel means, shape `[C]`.
    pub std_of_means: Tensor<B, 1>,
    /// Mean of channel means, shape `[C]`.
    pub mean_of_means: Tensor<B, 1>,
}

impl<B: Backend> PerChannelStatistics<B> {
    /// Construct with identity statistics.
    #[must_use]
    pub fn identity(channels: usize, device: &B::Device) -> Self {
        Self {
            std_of_means: Tensor::ones([channels], device),
            mean_of_means: Tensor::zeros([channels], device),
        }
    }

    /// Undo per-channel normalisation: `x_raw = x_norm * std + mean`.
    ///
    /// `x` is channels-first `[B, C, F, H, W]`.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    pub fn un_normalize(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_b, c, _f, _h, _w] = x.dims();
        let std_view = self.std_of_means.clone().reshape([1, c, 1, 1, 1]);
        let mean_view = self.mean_of_means.clone().reshape([1, c, 1, 1, 1]);
        x * std_view + mean_view
    }
}
