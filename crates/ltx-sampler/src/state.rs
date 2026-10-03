//! Patchified latent state for the Euler denoising loop.
//!
//! Mirrors `ltx_core.types.LatentState`.  All tensors live in the patchified
//! token space (shape `[batch, tokens, channels]` for latent/mask, and
//! `[batch, 3, tokens, 2]` for positions) rather than the 5-D pixel layout.

use burn::prelude::Backend;
use burn::tensor::Tensor;

/// State of one video latent during the denoising loop.
///
/// Shapes follow the patchified convention (`VideoLatentPatchifier(patch_size=1)`):
/// - `latent`, `clean_latent`: `[B, T, C]`
/// - `denoise_mask`: `[B, T, 1]` — 1 = denoise, 0 = frozen
/// - `positions`: `[B, 3, T, 2]` — pixel-space `[start, end)` per axis
/// - `attention_mask`: `[B, T, T]` when present
/// - `keyframes_mask`: `[B, T, 1]` — marks the first latent frame (causal)
///   and any generated-keyframe slots
pub struct LatentState<B: Backend> {
    /// Current noisy latent tokens.
    pub latent: Tensor<B, 3>,
    /// Denoising weight per token (1 = fully denoise, 0 = frozen/conditioning).
    pub denoise_mask: Tensor<B, 3>,
    /// Pixel-space position bounds for each token.
    pub positions: Tensor<B, 4>,
    /// Initial clean state; reference tokens hold their encoded latent here.
    pub clean_latent: Tensor<B, 3>,
    /// Optional self-attention mask; `None` = full attention.
    pub attention_mask: Option<Tensor<B, 3>>,
    /// Optional first-frame / keyframe marker.
    pub keyframes_mask: Option<Tensor<B, 3>>,
}

impl<B: Backend> LatentState<B> {
    /// Number of tokens (target + conditioning) in the sequence.
    pub fn total_tokens(&self) -> usize {
        self.latent.dims()[1]
    }
}
