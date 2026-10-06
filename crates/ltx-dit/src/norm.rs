//! Normalization helpers.
//!
//! The transformer blocks use a **no-affine** RMS norm for their pre-norms
//! (equivalent to `torch.nn.functional.rms_norm(x, (d,), weight=None, eps=eps)`)
//! and a **no-affine** layer norm for the final output modulation
//! (`torch.nn.LayerNorm(d, elementwise_affine=False)`).
//!
//! Q and K inside each attention layer use [`burn::nn::RmsNorm`], which
//! carries a learnable `gamma` weight initialised to ones.

use burn::prelude::*;
use burn::tensor::DType;

/// RMS-normalise `x` along the last dimension without a learned scale.
///
/// `y = x / sqrt(mean(x²) + eps)`
///
/// Computation is promoted to f32 and cast back to the input dtype.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor operators run on the compute backend; no integer overflow possible"
)]
pub fn rms_norm<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f32) -> Tensor<B, D> {
    let dtype = x.dtype();
    let last = D.saturating_sub(1);
    let rms = (x
        .clone()
        .cast(DType::F32)
        .powf_scalar(2.0_f32)
        .mean_dim(last)
        .add_scalar(eps))
    .sqrt()
    .cast(dtype);
    x / rms
}

/// Layer-normalise `x` along the last dimension without learned affine params.
///
/// `y = (x - mean(x)) / sqrt(var(x) + eps)`
///
/// Equivalent to `F.layer_norm(x, (d,), weight=None, bias=None, eps=eps)`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor operators run on the compute backend; no integer overflow possible"
)]
pub fn layer_norm_no_affine<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f32) -> Tensor<B, D> {
    let last = D.saturating_sub(1);
    let mean = x.clone().mean_dim(last);
    let shifted = x - mean;
    let var = shifted.clone().powf_scalar(2.0_f32).mean_dim(last);
    let std = var.add_scalar(eps).sqrt();
    shifted / std
}
