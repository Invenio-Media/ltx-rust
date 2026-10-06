//! Pre-norm transformer block for deterministic NA stages 1–4.
//!
//! Forward: `x = x + NA(norm1(x))`, then `x = x + swiglu(norm2(x))`.

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, backend::Backend},
};

use crate::nn::{
    attention::NeighborhoodAttention3D,
    swiglu::{SwiGlu, plain_mlp},
};

/// Deterministic NA block with pre-norm and residual connections.
#[derive(Module, Debug)]
pub struct NaBlock<B: Backend> {
    /// Pre-norm before attention.
    pub norm1: nn::RmsNorm<B>,
    /// 3-D neighbourhood attention.
    pub attn: NeighborhoodAttention3D<B>,
    /// Pre-norm before MLP.
    pub norm2: nn::RmsNorm<B>,
    /// `SwiGLU` feed-forward.
    pub mlp: SwiGlu<B>,
}

impl<B: Backend> NaBlock<B> {
    /// Channels-last `[B, T, H, W, C]` → same shape out.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    pub fn forward(&self, x: Tensor<B, 5>, device: &B::Device) -> Tensor<B, 5> {
        let attn_in: Tensor<B, 5> = self.norm1.forward(x.clone());
        let x_attn = x + self.attn.forward(attn_in, device);
        plain_mlp(x_attn, &self.mlp, &self.norm2)
    }
}
