//! Gated MLP (SwiGLU): `w_down(silu(w_gate(x)) · w_up(x))`.
//!
//! Weight names follow the Python `SwiGLU` class: `w_up`, `w_gate`, `w_down`
//! (no bias on any of them).

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, activation::silu, backend::Backend},
};

/// `SwiGLU` gated MLP.
#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    /// Up-projection `[dim → hidden]`.
    pub w_up: nn::Linear<B>,
    /// Gate projection `[dim → hidden]`.
    pub w_gate: nn::Linear<B>,
    /// Down-projection `[hidden → dim]`.
    pub w_down: nn::Linear<B>,
}

impl<B: Backend> SwiGlu<B> {
    /// `w_down(silu(w_gate(x)) · w_up(x))` for any leading dimensions.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = silu(self.w_gate.forward(x.clone()));
        let up = self.w_up.forward(x);
        self.w_down.forward(gate * up)
    }
}

/// Det-stage residual MLP: `x = x + swiglu(norm(x))`.
///
/// Used by `NABlock` (no `AdaLN` modulation).
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
)]
pub fn plain_mlp<B: Backend>(
    x: Tensor<B, 5>,
    mlp: &SwiGlu<B>,
    norm: &nn::RmsNorm<B>,
) -> Tensor<B, 5> {
    let y: Tensor<B, 5> = norm.forward(x.clone());
    x + mlp.forward(y)
}

/// Diffusion residual MLP: `x = x + swiglu(modulate(norm(x)))`.
///
/// `scale` and `shift` broadcast from `[B, 1, 1, 1, C]`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
)]
pub fn adaln_mlp<B: Backend>(
    x: Tensor<B, 5>,
    mlp: &SwiGlu<B>,
    norm: &nn::RmsNorm<B>,
    scale: Tensor<B, 5>,
    shift: Tensor<B, 5>,
) -> Tensor<B, 5> {
    let y: Tensor<B, 5> = norm.forward(x.clone());
    let modulated = y * (scale + 1.0) + shift;
    x + mlp.forward(modulated)
}
