//! Position-wise feed-forward network.
//!
//! `FeedForward` matches the reference `FeedForward` in `feed_forward.py`:
//!
//! `x → GELUApprox(linear_in) → linear_out`
//!
//! where `GELUApprox(d_in, d_out)` = `gelu(linear(d_in, d_out), approximate='tanh')`.
//! The expansion factor is 4 (`inner_dim = 4 × dim`).

use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;

/// Point-wise feed-forward with tanh-approximate GELU and 4× expansion.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    /// First linear: `dim → 4 × dim`.
    pub linear_in: Linear<B>,
    /// GELU activation (tanh approximation, matching `gelu-approximate` in the config).
    pub act: Gelu,
    /// Second linear: `4 × dim → dim`.
    pub linear_out: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Build from `dim` (both input and output).  `bias` controls whether
    /// both linear layers carry a bias term (LTX-2.5 22B sets `ff_bias=false`).
    pub fn new(dim: usize, bias: bool, device: &B::Device) -> Self {
        let inner = dim.saturating_mul(4);
        Self {
            linear_in: LinearConfig::new(dim, inner).with_bias(bias).init(device),
            // tanh-approximate GELU matches `gelu-approximate` activation_fn.
            act: Gelu::new_approximate(),
            linear_out: LinearConfig::new(inner, dim).with_bias(bias).init(device),
        }
    }

    /// `(…, dim) → (…, dim)`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let h = self.linear_in.forward(x);
        let h = self.act.forward(h);
        self.linear_out.forward(h)
    }
}
