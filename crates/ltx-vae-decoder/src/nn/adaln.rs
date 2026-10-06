//! AdaLN-Zero: timestep embedding → 7 (scale/shift/gate) chunks per block.
//!
//! The output projection is zero-initialised so every block starts as identity.
//! Gate chunks (indices 2, 5, 6) are unused in the combined pathway; the
//! reference folds legacy static gates into `Linear` weights at load time.

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, activation::silu, backend::Backend},
};

/// Number of modulation chunks produced (matches the reference).
pub const NUM_CHUNKS: usize = 7;

/// AdaLN-Zero modulation layer.
///
/// `t_emb [B, t_emb_dim]` → 7 chunks, each `[B, 1, 1, 1, dim]`.
#[derive(Module, Debug)]
pub struct AdaLnZero<B: Backend> {
    /// Zero-initialised projection `t_emb_dim → NUM_CHUNKS × dim`.
    pub proj: nn::Linear<B>,
}

impl<B: Backend> AdaLnZero<B> {
    /// Produce the 7 modulation tensors.
    ///
    /// Returns `[scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp, gate_ctx]`,
    /// each with shape `[B, 1, 1, 1, dim]`.
    pub fn forward(&self, t_emb: Tensor<B, 2>) -> [Tensor<B, 5>; NUM_CHUNKS] {
        let [b, _] = t_emb.dims();
        let h: Tensor<B, 2> = self.proj.forward(silu(t_emb));
        let [_, total] = h.dims();
        let chunk = total.checked_div(NUM_CHUNKS).unwrap_or(0);
        std::array::from_fn(|i| {
            let start = i.saturating_mul(chunk);
            let end = start.saturating_add(chunk);
            h.clone()
                .slice([0..b, start..end])
                .reshape([b, 1, 1, 1, chunk])
        })
    }
}
