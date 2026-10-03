//! Combined-context diffusion NA block with shared AdaLN-Zero modulation.
//!
//! `CombinedDiffusionNaBlock` takes a `[B, T, H, W, C_ctx + C5]` buffer
//! (latent context concatenated with noised-pixel features) and the
//! pre-computed modulation tuple, and returns the updated x half.
//!
//! This is `CombinedDiffusionNABlock.forward_combined` from the reference.

use burn::{
    module::{Module, Param},
    nn,
    tensor::{Tensor, backend::Backend},
};

use crate::nn::{
    adaln::NUM_CHUNKS,
    attention::NeighborhoodAttention3D,
    swiglu::{SwiGlu, adaln_mlp},
};

/// Diffusion NA block with combined context injection and AdaLN-Zero modulation.
#[derive(Module, Debug)]
pub struct CombinedDiffusionNaBlock<B: Backend> {
    /// Linear projection `C_ctx → C5` applied to the context half.
    pub context_proj: nn::Linear<B>,
    /// Per-block `AdaLN` offsets, shape `[NUM_CHUNKS, C5]`.
    pub scale_shift_table: Param<Tensor<B, 2>>,
    /// Pre-norm before attention.
    pub norm1: nn::RmsNorm<B>,
    /// 3-D neighbourhood attention.
    pub attn: NeighborhoodAttention3D<B>,
    /// Pre-norm before MLP.
    pub norm2: nn::RmsNorm<B>,
    /// `SwiGLU` feed-forward.
    pub mlp: SwiGlu<B>,
    /// Context channel width.
    pub context_channels: usize,
}

impl<B: Backend> CombinedDiffusionNaBlock<B> {
    /// Decompose shared modulation into per-block scale/shift tensors.
    ///
    /// Adds the learnable `scale_shift_table` offset to each global modulation
    /// chunk.  Gate slots (2, 5, 6) are computed but not used in this path.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    fn modulation(
        &self,
        modulation: &[Tensor<B, 5>; NUM_CHUNKS],
    ) -> (Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 5>) {
        let [_, _, _, _, c5] = modulation[0].dims();
        let tbl = self.scale_shift_table.val();

        let offset = |i: usize| -> Tensor<B, 5> {
            tbl.clone()
                .slice([i..i.saturating_add(1), 0..c5])
                .reshape([1, 1, 1, 1, c5])
        };

        (
            modulation[0].clone() + offset(0), // scale_msa
            modulation[1].clone() + offset(1), // shift_msa
            modulation[3].clone() + offset(3), // scale_mlp
            modulation[4].clone() + offset(4), // shift_mlp
        )
    }

    /// Combined forward (non-keyframe path).
    ///
    /// `context_and_x`: `[B, T, H, W, C_ctx + C5]`.
    /// `modulation`: 7 tensors from [`AdaLnZero`], each `[B, 1, 1, 1, C5]`.
    ///
    /// Returns the updated x half: `[B, T, H, W, C5]`.
    ///
    /// [`AdaLnZero`]: crate::nn::adaln::AdaLnZero
    #[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    pub fn forward_combined(
        &self,
        context_and_x: Tensor<B, 5>,
        modulation: &[Tensor<B, 5>; NUM_CHUNKS],
        device: &B::Device,
    ) -> Tensor<B, 5> {
        let (scale_msa, shift_msa, scale_mlp, shift_mlp) = self.modulation(modulation);
        let [b, t, h, w, full_c] = context_and_x.dims();
        let cc = self.context_channels;

        // Split context and x halves.
        let ctx = context_and_x.clone().slice([0..b, 0..t, 0..h, 0..w, 0..cc]);
        let mut x = context_and_x.slice([0..b, 0..t, 0..h, 0..w, cc..full_c]);

        // Inject context: x = x + context_proj(context).
        x = x + self.context_proj.forward(ctx);

        // AdaLN residual attention.
        {
            let y: Tensor<B, 5> = self.norm1.forward(x.clone());
            let modulated = y * (scale_msa + 1.0) + shift_msa;
            x = x + self.attn.forward(modulated, device);
        }

        // AdaLN residual MLP.
        adaln_mlp(x, &self.mlp, &self.norm2, scale_mlp, shift_mlp)
    }
}
