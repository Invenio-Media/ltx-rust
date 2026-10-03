//! 3-D Neighbourhood Attention with absolute `RoPE`.
//!
//! The forward pass:
//! 1. Q/K/V linear projections (three separate GEMMs).
//! 2. Per-head RMS norm and scale on Q/K.
//! 3. Absolute `RoPE` on Q and K (per-axis: T, H, W).
//! 4. Eager NA3D (O(N²) score matrix — correct for parity tests on CPU).
//! 5. Output projection.
//!
//! ## Memory note
//!
//! `na3d` materialises a `[B·NH, N, N]` score matrix.  NATTEN is required
//! for production shapes.

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, backend::Backend},
};

use crate::{
    na3d::na3d,
    rope::{apply_rope, inv_freqs},
};

/// 3-D Neighbourhood Attention.
#[derive(Module, Debug)]
pub struct NeighborhoodAttention3D<B: Backend> {
    /// Q projection `[dim → dim]`.
    pub to_q: nn::Linear<B>,
    /// K projection `[dim → dim]`.
    pub to_k: nn::Linear<B>,
    /// V projection `[dim → dim]`.
    pub to_v: nn::Linear<B>,
    /// Output projection `[dim → dim]`.
    pub proj: nn::Linear<B>,
    /// Per-head RMS norm applied to Q after projection.
    pub q_norm: nn::RmsNorm<B>,
    /// Per-head RMS norm applied to K after projection.
    pub k_norm: nn::RmsNorm<B>,

    // ── Config ───────────────────────────────────────────────────────────
    /// Full channel dim.
    pub dim: usize,
    /// Number of attention heads (`dim / head_dim`).
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// NA kernel `[kt, kh, kw]`.
    pub kernel_size: [usize; 3],
    /// `head_dim^{-0.5}` scale applied to Q after norm.
    pub scale: f32,
    /// `RoPE` dim split `[d_t, d_h, d_w]` (must sum to `head_dim`).
    pub rope_dim_split: [usize; 3],
    /// `RoPE` base frequency (default `10_000.0`).
    pub rope_base: f64,
}

impl<B: Backend> NeighborhoodAttention3D<B> {
    /// Project Q, K, V and reshape each to `[B, T, H, W, NH, HD]`.
    #[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
    fn project_qkv(&self, x: Tensor<B, 5>) -> (Tensor<B, 6>, Tensor<B, 6>, Tensor<B, 6>) {
        let [b, t, h, w, _] = x.dims();
        let nh = self.num_heads;
        let hd = self.head_dim;
        let q = self.to_q.forward(x.clone()).reshape([b, t, h, w, nh, hd]);
        let k = self.to_k.forward(x.clone()).reshape([b, t, h, w, nh, hd]);
        let v = self.to_v.forward(x).reshape([b, t, h, w, nh, hd]);
        (q, k, v)
    }

    /// Norm, scale, and `RoPE` Q and K; return `(q, k, v)` for attention.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    fn qkv_with_rope(
        &self,
        x: Tensor<B, 5>,
        device: &B::Device,
    ) -> (Tensor<B, 6>, Tensor<B, 6>, Tensor<B, 6>) {
        let (q_raw, k_raw, v) = self.project_qkv(x);

        let q_normed = apply_head_norm(&self.q_norm, q_raw) * self.scale;
        let k_normed = apply_head_norm(&self.k_norm, k_raw);

        let [d_t, d_h, d_w] = self.rope_dim_split;
        let inv_t = inv_freqs(d_t, self.rope_base);
        let inv_h = inv_freqs(d_h, self.rope_base);
        let inv_w = inv_freqs(d_w, self.rope_base);

        let q_rope = apply_rope(
            q_normed,
            self.rope_dim_split,
            &inv_t,
            &inv_h,
            &inv_w,
            device,
        );
        let k_rope = apply_rope(
            k_normed,
            self.rope_dim_split,
            &inv_t,
            &inv_h,
            &inv_w,
            device,
        );
        (q_rope, k_rope, v)
    }

    /// Standard NA forward: channels-last `[B, T, H, W, C]` in → same shape out.
    #[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
    pub fn forward(&self, x: Tensor<B, 5>, device: &B::Device) -> Tensor<B, 5> {
        let [b, t, h, w, _] = x.dims();
        let (q_out, k_out, v_out) = self.qkv_with_rope(x, device);
        let out = na3d(q_out, k_out, v_out, self.kernel_size, device);
        let out = out.reshape([b, t, h, w, self.dim]);
        self.proj.forward(out)
    }
}

/// Apply per-head RMS norm to `[B, T, H, W, NH, HD]` by flattening into 2-D.
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
pub(crate) fn apply_head_norm<B: Backend>(norm: &nn::RmsNorm<B>, x: Tensor<B, 6>) -> Tensor<B, 6> {
    let [b, t, h, w, nh, hd] = x.dims();
    let flat_n = b
        .saturating_mul(t)
        .saturating_mul(h)
        .saturating_mul(w)
        .saturating_mul(nh);
    let flat: Tensor<B, 2> = x.reshape([flat_n, hd]);
    let normed: Tensor<B, 2> = norm.forward(flat);
    normed.reshape([b, t, h, w, nh, hd])
}
