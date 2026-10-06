//! Multi-head attention for the LTX-2.5 video `DiT`.
//!
//! # Features
//! - Q/K RMS-norms (learnable `gamma`, matching `torch.nn.RMSNorm` in the reference).
//! - Optional 3-D `RoPE` applied after Q/K norm.
//! - Optional per-head sigmoid gating (`apply_gated_attention`).
//! - Additive attention-bias mask (float log-space, shape `(B, 1, T_q, T_k)`).
//! - Self-attention when no explicit context is given.
//!
//! # Burn MHA vs. custom
//!
//! [`burn::nn::MultiHeadAttention`] computes `query.matmul(key.transpose())`
//! explicitly, materialising a full `(B, H, T, T)` score matrix on every
//! backend including `ndarray` (CPU) and `wgpu`.  It also does not support
//! QK-norm or `RoPE`, so we implement attention from scratch here.
//!
//! The same materialisation happens in our own [`sdp_attention`]: the full
//! `(B, H, T_q, T_k)` score matrix is computed on all backends.
//!
//! # Chunked attention
//!
//! The `q_chunk` field, when `Some(chunk_size)`, splits the query along the
//! sequence dimension and processes each chunk independently.  This bounds peak
//! memory to `O(H × chunk_size × T_k)` instead of `O(H × T_q × T_k)` while
//! producing mathematically identical attention rows (only Q is chunked; the
//! full K/V remain on device, so the per-row softmax inputs are identical).

use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};

use crate::config::RopeType;
use crate::rope::{apply_interleaved_rope, apply_split_rope};

/// Multi-head attention with QK-norm, optional `RoPE`, optional gating, and an
/// optional chunked query-block path.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    /// Query projection: `query_dim → heads × d_head`.
    pub to_q: Linear<B>,
    /// Key projection: `context_dim → heads × d_head`.
    pub to_k: Linear<B>,
    /// Value projection: `context_dim → heads × d_head`.
    pub to_v: Linear<B>,
    /// RMS-norm applied to the projected queries (learnable `gamma`).
    pub q_norm: RmsNorm<B>,
    /// RMS-norm applied to the projected keys (learnable `gamma`).
    pub k_norm: RmsNorm<B>,
    /// Output projection: `heads × d_head → query_dim`.
    pub to_out: Linear<B>,
    /// Per-head sigmoid gate (optional): `query_dim → heads`.
    pub to_gate_logits: Option<Linear<B>>,
    /// Number of attention heads.
    pub heads: usize,
    /// Feature dimension per head.
    pub d_head: usize,
    /// Pre-computed `1 / sqrt(d_head)` attention scale.
    pub attn_scale: f32,
    /// `RoPE` variant.
    pub rope_type: RopeType,
    /// Optional chunk size along Q's sequence dimension.
    /// `None` uses full O(T²) attention; `Some(c)` processes Q in chunks of c.
    pub q_chunk: Option<usize>,
}

impl<B: Backend> Attention<B> {
    /// Build an `Attention` module.
    ///
    /// - `query_dim`: model's hidden dimension for the query modality.
    /// - `context_dim`: key/value modality dimension (`None` = self-attention).
    /// - `heads`: number of attention heads.
    /// - `d_head`: per-head feature dimension.
    /// - `norm_eps`: epsilon for Q/K RMS norms.
    /// - `rope_type`: `Split` or `Interleaved`.
    /// - `gated`: enable per-head sigmoid gating.
    /// - `q_chunk`: optional Q chunk size.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        d_head: usize,
        norm_eps: f64,
        rope_type: RopeType,
        gated: bool,
        q_chunk: Option<usize>,
        device: &B::Device,
    ) -> Self {
        let inner = heads.saturating_mul(d_head);
        let ctx = context_dim.unwrap_or(query_dim);
        let to_gate_logits = if gated {
            Some(LinearConfig::new(query_dim, heads).init(device))
        } else {
            None
        };
        // attn_scale = 1 / sqrt(d_head).  d_head ≤ 128 for the 22B model,
        // well within f32's precision range.
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "d_head ≤ 128 in all supported configs; exact in f32"
        )]
        let attn_scale = 1.0_f32 / (d_head as f32).sqrt();

        Self {
            to_q: LinearConfig::new(query_dim, inner).init(device),
            to_k: LinearConfig::new(ctx, inner).init(device),
            to_v: LinearConfig::new(ctx, inner).init(device),
            q_norm: RmsNormConfig::new(inner)
                .with_epsilon(norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(inner)
                .with_epsilon(norm_eps)
                .init(device),
            to_out: LinearConfig::new(inner, query_dim).init(device),
            to_gate_logits,
            heads,
            d_head,
            attn_scale,
            rope_type,
            q_chunk,
        }
    }

    /// Forward pass.
    ///
    /// - `x`: query tokens `(B, T_q, query_dim)`.
    /// - `context`: key/value tokens `(B, T_k, context_dim)` (`None` = self-attention).
    /// - `mask`: optional additive attention bias `(B, 1, T_q, T_k)` in log-space.
    ///   Log-space means `0.0` = full attention, very negative = masked out.
    /// - `pe`: optional `RoPE` `(cos, sin)` for Q and K.
    /// - `k_pe`: optional separate `RoPE` for K; if absent, `pe` is reused.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor operators run on the compute backend; no integer overflow possible"
    )]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Option<Tensor<B, 3>>,
        mask: Option<&Tensor<B, 4>>,
        pe: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        k_pe: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> Tensor<B, 3> {
        let kv = context.unwrap_or_else(|| x.clone());
        let [batch, t_q, _] = x.dims();
        let [_, t_k, _] = kv.dims();
        let inner = self.heads.saturating_mul(self.d_head);

        let mut q = self.to_q.forward(x.clone());
        let mut k = self.to_k.forward(kv.clone());
        let v = self.to_v.forward(kv);

        q = self.q_norm.forward(q);
        k = self.k_norm.forward(k);

        if let Some((cos_q, sin_q)) = pe {
            let (cos_k, sin_k) = k_pe.unwrap_or_else(|| (cos_q.clone(), sin_q.clone()));
            q = apply_rope(q, &cos_q, &sin_q, self.heads, self.rope_type);
            k = apply_rope(k, &cos_k, &sin_k, self.heads, self.rope_type);
        }

        // (B, H, T, d_head)
        let q4 = q
            .reshape([batch, t_q, self.heads, self.d_head])
            .swap_dims(1, 2);
        let k4 = k
            .reshape([batch, t_k, self.heads, self.d_head])
            .swap_dims(1, 2);
        let v4 = v
            .reshape([batch, t_k, self.heads, self.d_head])
            .swap_dims(1, 2);

        let ctx4 = if let Some(chunk_size) = self.q_chunk {
            chunked_sdp_attention(q4, k4, v4, mask, self.attn_scale, chunk_size)
        } else {
            sdp_attention(q4, k4, v4, mask, self.attn_scale)
        };

        let out = ctx4.swap_dims(1, 2).reshape([batch, t_q, inner]);

        let out = if let Some(gate_proj) = &self.to_gate_logits {
            let gate_logits = gate_proj.forward(x);
            // 2 × sigmoid matches `2.0 * torch.sigmoid(logits)` in the reference.
            let gates = sigmoid(gate_logits)
                .mul_scalar(2.0_f32)
                .reshape([batch, t_q, self.heads, 1]);
            let out4 = out.reshape([batch, t_q, self.heads, self.d_head]);
            (out4 * gates).reshape([batch, t_q, inner])
        } else {
            out
        };

        self.to_out.forward(out)
    }
}

// ---------------------------------------------------------------------------
// RoPE dispatch
// ---------------------------------------------------------------------------

fn apply_rope<B: Backend>(
    x: Tensor<B, 3>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
    heads: usize,
    rope_type: RopeType,
) -> Tensor<B, 3> {
    match rope_type {
        RopeType::Split => apply_split_rope(x, cos.clone(), sin.clone(), heads),
        RopeType::Interleaved => apply_interleaved_rope(x, cos.clone(), sin.clone()),
    }
}

// ---------------------------------------------------------------------------
// Attention kernels
// ---------------------------------------------------------------------------

/// Scaled dot-product attention (full O(T²) score matrix).
///
/// All backends (`ndarray`, wgpu, cuda) materialise the full `(B, H, T_q, T_k)`
/// matrix here.  Use [`chunked_sdp_attention`] to bound peak memory
/// linearly in `chunk_size`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor operators run on the compute backend; no integer overflow possible"
)]
fn sdp_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<&Tensor<B, 4>>,
    scale: f32,
) -> Tensor<B, 4> {
    let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
    let scores = if let Some(m) = mask {
        scores + m.clone()
    } else {
        scores
    };
    softmax(scores, 3).matmul(v)
}

/// Chunked scaled dot-product attention.
///
/// Splits Q into `ceil(T_q / chunk_size)` chunks. Each chunk computes attention
/// against the full K and V. Because the softmax is per-row and K/V are not
/// chunked, the output is mathematically identical to the full-matrix path.
///
/// Peak device memory: `O(B × H × chunk_size × T_k)` instead of
/// `O(B × H × T_q × T_k)`.
fn chunked_sdp_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<&Tensor<B, 4>>,
    scale: f32,
    chunk_size: usize,
) -> Tensor<B, 4> {
    if chunk_size == 0 {
        return sdp_attention(q, k, v, mask, scale);
    }
    let [_batch, _heads, t_q, _d_head] = q.dims();
    let n_chunks = t_q.div_ceil(chunk_size);
    if n_chunks <= 1 {
        return sdp_attention(q, k, v, mask, scale);
    }

    let q_chunks = q.chunk(n_chunks, 2);
    let mut out_chunks = Vec::with_capacity(q_chunks.len());
    let mut q_start: usize = 0;
    for q_c in q_chunks {
        let chunk_len = q_c.dims()[2];
        let mask_c = mask.map(|m| m.clone().narrow(2, q_start, chunk_len));
        out_chunks.push(sdp_attention(
            q_c,
            k.clone(),
            v.clone(),
            mask_c.as_ref(),
            scale,
        ));
        q_start = q_start.saturating_add(chunk_len);
    }
    Tensor::cat(out_chunks, 2)
}
