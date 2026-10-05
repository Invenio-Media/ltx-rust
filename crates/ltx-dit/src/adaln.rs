//! Adaptive layer-norm single (adaLN-single) from PixArt-α.
//!
//! Maps a flat vector of scalar timesteps to a stacked modulation tensor that
//! each transformer block uses to scale and shift its pre-normalised activations.
//!
//! Forward pass (matches `AdaLayerNormSingle` + the combined embedding in the
//! reference `adaln.py` / `timestep_embedding.py`):
//!
//! 1. Sinusoidal embedding: `t → e` with 256 channels, no learnable params.
//! 2. Two-layer MLP `e → inner_dim` (`linear_1 → SiLU → linear_2`).
//!    — this produces `embedded_timestep`, returned as the second output.
//! 3. `SiLU(embedded_timestep) → linear(coeff × inner_dim)` = first output.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::activation::silu;

// Half-dim for the sinusoidal embedding (256 channels total).
const SINUSOIDAL_HALF: usize = 128;

// ---------------------------------------------------------------------------
// Sinusoidal timestep embedding (no learnable parameters)
// ---------------------------------------------------------------------------

/// Compute sinusoidal timestep embeddings.
///
/// `timesteps`: flat `(N,)` vector of (scaled) time values.
///
/// Returns `(N, 256)` with cosines in the first 128 channels and sines in the
/// last 128 channels (`flip_sin_to_cos = true`, `downscale_freq_shift = 0`).
///
/// # Precision
///
/// Index-to-frequency conversion uses f32 arithmetic, which is exact for all
/// indices in `0..128` because all values are well within the mantissa range.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "loop indices i in 0..128 and constant SINUSOIDAL_HALF=128 are ≤ 128; \
              lossless f32 conversion and no overflow risk"
)]
pub fn sinusoidal_emb<B: Backend>(timesteps: Tensor<B, 1>, device: &B::Device) -> Tensor<B, 2> {
    let n = timesteps.dims()[0];
    // freqs[i] = 10000^(-i/128) = exp(-ln(10000) × i / 128)
    let log10k = (10_000.0_f32).ln();
    let freq_vec: Vec<f32> = (0..SINUSOIDAL_HALF)
        .map(|i| (-log10k * i as f32 / SINUSOIDAL_HALF as f32).exp())
        .collect();
    let freqs = Tensor::<B, 1>::from_floats(freq_vec.as_slice(), device); // (128,)

    // (N, 1) × (1, 128) → (N, 128)
    let t = timesteps.cast(DType::F32).reshape([n, 1]);
    let freq_2d = freqs.unsqueeze_dim::<2>(0); // (1, 128)
    let angles = t * freq_2d; // (N, 128)

    // flip_sin_to_cos=True → [cos, sin]
    Tensor::cat(vec![angles.clone().cos(), angles.sin()], 1) // (N, 256)
}

// ---------------------------------------------------------------------------
// Two-layer MLP (TimestepEmbedding in the reference)
// ---------------------------------------------------------------------------

/// `linear_1(256 → inner_dim) → SiLU → linear_2(inner_dim → inner_dim)`
#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    /// First linear layer: 256 → `inner_dim`.
    pub linear_1: Linear<B>,
    /// Second linear layer: `inner_dim` → `inner_dim`.
    pub linear_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedding<B> {
    /// Initialise with Kaiming weights.
    pub fn new(inner_dim: usize, device: &B::Device) -> Self {
        Self {
            linear_1: LinearConfig::new(256, inner_dim).init(device),
            linear_2: LinearConfig::new(inner_dim, inner_dim).init(device),
        }
    }

    /// `(N, 256) → (N, inner_dim)`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.linear_1.forward(x);
        let h = silu(h);
        self.linear_2.forward(h)
    }
}

// ---------------------------------------------------------------------------
// AdaLayerNormSingle
// ---------------------------------------------------------------------------

/// Combined sinusoidal embedding + MLP + projection.
///
/// Wraps the reference `AdaLayerNormSingle` which in turn wraps
/// `PixArtAlphaCombinedTimestepSizeEmbeddings` (the size-embedding branch is
/// unused in the LTX forward pass).
///
/// Weight keys (relative to parent prefix):
///
/// ```text
/// emb.timestep_embedder.linear_1.{weight,bias}
/// emb.timestep_embedder.linear_2.{weight,bias}
/// linear.{weight,bias}
/// ```
#[derive(Module, Debug)]
pub struct AdaLayerNormSingle<B: Backend> {
    /// The two-layer MLP.  In the checkpoint it lives under `emb.timestep_embedder`.
    pub timestep_embedder: TimestepEmbedding<B>,
    /// Output projection: `inner_dim → coeff × inner_dim`.
    pub linear: Linear<B>,
}

impl<B: Backend> AdaLayerNormSingle<B> {
    /// Build with default initialisation.
    pub fn new(inner_dim: usize, coeff: usize, device: &B::Device) -> Self {
        Self {
            timestep_embedder: TimestepEmbedding::new(inner_dim, device),
            linear: LinearConfig::new(inner_dim, inner_dim.saturating_mul(coeff)).init(device),
        }
    }

    /// Embed `timesteps` (flat `(N,)`) and project.
    ///
    /// Returns `(modulation, embedded_timestep)`:
    /// - `modulation`: `(N, coeff × inner_dim)` — per-block AdaLN weights.
    /// - `embedded_timestep`: `(N, inner_dim)` — used by the output norm.
    pub fn forward(
        &self,
        timesteps: Tensor<B, 1>,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Sinusoidal embedding (no params, computed every call).
        let sin_emb = sinusoidal_emb::<B>(timesteps, device); // (N, 256)
        // Two-layer MLP.
        let embedded_timestep = self.timestep_embedder.forward(sin_emb); // (N, inner_dim)
        // SiLU then project (matches `self.linear(self.silu(embedded_timestep))`).
        let modulation = self.linear.forward(silu(embedded_timestep.clone()));
        (modulation, embedded_timestep)
    }
}
