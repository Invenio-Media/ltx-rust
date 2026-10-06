//! Timestep embeddings matching `PixArtAlphaCombinedTimestepSizeEmbeddings`.
//!
//! Chain: `t [B]` → sinusoidal `[B, 256]` → linear → `SiLU` → linear → `[B, t_emb_dim]`.

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, TensorData, activation::silu, backend::Backend},
};

/// Learnable linear pair that maps the sinusoidal projection to the embedding.
#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    /// First linear `[256 → t_emb_dim]`.
    pub linear_1: nn::Linear<B>,
    /// Second linear `[t_emb_dim → t_emb_dim]`.
    pub linear_2: nn::Linear<B>,
}

impl<B: Backend> TimestepEmbedding<B> {
    /// `[B, 256]` → `[B, t_emb_dim]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x: Tensor<B, 2> = self.linear_1.forward(x);
        let x: Tensor<B, 2> = silu(x);
        self.linear_2.forward(x)
    }
}

/// Combined PixArt-Alpha timestep size embeddings.
#[derive(Module, Debug)]
pub struct PixArtEmbeddings<B: Backend> {
    /// Learnable linear pair.
    pub timestep_embedder: TimestepEmbedding<B>,
    /// Sinusoidal channel count (always 256).
    pub num_channels: usize,
}

impl<B: Backend> PixArtEmbeddings<B> {
    /// `t [B]` (float, range 0–1) → `[B, t_emb_dim]`.
    pub fn forward(&self, t: Tensor<B, 1>, device: &B::Device) -> Tensor<B, 2> {
        let proj = sinusoidal_proj(t, self.num_channels, device);
        self.timestep_embedder.forward(proj)
    }
}

/// Sinusoidal position embedding with `flip_sin_to_cos=True, shift=0`.
///
/// Output layout: `[cos(f_0·t), …, cos(f_{N/2-1}·t), sin(f_0·t), …]`.
fn sinusoidal_proj<B: Backend>(
    t: Tensor<B, 1>,
    num_channels: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [b] = t.dims();
    let half = num_channels.checked_div(2).unwrap_or(0);

    // inv_freq[i] = 10000^(-i/half)
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "freq index is small (≤ 127); precision loss is negligible"
    )]
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 10_000_f64.powf(-(i as f64) / (half as f64)) as f32)
        .collect();

    let inv = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);
    // t: [B, 1] × inv: [1, half] → angles: [B, half]
    let angles = t.reshape([b, 1]).matmul(inv.reshape([1, half]));

    // flip_sin_to_cos=True → output is [cos, sin]
    let cos = angles.clone().cos();
    let sin = angles.sin();
    Tensor::cat(vec![cos, sin], 1)
}

/// Build a `[B]` timestep tensor from a scalar for a given batch size.
///
/// # Errors
///
/// Returns an error if `batch_size` would overflow.
pub fn scalar_timestep<B: Backend>(
    t_val: f32,
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let data: Vec<f32> = vec![t_val; batch_size];
    Tensor::<B, 1>::from_data(TensorData::new(data, vec![batch_size]), device)
}
