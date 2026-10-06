//! Per-frame spatial self-attention block.
//!
//! Port of `AttnBlock3D` from
//! `ltx_core/model/video_vae/attention.py` at commit 9ec55f9.
//!
//! Frames are folded into the batch dimension; attention is computed over the
//! `H×W` spatial locations independently per frame — no temporal interaction.
//! The block is single-head (`head_dim == in_channels`).
//!
//! `_RMSNorm2D` in the reference normalises each `(H, W)` position across
//! channels using L2 normalisation scaled by `sqrt(channels) * gamma`, which
//! is equivalent to RMS normalisation with a learnable per-channel gain.

use burn::{
    module::{Module, Param},
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Tensor, activation::softmax, backend::Backend},
};

use crate::error::VaeError;

// ── RmsNorm2d ─────────────────────────────────────────────────────────────────

/// Channel-first RMS normalisation for 4-D tensors.
///
/// `y = x / sqrt(mean(x², dim=1) + ε) * gamma`
///
/// `gamma` is a learnable per-channel gain, shape `(C, 1, 1)`.
#[derive(Module, Debug)]
pub struct RmsNorm2d<B: Backend> {
    /// Learnable per-channel scale `(C, 1, 1)`.
    gamma: Tensor<B, 3>,
}

impl<B: Backend> RmsNorm2d<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Tensor::ones([channels, 1, 1], device),
        }
    }

    /// Normalise `x` of shape `(BT, C, H, W)`.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mean_sq = x.clone().powf_scalar(2.0_f32).mean_dim(1);
        let rms = mean_sq.add_scalar(1e-8_f32).sqrt();
        let normalised = x.div(rms);
        let gamma = self.gamma.clone().unsqueeze::<4>();
        normalised.mul(gamma)
    }

    /// Load `gamma` from a [`ltx_weights::Scope`].
    ///
    /// Reads `gamma` (shape `[C, 1, 1]`) matching the reference
    /// `_RMSNorm2D.state_dict()` key name.
    ///
    /// # Errors
    /// Returns [`VaeError::Load`] if the tensor is missing or wrong rank.
    fn load_weights_from_scope(
        &mut self,
        scope: &ltx_weights::Scope<'_>,
        device: &B::Device,
    ) -> Result<(), VaeError> {
        self.gamma = scope.tensor("gamma", device)?;
        Ok(())
    }
}

// ── AttnBlock3D ───────────────────────────────────────────────────────────────

/// Single-head per-frame spatial self-attention block.
///
/// Architecture:
/// ```text
/// identity = x
/// x = fold_frames_into_batch(x)  # (B, C, T, H, W) → (B·T, C, H, W)
/// x = rms_norm(x)
/// qkv = to_qkv(x)                # (B·T, 3C, H, W)
/// q, k, v each (B·T, H·W, C)
/// x = scaled_dot_product(q, k, v)
/// x = proj(x)
/// return identity + unfold_frames(x)
/// ```
#[derive(Module, Debug)]
pub struct AttnBlock3D<B: Backend> {
    norm: RmsNorm2d<B>,
    /// Fused Q/K/V projection: 1×1 conv `C → 3C`.
    to_qkv: Conv2d<B>,
    /// Output projection: 1×1 conv `C → C`.
    proj: Conv2d<B>,
}

impl<B: Backend> AttnBlock3D<B> {
    /// Build and initialise.
    ///
    /// # Errors
    /// Returns [`VaeError::DimOverflow`] if `3 * in_channels` overflows `usize`.
    pub fn new(in_channels: usize, device: &B::Device) -> Result<Self, VaeError> {
        let qkv_channels = in_channels.checked_mul(3).ok_or(VaeError::DimOverflow)?;

        let to_qkv = Conv2dConfig::new([in_channels, qkv_channels], [1, 1])
            .with_bias(true)
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        let proj = Conv2dConfig::new([in_channels, in_channels], [1, 1])
            .with_bias(true)
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        Ok(Self {
            norm: RmsNorm2d::new(in_channels, device),
            to_qkv,
            proj,
        })
    }

    /// Forward. Input `(B, C, T, H, W)`.
    ///
    /// # Errors
    /// Returns [`VaeError::DimOverflow`] on shape arithmetic overflow.
    pub fn forward(&self, x: Tensor<B, 5>) -> Result<Tensor<B, 5>, VaeError> {
        let [nb, nc, nt, nh, nw] = x.dims();
        let hw = nh.checked_mul(nw).ok_or(VaeError::DimOverflow)?;
        let bt = nb.checked_mul(nt).ok_or(VaeError::DimOverflow)?;
        let identity = x.clone();

        // Fold frames into batch: (B, C, T, H, W) → (B·T, C, H, W)
        let x4: burn::tensor::Tensor<B, 4> = x.swap_dims(1, 2).reshape([bt, nc, nh, nw]);

        // RMS norm
        let x4 = self.norm.forward(x4);

        // QKV: (B·T, 3C, H, W)
        let qkv = self.to_qkv.forward(x4);

        // Flatten spatial: (B·T, 3C, H·W) → (B·T, H·W, 3C)
        let nc3 = nc.checked_mul(3).ok_or(VaeError::DimOverflow)?;
        let qkv3: burn::tensor::Tensor<B, 3> = qkv.reshape([bt, nc3, hw]).swap_dims(1, 2);

        let query = qkv3.clone().narrow(2, 0, nc);
        let key = qkv3.clone().narrow(2, nc, nc);
        let nc2 = nc.checked_mul(2).ok_or(VaeError::DimOverflow)?;
        let value = qkv3.narrow(2, nc2, nc);

        // Scaled dot-product attention (single head, head_dim = C).
        // Use f32::from(u16) to avoid `as_conversions`; channel counts always < 65536.
        let scale = f32::from(u16::try_from(nc).unwrap_or(u16::MAX))
            .sqrt()
            .recip();
        let scores = query.mul_scalar(scale).matmul(key.swap_dims(1, 2));
        let attn = softmax(scores, 2);
        let out3 = attn.matmul(value); // (B·T, H·W, C)

        // (B·T, H·W, C) → (B·T, C, H, W)
        let out4: burn::tensor::Tensor<B, 4> = out3.swap_dims(1, 2).reshape([bt, nc, nh, nw]);

        // Output projection
        let out4 = self.proj.forward(out4);

        // Unfold frames: (B·T, C, H, W) → (B, T, C, H, W) → (B, C, T, H, W)
        let out5: Tensor<B, 5> = out4.reshape([nb, nt, nc, nh, nw]).swap_dims(1, 2);

        Ok(identity.add(out5))
    }

    /// Load all parameters from a [`ltx_weights::Scope`].
    ///
    /// Key names match the reference `AttnBlock3D.state_dict()`:
    /// - `norm.gamma`
    /// - `to_qkv.weight`, `to_qkv.bias`
    /// - `proj.weight`, `proj.bias`
    ///
    /// # Errors
    /// Returns [`VaeError::Load`] if any tensor is missing or wrong rank.
    pub(crate) fn load_weights_from_scope(
        &mut self,
        scope: &ltx_weights::Scope<'_>,
        device: &B::Device,
    ) -> Result<(), VaeError> {
        self.norm
            .load_weights_from_scope(&scope.scope("norm"), device)?;
        self.to_qkv.weight = Param::from_tensor(scope.tensor("to_qkv.weight", device)?);
        self.to_qkv.bias = Some(Param::from_tensor(scope.tensor("to_qkv.bias", device)?));
        self.proj.weight = Param::from_tensor(scope.tensor("proj.weight", device)?);
        self.proj.bias = Some(Param::from_tensor(scope.tensor("proj.bias", device)?));
        Ok(())
    }
}
