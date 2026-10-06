//! Residual blocks for the VAE encoder.
//!
//! Ports of `ResnetBlock3D` and `UNetMidBlock3D` from
//! `ltx_core/model/video_vae/resnet.py` at commit 9ec55f9.
//!
//! Both blocks run in causal mode inside the encoder (no timestep conditioning,
//! no noise injection, no dropout).

use burn::{
    module::{Module, Param},
    nn::{
        PaddingConfig3d,
        conv::{Conv3d, Conv3dConfig},
    },
    tensor::{Tensor, activation::silu, backend::Backend},
};

use crate::{conv::CausalConv3d, error::VaeError, norm::NormLayer};

// ── shortcut conv (1×1×1 for channel-changing resnets) ────────────────────────

/// A 1×1×1 convolution used as the shortcut projection.
#[derive(Module, Debug)]
struct ShortcutConv<B: Backend> {
    conv: Conv3d<B>,
}

impl<B: Backend> ShortcutConv<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv = Conv3dConfig::new([in_channels, out_channels], [1, 1, 1])
            .with_bias(true)
            .with_padding(PaddingConfig3d::Explicit(0, 0, 0))
            .init(device);
        Self { conv }
    }

    fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        self.conv.forward(x)
    }
}

// ── ResnetBlock3D ─────────────────────────────────────────────────────────────

/// One causal 3-D residual block.
///
/// Layout:
/// ```text
/// norm1 → SiLU → conv1 → norm2 → SiLU → conv2
///   │                                       │
///   └─── [norm3 →] shortcut_conv [optional] ┘  (+residual)
/// ```
/// `shortcut_conv` is `Some(1×1×1 conv)` when channels change, `None` otherwise.
/// `shortcut_norm` is `Some(GroupNorm(1))` before the shortcut projection when
/// channels change, `None` otherwise.
#[derive(Module, Debug)]
pub struct ResnetBlock3D<B: Backend> {
    norm1: NormLayer<B>,
    conv1: CausalConv3d<B>,
    norm2: NormLayer<B>,
    conv2: CausalConv3d<B>,
    /// `Some` when `in_channels != out_channels`.
    shortcut_conv: Option<ShortcutConv<B>>,
    /// `GroupNorm(1)` pre-norm on the residual branch when channels change.
    shortcut_norm: Option<burn::nn::GroupNorm<B>>,
}

impl<B: Backend> ResnetBlock3D<B> {
    /// Build a `ResnetBlock3D`.
    ///
    /// # Errors
    /// Propagates errors from [`CausalConv3d::new`].
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        norm: &NormLayer<B>,
        spatial_padding_mode: &str,
        eps: f64,
        device: &B::Device,
    ) -> Result<Self, VaeError> {
        let norm1 = clone_norm(norm, in_channels, eps, device);
        let norm2 = clone_norm(norm, out_channels, eps, device);

        let conv1 = CausalConv3d::new(
            in_channels,
            out_channels,
            3,
            [1, 1, 1],
            1,
            1,
            true,
            spatial_padding_mode,
            device,
        )?;
        let conv2 = CausalConv3d::new(
            out_channels,
            out_channels,
            3,
            [1, 1, 1],
            1,
            1,
            true,
            spatial_padding_mode,
            device,
        )?;

        let (shortcut_conv, shortcut_norm) = if in_channels == out_channels {
            (None, None)
        } else {
            let sc = ShortcutConv::new(in_channels, out_channels, device);
            let gn = burn::nn::GroupNormConfig::new(1, in_channels)
                .with_epsilon(eps)
                .init(device);
            (Some(sc), Some(gn))
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut_conv,
            shortcut_norm,
        })
    }

    /// Causal forward pass. `x` shape: `(B, C_in, F, H, W)`.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        // Residual branch: optionally norm + project.
        let residual = self.shortcut_norm.as_ref().map_or_else(
            || x.clone(),
            |shortcut_norm| {
                let normed = shortcut_norm.forward(x.clone());
                match self.shortcut_conv.as_ref() {
                    Some(shortcut_conv) => shortcut_conv.forward(normed),
                    None => normed,
                }
            },
        );

        // Main branch: norm → SiLU → conv → norm → SiLU → conv.
        let h = self.norm1.forward(x);
        let h = silu(h);
        let h = self.conv1.forward(h, true);
        let h = self.norm2.forward(h);
        let h = silu(h);
        let h = self.conv2.forward(h, true);

        residual.add(h)
    }

    /// Load all parameters from a [`ltx_weights::Scope`].
    ///
    /// Key names match the reference `ResnetBlock3D.state_dict()`:
    /// - `conv1.conv.weight`, `conv1.conv.bias`
    /// - `conv2.conv.weight`, `conv2.conv.bias`
    /// - `conv_shortcut.weight`, `conv_shortcut.bias` (when channels change)
    /// - `norm3.weight`, `norm3.bias` (when channels change)
    /// - `norm1.weight`, `norm1.bias` (when using `GroupNorm`)
    /// - `norm2.weight`, `norm2.bias` (when using `GroupNorm`)
    ///
    /// # Errors
    /// Returns [`VaeError::Load`] if any expected tensor is missing or wrong rank.
    pub(crate) fn load_weights_from_scope(
        &mut self,
        scope: &ltx_weights::Scope<'_>,
        device: &B::Device,
    ) -> Result<(), VaeError> {
        self.norm1
            .load_weights_from_scope(&scope.scope("norm1"), device)?;
        self.conv1
            .load_weights_from_scope(&scope.scope("conv1"), device)?;
        self.norm2
            .load_weights_from_scope(&scope.scope("norm2"), device)?;
        self.conv2
            .load_weights_from_scope(&scope.scope("conv2"), device)?;
        if let Some(sc) = &mut self.shortcut_conv {
            sc.conv.weight = Param::from_tensor(scope.tensor("conv_shortcut.weight", device)?);
            sc.conv.bias = Some(Param::from_tensor(
                scope.tensor("conv_shortcut.bias", device)?,
            ));
        }
        if let Some(sn) = &mut self.shortcut_norm {
            sn.gamma = Some(Param::from_tensor(scope.tensor("norm3.weight", device)?));
            sn.beta = Some(Param::from_tensor(scope.tensor("norm3.bias", device)?));
        }
        Ok(())
    }
}

// ── UNetMidBlock3D ────────────────────────────────────────────────────────────

/// A stack of `num_layers` causal residual blocks with the same channel width.
///
/// This is the `"res_x"` encoder block type.
#[derive(Module, Debug)]
pub struct UNetMidBlock3D<B: Backend> {
    res_blocks: Vec<ResnetBlock3D<B>>,
}

impl<B: Backend> UNetMidBlock3D<B> {
    /// Build a `UNetMidBlock3D`.
    ///
    /// # Errors
    /// Propagates errors from [`ResnetBlock3D::new`].
    pub fn new(
        in_channels: usize,
        num_layers: usize,
        norm: &NormLayer<B>,
        spatial_padding_mode: &str,
        eps: f64,
        device: &B::Device,
    ) -> Result<Self, VaeError> {
        let mut res_blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            res_blocks.push(ResnetBlock3D::new(
                in_channels,
                in_channels,
                norm,
                spatial_padding_mode,
                eps,
                device,
            )?);
        }
        Ok(Self { res_blocks })
    }

    /// Forward through all residual blocks.
    pub fn forward(&self, mut x: Tensor<B, 5>) -> Tensor<B, 5> {
        for block in &self.res_blocks {
            x = block.forward(x);
        }
        x
    }

    /// Load all parameters from a [`ltx_weights::Scope`].
    ///
    /// Reads `res_blocks.{i}.*` for each block, matching the reference
    /// `UNetMidBlock3D.state_dict()` key names.
    ///
    /// # Errors
    /// Returns [`VaeError::Load`] if any tensor is missing or wrong rank.
    pub(crate) fn load_weights_from_scope(
        &mut self,
        scope: &ltx_weights::Scope<'_>,
        device: &B::Device,
    ) -> Result<(), VaeError> {
        for (i, block) in self.res_blocks.iter_mut().enumerate() {
            block.load_weights_from_scope(&scope.scope(&format!("res_blocks.{i}")), device)?;
        }
        Ok(())
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Clone a `NormLayer` configuration for a given channel count and epsilon.
fn clone_norm<B: Backend>(
    template: &NormLayer<B>,
    channels: usize,
    eps: f64,
    device: &B::Device,
) -> NormLayer<B> {
    match template {
        NormLayer::Pixel(_) => NormLayer::Pixel(crate::norm::PixelNorm::new()),
        NormLayer::Group(_) => {
            let gn = burn::nn::GroupNormConfig::new(32, channels)
                .with_epsilon(eps)
                .init(device);
            NormLayer::Group(gn)
        }
    }
}
