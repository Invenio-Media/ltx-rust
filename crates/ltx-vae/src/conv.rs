//! Causal 3-D convolution.
//!
//! Port of `CausalConv3d` in
//! `ltx_core/model/video_vae/convolution.py` at commit 9ec55f9.
//!
//! **Causal padding**: the temporal dimension is padded by replicating the
//! first frame `(time_kernel_size - 1) * dilation` times before the convolution,
//! so the output at time `t` depends only on frames `<= t`.  The spatial
//! dimensions use zero-padding (only `"zeros"` mode is supported).

use burn::{
    module::Module,
    nn::{
        PaddingConfig3d,
        conv::{Conv3d, Conv3dConfig},
    },
    tensor::{Tensor, backend::Backend},
};

use crate::error::VaeError;

/// A causal 3-D convolution with first-frame temporal padding.
#[derive(Module, Debug)]
pub struct CausalConv3d<B: Backend> {
    conv: Conv3d<B>,
    /// Number of frames in the temporal kernel; used for padding.
    time_kernel_size: usize,
    /// Temporal dilation; used for padding.
    time_dilation: usize,
}

impl<B: Backend> CausalConv3d<B> {
    /// Build and initialise a `CausalConv3d`.
    ///
    /// # Parameters
    /// - `in_channels` / `out_channels`: channel counts.
    /// - `kernel_size`: applied in all three dimensions.
    /// - `stride`: `[time, height, width]`.
    /// - `dilation`: temporal dilation only (spatial dilation = 1).
    /// - `groups`: convolution groups.
    /// - `bias`: whether to include a bias term.
    /// - `spatial_padding_mode`: only `"zeros"` is supported.
    ///
    /// # Errors
    /// Returns [`VaeError::UnsupportedPaddingMode`] if `spatial_padding_mode`
    /// is not `"zeros"`.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the reference CausalConv3d init signature exactly"
    )]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: [usize; 3],
        dilation: usize,
        groups: usize,
        bias: bool,
        spatial_padding_mode: &str,
        device: &B::Device,
    ) -> Result<Self, VaeError> {
        if spatial_padding_mode != "zeros" {
            return Err(VaeError::UnsupportedPaddingMode(
                spatial_padding_mode.to_owned(),
            ));
        }

        let height_pad = kernel_size.checked_div(2).ok_or(VaeError::DimOverflow)?;
        let width_pad = height_pad;

        let conv = Conv3dConfig::new(
            [in_channels, out_channels],
            [kernel_size, kernel_size, kernel_size],
        )
        .with_stride(stride)
        .with_dilation([dilation, 1, 1])
        .with_padding(PaddingConfig3d::Explicit(0, height_pad, width_pad))
        .with_groups(groups)
        .with_bias(bias)
        .init(device);

        Ok(Self {
            conv,
            time_kernel_size: kernel_size,
            time_dilation: dilation,
        })
    }

    /// Forward pass.
    ///
    /// When `causal` is `true` (the encoder always uses `true`), the input is
    /// padded by replicating the first frame `(time_kernel_size - 1) * dilation`
    /// times. When `false`, symmetric replication padding is applied instead.
    ///
    /// Input shape: `(B, C, F, H, W)`.
    pub fn forward(&self, x: Tensor<B, 5>, causal: bool) -> Tensor<B, 5> {
        let x = self.apply_temporal_pad(x, causal);
        self.conv.forward(x)
    }

    fn apply_temporal_pad(&self, x: Tensor<B, 5>, causal: bool) -> Tensor<B, 5> {
        let pad_total = self
            .time_kernel_size
            .saturating_sub(1)
            .saturating_mul(self.time_dilation);
        if pad_total == 0 {
            return x;
        }
        if causal {
            // Prepend `pad_total` copies of the first frame.
            let first = x.clone().narrow(2, 0, 1);
            let pad = first.repeat_dim(2, pad_total);
            Tensor::cat(vec![pad, x], 2)
        } else {
            // Symmetric: prepend floor(pad/2) frames and append the remainder.
            let left = pad_total.saturating_div(2);
            let right = pad_total.saturating_sub(left);
            let first = if left > 0 {
                Some(x.clone().narrow(2, 0, 1).repeat_dim(2, left))
            } else {
                None
            };
            let [_nb, _nc, frames, _nh, _nw] = x.dims();
            let last = if right > 0 {
                Some(
                    x.clone()
                        .narrow(2, frames.saturating_sub(1), 1)
                        .repeat_dim(2, right),
                )
            } else {
                None
            };
            let mut parts = Vec::new();
            if let Some(first) = first {
                parts.push(first);
            }
            parts.push(x);
            if let Some(last) = last {
                parts.push(last);
            }
            Tensor::cat(parts, 2)
        }
    }
}
