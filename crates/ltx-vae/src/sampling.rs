//! Space-to-depth downsampling block.
//!
//! Port of `SpaceToDepthDownsample` from
//! `ltx_core/model/video_vae/sampling.py` at commit 9ec55f9.
//!
//! The block applies pixel-shuffle to fold spatial/temporal strides into the
//! channel axis (matching the reference einops pattern
//! `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`), runs a parallel
//! stride-1 causal conv + pixel-shuffle path, then returns their element-wise
//! sum.
//!
//! For temporal stride 2, a copy of the first frame is prepended *before* the
//! `CausalConv3d`, giving `F + 1` frames. `CausalConv3d` then adds its own
//! two-frame causal pad, so the conv output has `F + 1` frames; the subsequent
//! pixel-shuffle reduces this to `(F + 1) / 2`.  With `F = 8k + 1`,
//! `(F + 1) / 2 = 4k + 1`, which is still a valid `8m + 1` count.

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

use crate::{conv::CausalConv3d, error::VaeError};

/// Downsamples via space-to-depth (pixel-shuffle) with a learned conv on the
/// residual path.
///
/// Supported stride triplets `[stride_t, stride_h, stride_w]`:
/// - `[2, 1, 1]` – temporal only (`"compress_time_res"`)
/// - `[1, 2, 2]` – spatial only (`"compress_space_res"`)
/// - `[2, 2, 2]` – all dimensions (`"compress_all_res"`)
#[derive(Module, Debug)]
pub struct SpaceToDepthDownsample<B: Backend> {
    /// Stride-1 causal conv on the residual path.
    conv: CausalConv3d<B>,
    stride_t: usize,
    stride_h: usize,
    stride_w: usize,
    /// Groups to average when constructing the skip connection.
    group_size: usize,
}

impl<B: Backend> SpaceToDepthDownsample<B> {
    /// Build and initialise.
    ///
    /// `out_channels` must divide `in_channels * stride_t * stride_h * stride_w`
    /// evenly; the quotient is `group_size`.
    ///
    /// # Errors
    /// Propagates errors from [`CausalConv3d::new`] and returns
    /// [`VaeError::DimOverflow`] on shape arithmetic overflow.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: [usize; 3],
        spatial_padding_mode: &str,
        device: &B::Device,
    ) -> Result<Self, VaeError> {
        let [stride_t, stride_h, stride_w] = stride;
        let stride_prod = stride_t
            .checked_mul(stride_h)
            .and_then(|v| v.checked_mul(stride_w))
            .ok_or(VaeError::DimOverflow)?;

        let pixel_channels = in_channels
            .checked_mul(stride_prod)
            .ok_or(VaeError::DimOverflow)?;

        if !pixel_channels.is_multiple_of(out_channels) {
            return Err(VaeError::Config(format!(
                "SpaceToDepth: in_channels={in_channels} * stride_prod={stride_prod} \
                 = {pixel_channels} is not divisible by out_channels={out_channels}"
            )));
        }
        let group_size = pixel_channels
            .checked_div(out_channels)
            .ok_or(VaeError::DimOverflow)?;

        let conv_out = out_channels
            .checked_div(stride_prod)
            .ok_or(VaeError::DimOverflow)?;

        let conv = CausalConv3d::new(
            in_channels,
            conv_out,
            3,
            [1, 1, 1],
            1,
            1,
            true,
            spatial_padding_mode,
            device,
        )?;

        Ok(Self {
            conv,
            stride_t,
            stride_h,
            stride_w,
            group_size,
        })
    }

    /// Forward. Input `(B, C, F, H, W)`.
    ///
    /// # Errors
    /// Returns [`VaeError::DimOverflow`] on shape arithmetic overflow.
    pub fn forward(&self, x: Tensor<B, 5>) -> Result<Tensor<B, 5>, VaeError> {
        let x = if self.stride_t == 2 {
            let first = x.clone().narrow(2, 0, 1);
            Tensor::cat(vec![first, x], 2)
        } else {
            x
        };

        let x_skip = pixel_shuffle_5d(x.clone(), self.stride_t, self.stride_h, self.stride_w)?;
        let x_skip = channel_avg(x_skip, self.group_size)?;

        let x_conv = self.conv.forward(x, true);
        let x_conv = pixel_shuffle_5d(x_conv, self.stride_t, self.stride_h, self.stride_w)?;

        Ok(x_conv.add(x_skip))
    }
}

// ── pixel_shuffle_5d ──────────────────────────────────────────────────────────

/// Fold temporal and spatial strides into the channel axis.
///
/// Implements `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`
/// (the reference einops pattern) in three sequential passes over 5-D tensors.
///
/// Output channel ordering per base channel `c`:
/// `c * st * sh * sw + p1 * sh * sw + p2 * sw + p3`.
fn pixel_shuffle_5d<B: Backend>(
    x: Tensor<B, 5>,
    stride_t: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor<B, 5>, VaeError> {
    let x = if stride_t > 1 {
        fold_temporal(x, stride_t)?
    } else {
        x
    };
    let x = if stride_h > 1 {
        fold_height(x, stride_h)?
    } else {
        x
    };
    let x = if stride_w > 1 {
        fold_width(x, stride_w)?
    } else {
        x
    };
    Ok(x)
}

/// `(B, C, F*st, H, W)` → `(B, C*st, F, H, W)`.
fn fold_temporal<B: Backend>(x: Tensor<B, 5>, stride_t: usize) -> Result<Tensor<B, 5>, VaeError> {
    let [nb, nc, f_in, nh, nw] = x.dims();
    let f_out = f_in.checked_div(stride_t).ok_or(VaeError::DimOverflow)?;
    let c_out = nc.checked_mul(stride_t).ok_or(VaeError::DimOverflow)?;
    let bc = nb.checked_mul(nc).ok_or(VaeError::DimOverflow)?;
    let hw = nh.checked_mul(nw).ok_or(VaeError::DimOverflow)?;

    let x3: burn::tensor::Tensor<B, 3> = x.reshape([bc, f_in, hw]);
    let x4: burn::tensor::Tensor<B, 4> = x3.reshape([bc, f_out, stride_t, hw]);
    let x4 = x4.swap_dims(1, 2);
    let bc_st = bc.checked_mul(stride_t).ok_or(VaeError::DimOverflow)?;
    // Reshape (B·C·st, f_out, H·W, 1) → (B, C·st, f_out, H, W)
    Ok(x4
        .reshape([bc_st, f_out, hw, 1])
        .reshape([nb, c_out, f_out, nh, nw]))
}

/// `(B, C, F, H*sh, W)` → `(B, C*sh, F, H, W)`.
fn fold_height<B: Backend>(x: Tensor<B, 5>, stride_h: usize) -> Result<Tensor<B, 5>, VaeError> {
    let [nb, nc, nf, h_in, nw] = x.dims();
    let h_out = h_in.checked_div(stride_h).ok_or(VaeError::DimOverflow)?;
    let c_out = nc.checked_mul(stride_h).ok_or(VaeError::DimOverflow)?;
    let bcf = nb
        .checked_mul(nc)
        .and_then(|v| v.checked_mul(nf))
        .ok_or(VaeError::DimOverflow)?;

    let x4: burn::tensor::Tensor<B, 4> = x.reshape([bcf, h_out, stride_h, nw]);
    let x4 = x4.swap_dims(1, 2);
    let bcf_sh = bcf.checked_mul(stride_h).ok_or(VaeError::DimOverflow)?;
    Ok(x4
        .reshape([bcf_sh, h_out, nw, 1])
        .reshape([nb, c_out, nf, h_out, nw]))
}

/// `(B, C, F, H, W*sw)` → `(B, C*sw, F, H, W)`.
fn fold_width<B: Backend>(x: Tensor<B, 5>, stride_w: usize) -> Result<Tensor<B, 5>, VaeError> {
    let [nb, nc, nf, nh, w_in] = x.dims();
    let w_out = w_in.checked_div(stride_w).ok_or(VaeError::DimOverflow)?;
    let c_out = nc.checked_mul(stride_w).ok_or(VaeError::DimOverflow)?;
    let bcfh = nb
        .checked_mul(nc)
        .and_then(|v| v.checked_mul(nf))
        .and_then(|v| v.checked_mul(nh))
        .ok_or(VaeError::DimOverflow)?;

    let x3: burn::tensor::Tensor<B, 3> = x.reshape([bcfh, w_out, stride_w]);
    let x3 = x3.swap_dims(1, 2);
    let bcfh_sw = bcfh.checked_mul(stride_w).ok_or(VaeError::DimOverflow)?;
    Ok(x3
        .reshape([bcfh_sw, w_out, 1])
        .reshape([nb, c_out, nf, nh, w_out]))
}

// ── channel_avg ───────────────────────────────────────────────────────────────

/// Average groups of `group_size` consecutive channels.
///
/// `(B, C*g, F, H, W)` → `(B, C, F, H, W)`.
fn channel_avg<B: Backend>(x: Tensor<B, 5>, group_size: usize) -> Result<Tensor<B, 5>, VaeError> {
    if group_size == 1 {
        return Ok(x);
    }
    let [nb, c_in, nf, nh, nw] = x.dims();
    let c_out = c_in.checked_div(group_size).ok_or(VaeError::DimOverflow)?;
    let fhw = nf
        .checked_mul(nh)
        .and_then(|v| v.checked_mul(nw))
        .ok_or(VaeError::DimOverflow)?;
    let b_c_out = nb.checked_mul(c_out).ok_or(VaeError::DimOverflow)?;

    let x4: burn::tensor::Tensor<B, 4> = x.reshape([nb, c_out, group_size, fhw]);
    let x4_mean: burn::tensor::Tensor<B, 4> = x4.mean_dim(2);
    let x2: burn::tensor::Tensor<B, 2> = x4_mean.reshape([b_c_out, fhw]);
    Ok(x2.reshape([nb, c_out, nf, nh, nw]))
}
