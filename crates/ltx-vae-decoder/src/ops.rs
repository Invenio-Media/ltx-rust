//! Space-to-depth (patchify) and depth-to-space (unpatchify) ops.
//!
//! Layout convention: pixel tensors are channels-first `[B, C, F, H, W]`.
//! Temporal patch size is always 1 in the decoder (only spatial patchification
//! is used at stage 5).

use burn::tensor::{Tensor, backend::Backend};

use crate::error::VaeDecoderError;

/// Space-to-depth: `[B, C, F, H, W]` → `[B, C·p², F, H/p, W/p]`.
///
/// When `patch_size_hw == 1` the input is returned unchanged.
///
/// # Errors
///
/// Returns [`VaeDecoderError::InvalidArgument`] when H or W are not divisible by
/// `patch_size_hw`.
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
pub fn patchify<B: Backend>(
    x: Tensor<B, 5>,
    patch_size_hw: usize,
) -> Result<Tensor<B, 5>, VaeDecoderError> {
    if patch_size_hw == 1 {
        return Ok(x);
    }
    let [b, c, f, h, w] = x.dims();
    let h_rem = h.checked_rem(patch_size_hw).unwrap_or(1);
    if h_rem != 0 {
        return Err(VaeDecoderError::InvalidArgument {
            detail: format!("patchify: H={h} not divisible by patch_size_hw={patch_size_hw}"),
        });
    }
    let w_rem = w.checked_rem(patch_size_hw).unwrap_or(1);
    if w_rem != 0 {
        return Err(VaeDecoderError::InvalidArgument {
            detail: format!("patchify: W={w} not divisible by patch_size_hw={patch_size_hw}"),
        });
    }
    let patch = patch_size_hw;
    let h_out = h.checked_div(patch).unwrap_or(0);
    let w_out = w.checked_div(patch).unwrap_or(0);
    let c_out = c.saturating_mul(patch).saturating_mul(patch);
    // [B, C, F, H, W] → [B, C, F, h_out, patch, w_out, patch]
    let x = x.reshape([b, c, f, h_out, patch, w_out, patch]);
    // → [B, C, patch, patch, F, h_out, w_out]
    let x = x.permute([0, 1, 4, 6, 2, 3, 5]);
    // → [B, C·patch², F, h_out, w_out]
    Ok(x.reshape([b, c_out, f, h_out, w_out]))
}

/// Depth-to-space: `[B, C·p², F, H/p, W/p]` → `[B, C, F, H, W]`.
///
/// When `patch_size_hw == 1` the input is returned unchanged.
///
/// # Errors
///
/// Returns [`VaeDecoderError::InvalidArgument`] when C is not divisible by `p²`.
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
pub fn unpatchify<B: Backend>(
    x: Tensor<B, 5>,
    patch_size_hw: usize,
) -> Result<Tensor<B, 5>, VaeDecoderError> {
    if patch_size_hw == 1 {
        return Ok(x);
    }
    let [b, c_in, f, h_small, w_small] = x.dims();
    let patch = patch_size_hw;
    let p2 = patch.saturating_mul(patch);
    let c_rem = c_in.checked_rem(p2).unwrap_or(1);
    if c_rem != 0 {
        return Err(VaeDecoderError::InvalidArgument {
            detail: format!("unpatchify: C={c_in} not divisible by patch²={p2}"),
        });
    }
    let channels = c_in.checked_div(p2).unwrap_or(0);
    let h_out = h_small.saturating_mul(patch);
    let w_out = w_small.saturating_mul(patch);
    // [B, C·p², F, h_small, w_small] → [B, C, p, p, F, h_small, w_small]
    let x = x.reshape([b, channels, patch, patch, f, h_small, w_small]);
    // → [B, C, F, h_small, p, w_small, p]
    let x = x.permute([0, 1, 4, 5, 2, 6, 3]);
    // → [B, C, F, H, W]
    Ok(x.reshape([b, channels, f, h_out, w_out]))
}
