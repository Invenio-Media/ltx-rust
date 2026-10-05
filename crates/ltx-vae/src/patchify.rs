//! Spatial space-to-depth (patchify).
//!
//! Port of `patchify` in `ltx_core/model/video_vae/ops.py` at commit 9ec55f9.
//!
//! The einops rearrangement used in the reference is:
//! ```text
//! b c (f p) (h q) (w r) -> b (c p r q) f h w
//! ```
//! with `p = patch_size_t = 1`, `q = r = patch_size_hw`.
//!
//! For the VAE encoder `patch_size_t` is always 1, so this reduces to:
//! ```text
//! b c f (h q) (w r) -> b (c r q) f h w
//! ```
//! which folds each `q×r` spatial patch into the channel axis.  The channel
//! ordering is `(c, r, q)` outer→inner: r indexes width-within-patch,
//! q indexes height-within-patch.
//!
//! # Implementation
//! We implement the permutation via reshape + two `swap_dims` calls on a
//! 5-D tensor `(B·C·F, h_out, q, w_out, r)`:
//! 1. `swap_dims(1, 4)` → `(B·C·F, r, q, w_out, h_out)`
//! 2. `swap_dims(3, 4)` → `(B·C·F, r, q, h_out, w_out)`
//!    Then reshape to `(B, C·r·q, F, h_out, w_out)`.

use burn::tensor::{Tensor, backend::Backend};

use crate::error::VaeError;

/// Fold `H×W` spatial patches into the channel axis.
///
/// Input:  `(B, C, F, H, W)` — `H` and `W` must be divisible by `patch_size`.
/// Output: `(B, C·P², F, H/P, W/P)` where `P = patch_size`.
///
/// # Errors
/// Returns [`VaeError::PatchifyDim`] if `H` or `W` is not divisible by
/// `patch_size`, or [`VaeError::DimOverflow`] on shape arithmetic overflow.
pub fn patchify<B: Backend>(x: Tensor<B, 5>, patch_size: usize) -> Result<Tensor<B, 5>, VaeError> {
    if patch_size <= 1 {
        return Ok(x);
    }

    let [nb, nc, nf, nh, nw] = x.dims();

    let h_out = nh
        .checked_div(patch_size)
        .filter(|&v| v.checked_mul(patch_size) == Some(nh))
        .ok_or(VaeError::PatchifyDim {
            value: nh,
            patch_size,
        })?;

    let w_out = nw
        .checked_div(patch_size)
        .filter(|&v| v.checked_mul(patch_size) == Some(nw))
        .ok_or(VaeError::PatchifyDim {
            value: nw,
            patch_size,
        })?;

    let bcf = nb
        .checked_mul(nc)
        .and_then(|v| v.checked_mul(nf))
        .ok_or(VaeError::DimOverflow)?;

    let c_out = nc
        .checked_mul(patch_size)
        .and_then(|v| v.checked_mul(patch_size))
        .ok_or(VaeError::DimOverflow)?;

    // Reshape (B, C, F, H, W) → (B·C·F, h_out, q, w_out, r)
    // where q = r = patch_size.
    let x5: Tensor<B, 5> = x.reshape([bcf, h_out, patch_size, w_out, patch_size]);

    // Dims: [0=bcf, 1=h_out, 2=q, 3=w_out, 4=r]
    // Target: [0=bcf, 1=r, 2=q, 3=h_out, 4=w_out]
    //   swap(1, 4): [bcf, r, q, w_out, h_out]
    //   swap(3, 4): [bcf, r, q, h_out, w_out]
    let x5 = x5.swap_dims(1, 4);
    let x5 = x5.swap_dims(3, 4);

    Ok(x5.reshape([nb, c_out, nf, h_out, w_out]))
}

/// Inverse of [`patchify`]: restore `H×W` from the channel axis.
///
/// Input:  `(B, C·P², F, H/P, W/P)`.
/// Output: `(B, C, F, H, W)`.
///
/// # Errors
/// Returns [`VaeError::PatchifyDim`] if the channel count is not divisible by
/// `patch_size²`, or [`VaeError::DimOverflow`] on overflow.
pub fn unpatchify<B: Backend>(
    x: Tensor<B, 5>,
    patch_size: usize,
) -> Result<Tensor<B, 5>, VaeError> {
    if patch_size <= 1 {
        return Ok(x);
    }

    let [nb, c_in, nf, h_out, w_out] = x.dims();

    let p2 = patch_size
        .checked_mul(patch_size)
        .ok_or(VaeError::DimOverflow)?;

    let nc = c_in
        .checked_div(p2)
        .filter(|&v| v.checked_mul(p2) == Some(c_in))
        .ok_or(VaeError::PatchifyDim {
            value: c_in,
            patch_size,
        })?;

    let bcf = nb
        .checked_mul(nc)
        .and_then(|v| v.checked_mul(nf))
        .ok_or(VaeError::DimOverflow)?;

    let nh = h_out.checked_mul(patch_size).ok_or(VaeError::DimOverflow)?;
    let nw = w_out.checked_mul(patch_size).ok_or(VaeError::DimOverflow)?;

    // Reshape (B, C·P², F, h_out, w_out) → (B·C·F, r, q, h_out, w_out)
    let x5: Tensor<B, 5> = x.reshape([bcf, patch_size, patch_size, h_out, w_out]);

    // Inverse permutation of patchify:
    //   swap(3, 4): [bcf, r, q, w_out, h_out]
    //   swap(1, 4): [bcf, h_out, q, w_out, r]
    let x5 = x5.swap_dims(3, 4);
    let x5 = x5.swap_dims(1, 4);

    Ok(x5.reshape([nb, nc, nf, nh, nw]))
}
