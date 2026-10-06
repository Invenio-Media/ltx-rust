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
//!
//! The key invariant: the frame dimension `F` must NOT be mixed with the patch
//! indices `(r, q)` during the reshape.  We therefore swap `C` and `F` first
//! so the batch-frame-channel (`B·F·C`) flat index puts `C` as the innermost
//! (fastest-varying) axis, matching what the final `(B, C·r·q, F, H, W)`
//! channel ordering requires.
//!
//! Steps:
//! 1. `(B, C, F, H, W)` → `swap_dims(1,2)` → `(B, F, C, H, W)`
//! 2. Reshape to `(B·F·C, h_out, q, w_out, r)` — the combined dim is now `B·F·C`
//!    where `C` varies fastest; swapping later will correctly separate `C·r·q` from `F`.
//! 3. `swap_dims(1,4)` → `(B·F·C, r, q, w_out, h_out)`
//! 4. `swap_dims(3,4)` → `(B·F·C, r, q, h_out, w_out)`
//! 5. Reshape to `(B·F, C·r·q, h_out, w_out)` [4-D] — now `C·r·q` channels are correct.
//! 6. Reshape to `(B, F, C·r·q, h_out, w_out)` [5-D].
//! 7. `swap_dims(1,2)` → `(B, C·r·q, F, h_out, w_out)`.

use burn::tensor::{Tensor, backend::Backend};

use crate::error::VaeError;

/// Fold `H×W` spatial patches into the channel axis.
///
/// Input:  `(B, C, F, H, W)` — `H` and `W` must be divisible by `patch_size`.
/// Output: `(B, C·P², F, H/P, W/P)` where `P = patch_size`.
///
/// Channel ordering: `(c, r, q)` outer→inner, matching the reference's
/// `b (c r q) f h w` layout.
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

    let bfc = nb
        .checked_mul(nf)
        .and_then(|v| v.checked_mul(nc))
        .ok_or(VaeError::DimOverflow)?;

    let bf = nb.checked_mul(nf).ok_or(VaeError::DimOverflow)?;

    let c_out = nc
        .checked_mul(patch_size)
        .and_then(|v| v.checked_mul(patch_size))
        .ok_or(VaeError::DimOverflow)?;

    // Step 1: (B, C, F, H, W) → (B, F, C, H, W)
    let x = x.swap_dims(1, 2);

    // Step 2: Reshape (B, F, C, H, W) → (B·F·C, h_out, q, w_out, r)
    // C is now innermost in the combined dim, so the later reshape won't mix F with (r,q).
    let x5: Tensor<B, 5> = x.reshape([bfc, h_out, patch_size, w_out, patch_size]);

    // Steps 3–4: swap to (B·F·C, r, q, h_out, w_out)
    //   [0=bfc, 1=h_out, 2=q, 3=w_out, 4=r]
    //   swap(1,4) → [bfc, r, q, w_out, h_out]
    //   swap(3,4) → [bfc, r, q, h_out, w_out]
    let x5 = x5.swap_dims(1, 4);
    let x5 = x5.swap_dims(3, 4);

    // Step 5: (B·F·C, r, q, h_out, w_out) → (B·F, C·r·q, h_out, w_out)
    // This reshape correctly assigns channel index = c·r_max·q_max + r·q_max + q.
    let x4: Tensor<B, 4> = x5.reshape([bf, c_out, h_out, w_out]);

    // Steps 6–7: (B·F, C·r·q, h_out, w_out) → (B, F, C_out, h, w) → (B, C_out, F, h, w)
    let x5: Tensor<B, 5> = x4.reshape([nb, nf, c_out, h_out, w_out]);
    Ok(x5.swap_dims(1, 2))
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

    let bfc = nb
        .checked_mul(nf)
        .and_then(|v| v.checked_mul(nc))
        .ok_or(VaeError::DimOverflow)?;

    let bf = nb.checked_mul(nf).ok_or(VaeError::DimOverflow)?;

    let nh = h_out.checked_mul(patch_size).ok_or(VaeError::DimOverflow)?;
    let nw = w_out.checked_mul(patch_size).ok_or(VaeError::DimOverflow)?;

    // Step 1: (B, C_out, F, h_out, w_out) → (B, F, C_out, h_out, w_out)
    let x = x.swap_dims(1, 2);

    // Step 2: → (B·F, C_out, h_out, w_out) [4-D]
    let x4: Tensor<B, 4> = x.reshape([bf, c_in, h_out, w_out]);

    // Step 3: → (B·F·C, r, q, h_out, w_out) — inverse of step 5 in patchify.
    let x5: Tensor<B, 5> = x4.reshape([bfc, patch_size, patch_size, h_out, w_out]);

    // Steps 4–5: inverse permutation of patchify steps 3–4.
    //   [bfc, r, q, h_out, w_out]
    //   swap(3,4) → [bfc, r, q, w_out, h_out]
    //   swap(1,4) → [bfc, h_out, q, w_out, r]
    let x5 = x5.swap_dims(3, 4);
    let x5 = x5.swap_dims(1, 4);
    // Step 6: (B·F·C, h_out, q, w_out, r) → (B·F·C, H, W)
    //   The flat layout h*(q·w_out·r) + q*(w_out·r) + w·r + r correctly maps
    //   to H=h·q_max+q and W=w·r_max+r, so the 5→3-D reshape is valid.
    let x3: Tensor<B, 3> = x5.reshape([bfc, nh, nw]);

    // Step 7: (B·F·C, H, W) → (B, F, C, H, W) → (B, C, F, H, W)
    let x5: Tensor<B, 5> = x3.reshape([nb, nf, nc, nh, nw]);
    Ok(x5.swap_dims(1, 2))
}
