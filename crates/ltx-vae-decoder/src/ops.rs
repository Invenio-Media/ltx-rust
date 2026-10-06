//! Space-to-depth (patchify) and depth-to-space (unpatchify) ops.
//!
//! Layout convention: pixel tensors are channels-first `[B, C, F, H, W]`.
//! Temporal patch size is always 1 in the decoder (only spatial patchification
//! is used at stage 5).
//!
//! Channel ordering matches the Python reference (einops `(c r q)` pattern):
//!   `c_flat = C · pw · ph + pw_sub · ph + ph_sub`
//! where `pw` is the W-direction stride (outer) and `ph` is the H-direction
//! stride (inner).  Both spatial shuffles are performed as two successive 1-D
//! folds to keep all intermediate tensors ≤ 6-D (`NdArray` limit).

use burn::tensor::{Tensor, backend::Backend};

use crate::error::VaeDecoderError;

/// Space-to-depth: `[B, C, F, H, W]` → `[B, C·p², F, H/p, W/p]`.
///
/// Channel ordering: `(C, pw_sub, ph_sub)` in C-order — W-stride is the more
/// significant index — matching the Python einops `(c r q)` pattern.
///
/// When `patch_size_hw == 1` the input is returned unchanged.
///
/// # Errors
///
/// Returns [`VaeDecoderError::InvalidArgument`] when H or W are not divisible by
/// `patch_size_hw`.
#[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
)]
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
    let p = patch_size_hw;
    let h_out = h.checked_div(p).unwrap_or(0);
    let w_out = w.checked_div(p).unwrap_or(0);

    // ── W pixel-fold (pw_sub → outer channel dim) ─────────────────────────
    // Split W into (w_out, pw): [B, C, F, H, W] → [B, C, F, H, w_out, pw]
    let x: Tensor<B, 6> = x.reshape([b, c, f, h, w_out, p]);
    // Move pw adjacent to C: [B, C, pw, F, H, w_out]
    let x: Tensor<B, 6> = x.permute([0, 1, 5, 2, 3, 4]);
    // Merge C and pw: [B, C*pw, F, H, w_out]
    let c_pw = c * p;
    let x: Tensor<B, 5> = x.reshape([b, c_pw, f, h, w_out]);

    // ── H pixel-fold (ph_sub → inner channel dim) ─────────────────────────
    // Split H into (h_out, ph): [B, C*pw, F, H, w_out] → [B, C*pw, F, h_out, ph, w_out]
    let x: Tensor<B, 6> = x.reshape([b, c_pw, f, h_out, p, w_out]);
    // Move ph adjacent to C*pw: [B, C*pw, ph, F, h_out, w_out]
    let x: Tensor<B, 6> = x.permute([0, 1, 4, 2, 3, 5]);
    // Merge C*pw and ph → C*p²: [B, C*p², F, h_out, w_out]
    let c_out = c_pw * p;
    Ok(x.reshape([b, c_out, f, h_out, w_out]))
}

/// Depth-to-space: `[B, C·p², F, H/p, W/p]` → `[B, C, F, H, W]`.
///
/// Inverse of [`patchify`]: recovers the original `(C, pw_sub, ph_sub)` channel
/// ordering by splitting the H dimension first (ph = innermost factor), then W.
///
/// When `patch_size_hw == 1` the input is returned unchanged.
///
/// # Errors
///
/// Returns [`VaeDecoderError::InvalidArgument`] when C is not divisible by `p²`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
)]
pub fn unpatchify<B: Backend>(
    x: Tensor<B, 5>,
    patch_size_hw: usize,
) -> Result<Tensor<B, 5>, VaeDecoderError> {
    if patch_size_hw == 1 {
        return Ok(x);
    }
    let [b, c_in, f, h_small, w_small] = x.dims();
    let p = patch_size_hw;
    let p2 = p * p;
    let c_rem = c_in.checked_rem(p2).unwrap_or(1);
    if c_rem != 0 {
        return Err(VaeDecoderError::InvalidArgument {
            detail: format!("unpatchify: C={c_in} not divisible by patch²={p2}"),
        });
    }
    let channels = c_in.checked_div(p2).unwrap_or(0);
    // c_in = C*pw*ph; split as (C*pw, ph) since ph is innermost.
    let c_pw = channels * p; // C * pw = c_in / p

    // ── H pixel-shuffle (ph = innermost factor; undo H-fold in patchify) ──
    // Split c_in into (C*pw, ph): inner ph → H axis.
    let x: Tensor<B, 6> = x.reshape([b, c_pw, p, f, h_small, w_small]);
    // Move ph adjacent to h_small: [B, C*pw, F, h_small, ph, w_small]
    let x: Tensor<B, 6> = x.permute([0, 1, 3, 4, 2, 5]);
    // Merge h_small and ph → H: [B, C*pw, F, H, w_small]
    let h_out = h_small * p;
    let x: Tensor<B, 5> = x.reshape([b, c_pw, f, h_out, w_small]);

    // ── W pixel-shuffle (pw = outer factor; undo W-fold in patchify) ───────
    // Split C*pw into (C, pw): inner pw → W axis.
    let x: Tensor<B, 6> = x.reshape([b, channels, p, f, h_out, w_small]);
    // Move pw adjacent to w_small: [B, C, F, H, w_small, pw]
    let x: Tensor<B, 6> = x.permute([0, 1, 3, 4, 5, 2]);
    // Merge w_small and pw → W: [B, C, F, H, W]
    let w_out = w_small * p;
    Ok(x.reshape([b, channels, f, h_out, w_out]))
}
