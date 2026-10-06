//! Linear channel-expand + channels-last pixel-shuffle upsample.
//!
//! Implements the reference `LinearPixelShuffleUpsample`:
//! ```text
//! "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c"
//! ```
//! which expands channels and then rearranges them into the spatial axes.
//!
//! The 3-D pixel shuffle is performed axis by axis (W → H → T) to avoid
//! intermediate tensors with more than 6 dimensions (`NdArray`'s limit).
//!
//! When `stride[0] == 2 && drop_leading_frame`, the first output time step
//! is dropped.  This models the causal `1:2` temporal mapping (one latent
//! frame → two pixel frames, but the duplicate first frame is discarded).
//! Only the **origin tile** (`t = 0`) should pass `drop_leading_frame = true`.

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, backend::Backend},
};

/// Channels-last pixel-shuffle upsample.
#[derive(Module, Debug)]
pub struct LinearPixelShuffleUpsample<B: Backend> {
    /// Linear channel expansion before pixel shuffle.
    pub proj: nn::Linear<B>,
    /// Pixel-shuffle strides `[t, h, w]`.
    pub stride: [usize; 3],
    /// Output channel count after the pixel shuffle.
    pub out_channels: usize,
}

impl<B: Backend> LinearPixelShuffleUpsample<B> {
    /// Upsample channels-last `[B, T, H, W, C]`.
    ///
    /// When `stride[0] == 2`, temporal stride doubles the time axis and the
    /// duplicate leading frame is dropped iff `drop_leading_frame = true`.
    ///
    /// The 3-D pixel shuffle is decomposed into three 1-D shuffles (W, H, T)
    /// to keep all intermediate tensors ≤ 6-D (`NdArray` limit).
    #[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    pub fn forward(&self, x: Tensor<B, 5>, drop_leading_frame: bool) -> Tensor<B, 5> {
        let [b, t, h, w, _] = x.dims();
        let [st, sh, sw] = self.stride;
        let c_out = self.out_channels;

        // Expand channels: [B, T, H, W, C] → [B, T, H, W, C_out * st * sh * sw]
        let x = self.proj.forward(x);

        // ── W pixel-shuffle ────────────────────────────────────────────────
        // Channels ordered as [C_out, st, sh, sw] in C-order (outermost..innermost).
        // Isolate sw: reshape last dim to [..., C_out*st*sh, sw] (6-D)
        let c_sh_t = c_out * st * sh; // channels after removing sw
        let x: Tensor<B, 6> = x.reshape([b, t, h, w, c_sh_t, sw]);
        // Permute to [..., W, sw, C_out*st*sh] then merge W and sw.
        let x: Tensor<B, 6> = x.permute([0, 1, 2, 3, 5, 4]);
        let w_up = w * sw;
        let x: Tensor<B, 5> = x.reshape([b, t, h, w_up, c_sh_t]);

        // ── H pixel-shuffle ────────────────────────────────────────────────
        // Channels are now [C_out, st, sh] in C-order; isolate sh.
        let c_t = c_out * st; // channels after removing sh
        let x: Tensor<B, 6> = x.reshape([b, t, h, w_up, c_t, sh]);
        // Permute to [..., H, sh, W_new, C_out*st] then merge H and sh.
        let x: Tensor<B, 6> = x.permute([0, 1, 2, 5, 3, 4]);
        let h_up = h * sh;
        let x: Tensor<B, 5> = x.reshape([b, t, h_up, w_up, c_t]);

        // ── T pixel-shuffle ────────────────────────────────────────────────
        // Channels are now [C_out, st] in C-order; isolate st.
        let x: Tensor<B, 6> = x.reshape([b, t, h_up, w_up, c_out, st]);
        // Permute to [..., T, st, H_new, W_new, C_out] then merge T and st.
        let x: Tensor<B, 6> = x.permute([0, 1, 5, 2, 3, 4]);
        let t_up = t * st;
        let x: Tensor<B, 5> = x.reshape([b, t_up, h_up, w_up, c_out]);

        // Drop the duplicate leading frame produced by temporal pixel-shuffle.
        if st == 2 && drop_leading_frame && t_up > 1 {
            x.slice([0..b, 1..t_up, 0..h_up, 0..w_up, 0..c_out])
        } else {
            x
        }
    }
}
