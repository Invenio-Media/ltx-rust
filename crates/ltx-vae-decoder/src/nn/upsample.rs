//! Linear channel-expand + channels-last pixel-shuffle upsample.
//!
//! Implements the reference `LinearPixelShuffleUpsample`:
//! ```text
//! "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c"
//! ```
//! which expands channels and then rearranges them into the spatial axes.
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
    #[expect(clippy::many_single_char_names, reason = "tensor dim variables")]
    pub fn forward(&self, x: Tensor<B, 5>, drop_leading_frame: bool) -> Tensor<B, 5> {
        let [b, t, h, w, _] = x.dims();
        let [st, sh, sw] = self.stride;
        let c_out = self.out_channels;

        // Expand channels: [B, T, H, W, C] → [B, T, H, W, C_out * st * sh * sw]
        let x = self.proj.forward(x);

        // Reshape: [B, T, H, W, C_out, st, sh, sw]
        let x = x.reshape([b, t, h, w, c_out, st, sh, sw]);

        // Permute: → [B, T, st, H, sh, W, sw, C_out]
        let x = x.permute([0, 1, 5, 2, 6, 3, 7, 4]);

        // Reshape: → [B, T*st, H*sh, W*sw, C_out]
        let t_up = t.saturating_mul(st);
        let h_up = h.saturating_mul(sh);
        let w_up = w.saturating_mul(sw);
        let x = x.reshape([b, t_up, h_up, w_up, c_out]);

        // Drop the duplicate leading frame produced by temporal pixel-shuffle.
        if st == 2 && drop_leading_frame && t_up > 1 {
            x.slice([0..b, 1..t_up, 0..h_up, 0..w_up, 0..c_out])
        } else {
            x
        }
    }
}
