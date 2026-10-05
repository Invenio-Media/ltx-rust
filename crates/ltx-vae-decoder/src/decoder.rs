//! Diffusion video VAE decoder.
//!
//! ## Architecture
//!
//! ```text
//! latent [B, C_lat, F_l, H_l, W_l]
//!   → un_normalize + permute → [B, F_l, H_l, W_l, C_lat]  (channels-last)
//!   → conv_in               → [B, F_l, H_l, W_l, stage_channels[0]]
//!   → det_stage1: NABlock×d + upsample1
//!   → det_stage2: NABlock×d + upsample2
//!   → det_stage3: NABlock×d + upsample3
//!   → det_stage4: NABlock×d + upsample4 + ghost-crop → context
//!
//! x_t [B, C_out, F5, H5·p, W5·p]  (pre-generated noise)
//!   → patchify + conv_in_x_t → [B, F5, H5, W5, C5]
//!   → cat[context, x]        → [B, F5, H5, W5, C_ctx + C5]
//!
//! Euler loop:
//!   → t_embedder(t) → shared_adaln → 7 modulation tensors
//!   → diff_blocks (CombinedDiffusionNABlock × d5)
//!   → norm_out → conv_out → permute → unpatchify → pixels
//! ```
//!
//! ## Memory
//!
//! The eager NA3D materialises a full `[N, N]` score matrix (N = T·H·W).
//! NATTEN/Triton kernels are required for production; CPU parity tests with
//! N ≲ 100 are tractable.

use std::path::Path;

use burn::{
    module::{Module, Param},
    nn,
    tensor::{Tensor, backend::Backend},
};
use safetensors::SafeTensors;

use crate::{
    config::{DecoderConfig, ModelOutputType},
    error::VaeDecoderError,
    load::{load_1d, load_2d_raw, load_linear_weight},
    nn::{
        adaln::AdaLnZero,
        attention::NeighborhoodAttention3D,
        diff_block::CombinedDiffusionNaBlock,
        na_block::NaBlock,
        per_channel_stats::PerChannelStatistics,
        swiglu::SwiGlu,
        timestep_emb::{PixArtEmbeddings, TimestepEmbedding, scalar_timestep},
        upsample::LinearPixelShuffleUpsample,
    },
    ops::{patchify, unpatchify},
    tiling::{crop_trailing_context, pad_trailing_latent, stage4_thw},
};

// ─── Decoder struct ──────────────────────────────────────────────────────────

/// Diffusion video VAE decoder in Burn.
#[derive(Module, Debug)]
pub struct DiffusionVideoDecoder<B: Backend> {
    /// Per-channel latent (de)normalisation.
    pub per_channel_stats: PerChannelStatistics<B>,
    /// Latent-to-stage-1 feature projection (channels-last Linear).
    pub conv_in: nn::Linear<B>,
    /// Keyframe type tag (zero for non-keyframe decode).
    pub type_emb: Param<Tensor<B, 1>>,

    // ── Det stages ───────────────────────────────────────────────────────
    /// Stage-1 NA blocks.
    pub det_stage1: Vec<NaBlock<B>>,
    /// Stage-2 NA blocks.
    pub det_stage2: Vec<NaBlock<B>>,
    /// Stage-3 NA blocks.
    pub det_stage3: Vec<NaBlock<B>>,
    /// Stage-4 NA blocks.
    pub det_stage4: Vec<NaBlock<B>>,

    // ── Upsamples ────────────────────────────────────────────────────────
    /// Stage-1 → stage-2 upsample.
    pub up1: LinearPixelShuffleUpsample<B>,
    /// Stage-2 → stage-3 upsample.
    pub up2: LinearPixelShuffleUpsample<B>,
    /// Stage-3 → stage-4 upsample.
    pub up3: LinearPixelShuffleUpsample<B>,
    /// Stage-4 → stage-5 upsample.
    pub up4: LinearPixelShuffleUpsample<B>,

    // ── Diffusion stage ──────────────────────────────────────────────────
    /// Timestep → `t_emb_dim` embedding.
    pub t_embedder: PixArtEmbeddings<B>,
    /// Shared AdaLN-Zero.
    pub shared_adaln: AdaLnZero<B>,
    /// Noised-pixel patches → stage-5 features.
    pub conv_in_x_t: nn::Linear<B>,
    /// Stage-5 diffusion NA blocks.
    pub diff_blocks: Vec<CombinedDiffusionNaBlock<B>>,
    /// Stage-5 output RMS norm.
    pub norm_out: nn::RmsNorm<B>,
    /// Stage-5 feature → noised-pixel-patch channels.
    pub conv_out: nn::Linear<B>,

    // ── Non-learnable config ─────────────────────────────────────────────
    /// Number of Euler denoising steps.
    pub num_inference_steps: usize,
    /// Multiplier on timestep before embedding.
    pub timestep_scale_multiplier: f32,
    /// Velocity or x₀ prediction mode.
    pub model_output_type: ModelOutputType,
    /// Latent channel count.
    pub in_channels: usize,
    /// Pixel channel count.
    pub out_channels: usize,
    /// Spatial patch size.
    pub patch_size: usize,
    /// Context channel width (`stage_channels`[3]).
    pub context_channels: usize,
    /// NATTEN ghost-pad frame count.
    pub natten_trailing_pad: usize,
    /// Stage-5 kernel temporal size for ghost-crop floor.
    pub stage5_kernel_t: usize,
    /// Cumulative time-stride from the first three upsamples.
    pub time_scale: usize,
    /// Upsample-4 stride (for window reporting).
    pub up4_stride: [usize; 3],
}

// ─── Forward pass ─────────────────────────────────────────────────────────────

impl<B: Backend> DiffusionVideoDecoder<B> {
    const fn pixel_time_scale(&self) -> usize {
        self.time_scale.saturating_mul(self.up4_stride[0])
    }

    /// Decode a latent tensor with pre-generated noise.
    ///
    /// `latent`: channels-first `[B, C_lat, F_l, H_l, W_l]`.
    /// `noise`:  pixel-canvas noise `[B, C_out, F_pix, H_pix, W_pix]`;
    ///           `None` uses zeros (suitable for single-step x₀ models).
    ///
    /// Returns pixels `[B, C_out, F_pix, H_pix, W_pix]` in `[-1, 1]`.
    ///
    /// # Errors
    ///
    /// Returns a [`VaeDecoderError`] if patchification or the diff step fails.
    pub fn decode(
        &self,
        latent: Tensor<B, 5>,
        noise: Option<Tensor<B, 5>>,
        device: &B::Device,
    ) -> Result<Tensor<B, 5>, VaeDecoderError> {
        let [b, _c, f_l, h_l, w_l] = latent.dims();

        // Pixel content extents (for final crop).
        let pixel_f = f_l
            .saturating_sub(1)
            .saturating_mul(self.pixel_time_scale())
            .saturating_add(1);
        let pixel_h = h_l.saturating_mul(32);
        let pixel_w = w_l.saturating_mul(32);

        // NATTEN ghost pad + stages 1-3.
        let latent_padded = pad_trailing_latent(latent, self.natten_trailing_pad);
        let feat_s4 = self.forward_stages_1_to_3(latent_padded, true, device);

        // Stage 4 + ghost crop → context.
        let feat_ctx = self.forward_stage_4(feat_s4, true, device);
        let [_, ctx_t, ctx_h, ctx_w, _] = feat_ctx.dims();

        // Pixel canvas shape.
        let canvas_h = ctx_h.saturating_mul(self.patch_size);
        let canvas_w = ctx_w.saturating_mul(self.patch_size);

        let x_t = noise.unwrap_or_else(|| {
            Tensor::zeros([b, self.out_channels, ctx_t, canvas_h, canvas_w], device)
        });

        // Timestep schedule: linspace(1, 1/n_steps, n_steps).
        let n = self.num_inference_steps;
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "n_steps ≤ 100; f32 precision is sufficient for timestep schedule"
        )]
        let timesteps: Vec<f32> = (0..n)
            .map(|i| {
                if n <= 1 {
                    1.0_f32
                } else {
                    let frac = i as f32 / (n as f32 - 1.0);
                    1.0_f32.mul_add(1.0 - frac, frac / n as f32)
                }
            })
            .collect();

        let pixels = self.decode_one_tile(&feat_ctx, x_t, &timesteps, device)?;

        // Crop to content.
        let [b_out, c_out, f_out, h_out, w_out] = pixels.dims();
        let f_keep = f_out.min(pixel_f);
        let h_keep = h_out.min(pixel_h);
        let w_keep = w_out.min(pixel_w);
        Ok(pixels.slice([0..b_out, 0..c_out, 0..f_keep, 0..h_keep, 0..w_keep]))
    }

    // ── Stages 1-3 ────────────────────────────────────────────────────────

    fn forward_stages_1_to_3(
        &self,
        latent: Tensor<B, 5>,
        drop_leading: bool,
        device: &B::Device,
    ) -> Tensor<B, 5> {
        let x = self.per_channel_stats.un_normalize(latent);
        // channels-first → channels-last
        let x = x.permute([0, 2, 3, 4, 1]);
        let mut x = self.conv_in.forward(x);

        for blk in &self.det_stage1 {
            x = blk.forward(x, device);
        }
        x = self.up1.forward(x, drop_leading);

        for blk in &self.det_stage2 {
            x = blk.forward(x, device);
        }
        x = self.up2.forward(x, drop_leading);

        for blk in &self.det_stage3 {
            x = blk.forward(x, device);
        }
        self.up3.forward(x, drop_leading)
    }

    // ── Stage 4 ────────────────────────────────────────────────────────────

    fn forward_stage_4(
        &self,
        feat: Tensor<B, 5>,
        drop_leading: bool,
        device: &B::Device,
    ) -> Tensor<B, 5> {
        let mut x = feat;
        for blk in &self.det_stage4 {
            x = blk.forward(x, device);
        }
        x = self.up4.forward(x, drop_leading);
        // Ghost crop: crop back latent-space ghost frames after all temporal upsamples.
        crop_trailing_context(
            x,
            self.natten_trailing_pad,
            self.pixel_time_scale(),
            self.stage5_kernel_t,
        )
    }

    // ── One diffusion step ────────────────────────────────────────────────

    fn forward_diff_step(
        &self,
        context: &Tensor<B, 5>,
        x_t: Tensor<B, 5>,
        t_scalar: f32,
        device: &B::Device,
    ) -> Result<Tensor<B, 5>, VaeDecoderError> {
        let [b, c_ctx, _ctx_t, _ctx_h, _ctx_w] = context.dims();

        // Patchify and project noised pixels.
        let patched = patchify(x_t, self.patch_size)?;
        // channels-first → channels-last, then project
        let x_feat = self.conv_in_x_t.forward(patched.permute([0, 2, 3, 4, 1]));

        // cat[context | x]
        let context_and_x = Tensor::cat(vec![context.clone(), x_feat], 4);

        // Timestep embedding.
        let t_scaled = t_scalar * self.timestep_scale_multiplier;
        let t_tensor = scalar_timestep(t_scaled, b, device);
        let t_emb = self.t_embedder.forward(t_tensor, device);

        // Shared AdaLN modulation.
        let modulation = self.shared_adaln.forward(t_emb);

        // Run diff blocks.
        let mut cx = context_and_x;
        for blk in &self.diff_blocks {
            let x_new = blk.forward_combined(cx.clone(), &modulation, device);
            let [_, t5, h5, w5, _] = cx.dims();
            let ctx_part = cx.clone().slice([0..b, 0..t5, 0..h5, 0..w5, 0..c_ctx]);
            cx = Tensor::cat(vec![ctx_part, x_new], 4);
        }

        // Extract x half and produce pixel prediction.
        let [_, t5, h5, w5, full_c] = cx.dims();
        let x_half: Tensor<B, 5> = cx.slice([0..b, 0..t5, 0..h5, 0..w5, c_ctx..full_c]);

        let x_out: Tensor<B, 5> = self.norm_out.forward(x_half);
        let x_out = self.conv_out.forward(x_out);
        // channels-last → channels-first
        let x_out = x_out.permute([0, 4, 1, 2, 3]);
        unpatchify(x_out, self.patch_size)
    }

    // ── Euler step ────────────────────────────────────────────────────────

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Burn tensor ops run on device; Rust host integer overflow is not possible"
    )]
    fn euler_step(
        x_t: Tensor<B, 5>,
        model_out: Tensor<B, 5>,
        t_now: f32,
        t_next: f32,
    ) -> Tensor<B, 5> {
        let dt = t_now - t_next;
        x_t - model_out * dt
    }

    // ── Full tile decode ──────────────────────────────────────────────────

    /// Decode one stage-4 context tile through the Euler denoising loop.
    ///
    /// `context`: channels-last `[B, T, H, W, C_ctx]`.
    /// `x_t_init`: channels-first noise `[B, C_out, F_pix, H_pix, W_pix]`.
    /// `timesteps`: decreasing schedule, e.g. `[1.0, 0.5]` for 2 steps.
    ///
    /// # Errors
    ///
    /// Returns a [`VaeDecoderError`] if `timesteps` is empty or a diff step
    /// fails.
    #[expect(
        clippy::indexing_slicing,
        reason = "bounds are verified by construction"
    )]
    pub fn decode_one_tile(
        &self,
        context: &Tensor<B, 5>,
        x_t_init: Tensor<B, 5>,
        timesteps: &[f32],
        device: &B::Device,
    ) -> Result<Tensor<B, 5>, VaeDecoderError> {
        if timesteps.is_empty() {
            return Err(VaeDecoderError::InvalidArgument {
                detail: "timesteps must not be empty".to_owned(),
            });
        }

        let mut x_t = x_t_init;
        let n = timesteps.len();

        for i in 0..n.saturating_sub(1) {
            let t_now = timesteps[i];
            let t_next = timesteps[i.saturating_add(1)];
            let out = self.forward_diff_step(context, x_t.clone(), t_now, device)?;
            x_t = Self::euler_step(x_t, out, t_now, t_next);
        }

        let t_last = timesteps[n.saturating_sub(1)];
        let model_out = self.forward_diff_step(context, x_t.clone(), t_last, device)?;

        if self.model_output_type == ModelOutputType::X0 {
            return Ok(model_out);
        }
        Ok(Self::euler_step(x_t, model_out, t_last, 0.0))
    }

    // ── Window reporting ──────────────────────────────────────────────────

    /// Report the pixel canvas `[frames, height, width]` for memory budgeting.
    ///
    /// `ltx-budget` calls this to estimate peak decode memory.
    ///
    /// # Errors
    ///
    /// Returns [`VaeDecoderError::InvalidArgument`] if any dimension is zero.
    pub fn decode_window_pixels(
        &self,
        latent_t: usize,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<[usize; 3], VaeDecoderError> {
        let [s4_t, s4_h, s4_w] = stage4_thw(&[], latent_t, latent_h, latent_w);
        crate::tiling::decode_window_pixels(
            s4_t,
            s4_h,
            s4_w,
            self.up4_stride,
            self.patch_size,
            true,
        )
    }
}

// ─── Construction ─────────────────────────────────────────────────────────────

impl<B: Backend> DiffusionVideoDecoder<B> {
    /// Load weights from a safetensors file.
    ///
    /// `key_prefix` is prepended to every state-dict key (`""` for a fixture,
    /// `"vae.decoder."` for a full model checkpoint).
    ///
    /// # Errors
    ///
    /// Returns a [`VaeDecoderError`] if the file cannot be read, the config is
    /// invalid, or any required weight key is missing.
    pub fn load(
        path: &Path,
        config: &DecoderConfig,
        key_prefix: &str,
        device: &B::Device,
    ) -> Result<Self, VaeDecoderError> {
        let bytes = std::fs::read(path).map_err(|e| VaeDecoderError::InvalidArgument {
            detail: format!("cannot read {}: {e}", path.display()),
        })?;
        let st = SafeTensors::deserialize(&bytes)?;
        Self::from_safetensors(&st, config, key_prefix, device)
    }

    /// Build from an already-open safetensors archive.
    ///
    /// # Errors
    ///
    /// Returns a [`VaeDecoderError`] if the config is invalid or a required
    /// weight key is missing.
    #[expect(
        clippy::too_many_lines,
        reason = "checkpoint key mapping is kept linear so field names stay auditable against safetensors"
    )]
    pub fn from_safetensors(
        st: &SafeTensors<'_>,
        config: &DecoderConfig,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self, VaeDecoderError> {
        config.validate()?;
        let rope_split = config.rope_dim_split_resolved()?;
        let p = |s: &str| format!("{prefix}{s}");

        // ── per_channel_stats ─────────────────────────────────────────────
        let std_of_means = load_optional_1d(
            st,
            &p("per_channel_statistics.std-of-means"),
            config.in_channels,
            device,
            || Tensor::ones([config.in_channels], device),
        )?;
        let mean_of_means = load_optional_1d(
            st,
            &p("per_channel_statistics.mean-of-means"),
            config.in_channels,
            device,
            || Tensor::zeros([config.in_channels], device),
        )?;
        let per_channel_stats = PerChannelStatistics {
            std_of_means,
            mean_of_means,
        };

        // ── conv_in ───────────────────────────────────────────────────────
        let c0 = config.stage_channels.first().copied().unwrap_or(128);
        let conv_in = build_linear(st, &p("conv_in"), config.in_channels, c0, true, device)?;

        // ── type_emb ──────────────────────────────────────────────────────
        let type_emb_val = load_1d(st, &p("type_emb"), device)
            .unwrap_or_else(|_| Tensor::zeros([config.in_channels], device));
        let type_emb = Param::from_tensor(type_emb_val);

        // ── deterministic stages ──────────────────────────────────────────
        let det_stage1 = load_na_stage(st, prefix, 0, config, rope_split, device)?;
        let det_stage2 = load_na_stage(st, prefix, 1, config, rope_split, device)?;
        let det_stage3 = load_na_stage(st, prefix, 2, config, rope_split, device)?;
        let det_stage4 = load_na_stage(st, prefix, 3, config, rope_split, device)?;

        // ── upsamples ─────────────────────────────────────────────────────
        let up1 = load_upsample(st, prefix, 0, config, device)?;
        let up2 = load_upsample(st, prefix, 1, config, device)?;
        let up3 = load_upsample(st, prefix, 2, config, device)?;
        let up4 = load_upsample(st, prefix, 3, config, device)?;

        // ── t_embedder ────────────────────────────────────────────────────
        let t_emb_dim = config.t_emb_dim;
        let te = "t_embedder.timestep_embedder.";
        let l1 = build_linear(
            st,
            &p(&format!("{te}linear_1")),
            256,
            t_emb_dim,
            true,
            device,
        )?;
        let l2 = build_linear(
            st,
            &p(&format!("{te}linear_2")),
            t_emb_dim,
            t_emb_dim,
            true,
            device,
        )?;
        let t_embedder = PixArtEmbeddings {
            timestep_embedder: TimestepEmbedding {
                linear_1: l1,
                linear_2: l2,
            },
            num_channels: 256,
        };

        // ── shared_adaln ──────────────────────────────────────────────────
        let c5 = config.stage5_channels_resolved();
        let adaln_out = crate::nn::adaln::NUM_CHUNKS.saturating_mul(c5);
        let adaln_proj = build_linear(
            st,
            &p("shared_adaln.proj"),
            t_emb_dim,
            adaln_out,
            true,
            device,
        )?;
        let shared_adaln = AdaLnZero { proj: adaln_proj };

        // ── conv_in_x_t ───────────────────────────────────────────────────
        let noised_ch = config
            .out_channels
            .saturating_mul(config.patch_size.saturating_mul(config.patch_size));
        let conv_in_x_t = build_linear(st, &p("conv_in_x_t"), noised_ch, c5, true, device)?;

        // ── diff_blocks ───────────────────────────────────────────────────
        let d5 = config.stage_depths.last().copied().unwrap_or(8);
        let stage5_kernel = config.stage5_kernel;
        let n_det = config.stage_channels.len().saturating_sub(1);
        let c_ctx = config
            .stage_channels
            .get(n_det.saturating_sub(1))
            .copied()
            .unwrap_or(128);
        let mut diff_blocks = Vec::with_capacity(d5);
        for bi in 0..d5 {
            let bp = format!("diff_blocks.{bi}.");
            diff_blocks.push(load_diff_block(
                st,
                prefix,
                &bp,
                c5,
                c_ctx,
                stage5_kernel,
                config.head_dim,
                rope_split,
                device,
            )?);
        }

        // ── norm_out + conv_out ───────────────────────────────────────────
        let norm_out = build_rms_norm(st, &p("norm_out"), c5, device)?;
        let conv_out = build_linear(st, &p("conv_out"), c5, noised_ch, true, device)?;

        // ── config fields ─────────────────────────────────────────────────
        let up4_stride = config.upsamples.get(3).map_or([1, 1, 1], |u| u.stride);
        let time_scale: usize = config
            .upsamples
            .iter()
            .take(3)
            .map(|u| u.stride[0])
            .product();
        let natten_trailing_pad = config.stage_kernels.first().copied().unwrap_or([3, 3, 3])[0]
            .checked_div(2)
            .unwrap_or(0)
            .saturating_mul(2);

        Ok(Self {
            per_channel_stats,
            conv_in,
            type_emb,
            det_stage1,
            det_stage2,
            det_stage3,
            det_stage4,
            up1,
            up2,
            up3,
            up4,
            t_embedder,
            shared_adaln,
            conv_in_x_t,
            diff_blocks,
            norm_out,
            conv_out,
            num_inference_steps: config.default_num_inference_steps,
            timestep_scale_multiplier: config.timestep_scale_multiplier,
            model_output_type: config.model_output_type,
            in_channels: config.in_channels,
            out_channels: config.out_channels,
            patch_size: config.patch_size,
            context_channels: c_ctx,
            natten_trailing_pad,
            stage5_kernel_t: stage5_kernel[0],
            time_scale,
            up4_stride,
        })
    }
}

// ─── Weight-loading helpers ───────────────────────────────────────────────────

fn load_optional_1d<B: Backend>(
    st: &SafeTensors<'_>,
    key: &str,
    expected_len: usize,
    device: &B::Device,
    fallback: impl FnOnce() -> Tensor<B, 1>,
) -> Result<Tensor<B, 1>, VaeDecoderError> {
    match load_1d(st, key, device) {
        Ok(tensor) => {
            let [actual_len] = tensor.dims();
            if actual_len == expected_len {
                Ok(tensor)
            } else {
                Err(VaeDecoderError::ShapeMismatch {
                    key: key.to_owned(),
                    expected: vec![expected_len],
                    actual: vec![actual_len],
                })
            }
        }
        Err(VaeDecoderError::KeyNotFound { .. }) => Ok(fallback()),
        Err(err) => Err(err),
    }
}

fn build_linear<B: Backend>(
    st: &SafeTensors<'_>,
    prefix: &str,
    d_in: usize,
    d_out: usize,
    with_bias: bool,
    device: &B::Device,
) -> Result<nn::Linear<B>, VaeDecoderError> {
    let w = load_linear_weight(st, &format!("{prefix}.weight"), device)?;
    let b_opt: Option<Tensor<B, 1>> = if with_bias {
        load_1d(st, &format!("{prefix}.bias"), device).ok()
    } else {
        None
    };
    let mut linear = nn::LinearConfig::new(d_in, d_out)
        .with_bias(with_bias)
        .init(device);
    linear.weight = Param::from_tensor(w);
    if let Some(bias) = b_opt {
        linear.bias = Some(Param::from_tensor(bias));
    }
    Ok(linear)
}

fn build_rms_norm<B: Backend>(
    st: &SafeTensors<'_>,
    prefix: &str,
    dim: usize,
    device: &B::Device,
) -> Result<nn::RmsNorm<B>, VaeDecoderError> {
    let w = load_1d(st, &format!("{prefix}.weight"), device)?;
    let mut norm = nn::RmsNormConfig::new(dim).with_epsilon(1e-6).init(device);
    norm.gamma = Param::from_tensor(w);
    Ok(norm)
}

fn build_swiglu<B: Backend>(
    st: &SafeTensors<'_>,
    prefix: &str,
    dim: usize,
    hidden: usize,
    device: &B::Device,
) -> Result<SwiGlu<B>, VaeDecoderError> {
    let w_up = build_linear(st, &format!("{prefix}w_up"), dim, hidden, false, device)?;
    let w_gate = build_linear(st, &format!("{prefix}w_gate"), dim, hidden, false, device)?;
    let w_down = build_linear(st, &format!("{prefix}w_down"), hidden, dim, false, device)?;
    Ok(SwiGlu {
        w_up,
        w_gate,
        w_down,
    })
}

fn build_na_attn<B: Backend>(
    st: &SafeTensors<'_>,
    prefix: &str,
    dim: usize,
    kernel: [usize; 3],
    head_dim: usize,
    rope_split: [usize; 3],
    device: &B::Device,
) -> Result<NeighborhoodAttention3D<B>, VaeDecoderError> {
    let num_heads = dim.checked_div(head_dim).unwrap_or(1);
    let to_q = build_linear(st, &format!("{prefix}qkv.to_q"), dim, dim, true, device)?;
    let to_k = build_linear(st, &format!("{prefix}qkv.to_k"), dim, dim, true, device)?;
    let to_v = build_linear(st, &format!("{prefix}qkv.to_v"), dim, dim, true, device)?;
    let proj = build_linear(st, &format!("{prefix}proj"), dim, dim, true, device)?;
    let q_norm = build_rms_norm(st, &format!("{prefix}q_norm"), head_dim, device)?;
    let k_norm = build_rms_norm(st, &format!("{prefix}k_norm"), head_dim, device)?;
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "head_dim ≤ 256; no precision loss in f32"
    )]
    let scale = (head_dim as f32).powf(-0.5);
    Ok(NeighborhoodAttention3D {
        to_q,
        to_k,
        to_v,
        proj,
        q_norm,
        k_norm,
        dim,
        num_heads,
        head_dim,
        kernel_size: kernel,
        scale,
        rope_dim_split: rope_split,
        rope_base: 10_000.0,
    })
}

fn load_na_block<B: Backend>(
    st: &SafeTensors<'_>,
    prefix: &str,
    dim: usize,
    kernel: [usize; 3],
    head_dim: usize,
    rope_split: [usize; 3],
    device: &B::Device,
) -> Result<NaBlock<B>, VaeDecoderError> {
    let hidden = ((dim.saturating_mul(4))
        .saturating_add(15)
        .checked_div(16)
        .unwrap_or(1))
    .saturating_mul(16);
    let norm1 = build_rms_norm(st, &format!("{prefix}norm1"), dim, device)?;
    let attn = build_na_attn(
        st,
        &format!("{prefix}attn."),
        dim,
        kernel,
        head_dim,
        rope_split,
        device,
    )?;
    let norm2 = build_rms_norm(st, &format!("{prefix}norm2"), dim, device)?;
    let mlp = build_swiglu(st, &format!("{prefix}mlp."), dim, hidden, device)?;
    Ok(NaBlock {
        norm1,
        attn,
        norm2,
        mlp,
    })
}

fn load_na_stage<B: Backend>(
    st: &SafeTensors<'_>,
    global_prefix: &str,
    stage_idx: usize,
    config: &DecoderConfig,
    rope_split: [usize; 3],
    device: &B::Device,
) -> Result<Vec<NaBlock<B>>, VaeDecoderError> {
    let c = config.stage_channels.get(stage_idx).copied().unwrap_or(128);
    let depth = config.stage_depths.get(stage_idx).copied().unwrap_or(1);
    let kernel = config
        .stage_kernels
        .get(stage_idx)
        .copied()
        .unwrap_or([3, 3, 3]);
    (0..depth)
        .map(|bi| {
            let bp = format!("{global_prefix}det_stages.{stage_idx}.{bi}.");
            load_na_block(st, &bp, c, kernel, config.head_dim, rope_split, device)
        })
        .collect()
}

fn load_upsample<B: Backend>(
    st: &SafeTensors<'_>,
    global_prefix: &str,
    idx: usize,
    config: &DecoderConfig,
    device: &B::Device,
) -> Result<LinearPixelShuffleUpsample<B>, VaeDecoderError> {
    let up_spec = config
        .upsamples
        .get(idx)
        .ok_or_else(|| VaeDecoderError::InvalidConfig {
            detail: format!("missing upsamples[{idx}]"),
        })?;
    let c_in = config.stage_channels.get(idx).copied().unwrap_or(128);
    let c_out = c_in
        .checked_div(up_spec.out_channels_reduction_factor.max(1))
        .unwrap_or(c_in);
    let proj_out = c_out
        .saturating_mul(up_spec.stride[0])
        .saturating_mul(up_spec.stride[1])
        .saturating_mul(up_spec.stride[2]);
    let proj = build_linear(
        st,
        &format!("{global_prefix}upsamples.{idx}.proj"),
        c_in,
        proj_out,
        true,
        device,
    )?;
    Ok(LinearPixelShuffleUpsample {
        proj,
        stride: up_spec.stride,
        out_channels: c_out,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_diff_block<B: Backend>(
    st: &SafeTensors<'_>,
    global_prefix: &str,
    block_prefix: &str,
    dim: usize,
    context_channels: usize,
    kernel: [usize; 3],
    head_dim: usize,
    rope_split: [usize; 3],
    device: &B::Device,
) -> Result<CombinedDiffusionNaBlock<B>, VaeDecoderError> {
    let hidden = ((dim.saturating_mul(4))
        .saturating_add(15)
        .checked_div(16)
        .unwrap_or(1))
    .saturating_mul(16);
    let pfx = format!("{global_prefix}{block_prefix}");
    let context_proj = build_linear(
        st,
        &format!("{pfx}context_proj"),
        context_channels,
        dim,
        true,
        device,
    )?;
    let sst: Tensor<B, 2> = load_2d_raw(st, &format!("{pfx}scale_shift_table"), device)?;
    let scale_shift_table = Param::from_tensor(sst);
    let norm1 = build_rms_norm(st, &format!("{pfx}norm1"), dim, device)?;
    let attn = build_na_attn(
        st,
        &format!("{pfx}attn."),
        dim,
        kernel,
        head_dim,
        rope_split,
        device,
    )?;
    let norm2 = build_rms_norm(st, &format!("{pfx}norm2"), dim, device)?;
    let mlp = build_swiglu(st, &format!("{pfx}mlp."), dim, hidden, device)?;
    Ok(CombinedDiffusionNaBlock {
        context_proj,
        scale_shift_table,
        norm1,
        attn,
        norm2,
        mlp,
        context_channels,
    })
}
