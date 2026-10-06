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

use burn::{
    module::{Module, Param},
    nn,
    tensor::{Tensor, backend::Backend},
};

use crate::{
    config::{DecoderConfig, ModelOutputType, UpsampleSpec},
    error::VaeDecoderError,
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
    tiling::{crop_trailing_context, pad_trailing_latent},
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
    /// Cumulative temporal stride from the first three upsamples.
    pub time_scale: usize,
    /// Cumulative `[T, H, W]` stride from the first three upsamples.
    pub stage4_stride: [usize; 3],
    /// Leading-frame temporal drop after the first three upsamples.
    pub stage4_time_drop: usize,
    /// Leading-frame temporal drop after all four upsamples.
    pub pixel_time_drop: usize,
    /// Upsample-4 stride (for window reporting).
    pub up4_stride: [usize; 3],
}

// ─── Forward pass ─────────────────────────────────────────────────────────────

impl<B: Backend> DiffusionVideoDecoder<B> {
    const fn pixel_time_scale(&self) -> usize {
        self.time_scale.saturating_mul(self.up4_stride[0])
    }

    const fn pixel_time_extent(&self, latent_t: usize) -> usize {
        latent_t
            .saturating_mul(self.pixel_time_scale())
            .saturating_sub(self.pixel_time_drop)
    }

    const fn stage4_time_extent(&self, latent_t: usize) -> usize {
        latent_t
            .saturating_mul(self.stage4_stride[0])
            .saturating_sub(self.stage4_time_drop)
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
        let pixel_f = self.pixel_time_extent(f_l);
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
        let [b, _ctx_t, _ctx_h, _ctx_w, c_ctx] = context.dims();

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
        let s4_t = self.stage4_time_extent(latent_t);
        let s4_h = latent_h.saturating_mul(self.stage4_stride[1]);
        let s4_w = latent_w.saturating_mul(self.stage4_stride[2]);
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
    /// Load weights from a [`ltx_weights::Scope`].
    ///
    /// `scope` must point at the decoder root — an empty prefix for a fixture
    /// saved with `module.state_dict()`, or a scope returned by
    /// `store.scope("")` with `KeyMap::video_decoder()` for a full checkpoint.
    ///
    /// # Errors
    ///
    /// Returns a [`VaeDecoderError`] if the config is invalid or any required
    /// weight key is missing.
    #[expect(
        clippy::too_many_lines,
        reason = "checkpoint key mapping is kept linear so field names stay auditable against the state dict"
    )]
    pub fn load(
        scope: &ltx_weights::Scope<'_>,
        config: &DecoderConfig,
        device: &B::Device,
    ) -> Result<Self, VaeDecoderError> {
        config.validate()?;
        let rope_split = config.rope_dim_split_resolved()?;

        // ── per_channel_stats ─────────────────────────────────────────────
        let std_of_means = scope
            .optional::<B, 1>("per_channel_statistics.std-of-means", device)?
            .unwrap_or_else(|| Tensor::ones([config.in_channels], device));
        let mean_of_means = scope
            .optional::<B, 1>("per_channel_statistics.mean-of-means", device)?
            .unwrap_or_else(|| Tensor::zeros([config.in_channels], device));
        let per_channel_stats = PerChannelStatistics {
            std_of_means,
            mean_of_means,
        };

        // ── conv_in ───────────────────────────────────────────────────────
        let conv_in = scope_linear(&scope.scope("conv_in"), device)?;

        // ── type_emb (Bug 3 fix: shape = in_channels, not stage_channels[0]) ──
        let type_emb_val = scope
            .optional::<B, 1>("type_emb", device)?
            .unwrap_or_else(|| Tensor::zeros([config.in_channels], device));
        let type_emb = Param::from_tensor(type_emb_val);

        // ── deterministic stages ──────────────────────────────────────────
        let det_stage1 = scope_na_stage(scope, 0, config, rope_split, device)?;
        let det_stage2 = scope_na_stage(scope, 1, config, rope_split, device)?;
        let det_stage3 = scope_na_stage(scope, 2, config, rope_split, device)?;
        let det_stage4 = scope_na_stage(scope, 3, config, rope_split, device)?;

        // ── upsamples ─────────────────────────────────────────────────────
        let up1 = scope_upsample(scope, 0, config, device)?;
        let up2 = scope_upsample(scope, 1, config, device)?;
        let up3 = scope_upsample(scope, 2, config, device)?;
        let up4 = scope_upsample(scope, 3, config, device)?;

        // ── t_embedder ────────────────────────────────────────────────────
        let te_scope = scope.scope("t_embedder.timestep_embedder");
        let l1 = scope_linear(&te_scope.scope("linear_1"), device)?;
        let l2 = scope_linear(&te_scope.scope("linear_2"), device)?;
        let t_embedder = PixArtEmbeddings {
            timestep_embedder: TimestepEmbedding {
                linear_1: l1,
                linear_2: l2,
            },
            num_channels: 256,
        };

        // ── shared_adaln ──────────────────────────────────────────────────
        let c5 = config.stage5_channels_resolved();
        let adaln_proj = scope_linear(&scope.scope("shared_adaln.proj"), device)?;
        let shared_adaln = AdaLnZero { proj: adaln_proj };

        // ── conv_in_x_t ───────────────────────────────────────────────────
        let conv_in_x_t = scope_linear(&scope.scope("conv_in_x_t"), device)?;

        // ── diff_blocks ───────────────────────────────────────────────────
        let d5 = config.stage_depths.last().copied().unwrap_or(8);
        let stage5_kernel = config.stage5_kernel;
        // Bug 2 fix: context_channels = stage_channels.last() not stage_channels[len-2]
        let c_ctx = config.stage_channels.last().copied().unwrap_or(128);
        let mut diff_blocks = Vec::with_capacity(d5);
        for bi in 0..d5 {
            diff_blocks.push(scope_diff_block(
                &scope.scope(&format!("diff_blocks.{bi}")),
                c5,
                c_ctx,
                stage5_kernel,
                config.head_dim,
                rope_split,
                device,
            )?);
        }

        // ── norm_out + conv_out ───────────────────────────────────────────
        let norm_out = scope_rms_norm(&scope.scope("norm_out"), device)?;
        let conv_out = scope_linear(&scope.scope("conv_out"), device)?;

        // ── config fields ─────────────────────────────────────────────────
        let up4_stride = config.upsamples.get(3).map_or([1, 1, 1], |u| u.stride);
        let stage4_stride = cumulative_upsample_stride(&config.upsamples, 3);
        let stage4_time_drop = cumulative_temporal_drop(&config.upsamples, 3);
        let time_scale = stage4_stride[0];
        let pixel_time_drop = cumulative_temporal_drop(&config.upsamples, 4);
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
            stage4_stride,
            stage4_time_drop,
            pixel_time_drop,
            up4_stride,
        })
    }
}

// ─── Stride helpers (used by load and tests) ──────────────────────────────────

fn cumulative_upsample_stride(upsamples: &[UpsampleSpec], count: usize) -> [usize; 3] {
    upsamples
        .iter()
        .take(count)
        .fold([1_usize, 1_usize, 1_usize], |stride, upsample| {
            [
                stride[0].saturating_mul(upsample.stride[0]),
                stride[1].saturating_mul(upsample.stride[1]),
                stride[2].saturating_mul(upsample.stride[2]),
            ]
        })
}

fn cumulative_temporal_drop(upsamples: &[UpsampleSpec], count: usize) -> usize {
    upsamples
        .iter()
        .take(count)
        .fold(0_usize, |drop, upsample| {
            let scaled_drop = drop.saturating_mul(upsample.stride[0]);
            if upsample.stride[0] == 2 {
                scaled_drop.saturating_add(1)
            } else {
                scaled_drop
            }
        })
}

// ─── Scope-based weight-loading helpers ───────────────────────────────────────

/// Load a `Linear` layer from `scope`; transposes weight from `[out, in]` to `[in, out]`.
fn scope_linear<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
    device: &B::Device,
) -> Result<nn::Linear<B>, VaeDecoderError> {
    let w: Tensor<B, 2> = scope.tensor("weight", device)?;
    let [d_out, d_in] = w.dims();
    let w = w.swap_dims(0, 1); // PyTorch [out, in] → Burn [in, out]
    let bias_opt: Option<Tensor<B, 1>> = scope.optional("bias", device)?;
    let with_bias = bias_opt.is_some();
    let mut linear = nn::LinearConfig::new(d_in, d_out)
        .with_bias(with_bias)
        .init(device);
    linear.weight = Param::from_tensor(w);
    if let Some(bias) = bias_opt {
        linear.bias = Some(Param::from_tensor(bias));
    }
    Ok(linear)
}

/// Load an `RmsNorm` from `scope.weight`.
fn scope_rms_norm<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
    device: &B::Device,
) -> Result<nn::RmsNorm<B>, VaeDecoderError> {
    let w: Tensor<B, 1> = scope.tensor("weight", device)?;
    let [dim] = w.dims();
    let mut norm = nn::RmsNormConfig::new(dim).with_epsilon(1e-6).init(device);
    norm.gamma = Param::from_tensor(w);
    Ok(norm)
}

/// Load a `SwiGlu` MLP from `scope.{w_up,w_gate,w_down}`.
fn scope_swiglu<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
    device: &B::Device,
) -> Result<SwiGlu<B>, VaeDecoderError> {
    let w_up = scope_linear(&scope.scope("w_up"), device)?;
    let w_gate = scope_linear(&scope.scope("w_gate"), device)?;
    let w_down = scope_linear(&scope.scope("w_down"), device)?;
    Ok(SwiGlu {
        w_up,
        w_gate,
        w_down,
    })
}

/// Load `NeighborhoodAttention3D` from `scope.{qkv.to_{q,k,v},proj,q_norm,k_norm}`.
fn scope_na_attn<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
    dim: usize,
    kernel: [usize; 3],
    head_dim: usize,
    rope_split: [usize; 3],
    device: &B::Device,
) -> Result<NeighborhoodAttention3D<B>, VaeDecoderError> {
    let num_heads = dim.checked_div(head_dim).unwrap_or(1);
    let qkv = scope.scope("qkv");
    let to_q = scope_linear(&qkv.scope("to_q"), device)?;
    let to_k = scope_linear(&qkv.scope("to_k"), device)?;
    let to_v = scope_linear(&qkv.scope("to_v"), device)?;
    let proj = scope_linear(&scope.scope("proj"), device)?;
    let q_norm = scope_rms_norm(&scope.scope("q_norm"), device)?;
    let k_norm = scope_rms_norm(&scope.scope("k_norm"), device)?;
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

/// Load a single `NaBlock` from `scope.{norm1,attn,norm2,mlp}`.
fn scope_na_block<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
    dim: usize,
    kernel: [usize; 3],
    head_dim: usize,
    rope_split: [usize; 3],
    device: &B::Device,
) -> Result<NaBlock<B>, VaeDecoderError> {
    let norm1 = scope_rms_norm(&scope.scope("norm1"), device)?;
    let attn = scope_na_attn(
        &scope.scope("attn"),
        dim,
        kernel,
        head_dim,
        rope_split,
        device,
    )?;
    let norm2 = scope_rms_norm(&scope.scope("norm2"), device)?;
    let mlp = scope_swiglu(&scope.scope("mlp"), device)?;
    Ok(NaBlock {
        norm1,
        attn,
        norm2,
        mlp,
    })
}

/// Load all blocks for `det_stages.{stage_idx}.{0..depth}`.
fn scope_na_stage<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
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
            scope_na_block(
                &scope.scope(&format!("det_stages.{stage_idx}.{bi}")),
                c,
                kernel,
                config.head_dim,
                rope_split,
                device,
            )
        })
        .collect()
}

/// Load `upsamples.{idx}.proj`.
fn scope_upsample<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
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
    let proj = scope_linear(&scope.scope(&format!("upsamples.{idx}.proj")), device)?;
    Ok(LinearPixelShuffleUpsample {
        proj,
        stride: up_spec.stride,
        out_channels: c_out,
    })
}

/// Load a `CombinedDiffusionNaBlock` from scope.
#[allow(clippy::too_many_arguments)]
fn scope_diff_block<B: Backend>(
    scope: &ltx_weights::Scope<'_>,
    dim: usize,
    context_channels: usize,
    kernel: [usize; 3],
    head_dim: usize,
    rope_split: [usize; 3],
    device: &B::Device,
) -> Result<CombinedDiffusionNaBlock<B>, VaeDecoderError> {
    let context_proj = scope_linear(&scope.scope("context_proj"), device)?;
    // scale_shift_table is a raw parameter, not a Linear weight — do NOT transpose.
    let sst: Tensor<B, 2> = scope.tensor("scale_shift_table", device)?;
    let scale_shift_table = Param::from_tensor(sst);
    let norm1 = scope_rms_norm(&scope.scope("norm1"), device)?;
    let attn = scope_na_attn(
        &scope.scope("attn"),
        dim,
        kernel,
        head_dim,
        rope_split,
        device,
    )?;
    let norm2 = scope_rms_norm(&scope.scope("norm2"), device)?;
    let mlp = scope_swiglu(&scope.scope("mlp"), device)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_upsamples() -> [UpsampleSpec; 4] {
        [
            UpsampleSpec {
                stride: [1, 2, 2],
                out_channels_reduction_factor: 2,
            },
            UpsampleSpec {
                stride: [2, 1, 1],
                out_channels_reduction_factor: 2,
            },
            UpsampleSpec {
                stride: [2, 2, 2],
                out_channels_reduction_factor: 1,
            },
            UpsampleSpec {
                stride: [2, 2, 2],
                out_channels_reduction_factor: 2,
            },
        ]
    }

    #[test]
    fn cumulative_upsample_plan_matches_reference_window() {
        let upsamples = reference_upsamples();
        let stage4_stride = cumulative_upsample_stride(&upsamples, 3);
        let stage4_time_drop = cumulative_temporal_drop(&upsamples, 3);
        let pixel_time_drop = cumulative_temporal_drop(&upsamples, 4);

        assert_eq!(stage4_stride, [4, 4, 4]);
        assert_eq!(stage4_time_drop, 3);
        assert_eq!(pixel_time_drop, 7);

        let latent_t = 2_usize;
        let latent_h = 3_usize;
        let latent_w = 4_usize;
        let stage4_t = latent_t
            .saturating_mul(stage4_stride[0])
            .saturating_sub(stage4_time_drop);
        let stage4_h = latent_h.saturating_mul(stage4_stride[1]);
        let stage4_w = latent_w.saturating_mul(stage4_stride[2]);

        let window = crate::tiling::decode_window_pixels(
            stage4_t,
            stage4_h,
            stage4_w,
            upsamples[3].stride,
            4,
            true,
        )
        .unwrap();

        assert_eq!(window, [9, 96, 128]);
        assert_eq!(
            latent_t
                .saturating_mul(stage4_stride[0].saturating_mul(upsamples[3].stride[0]))
                .saturating_sub(pixel_time_drop),
            9
        );
    }
}
