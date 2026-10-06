//! [`BurnBackend`]: the pure Burn `AlphaBackend` implementation.
//!
//! See [`crate`] docs for the full pipeline description.

use std::num::NonZeroU32;

use burn::{prelude::Backend, tensor::Tensor};
use ltx_backend::{AlphaBackend, AlphaChunk, BackendError, Keyframes, MemSample, VideoChunk};
use ltx_dit::{DiTConfig, OffloadMode, VideoInput, VideoTransformer};
use ltx_sampler::{
    GaussianNoiser, GuidedDenoiser, GuiderParams, ImageKeyframeCondition, LTX2Scheduler,
    LatentState, SamplerError, SchedulerConfig, VideoDenoiserModel, VideoModelInput,
    VideoReferenceCondition, euler_denoising_loop, make_target_positions, patchify, unpatchify,
};
use ltx_shape::{IcLoraLayout, ScaleFactors};
use ltx_vae::VideoEncoder;
use ltx_vae_decoder::DiffusionVideoDecoder;
use ltx_weights::{KeyMap, LoraFile, WeightStore};

use crate::{
    BurnError,
    settings::{GenerationSettings, ModelFiles},
};

// ── Public struct ─────────────────────────────────────────────────────────────

/// Pure Burn implementation of [`AlphaBackend`].
///
/// Construct via [`BurnBackend::load`].  Generic over the Burn backend `B`;
/// typical choices are `NdArray<f32>` (CPU) and `Metal<half::bf16>` (Apple GPU).
///
/// [`AlphaBackend`]: ltx_backend::AlphaBackend
pub struct BurnBackend<B: Backend> {
    transformer: VideoTransformer<B>,
    encoder: VideoEncoder<B>,
    decoder: DiffusionVideoDecoder<B>,
    /// Conditioned text context `[1, S, D]` from the prompt-context file.
    positive_context: Tensor<B, 3>,
    /// Unconditioned text context for CFG; `None` when `cfg_scale == 1.0`.
    negative_context: Option<Tensor<B, 3>>,
    /// `IC-LoRA` reference layout resolved from `LoRA` file metadata.
    ic_layout: IcLoraLayout,
    settings: GenerationSettings,
    device: B::Device,
}

// ── Construction ──────────────────────────────────────────────────────────────

impl<B: Backend> BurnBackend<B> {
    /// Load all model components from disk and prepare for inference.
    ///
    /// Steps:
    /// 1. Open the transformer checkpoint with the transformer `KeyMap`.
    /// 2. Collect `IC-LoRA` reference scale factors; merge each `LoRA`.
    /// 3. Open the video VAE checkpoint; load encoder and decoder.
    /// 4. Open the prompt-context safetensors; validate format metadata.
    ///
    /// # Errors
    /// Returns [`BurnError`] on any weight loading, config parsing, or
    /// shape validation failure.
    pub fn load(
        files: &ModelFiles,
        settings: &GenerationSettings,
        device: &B::Device,
    ) -> Result<Self, BurnError> {
        // ── transformer ──────────────────────────────────────────────────────
        let mut t_store = WeightStore::open(
            std::slice::from_ref(&files.transformer),
            &KeyMap::transformer(),
        )?;
        let t_cfg_json = t_store.config()?;
        let transformer_json = t_cfg_json.get("transformer").unwrap_or(&t_cfg_json);
        let dit_config = DiTConfig::from_json(transformer_json)?;

        // ── IC-LoRA: open each file once; collect layout and merge in one pass ─
        let mut spatial: Vec<u32> = Vec::new();
        let mut temporal: Vec<u32> = Vec::new();
        for (path, strength) in &files.loras {
            let lora = LoraFile::open(path)?;
            let layout = lora.ic_layout()?;
            let ds = layout.reference_downscale();
            let ts = layout.reference_temporal();
            if ds != 1 {
                spatial.push(ds);
            }
            if ts != 1 {
                temporal.push(ts);
            }
            t_store.merge_lora(&lora, *strength)?;
        }
        spatial.sort_unstable();
        spatial.dedup();
        temporal.sort_unstable();
        temporal.dedup();
        if spatial.len() > 1 {
            return Err(BurnError::LoraScaleDisagreement { values: spatial });
        }
        if temporal.len() > 1 {
            return Err(BurnError::LoraTemporalDisagreement { values: temporal });
        }
        let ic_layout = IcLoraLayout::new(
            spatial.first().copied().unwrap_or(1),
            temporal.first().copied().unwrap_or(1),
        )
        .map_err(BurnError::Shape)?;
        let transformer = VideoTransformer::<B>::load(&t_store.scope(""), &dit_config, device)?;

        // ── video VAE encoder ─────────────────────────────────────────────────
        let enc_store = WeightStore::open(
            std::slice::from_ref(&files.video_vae),
            &KeyMap::video_encoder(),
        )?;
        let vae_cfg_json = enc_store.config()?;
        let vae_json = vae_cfg_json.get("vae").unwrap_or(&vae_cfg_json);
        let enc_cfg = ltx_vae::VaeEncoderConfig::from_vae_json(vae_json)?;
        let encoder = VideoEncoder::<B>::load(&enc_store.scope(""), &enc_cfg, device)?;

        // ── video VAE decoder ─────────────────────────────────────────────────
        let dec_store = WeightStore::open(
            std::slice::from_ref(&files.video_vae),
            &KeyMap::video_decoder(),
        )?;
        let dec_cfg = ltx_vae_decoder::DecoderConfig::from_vae_json(vae_json)?;
        let decoder = DiffusionVideoDecoder::<B>::load(&dec_store.scope(""), &dec_cfg, device)?;

        // ── prompt context ────────────────────────────────────────────────────
        let ctx_store = WeightStore::open(
            std::slice::from_ref(&files.prompt_context),
            &KeyMap::identity(),
        )?;
        let fmt = ctx_store.metadata("format").unwrap_or("");
        if fmt != "ltx-prompt-context/1" {
            return Err(BurnError::PromptContextFormat {
                found: fmt.to_owned(),
            });
        }
        let ctx_scope = ctx_store.scope("");
        let positive_context: Tensor<B, 3> = ctx_scope.tensor("positive.video_encoding", device)?;
        let negative_context: Option<Tensor<B, 3>> =
            ctx_scope.optional("negative.video_encoding", device)?;
        validate_settings(&settings, negative_context.is_some())?;

        Ok(Self {
            transformer,
            encoder,
            decoder,
            positive_context,
            negative_context,
            ic_layout,
            settings: settings.clone(),
            device: device.clone(),
        })
    }

    /// Construct a [`BurnBackend`] from pre-built components.
    ///
    /// Used by parity tests that load each model component directly from a
    /// fixture file rather than through the production `ModelFiles`-based loader.
    /// Not needed for normal inference; use [`BurnBackend::load`] instead.
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor mirrors the struct fields; no meaningful grouping"
    )]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "Tensor fields are not const-constructible"
    )]
    pub fn from_components(
        transformer: VideoTransformer<B>,
        encoder: VideoEncoder<B>,
        decoder: DiffusionVideoDecoder<B>,
        positive_context: Tensor<B, 3>,
        negative_context: Option<Tensor<B, 3>>,
        ic_layout: IcLoraLayout,
        settings: GenerationSettings,
        device: B::Device,
    ) -> Self {
        Self {
            transformer,
            encoder,
            decoder,
            positive_context,
            negative_context,
            ic_layout,
            settings,
            device,
        }
    }
}

// ── AlphaBackend impl ─────────────────────────────────────────────────────────

impl<B: Backend> AlphaBackend for BurnBackend<B> {
    fn probe(&self, _shape: ltx_shape::PixelShape) -> Result<MemSample, BackendError> {
        Err(BackendError::Unsupported(
            BurnError::ProbeNotSupported.to_string(),
        ))
    }

    fn run_chunk(
        &self,
        rgb: &VideoChunk,
        seed: u64,
        cond: Option<&Keyframes>,
    ) -> Result<AlphaChunk, BackendError> {
        self.run_chunk_impl(rgb, seed, cond, None, None)
            .map_err(|e| BackendError::RunnerError {
                msg: e.to_string(),
                stderr: String::new(),
            })
    }
}

// ── Core pipeline ─────────────────────────────────────────────────────────────

impl<B: Backend> BurnBackend<B> {
    /// Internal pipeline; accepts optional noise overrides for parity tests.
    ///
    /// `initial_noise`: noisy latent tokens `[1, T, C]` to use instead of
    /// sampling from `GaussianNoiser`.
    /// `decoder_noise`: pixel-canvas noise `[1, C_out, F_pix, H_pix, W_pix]` passed
    /// directly to the decoder.
    #[expect(
        clippy::too_many_lines,
        reason = "each step labelled and ordered to mirror the reference alpha_gen pipeline"
    )]
    fn run_chunk_impl(
        &self,
        rgb: &VideoChunk,
        seed: u64,
        cond: Option<&Keyframes>,
        initial_noise: Option<Tensor<B, 3>>,
        decoder_noise: Option<Tensor<B, 5>>,
    ) -> Result<AlphaChunk, BurnError> {
        let frames = usize::try_from(rgb.frame_count).map_err(|_| BurnError::Overflow)?;
        let height = usize::try_from(rgb.height).map_err(|_| BurnError::Overflow)?;
        let width = usize::try_from(rgb.width).map_err(|_| BurnError::Overflow)?;

        let tf = self.encoder.temporal_factor();
        let sf = self.encoder.spatial_factor();
        let lat_c = self.encoder.latent_channels();

        let lat_f = latent_frames(frames, tf)?;
        // Validate that spatial dims divide exactly — otherwise encoded reference and
        // target latents disagree on H/W and conditioning shapes become corrupt.
        if height.checked_rem(sf).ok_or(BurnError::Overflow)? != 0 {
            return Err(BurnError::InvalidInput(format!(
                "height {height} must be a multiple of spatial_factor {sf}"
            )));
        }
        if width.checked_rem(sf).ok_or(BurnError::Overflow)? != 0 {
            return Err(BurnError::InvalidInput(format!(
                "width {width} must be a multiple of spatial_factor {sf}"
            )));
        }
        let lat_h = height.checked_div(sf).ok_or(BurnError::Overflow)?;
        let lat_w = width.checked_div(sf).ok_or(BurnError::Overflow)?;

        let scale = encoder_scale(
            u32::try_from(tf).map_err(|_| BurnError::Overflow)?,
            u32::try_from(sf).map_err(|_| BurnError::Overflow)?,
        )?;

        // ── step 1: VAE-encode the reference video (IC-LoRA conditioning) ────
        let ref_latent = self.build_reference_latent(rgb, frames, height, width)?;

        // ── step 2: zero target latent ────────────────────────────────────────
        let zero_latent: Tensor<B, 5> =
            Tensor::zeros([1, lat_c, lat_f, lat_h, lat_w], &self.device);

        // ── step 3: patchify → initial LatentState ────────────────────────────
        let target_tokens = lat_f
            .checked_mul(lat_h)
            .and_then(|n| n.checked_mul(lat_w))
            .ok_or(BurnError::Overflow)?;

        let positions = make_target_positions::<B>(
            u32::try_from(lat_f).map_err(|_| BurnError::Overflow)?,
            u32::try_from(lat_h).map_err(|_| BurnError::Overflow)?,
            u32::try_from(lat_w).map_err(|_| BurnError::Overflow)?,
            1,
            scale,
            self.settings.frame_rate,
            &self.device,
        )?;

        let tokens = patchify(zero_latent)?;
        let state = LatentState {
            latent: tokens.clone(),
            denoise_mask: Tensor::ones([1, target_tokens, 1], &self.device),
            positions,
            clean_latent: tokens,
            attention_mask: None,
            keyframes_mask: None,
        };

        // ── step 4: append IC-LoRA reference tokens ───────────────────────────
        let ref_cond = VideoReferenceCondition {
            latent: ref_latent,
            downscale_factor: self.ic_layout.reference_downscale(),
            temporal_scale_factor: self.ic_layout.reference_temporal(),
            strength: 1.0,
            first_latent_frame: 0,
        };
        let state = ref_cond.apply_to(state, scale, self.settings.frame_rate, &self.device)?;

        // ── step 5: seam keyframe conditioning ────────────────────────────────
        let state = self.apply_keyframes(state, cond, lat_h, lat_w)?;

        // ── step 6: sigma schedule ────────────────────────────────────────────
        let total_tokens = u64::try_from(state.total_tokens()).map_err(|_| BurnError::Overflow)?;
        let sigmas = LTX2Scheduler::execute(
            self.settings.num_inference_steps,
            total_tokens,
            SchedulerConfig::default(),
        )?;
        let sigma_max = sigmas
            .first()
            .copied()
            .ok_or(SamplerError::EmptySchedule(0))?;

        // ── step 7: initial noise ─────────────────────────────────────────────
        let state = if let Some(noise) = initial_noise {
            GaussianNoiser::apply_with_noise(state, noise, sigma_max)
        } else {
            let mut noiser = GaussianNoiser::new(seed)?;
            noiser.apply(state, sigma_max, &self.device)?
        };

        // ── step 8: Euler denoising loop ──────────────────────────────────────
        let params = GuiderParams::alpha_gen(self.settings.cfg_scale);
        let denoiser = GuidedDenoiser {
            context: self.positive_context.clone(),
            negative_context: if params.needs_uncond() {
                self.negative_context.clone()
            } else {
                None
            },
            params,
        };
        let adapter = TransformerAdapter {
            transformer: &self.transformer,
            device: &self.device,
        };
        let final_state = euler_denoising_loop(
            &sigmas,
            state,
            target_tokens,
            &adapter,
            &denoiser,
            &self.device,
        )?;

        // ── step 9: unpatchify ────────────────────────────────────────────────
        let denoised_latent = unpatchify(final_state.latent, lat_f, lat_h, lat_w)?;

        // ── step 10: VAE decode ───────────────────────────────────────────────
        let pixels = self
            .decoder
            .decode(denoised_latent, decoder_noise, &self.device)?;

        // ── step 11: RGB [-1,1] → alpha [0,1] via Rec.709 luminance ──────────
        let [_b, _c, pix_f, pix_h, pix_w] = pixels.dims();
        let alpha_data = pixels_to_alpha::<B>(pixels, pix_f, pix_h, pix_w)?;

        Ok(AlphaChunk {
            start_frame: rgb.start_frame,
            width: u32::try_from(pix_w).map_err(|_| BurnError::Overflow)?,
            height: u32::try_from(pix_h).map_err(|_| BurnError::Overflow)?,
            data: alpha_data,
            frame_count: u32::try_from(pix_f).map_err(|_| BurnError::Overflow)?,
        })
    }

    /// Build the IC-LoRA reference latent from the input RGB chunk.
    ///
    /// 1. Convert `[0,1]` interleaved RGB to `[1,3,F,H,W]` in `[-1,1]`.
    /// 2. Spatial downsample by `reference_downscale_factor` (box filter).
    /// 3. Temporal subsample: keep frame 0, then every `reference_temporal_scale_factor`-th.
    /// 4. VAE encode.
    fn build_reference_latent(
        &self,
        rgb: &VideoChunk,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor<B, 5>, BurnError> {
        let ds = usize::try_from(self.ic_layout.reference_downscale())
            .map_err(|_| BurnError::Overflow)?;
        let ts = usize::try_from(self.ic_layout.reference_temporal())
            .map_err(|_| BurnError::Overflow)?;

        // Floor division matches the Python reference (`//` operator).
        // If ds does not divide height/width exactly, the remainder is silently
        // dropped (the encoder crops trailing partial-patch rows/columns anyway).
        let ref_h = height.checked_div(ds).ok_or(BurnError::Overflow)?;
        let ref_w = width.checked_div(ds).ok_or(BurnError::Overflow)?;

        let expected = frames
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(3))
            .ok_or(BurnError::Overflow)?;
        if rgb.frames.len() != expected {
            return Err(BurnError::InvalidInput(format!(
                "VideoChunk.frames len {} does not match {}*{}*{}*3",
                rgb.frames.len(),
                frames,
                height,
                width
            )));
        }

        // [0,1] → [-1,1].
        let scaled: Vec<f32> = rgb.frames.iter().map(|&v| v.mul_add(2.0, -1.0)).collect();
        let video_5d =
            interleaved_to_channels_first::<B>(&scaled, frames, height, width, &self.device)?;

        // Spatial downsample.
        let video_5d = if ds > 1 {
            resize_and_center_crop_5d::<B>(video_5d, ref_h, ref_w, &self.device)?
        } else {
            video_5d
        };

        // Temporal subsample.
        let video_5d = if ts > 1 {
            temporal_subsample::<B>(&video_5d, ts)?
        } else {
            video_5d
        };

        Ok(self.encoder.encode(video_5d)?)
    }

    /// Apply seam keyframe conditionings to `state`.
    ///
    /// Each keyframe in `cond` is VAE-encoded as a single-frame clip and injected
    /// at its chunk-local latent frame index via [`ImageKeyframeCondition`].
    fn apply_keyframes(
        &self,
        mut state: LatentState<B>,
        cond: Option<&Keyframes>,
        lat_h: usize,
        lat_w: usize,
    ) -> Result<LatentState<B>, BurnError> {
        let Some(kf) = cond else { return Ok(state) };
        if !kf.strength.is_finite() || !(0.0..=1.0).contains(&kf.strength) {
            return Err(BurnError::InvalidInput(format!(
                "Keyframes.strength {} must be finite and in [0, 1]",
                kf.strength
            )));
        }

        let kf_count = usize::try_from(kf.frame_count).map_err(|_| BurnError::Overflow)?;
        let kf_h = usize::try_from(kf.height).map_err(|_| BurnError::Overflow)?;
        let kf_w = usize::try_from(kf.width).map_err(|_| BurnError::Overflow)?;
        let tokens_per_frame = lat_h.checked_mul(lat_w).ok_or(BurnError::Overflow)?;

        let frame_pixels = kf_h
            .checked_mul(kf_w)
            .and_then(|n| n.checked_mul(3))
            .ok_or(BurnError::Overflow)?;
        let expected_data = kf_count
            .checked_mul(frame_pixels)
            .ok_or(BurnError::Overflow)?;

        if kf.indices.len() != kf_count {
            return Err(BurnError::InvalidInput(format!(
                "Keyframes.indices.len() {} != frame_count {}",
                kf.indices.len(),
                kf_count
            )));
        }
        if kf.frames.len() != expected_data {
            return Err(BurnError::InvalidInput(format!(
                "Keyframes.frames.len() {} != {}*{}*{}*3",
                kf.frames.len(),
                kf_count,
                kf_h,
                kf_w
            )));
        }

        for i in 0..kf_count {
            let slice_start = i.checked_mul(frame_pixels).ok_or(BurnError::Overflow)?;
            let slice_end = slice_start
                .checked_add(frame_pixels)
                .ok_or(BurnError::Overflow)?;
            let frame_slice = kf
                .frames
                .get(slice_start..slice_end)
                .ok_or(BurnError::Overflow)?;

            // [0,1] → [-1,1].
            let scaled: Vec<f32> = frame_slice.iter().map(|&v| v.mul_add(2.0, -1.0)).collect();

            // Build [1, 3, 1, kf_h, kf_w] and VAE-encode.
            let img_5d = interleaved_to_channels_first::<B>(&scaled, 1, kf_h, kf_w, &self.device)?;
            let encoded = self.encoder.encode(img_5d)?;

            let frame_idx = usize::try_from(kf.indices.get(i).copied().ok_or(BurnError::Overflow)?)
                .map_err(|_| BurnError::Overflow)?;

            let kf_cond = ImageKeyframeCondition {
                image_latent: encoded,
                latent_frame_index: frame_idx,
                tokens_per_frame,
                strength: kf.strength,
            };
            state = kf_cond.apply_to(state, &self.device)?;
        }
        Ok(state)
    }

    /// Run one chunk with caller-supplied noise tensors for deterministic parity.
    ///
    /// Production code calls [`AlphaBackend::run_chunk`] which draws noise from
    /// a seeded `RNG`.  This method injects exact noise tensors so the Rust result
    /// can be compared element-by-element against a Python reference run.
    /// For normal inference use [`AlphaBackend::run_chunk`] which draws from a seeded `RNG`.
    ///
    /// # Errors
    /// Same as [`AlphaBackend::run_chunk`]: returns [`BurnError`] on any shape,
    /// model, or sampler failure.
    pub fn run_chunk_with_noise(
        &self,
        rgb: &VideoChunk,
        seed: u64,
        cond: Option<&Keyframes>,
        initial_noise: Tensor<B, 3>,
        decoder_noise: Tensor<B, 5>,
    ) -> Result<AlphaChunk, BurnError> {
        self.run_chunk_impl(rgb, seed, cond, Some(initial_noise), Some(decoder_noise))
    }
}

// ── VideoDenoiserModel adapter ────────────────────────────────────────────────

/// Wraps [`VideoTransformer`] to implement [`VideoDenoiserModel`].
///
/// Bridges the sampler's `VideoModelInput` to the `DiT`'s `VideoInput`:
/// - `timesteps [B, T, 1]` → `[B, T]` (`DiT` wants 2-D per-token σ·mask).
/// - `attention_mask Option<[B, T, T]>` → `Option<[B, 1, T, T]>` (broadcast head dim).
/// - `context_mask` is always `None` for `alpha_gen` (helpers.py:622).
struct TransformerAdapter<'a, B: Backend> {
    transformer: &'a VideoTransformer<B>,
    device: &'a B::Device,
}

impl<B: Backend> VideoDenoiserModel<B> for TransformerAdapter<'_, B> {
    fn forward(&self, input: VideoModelInput<B>) -> Result<Tensor<B, 3>, SamplerError> {
        let [batch, tokens, _ch] = input.latent.dims();

        // [B, T, 1] → [B, T]
        let ts_2d = input.timesteps.reshape([batch, tokens]);

        // Option<[B, T, T]> → Option<[B, 1, T, T]>
        let self_attn_mask = input.attention_mask.map(|m| m.unsqueeze_dim::<4>(1));

        let video_input = VideoInput {
            latent: input.latent,
            timesteps: ts_2d,
            positions: input.positions,
            context: input.context,
            context_mask: None, // always None per helpers.py:622
            self_attn_mask,
        };

        Ok(self
            .transformer
            .forward(video_input, OffloadMode::Resident, self.device))
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Compute latent frame count from pixel frames and temporal compression.
///
/// `latent_f = 1 + (pixel_f - 1) / temporal_factor`
///
/// # Errors
/// Returns [`BurnError::InvalidInput`] when `temporal_factor` is zero (an
/// encoder with a zero temporal factor cannot have been constructed).
fn latent_frames(pixel_f: usize, temporal_factor: usize) -> Result<usize, BurnError> {
    if temporal_factor == 0 {
        return Err(BurnError::InvalidInput(
            "encoder temporal_factor is 0; this encoder state is invalid".into(),
        ));
    }
    pixel_f
        .saturating_sub(1)
        .checked_div(temporal_factor)
        .and_then(|q| q.checked_add(1))
        .ok_or(BurnError::Overflow)
}

/// Build `ScaleFactors` from the encoder's temporal and (square) spatial factor.
fn encoder_scale(temporal: u32, spatial: u32) -> Result<ScaleFactors, BurnError> {
    let time = NonZeroU32::new(temporal).ok_or(BurnError::Overflow)?;
    let height = NonZeroU32::new(spatial).ok_or(BurnError::Overflow)?;
    let width = NonZeroU32::new(spatial).ok_or(BurnError::Overflow)?;
    Ok(ScaleFactors {
        time,
        height,
        width,
    })
}

/// Convert interleaved RGB `f32` (frame×row×col×channel layout) to
/// channels-first `[1, 3, F, H, W]` on device.
///
/// `data` must contain exactly `F * H * W * 3` elements.
fn interleaved_to_channels_first<B: Backend>(
    data: &[f32],
    frames: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, BurnError> {
    use burn::tensor::TensorData;

    let n = frames
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_mul(3))
        .ok_or(BurnError::Overflow)?;
    if data.len() != n {
        return Err(BurnError::InvalidInput(format!(
            "data.len() {} != {}*{}*{}*3 (frames×height×width×channels)",
            data.len(),
            frames,
            height,
            width
        )));
    }
    // [F, H, W, 3] → [3, F, H, W] → [1, 3, F, H, W]
    let t: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(data.to_vec(), [frames, height, width, 3]),
        device,
    );
    Ok(t.permute([3, 0, 1, 2]).unsqueeze_dim::<5>(0))
}

/// Bilinear resize `[1, C, F, src_h, src_w]` → `[1, C, F, dst_h, dst_w]`
/// matching `F.interpolate(mode="bilinear", align_corners=False, antialias=False)`
/// with aspect-ratio-preserving scale and center crop, exactly matching
/// `resize_and_center_crop` in `ltx_pipelines.utils.media_io.resize`.
///
/// All computation is done in `f64` host-side to match PyTorch's internal precision.
/// The `ceil` step uses exact integer `div_ceil` to avoid floating-point off-by-one
/// errors that occur when `src * scale` is within machine epsilon of an integer.
///
/// # Panics / Errors
/// Returns `Err` when `batch != 1`, when tensor data cannot be read, or when the
/// internal geometry is inconsistent (should not happen given the validation above).
#[expect(
    clippy::doc_markdown,
    reason = "PyTorch is a proper noun; False is Python bool literal, not a Rust item"
)]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "bilinear coordinate math uses f64 for PyTorch parity; casts are range-checked by floor+clamp"
)]
#[expect(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    reason = "index arithmetic is bounded by validated dimensions; mul_add formula matches the reference exactly"
)]
fn resize_and_center_crop_5d<B: Backend>(
    tensor: Tensor<B, 5>,
    dst_h: usize,
    dst_w: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, BurnError> {
    use burn::tensor::TensorData;
    let [nb, nc, nf, src_h, src_w] = tensor.dims();
    if nb != 1 {
        return Err(BurnError::InvalidInput(format!(
            "batch must be 1, got {nb}"
        )));
    }
    if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return Err(BurnError::InvalidInput(format!(
            "resize dimensions must be non-zero: src={src_h}×{src_w} dst={dst_h}×{dst_w}"
        )));
    }

    // Exact-integer ceiling avoids FP off-by-one when src*scale is within
    // machine epsilon of an integer (matches Python math.ceil behaviour).
    // scale = max(dst_h/src_h, dst_w/src_w); represented as cross-multiplied comparison.
    let (new_h, new_w) = if dst_h * src_w >= dst_w * src_h {
        // h-dominant: scale = dst_h/src_h; new_h = dst_h exactly.
        let new_w = (src_w * dst_h).div_ceil(src_h);
        (dst_h, new_w)
    } else {
        // w-dominant: scale = dst_w/src_w; new_w = dst_w exactly.
        let new_h = (src_h * dst_w).div_ceil(src_w);
        (new_h, dst_w)
    };
    // Safety: new_h >= dst_h and new_w >= dst_w by construction above.
    let crop_top = new_h.checked_sub(dst_h).ok_or_else(|| {
        BurnError::InvalidInput(format!(
            "resize geometry error: new_h={new_h} < dst_h={dst_h}"
        ))
    })? / 2;
    let crop_left = new_w.checked_sub(dst_w).ok_or_else(|| {
        BurnError::InvalidInput(format!(
            "resize geometry error: new_w={new_w} < dst_w={dst_w}"
        ))
    })? / 2;

    // Pull to host. This forces a device sync on Metal/CUDA; acceptable here
    // because `build_reference_latent` is called once per chunk, not per token.
    let src_data: Vec<f32> = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| BurnError::InvalidInput(format!("tensor data read failed: {e}")))?;

    // src layout: [nb=1, nc, nf, src_h, src_w]
    let src_frame_stride = nc * src_h * src_w;
    let dst_frame_stride = nc * dst_h * dst_w;
    let mut out = vec![0.0_f32; nf * nc * dst_h * dst_w];

    for f in 0..nf {
        for c in 0..nc {
            let src_off = f * src_frame_stride + c * src_h * src_w;
            let dst_off = f * dst_frame_stride + c * dst_h * dst_w;
            for y_out in 0..dst_h {
                let y_new = y_out + crop_top;
                let ys = (y_new as f64 + 0.5) * (src_h as f64 / new_h as f64) - 0.5;
                let y0 = (ys.floor() as isize).clamp(0, (src_h - 1) as isize) as usize;
                let y1 = ((y0 as isize + 1).clamp(0, (src_h - 1) as isize)) as usize;
                let dy = (ys - ys.floor()).clamp(0.0, 1.0) as f32;
                for x_out in 0..dst_w {
                    let x_new = x_out + crop_left;
                    let xs = (x_new as f64 + 0.5) * (src_w as f64 / new_w as f64) - 0.5;
                    let x0 = (xs.floor() as isize).clamp(0, (src_w - 1) as isize) as usize;
                    let x1 = ((x0 as isize + 1).clamp(0, (src_w - 1) as isize)) as usize;
                    let dx = (xs - xs.floor()).clamp(0.0, 1.0) as f32;
                    let v00 = src_data[src_off + y0 * src_w + x0];
                    let v01 = src_data[src_off + y0 * src_w + x1];
                    let v10 = src_data[src_off + y1 * src_w + x0];
                    let v11 = src_data[src_off + y1 * src_w + x1];
                    let top_val = v00.mul_add(1.0 - dx, v01 * dx);
                    let bot_val = v10.mul_add(1.0 - dx, v11 * dx);
                    out[dst_off + y_out * dst_w + x_out] = top_val.mul_add(1.0 - dy, bot_val * dy);
                }
            }
        }
    }

    Ok(Tensor::<B, 5>::from_data(
        TensorData::new(out, [nb, nc, nf, dst_h, dst_w]),
        device,
    ))
}

/// Keep frame 0 plus every `ts`-th subsequent frame.
///
/// Mirrors `temporal_subsample` in `iclora_utils.py`:
/// `indices = [0, *range(1, frames, ts)]`.
fn temporal_subsample<B: Backend>(x: &Tensor<B, 5>, ts: usize) -> Result<Tensor<B, 5>, BurnError> {
    let nf = x.dims()[2];
    if nf == 0 {
        return Err(BurnError::InvalidInput("video has zero frames".into()));
    }
    let mut parts: Vec<Tensor<B, 5>> = vec![x.clone().narrow(2, 0, 1)];
    let mut i = 1_usize;
    while i < nf {
        parts.push(x.clone().narrow(2, i, 1));
        i = i.checked_add(ts).ok_or(BurnError::Overflow)?;
    }
    Ok(Tensor::cat(parts, 2))
}

/// Convert decoded pixels `[1, 3, F, H, W]` in `[-1, 1]` to alpha in `[0, 1]`.
///
/// Rec.709 luminance: `Y = 0.2126·R + 0.7152·G + 0.0722·B`, clamped to [0, 1].
/// Pixels are first remapped `[-1,1] → [0,1]`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Burn tensor `+` runs on the compute backend and cannot overflow a Rust integer"
)]
fn pixels_to_alpha<B: Backend>(
    pixels: Tensor<B, 5>,
    pix_f: usize,
    pix_h: usize,
    pix_w: usize,
) -> Result<Vec<f32>, BurnError> {
    let npix = pix_f
        .checked_mul(pix_h)
        .and_then(|n| n.checked_mul(pix_w))
        .ok_or(BurnError::Overflow)?;
    // [-1,1] → [0,1]
    let p01 = pixels.add_scalar(1.0_f32).div_scalar(2.0_f32);
    let r_ch = p01.clone().narrow(1, 0, 1); // [1, 1, F, H, W]
    let g_ch = p01.clone().narrow(1, 1, 1);
    let b_ch = p01.narrow(1, 2, 1);
    // Y = 0.2126·R + 0.7152·G + 0.0722·B
    let y_luma =
        r_ch.mul_scalar(0.2126_f32) + g_ch.mul_scalar(0.7152_f32) + b_ch.mul_scalar(0.0722_f32);
    let y_luma = y_luma.clamp(0.0_f32, 1.0_f32);
    // Flatten to [1, N] and extract as Vec<f32>.
    let flat = y_luma.reshape([1, npix]);
    flat.into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| BurnError::Overflow)
}
/// Validate `GenerationSettings` against the available context.
///
/// Factored out of [`BurnBackend::load`] so it can be unit-tested without
/// requiring real safetensors files.
fn validate_settings(settings: &GenerationSettings, has_negative: bool) -> Result<(), BurnError> {
    // NaN would silently bypass needs_uncond() (which checks abs distance,
    // returning false for NaN) and cause silent NaN propagation in the guider.
    if !settings.cfg_scale.is_finite() {
        return Err(BurnError::InvalidInput(format!(
            "cfg_scale {} must be finite",
            settings.cfg_scale
        )));
    }
    if ltx_sampler::GuiderParams::alpha_gen(settings.cfg_scale).needs_uncond() && !has_negative {
        return Err(BurnError::MissingNegativeContext {
            cfg_scale: settings.cfg_scale,
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::approx_constant,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
mod tests {
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use burn::tensor::{Tensor, TensorData};

    use super::{latent_frames, resize_and_center_crop_5d, temporal_subsample, validate_settings};
    use crate::{BurnError, GenerationSettings};

    type B = NdArray;

    #[test]
    fn latent_frames_formula() {
        assert_eq!(latent_frames(3, 2).unwrap(), 2);
        assert_eq!(latent_frames(1, 8).unwrap(), 1);
        assert_eq!(latent_frames(9, 8).unwrap(), 2);
    }

    #[test]
    fn latent_frames_rejects_zero_factor() {
        assert!(latent_frames(3, 0).is_err());
    }

    #[test]
    fn bilinear_resize_ds2_matches_expected() {
        let device = NdArrayDevice::default();
        // All-ones: bilinear resize from 4×4 to 2×2 must give all 1.0.
        let ones: Tensor<B, 5> = Tensor::ones([1, 1, 1, 4, 4], &device);
        let down = resize_and_center_crop_5d::<B>(ones, 2, 2, &device).unwrap();
        let v: Vec<f32> = down.into_data().convert::<f32>().to_vec().unwrap();
        assert!(
            v.iter().all(|&x| (x - 1.0_f32).abs() < 1e-5),
            "all-ones failed: {v:?}"
        );

        // Constant-value tensor: bilinear must preserve the value.
        let fives: Tensor<B, 5> = Tensor::full([1, 2, 1, 6, 6], 5.0_f32, &device);
        let down2 = resize_and_center_crop_5d::<B>(fives, 3, 3, &device).unwrap();
        let v2: Vec<f32> = down2.into_data().convert::<f32>().to_vec().unwrap();
        assert!(
            v2.iter().all(|&x| (x - 5.0_f32).abs() < 1e-5),
            "constant-5 failed: {v2:?}"
        );
    }

    #[test]
    fn bilinear_resize_identity_no_op() {
        let device = NdArrayDevice::default();
        // src == dst: resize is a no-op (exact same grid).
        let data: Vec<f32> = (0..12_u32).map(|i| i as f32 * 0.1).collect();
        let x: Tensor<B, 5> =
            Tensor::from_data(TensorData::new(data.clone(), [1, 1, 1, 4, 3]), &device);
        let out = resize_and_center_crop_5d::<B>(x, 4, 3, &device).unwrap();
        let got: Vec<f32> = out.into_data().convert::<f32>().to_vec().unwrap();
        for (g, e) in got.iter().zip(data.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "identity mismatch: got {g} expected {e}"
            );
        }
    }

    #[test]
    fn temporal_subsample_keeps_correct_frames() {
        let device = NdArrayDevice::default();
        // 4 frames [0,1,2,3]; ts=2 → keep [0, 1, 3].
        let data: Vec<f32> = (0..4_u32).map(|i| i as f32).collect();
        let x: Tensor<B, 5> = Tensor::from_data(TensorData::new(data, [1, 1, 4, 1, 1]), &device);
        let sub = temporal_subsample::<B>(&x, 2).unwrap();
        let kept: Vec<f32> = sub.into_data().convert::<f32>().to_vec().unwrap();
        assert_eq!(kept, vec![0.0, 1.0, 3.0]);
    }
    #[test]
    fn latent_frames_floors_and_rejects_zero_factor() {
        // Zero temporal_factor is the only hard error.
        assert!(latent_frames(1, 0).is_err());
        // Non-multiples silently floor (matches encoder behaviour).
        assert_eq!(latent_frames(4, 8).unwrap(), 1);
    }

    #[test]
    fn validate_settings_rejects_nan_cfg_scale() {
        let s = GenerationSettings {
            cfg_scale: f32::NAN,
            ..GenerationSettings::default()
        };
        assert!(validate_settings(&s, true).is_err());
    }

    #[test]
    fn validate_settings_rejects_inf_cfg_scale() {
        let s = GenerationSettings {
            cfg_scale: f32::INFINITY,
            ..GenerationSettings::default()
        };
        assert!(validate_settings(&s, true).is_err());
    }

    #[test]
    fn validate_settings_rejects_missing_negative_for_cfg_ne_1() {
        let s = GenerationSettings {
            cfg_scale: 2.0,
            ..GenerationSettings::default()
        };
        assert!(matches!(
            validate_settings(&s, false),
            Err(BurnError::MissingNegativeContext { .. })
        ));
    }

    #[test]
    fn validate_settings_accepts_cfg_1_without_negative() {
        let s = GenerationSettings {
            cfg_scale: 1.0,
            ..GenerationSettings::default()
        };
        assert!(validate_settings(&s, false).is_ok());
    }

    #[test]
    fn validate_settings_accepts_cfg_ne_1_with_negative() {
        let s = GenerationSettings {
            cfg_scale: 2.0,
            ..GenerationSettings::default()
        };
        assert!(validate_settings(&s, true).is_ok());
    }

    #[test]
    fn bilinear_resize_non_square_with_crop() {
        let device = NdArrayDevice::default();
        // Source 1×8 (h=1, w=8), target 1×4: scale = max(1/1, 4/8) = 1.0 (h-dominant).
        // new_h = 1, new_w = 8; crop_left = (8-4)/2 = 2.
        // Output is the center 4 columns of the source.
        let data: Vec<f32> = (0..8_u32).map(|i| i as f32).collect(); // [0,1,2,3,4,5,6,7]
        let x: Tensor<B, 5> = Tensor::from_data(TensorData::new(data, [1, 1, 1, 1, 8]), &device);
        let out = resize_and_center_crop_5d::<B>(x, 1, 4, &device).unwrap();
        let got: Vec<f32> = out.into_data().convert::<f32>().to_vec().unwrap();
        // crop_left=2 → columns 2,3,4,5
        assert_eq!(
            got,
            vec![2.0, 3.0, 4.0, 5.0],
            "non-square crop failed: {got:?}"
        );
    }
    #[test]
    fn bilinear_resize_fractional_scale_with_crop() {
        use burn::backend::ndarray::NdArrayDevice;
        let device = NdArrayDevice::default();
        // Source 3×5 → target 2×2.
        // h-dominant: scale = 2/3; new_h=2, new_w=ceil(5*2/3)=ceil(10/3)=4; crop_left=1.
        // Expected values computed from Python reference
        // (F.interpolate + center crop, both torch and resize_and_center_crop match):
        //   [[2.625, 3.875], [10.125, 11.375]]
        let data: Vec<f32> = (0..15_u32).map(|i| i as f32).collect();
        let x: Tensor<B, 5> = Tensor::from_data(TensorData::new(data, [1, 1, 1, 3, 5]), &device);
        let out = resize_and_center_crop_5d::<B>(x, 2, 2, &device).unwrap();
        let got: Vec<f32> = out.into_data().convert::<f32>().to_vec().unwrap();
        let expected = vec![2.625_f32, 3.875, 10.125, 11.375];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-4,
                "fractional-scale bilinear mismatch: got {got:?} expected {expected:?}"
            );
        }
    }
}
