//! Video VAE encoder.
//!
//! Port of `VideoEncoder` from `ltx_core/model/video_vae/video_vae.py` and
//! `_make_encoder_block` from the same file at commit 9ec55f9.
//!
//! # Architecture
//! ```text
//! RGB input (B, 3, F, H, W) ∈ [-1, 1]
//!   → patchify(patch_size=4)          (B, 48, F, H/4, W/4)
//!   → conv_in (CausalConv3d)          (B, 128, F, H/4, W/4)
//!   → down_blocks [ordered]
//!   → norm_out (GroupNorm or PixelNorm)
//!   → SiLU
//!   → conv_out (CausalConv3d)         (B, 129, F', H', W')  [uniform log-var]
//!   → extract means
//!   → per_channel_statistics.normalize
//! ```

use burn::{
    module::Module,
    nn::GroupNormConfig,
    tensor::{Tensor, activation::silu, backend::Backend},
};

use crate::{
    attention::AttnBlock3D,
    config::VaeEncoderConfig,
    conv::CausalConv3d,
    error::VaeError,
    norm::{NormLayer, PerChannelStatistics, PixelNorm},
    patchify::patchify,
    resnet::{ResnetBlock3D, UNetMidBlock3D},
    sampling::SpaceToDepthDownsample,
};

// ── EncoderBlock ──────────────────────────────────────────────────────────────

/// One block in the encoder's `down_blocks` list.
///
/// The `ResX` and `ResXY` variants hold `Vec`-backed modules that are
/// considerably larger than the causal-conv variants.
#[expect(
    clippy::large_enum_variant,
    reason = "ResX/ResXY hold Vec<ResnetBlock3D>; boxing would break Burn's Module derive"
)]
#[derive(Module, Debug)]
pub enum EncoderBlock<B: Backend> {
    /// `"res_x"` – stack of same-channel residual blocks.
    ResX(UNetMidBlock3D<B>),
    /// `"res_x_y"` – single residual block that changes channel count.
    ResXY(ResnetBlock3D<B>),
    /// `"compress_time"` – causal conv with temporal stride 2.
    CompressTime(CausalConv3d<B>),
    /// `"compress_space"` – causal conv with spatial stride 2.
    CompressSpace(CausalConv3d<B>),
    /// `"compress_all"` – causal conv with stride 2 in all dimensions.
    CompressAll(CausalConv3d<B>),
    /// `"compress_all_x_y"` – causal conv + channel expand, stride 2 all.
    CompressAllXY(CausalConv3d<B>),
    /// `"compress_all_res"` – space-to-depth all dimensions.
    CompressAllRes(SpaceToDepthDownsample<B>),
    /// `"compress_space_res"` – space-to-depth spatial only.
    CompressSpaceRes(SpaceToDepthDownsample<B>),
    /// `"compress_time_res"` – space-to-depth temporal only.
    CompressTimeRes(SpaceToDepthDownsample<B>),
    /// `"attn"` – per-frame spatial self-attention.
    Attn(AttnBlock3D<B>),
}

impl<B: Backend> EncoderBlock<B> {
    /// Forward through this block. Input/output shape: `(B, C, F, H, W)`.
    ///
    /// # Errors
    /// Propagates errors from space-to-depth or attention operations.
    pub fn forward(&self, x: Tensor<B, 5>) -> Result<Tensor<B, 5>, VaeError> {
        match self {
            Self::ResX(b) => Ok(b.forward(x)),
            Self::ResXY(b) => Ok(b.forward(x)),
            // Causal convs with stride variants — all use causal=true.
            Self::CompressTime(b)
            | Self::CompressSpace(b)
            | Self::CompressAll(b)
            | Self::CompressAllXY(b) => Ok(b.forward(x, true)),
            // Space-to-depth variants.
            Self::CompressAllRes(b) | Self::CompressSpaceRes(b) | Self::CompressTimeRes(b) => {
                b.forward(x)
            }
            Self::Attn(b) => b.forward(x),
        }
    }
}

impl<B: Backend> EncoderBlock<B> {
    /// Load weights from a [`ltx_weights::Scope`].
    ///
    /// Dispatches to the block-specific loader.  Key paths match the
    /// reference `VideoEncoder.down_blocks[i].state_dict()` names.
    ///
    /// # Errors
    /// Returns [`VaeError::Load`] if any tensor is missing or wrong rank.
    pub(crate) fn load_weights_from_scope(
        &mut self,
        scope: &ltx_weights::Scope<'_>,
        device: &B::Device,
    ) -> Result<(), VaeError> {
        match self {
            Self::ResX(b) => b.load_weights_from_scope(scope, device),
            Self::ResXY(b) => b.load_weights_from_scope(scope, device),
            Self::CompressTime(b)
            | Self::CompressSpace(b)
            | Self::CompressAll(b)
            | Self::CompressAllXY(b) => b.load_weights_from_scope(scope, device),
            Self::CompressAllRes(b) | Self::CompressSpaceRes(b) | Self::CompressTimeRes(b) => {
                b.load_weights_from_scope(scope, device)
            }
            Self::Attn(b) => b.load_weights_from_scope(scope, device),
        }
    }
}

// ── VideoEncoder ──────────────────────────────────────────────────────────────

/// LTX-2.5 video VAE encoder.
///
/// Use [`VideoEncoder::new`] to build from a [`VaeEncoderConfig`], and
/// [`VideoEncoder::encode`] or [`VideoEncoder::tiled_encode`] to run it.
#[derive(Module, Debug)]
pub struct VideoEncoder<B: Backend> {
    per_channel_statistics: PerChannelStatistics<B>,
    conv_in: CausalConv3d<B>,
    down_blocks: Vec<EncoderBlock<B>>,
    conv_norm_out: NormLayer<B>,
    conv_out: CausalConv3d<B>,

    /// Spatial patch size (applied before `conv_in`).
    patch_size: usize,
    /// Number of latent channels (encoder output width before log-var).
    latent_channels: usize,
    /// Temporal downscale factor derived from `encoder_blocks`.
    temporal_factor: usize,
    /// Spatial downscale factor (height and width).
    spatial_factor: usize,

    /// Log-variance mode string.
    ///
    /// - `"uniform"` – `conv_out` has `C+1` channels; last is shared log-var.
    /// - `"per_channel"` – `conv_out` has `2C` channels.
    /// - `"constant"` – last channel is replaced by −30.
    /// - `"none"` – `conv_out` has `C` channels, no log-var.
    latent_log_var: String,
}

impl<B: Backend> VideoEncoder<B> {
    /// Build and initialise from a [`VaeEncoderConfig`].
    ///
    /// # Errors
    /// Returns a [`VaeError`] when a block type is unknown, a dimension
    /// arithmetic overflows, or a convolution config is invalid.
    pub fn new(cfg: &VaeEncoderConfig, device: &B::Device) -> Result<Self, VaeError> {
        if cfg.convolution_dimensions != 3 {
            return Err(VaeError::UnsupportedDims(cfg.convolution_dimensions));
        }
        if cfg.encoder_spatial_padding_mode != "zeros" {
            return Err(VaeError::UnsupportedPaddingMode(
                cfg.encoder_spatial_padding_mode.clone(),
            ));
        }

        let (temporal_factor, spatial_factor) =
            scale_factors_from_blocks(&cfg.encoder_blocks, cfg.patch_size);

        let per_channel_statistics = PerChannelStatistics::new(cfg.out_channels, device);

        let patched_in = cfg
            .in_channels
            .checked_mul(cfg.patch_size)
            .and_then(|v| v.checked_mul(cfg.patch_size))
            .ok_or(VaeError::DimOverflow)?;

        let norm_template = build_norm_template(&cfg.norm_layer, cfg.out_channels, device)?;

        let conv_in = CausalConv3d::new(
            patched_in,
            cfg.out_channels,
            3,
            [1, 1, 1],
            1,
            1,
            true,
            &cfg.encoder_spatial_padding_mode,
            device,
        )?;

        let mut feature_channels = cfg.out_channels;
        let mut down_blocks = Vec::with_capacity(cfg.encoder_blocks.len());

        for spec in &cfg.encoder_blocks {
            let (block, out_c) = make_encoder_block(
                &spec.name,
                spec.params.num_layers,
                spec.params.multiplier,
                feature_channels,
                &norm_template,
                &cfg.encoder_spatial_padding_mode,
                device,
            )?;
            down_blocks.push(block);
            feature_channels = out_c;
        }

        let conv_norm_out = build_norm_template(&cfg.norm_layer, feature_channels, device)?;

        let conv_out_ch = conv_out_channels(cfg.out_channels, &cfg.latent_log_var)?;
        let conv_out = CausalConv3d::new(
            feature_channels,
            conv_out_ch,
            3,
            [1, 1, 1],
            1,
            1,
            true,
            &cfg.encoder_spatial_padding_mode,
            device,
        )?;

        Ok(Self {
            per_channel_statistics,
            conv_in,
            down_blocks,
            conv_norm_out,
            conv_out,
            patch_size: cfg.patch_size,
            latent_channels: cfg.out_channels,
            temporal_factor,
            spatial_factor,
            latent_log_var: cfg.latent_log_var.clone(),
        })
    }

    /// Encode `video` into normalised latent means.
    ///
    /// Input:  `(B, C_in, F, H, W)` in `[-1, 1]`.
    /// Output: `(B, latent_channels, F', H', W')`.
    ///
    /// Frames are cropped to the nearest valid count if `(F - 1)` is not
    /// divisible by `temporal_factor`.
    ///
    /// # Errors
    /// Returns [`VaeError`] on shape or block errors.
    pub fn encode(&self, video: Tensor<B, 5>) -> Result<Tensor<B, 5>, VaeError> {
        let frames = video.dims()[2];
        let tf = self.temporal_factor;
        let video = crop_to_valid_frames(video, frames, tf);

        let x = patchify(video, self.patch_size)?;
        let mut x = self.conv_in.forward(x, true);

        for block in &self.down_blocks {
            x = block.forward(x)?;
        }

        let x = self.conv_norm_out.forward(x);
        let x = silu(x);
        let x = self.conv_out.forward(x, true);

        let means = self.extract_means(x)?;
        Ok(self.per_channel_statistics.normalize(means))
    }

    /// Encode using temporal tiles to bound memory use.
    ///
    /// When `tile_frames` is `None`, the entire clip is encoded in one pass.
    /// When provided, only temporal tiling is used: the clip is split into
    /// overlapping temporal tiles blended with a trapezoidal weight mask.
    /// `tile_frames` must satisfy `1 + k * temporal_factor`, and
    /// `tile_frames - tile_overlap` must be a multiple of `temporal_factor` so
    /// every tile starts on the latent temporal grid.
    /// Input:  `(B, C, F, H, W)`.
    /// Output: `(B, latent_channels, F', H', W')`.
    ///
    /// # Errors
    /// Returns [`VaeError::Config`] for zero, fully-overlapped, or temporally
    /// misaligned tile settings. Propagates errors from [`encode`](Self::encode).
    pub fn tiled_encode(
        &self,
        video: Tensor<B, 5>,
        tile_frames: Option<usize>,
        tile_overlap: usize,
    ) -> Result<Tensor<B, 5>, VaeError> {
        let Some(tile_frames) = tile_frames else {
            return self.encode(video);
        };

        validate_tile_args(tile_frames, tile_overlap, self.temporal_factor)?;

        let [nb, _nc, f_total, nh, nw] = video.dims();
        let tf = self.temporal_factor;
        let f_valid = valid_frame_count(f_total, tf);
        let video = if f_valid < f_total {
            video.narrow(2, 0, f_valid)
        } else {
            video
        };

        if tile_frames >= f_valid {
            return self.encode(video);
        }

        let output_shape = self.compute_output_shape(nb, f_valid, nh, nw, tf)?;
        let (f_out, h_out, w_out) = output_shape;
        let lc = self.latent_channels;
        let device = video.device();

        let mut latent_buf: Tensor<B, 5> = Tensor::zeros([nb, lc, f_out, h_out, w_out], &device);
        let mut weight_buf: Tensor<B, 5> = Tensor::zeros([nb, lc, f_out, h_out, w_out], &device);

        let step = tile_frames.saturating_sub(tile_overlap);
        let mut start = 0_usize;

        loop {
            let end = (start.saturating_add(tile_frames)).min(f_valid);
            let tile = video.clone().narrow(2, start, end.saturating_sub(start));
            let latent_tile = self.encode(tile)?;
            let lt_f = latent_tile.dims()[2];

            let l_start = if tf > 0 {
                start.checked_div(tf).ok_or(VaeError::DimOverflow)?
            } else {
                start
            };
            let l_end = l_start.saturating_add(lt_f).min(f_out);
            let l_len = l_end.saturating_sub(l_start);
            let ramp = tile_overlap
                .checked_div(tf.max(1))
                .ok_or(VaeError::DimOverflow)?
                .min(l_len.saturating_div(2));

            let mask = trapezoidal_mask_1d::<B>(l_len, ramp, ramp, &device);
            let mask5 = mask
                .reshape([1, 1, l_len, 1, 1])
                .expand([nb, lc, l_len, h_out, w_out]);

            let latent_slice = if lt_f > l_len {
                latent_tile.narrow(2, 0, l_len)
            } else {
                latent_tile
            };

            let add_l = latent_slice.mul(mask5.clone());
            let existing_l = latent_buf.clone().narrow(2, l_start, l_len);
            let existing_w = weight_buf.clone().narrow(2, l_start, l_len);
            latent_buf = splice_dim2(latent_buf, existing_l.add(add_l), l_start, l_end);
            weight_buf = splice_dim2(weight_buf, existing_w.add(mask5), l_start, l_end);

            if end >= f_valid {
                break;
            }
            start = start.saturating_add(step);
        }

        let weight_buf = weight_buf.clamp_min(1e-8_f32);
        Ok(latent_buf.div(weight_buf))
    }

    // ── private ──────────────────────────────────────────────────────────────

    fn extract_means(&self, x: Tensor<B, 5>) -> Result<Tensor<B, 5>, VaeError> {
        let nc = self.latent_channels;
        match self.latent_log_var.as_str() {
            "per_channel" | "uniform" | "constant" => Ok(x.narrow(1, 0, nc)),
            "none" => Ok(x),
            other => Err(VaeError::Config(format!(
                "unknown latent_log_var: {other:?}"
            ))),
        }
    }

    fn compute_output_shape(
        &self,
        _nb: usize,
        f: usize,
        nh: usize,
        nw: usize,
        tf: usize,
    ) -> Result<(usize, usize, usize), VaeError> {
        let f_out = if tf > 0 {
            f.saturating_sub(1)
                .checked_div(tf)
                .and_then(|v| v.checked_add(1))
                .ok_or(VaeError::DimOverflow)?
        } else {
            f
        };
        let sf = self.spatial_factor.max(1);
        let h_out = nh.checked_div(sf).ok_or(VaeError::DimOverflow)?;
        let w_out = nw.checked_div(sf).ok_or(VaeError::DimOverflow)?;
        Ok((f_out, h_out, w_out))
    }

    /// Temporal downscale factor.
    #[must_use]
    pub const fn temporal_factor(&self) -> usize {
        self.temporal_factor
    }

    /// Spatial downscale factor.
    #[must_use]
    pub const fn spatial_factor(&self) -> usize {
        self.spatial_factor
    }

    /// Number of latent channels.
    #[must_use]
    pub const fn latent_channels(&self) -> usize {
        self.latent_channels
    }
}

impl<B: Backend> VideoEncoder<B> {
    /// Build a `VideoEncoder` from checkpoint weights.
    ///
    /// # Key mapping
    ///
    /// For parity fixtures saved with `module.state_dict()` names, open the
    /// [`ltx_weights::WeightStore`] with [`ltx_weights::KeyMap::identity`] and
    /// pass the root scope.
    ///
    /// For real LTX-2.5 diffusion-VAE checkpoints, open with
    /// [`ltx_weights::KeyMap::video_encoder`]; the map strips `vae.encoder.` /
    /// `encoder.` prefixes and renames `vae.per_channel_statistics.*` →
    /// `per_channel_statistics.*`, after which call this function with the
    /// root scope.  This code path follows the reference `VAE_ENCODER_COMFY_KEYS_FILTER`
    /// `SDOps` exactly; it is documented but not integration-tested in this crate
    /// (no real Lightricks checkpoint is available locally).
    ///
    /// # Conv weight layout
    ///
    /// `CausalConv3d` and all conv blocks store weights as `PyTorch`
    /// `[out, in/groups, kT, kH, kW]` — Burn uses the same layout for 3-D
    /// convolutions, so no transposition is applied.
    ///
    /// # Errors
    /// Returns [`VaeError`] if construction fails, if any weight tensor is
    /// missing or has the wrong rank, or if a loaded shape does not match the
    /// config-derived expectation.
    pub fn load(
        scope: &ltx_weights::Scope<'_>,
        config: &VaeEncoderConfig,
        device: &B::Device,
    ) -> Result<Self, VaeError> {
        let mut encoder = Self::new(config, device)?;

        // ── per-channel normalisation statistics ──────────────────────────────
        let stats_scope = scope.scope("per_channel_statistics");
        let stats = PerChannelStatistics::load_from_scope(&stats_scope, device)?;
        // Validate that the checkpoint's latent channel count matches the config.
        let got_ch = stats.std_of_means.dims()[0];
        if got_ch != config.out_channels {
            return Err(VaeError::Config(format!(
                "`per_channel_statistics.std-of-means`: expected [{0}], got [{1}]",
                config.out_channels, got_ch
            )));
        }
        encoder.per_channel_statistics = stats;

        // ── conv_in ───────────────────────────────────────────────────────────
        encoder
            .conv_in
            .load_weights_from_scope(&scope.scope("conv_in"), device)?;

        // ── encoder blocks ────────────────────────────────────────────────────
        for (i, block) in encoder.down_blocks.iter_mut().enumerate() {
            block.load_weights_from_scope(&scope.scope(&format!("down_blocks.{i}")), device)?;
        }

        // ── conv_norm_out (PixelNorm: no params; GroupNorm: weight + bias) ────
        encoder
            .conv_norm_out
            .load_weights_from_scope(&scope.scope("conv_norm_out"), device)?;

        // ── conv_out ──────────────────────────────────────────────────────────
        encoder
            .conv_out
            .load_weights_from_scope(&scope.scope("conv_out"), device)?;

        Ok(encoder)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Crop `video` so that `(F - 1)` is divisible by `temporal_factor`.
fn crop_to_valid_frames<B: Backend>(
    video: Tensor<B, 5>,
    frames: usize,
    temporal_factor: usize,
) -> Tensor<B, 5> {
    if temporal_factor == 0 {
        return video;
    }
    let excess = frames
        .saturating_sub(1)
        .checked_rem(temporal_factor)
        .unwrap_or(0);
    if excess == 0 {
        video
    } else {
        video.narrow(2, 0, frames.saturating_sub(excess))
    }
}

/// Largest valid frame count ≤ `frames` satisfying `1 + k * temporal_factor`.
fn valid_frame_count(frames: usize, temporal_factor: usize) -> usize {
    if temporal_factor == 0 {
        return frames;
    }
    let excess = frames
        .saturating_sub(1)
        .checked_rem(temporal_factor)
        .unwrap_or(0);
    frames.saturating_sub(excess)
}

/// Compute temporal and spatial scale factors from the encoder block list.
fn scale_factors_from_blocks(
    blocks: &[crate::config::EncoderBlockSpec],
    patch_size: usize,
) -> (usize, usize) {
    let mut temporal: usize = 1;
    let mut spatial: usize = patch_size;

    for spec in blocks {
        match spec.name.as_str() {
            "compress_time" | "compress_time_res" => {
                temporal = temporal.saturating_mul(2);
            }
            "compress_space" | "compress_space_res" => {
                spatial = spatial.saturating_mul(2);
            }
            "compress_all" | "compress_all_x_y" | "compress_all_res" => {
                temporal = temporal.saturating_mul(2);
                spatial = spatial.saturating_mul(2);
            }
            _ => {}
        }
    }

    (temporal, spatial)
}

/// Number of output channels for `conv_out` based on `latent_log_var`.
fn conv_out_channels(latent_channels: usize, latent_log_var: &str) -> Result<usize, VaeError> {
    match latent_log_var {
        "per_channel" => latent_channels.checked_mul(2).ok_or(VaeError::DimOverflow),
        "uniform" | "constant" => latent_channels.checked_add(1).ok_or(VaeError::DimOverflow),
        "none" => Ok(latent_channels),
        other => Err(VaeError::Config(format!(
            "unknown latent_log_var: {other:?}"
        ))),
    }
}

/// Build a `NormLayer` for the given config string.
fn build_norm_template<B: Backend>(
    norm_layer: &str,
    channels: usize,
    device: &B::Device,
) -> Result<NormLayer<B>, VaeError> {
    match norm_layer {
        "pixel_norm" => Ok(NormLayer::Pixel(PixelNorm::new())),
        "group_norm" => {
            let gn = GroupNormConfig::new(32, channels)
                .with_epsilon(1e-6)
                .init(device);
            Ok(NormLayer::Group(gn))
        }
        other => Err(VaeError::Config(format!("unknown norm_layer: {other:?}"))),
    }
}

/// Build a single encoder block from its name and parameters.
#[expect(
    clippy::too_many_lines,
    reason = "the reference VAE block switch is kept in one place so block names map directly to Python"
)]
fn make_encoder_block<B: Backend>(
    name: &str,
    num_layers: usize,
    multiplier: usize,
    in_channels: usize,
    norm_template: &NormLayer<B>,
    spatial_padding_mode: &str,
    device: &B::Device,
) -> Result<(EncoderBlock<B>, usize), VaeError> {
    const EPS: f64 = 1e-6;

    match name {
        "res_x" => {
            let block = UNetMidBlock3D::new(
                in_channels,
                num_layers,
                norm_template,
                spatial_padding_mode,
                EPS,
                device,
            )?;
            Ok((EncoderBlock::ResX(block), in_channels))
        }
        "res_x_y" => {
            let out_ch = in_channels
                .checked_mul(multiplier)
                .ok_or(VaeError::DimOverflow)?;
            let block = ResnetBlock3D::new(
                in_channels,
                out_ch,
                norm_template,
                spatial_padding_mode,
                EPS,
                device,
            )?;
            Ok((EncoderBlock::ResXY(block), out_ch))
        }
        "compress_time" => {
            let block = CausalConv3d::new(
                in_channels,
                in_channels,
                3,
                [2, 1, 1],
                1,
                1,
                true,
                spatial_padding_mode,
                device,
            )?;
            Ok((EncoderBlock::CompressTime(block), in_channels))
        }
        "compress_space" => {
            let block = CausalConv3d::new(
                in_channels,
                in_channels,
                3,
                [1, 2, 2],
                1,
                1,
                true,
                spatial_padding_mode,
                device,
            )?;
            Ok((EncoderBlock::CompressSpace(block), in_channels))
        }
        "compress_all" => {
            let block = CausalConv3d::new(
                in_channels,
                in_channels,
                3,
                [2, 2, 2],
                1,
                1,
                true,
                spatial_padding_mode,
                device,
            )?;
            Ok((EncoderBlock::CompressAll(block), in_channels))
        }
        "compress_all_x_y" => {
            let out_ch = in_channels
                .checked_mul(multiplier)
                .ok_or(VaeError::DimOverflow)?;
            let block = CausalConv3d::new(
                in_channels,
                out_ch,
                3,
                [2, 2, 2],
                1,
                1,
                true,
                spatial_padding_mode,
                device,
            )?;
            Ok((EncoderBlock::CompressAllXY(block), out_ch))
        }
        "compress_all_res" => {
            let out_ch = in_channels
                .checked_mul(multiplier)
                .ok_or(VaeError::DimOverflow)?;
            let block = SpaceToDepthDownsample::new(
                in_channels,
                out_ch,
                [2, 2, 2],
                spatial_padding_mode,
                device,
            )?;
            Ok((EncoderBlock::CompressAllRes(block), out_ch))
        }
        "compress_space_res" => {
            let out_ch = in_channels
                .checked_mul(multiplier)
                .ok_or(VaeError::DimOverflow)?;
            let block = SpaceToDepthDownsample::new(
                in_channels,
                out_ch,
                [1, 2, 2],
                spatial_padding_mode,
                device,
            )?;
            Ok((EncoderBlock::CompressSpaceRes(block), out_ch))
        }
        "compress_time_res" => {
            let out_ch = in_channels
                .checked_mul(multiplier)
                .ok_or(VaeError::DimOverflow)?;
            let block = SpaceToDepthDownsample::new(
                in_channels,
                out_ch,
                [2, 1, 1],
                spatial_padding_mode,
                device,
            )?;
            Ok((EncoderBlock::CompressTimeRes(block), out_ch))
        }
        "attn" => {
            let block = AttnBlock3D::new(in_channels, device)?;
            Ok((EncoderBlock::Attn(block), in_channels))
        }
        other => Err(VaeError::UnknownBlockType(other.to_owned())),
    }
}

// ── tiling helpers ────────────────────────────────────────────────────────────

/// Ramp-and-plateau 1-D trapezoidal weight mask of length `n`.
///
/// Elements ramp from 0 to 1 over `ramp_left` entries, stay at 1, then ramp
/// from 1 to 0 over `ramp_right` entries.
fn trapezoidal_mask_1d<B: Backend>(
    n: usize,
    ramp_left: usize,
    ramp_right: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    if n == 0 {
        return Tensor::zeros([0], device);
    }
    let mut data: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let v = ramp_value(i, n, ramp_left, ramp_right);
        data.push(v);
    }
    Tensor::from_floats(data.as_slice(), device)
}

/// Compute one weight value for the trapezoidal mask.
///
/// All values in `[0.0, 1.0]`.
///
/// Uses a direct integer-to-float cast. Tile lengths are far below the range
/// where `usize` cannot be exactly represented as `f32` in normal use, and the
/// mask is only a blend weight.
fn ramp_value(i: usize, n: usize, ramp_left: usize, ramp_right: usize) -> f32 {
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "tile-mask indices are small and only become blend weights"
    )]
    #[inline]
    const fn to_f32(v: usize) -> f32 {
        v as f32
    }
    if ramp_left > 0 && i < ramp_left {
        to_f32(i.saturating_add(1)) / to_f32(ramp_left.saturating_add(1))
    } else if ramp_right > 0 && i >= n.saturating_sub(ramp_right) {
        let from_end = n.saturating_sub(1).saturating_sub(i);
        to_f32(from_end.saturating_add(1)) / to_f32(ramp_right.saturating_add(1))
    } else {
        1.0_f32
    }
}

fn validate_tile_args(
    tile_frames: usize,
    tile_overlap: usize,
    temporal_factor: usize,
) -> Result<(), VaeError> {
    if tile_frames == 0 {
        return Err(VaeError::Config(
            "tile_frames must be greater than zero".into(),
        ));
    }
    if tile_overlap >= tile_frames {
        return Err(VaeError::Config(
            "tile_overlap must be smaller than tile_frames".into(),
        ));
    }
    if temporal_factor > 0 && valid_frame_count(tile_frames, temporal_factor) != tile_frames {
        return Err(VaeError::Config(format!(
            "tile_frames {tile_frames} must be 1 + k*{temporal_factor}"
        )));
    }
    let step = tile_frames.saturating_sub(tile_overlap);
    if temporal_factor > 0 && !step.is_multiple_of(temporal_factor) {
        return Err(VaeError::Config(format!(
            "tile_frames - tile_overlap ({step}) must be a multiple of temporal_factor {temporal_factor}"
        )));
    }
    Ok(())
}

/// Replace `buf` along dim 2 from `l_start` to `l_end` (exclusive) with `patch`.
fn splice_dim2<B: Backend>(
    buf: Tensor<B, 5>,
    patch: Tensor<B, 5>,
    l_start: usize,
    l_end: usize,
) -> Tensor<B, 5> {
    let f_total = buf.dims()[2];
    let mut parts: Vec<Tensor<B, 5>> = Vec::new();
    if l_start > 0 {
        parts.push(buf.clone().narrow(2, 0, l_start));
    }
    parts.push(patch);
    if l_end < f_total {
        let len = f_total.saturating_sub(l_end);
        parts.push(buf.narrow(2, l_end, len));
    }
    Tensor::cat(parts, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_args_reject_degenerate_values() {
        assert!(validate_tile_args(0, 0, 8).is_err());
        assert!(validate_tile_args(17, 17, 8).is_err());
        assert!(validate_tile_args(17, 18, 8).is_err());
    }

    #[test]
    fn tile_args_require_grid_aligned_step() {
        assert!(validate_tile_args(17, 8, 8).is_err());
        assert!(validate_tile_args(17, 9, 8).is_ok());
        assert!(validate_tile_args(25, 9, 8).is_ok());
    }

    #[test]
    fn ramp_value_does_not_saturate_large_lengths() {
        let value = ramp_value(69_999, 70_000, 70_000, 0);
        assert!(value > 0.99_f32, "unexpected ramp value {value}");
    }
}
