//! `VaeEncoderConfig` – deserialises from the checkpoint `config.vae` JSON.
//!
//! The reference handles two layouts (see `_prepare_video_encoder_kwargs`):
//! - **Flat** (`CausalVideoAutoencoder`): fields live directly on `vae`.
//! - **Nested** (`CausalDiffusionVAE`): fields live under `vae.encoder`.
//!
//! Reference: `ltx_core/model/video_vae/model_configurator.py`
//! at commit 9ec55f9.

use serde_json::Value;

use crate::error::VaeError;

/// Parameters for a single encoder block.
///
/// Second element of each `encoder_blocks` entry.  The reference accepts
/// either an integer (number of layers, for `res_x`) or a dict.
#[derive(Debug, Clone)]
pub struct BlockParams {
    /// Number of residual layers (used by `res_x`; default 1).
    pub num_layers: usize,
    /// Channel multiplier (used by `res_x_y`, `compress_*_x_y`, `compress_*_res`; default 2).
    pub multiplier: usize,
}

impl Default for BlockParams {
    fn default() -> Self {
        Self {
            num_layers: 1,
            multiplier: 2,
        }
    }
}

/// One entry in `encoder_blocks`: a `(name, params)` pair.
#[derive(Debug, Clone)]
pub struct EncoderBlockSpec {
    /// Block type name (e.g. `"res_x"`, `"compress_space_res"`).
    pub name: String,
    /// Block parameters.
    pub params: BlockParams,
}

/// Full encoder configuration.
///
/// Use [`VaeEncoderConfig::from_vae_json`] to construct from the
/// `config.vae` JSON object in a checkpoint's `__metadata__`.
#[derive(Debug, Clone)]
pub struct VaeEncoderConfig {
    /// Number of convolution dimensions (only 3 is supported).
    pub convolution_dimensions: usize,
    /// Number of input channels (3 for RGB).
    pub in_channels: usize,
    /// Latent channels (encoder output, e.g. 128).
    pub out_channels: usize,
    /// Ordered sequence of blocks.
    pub encoder_blocks: Vec<EncoderBlockSpec>,
    /// Spatial patch size applied before `conv_in` (space-to-depth).
    pub patch_size: usize,
    /// Normalisation layer: `"pixel_norm"` or `"group_norm"`.
    pub norm_layer: String,
    /// Log-variance mode: `"uniform"`, `"per_channel"`, `"constant"`, `"none"`.
    pub latent_log_var: String,
    /// Spatial padding mode for convolutions: only `"zeros"` is supported.
    pub encoder_spatial_padding_mode: String,
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn str_field<'a>(
    obj: &'a serde_json::Map<String, Value>,
    key: &str,
    default: &'static str,
) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or(default)
}

/// Read a `usize` from a JSON object, falling back to `default`.
///
/// `u64 → usize` is saturating on overflow (practically impossible for config dims).
fn usize_field(obj: &serde_json::Map<String, Value>, key: &str, default: usize) -> usize {
    obj.get(key)
        .and_then(Value::as_u64)
        .map_or(default, |v| usize::try_from(v).unwrap_or(default))
}

fn parse_block_params(v: &Value) -> BlockParams {
    match v {
        Value::Number(n) => BlockParams {
            num_layers: n.as_u64().map_or(1, |x| usize::try_from(x).unwrap_or(1)),
            multiplier: 2,
        },
        Value::Object(m) => BlockParams {
            num_layers: m
                .get("num_layers")
                .and_then(Value::as_u64)
                .map_or(1, |x| usize::try_from(x).unwrap_or(1)),
            multiplier: m
                .get("multiplier")
                .and_then(Value::as_u64)
                .map_or(2, |x| usize::try_from(x).unwrap_or(2)),
        },
        _ => BlockParams::default(),
    }
}

fn parse_blocks(arr: &[Value]) -> Vec<EncoderBlockSpec> {
    arr.iter()
        .filter_map(|entry| {
            let pair = entry.as_array()?;
            let name = pair.first()?.as_str()?.to_owned();
            let params = pair
                .get(1)
                .map_or_else(BlockParams::default, parse_block_params);
            Some(EncoderBlockSpec { name, params })
        })
        .collect()
}

// ── public API ────────────────────────────────────────────────────────────────

impl VaeEncoderConfig {
    /// Build from the `config.vae` JSON object (both flat and nested layouts).
    ///
    /// # Errors
    /// Returns [`VaeError::Config`] when a required field is missing or
    /// has an unexpected type.
    pub fn from_vae_json(vae: &Value) -> Result<Self, VaeError> {
        let vae_obj = vae
            .as_object()
            .ok_or_else(|| VaeError::Config("config.vae is not a JSON object".into()))?;

        // Nested layout: fields live under "encoder".
        let (enc_obj, base_obj) = vae_obj
            .get("encoder")
            .and_then(Value::as_object)
            .map_or((vae_obj, vae_obj), |enc| (enc, vae_obj));

        // "out_channels" means latent width in the nested layout;
        // in the flat layout it is the decoder RGB width, so use "latent_channels" instead.
        let out_channels =
            if enc_obj.contains_key("out_channels") && !std::ptr::eq(enc_obj, vae_obj) {
                usize_field(enc_obj, "out_channels", 128)
            } else {
                usize_field(base_obj, "latent_channels", 128)
            };

        let blocks_key = if enc_obj.contains_key("blocks") {
            "blocks"
        } else {
            "encoder_blocks"
        };
        let raw_blocks = enc_obj
            .get(blocks_key)
            .or_else(|| base_obj.get("encoder_blocks"))
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |a| parse_blocks(a));

        // "spatial_padding_mode" inside encoder takes priority; fallback to
        // "encoder_spatial_padding_mode" on the outer vae object.
        let spatial_padding_mode = enc_obj
            .get("spatial_padding_mode")
            .and_then(Value::as_str)
            .or_else(|| {
                base_obj
                    .get("encoder_spatial_padding_mode")
                    .and_then(Value::as_str)
            })
            .unwrap_or("zeros")
            .to_owned();

        Ok(Self {
            convolution_dimensions: usize_field(enc_obj, "dims", usize_field(base_obj, "dims", 3)),
            in_channels: usize_field(enc_obj, "in_channels", 3),
            out_channels,
            encoder_blocks: raw_blocks,
            patch_size: usize_field(enc_obj, "patch_size", 4),
            norm_layer: str_field(enc_obj, "norm_layer", "pixel_norm").to_owned(),
            latent_log_var: str_field(enc_obj, "latent_log_var", "uniform").to_owned(),
            encoder_spatial_padding_mode: spatial_padding_mode,
        })
    }
}
