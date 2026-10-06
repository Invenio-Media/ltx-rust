//! Decoder configuration loaded from checkpoint `__metadata__["config"]["vae"]`.

use crate::error::VaeDecoderError;

/// Temporal/spatial upsampling step between adjacent deterministic stages.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct UpsampleSpec {
    /// Pixel-shuffle stride `[t, h, w]`.
    pub stride: [usize; 3],
    /// Channel reduction factor applied by the pixel shuffle.
    pub out_channels_reduction_factor: usize,
}

/// Whether the model predicts velocity (`"v"`) or x₀ (`"x0"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelOutputType {
    /// Velocity prediction (default).
    #[default]
    V,
    /// Direct x₀ prediction.
    X0,
}

/// Architecture config for the diffusion video VAE decoder.
///
/// Mirrors the fields that `_build_diffusion_video_decoder` reads, with sane
/// defaults so incomplete JSON (e.g. the test fixture) still deserialises.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecoderConfig {
    /// Latent channel count (encoder output width).
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,

    /// Pixel channel count (`RGB = 3`).
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,

    /// Spatial patch size applied before and after stage 5.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Attention head dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    /// Timestep embedding dimension fed to `AdaLNZero`.
    #[serde(default = "default_t_emb_dim")]
    pub t_emb_dim: usize,

    /// Number of reverse-diffusion Euler steps at inference time.
    #[serde(default = "default_num_inference_steps")]
    pub default_num_inference_steps: usize,

    /// Multiplier applied to the timestep before embedding.
    #[serde(default = "default_timestep_scale")]
    pub timestep_scale_multiplier: f32,

    /// Velocity or x₀ prediction mode.
    #[serde(default)]
    pub model_output_type: ModelOutputType,

    /// Channel widths for each of the 5 stages.
    #[serde(default = "default_stage_channels")]
    pub stage_channels: Vec<usize>,

    /// Block depths per stage.
    #[serde(default = "default_stage_depths")]
    pub stage_depths: Vec<usize>,

    /// 3-D NA kernel `[kt, kh, kw]` per stage (5 entries).
    #[serde(default = "default_stage_kernels")]
    pub stage_kernels: Vec<[usize; 3]>,

    /// Upsample specs connecting adjacent stages (4 entries).
    #[serde(default = "default_upsamples")]
    pub upsamples: Vec<UpsampleSpec>,

    /// NA kernel for the diffusion (stage 5) blocks.
    #[serde(default = "default_stage5_kernel")]
    pub stage5_kernel: [usize; 3],

    /// Stage-5 feature width; `None` ⇒ `stage_channels.last()`.
    #[serde(default)]
    pub stage5_channels: Option<usize>,

    /// Optional `RoPE` dim split `[d_t, d_h, d_w]`; `None` ⇒ computed from `head_dim`.
    #[serde(default)]
    pub rope_dim_split: Option<[usize; 3]>,
}

impl DecoderConfig {
    /// Build from the `vae` sub-object of the checkpoint `config` key.
    ///
    /// # Errors
    ///
    /// Returns [`VaeDecoderError::Json`] on parse failure or
    /// [`VaeDecoderError::InvalidConfig`] on validation failure.
    pub fn from_vae_json(vae: &serde_json::Value) -> Result<Self, VaeDecoderError> {
        let src = vae.get("decoder").unwrap_or(vae);
        let mut cfg: Self = serde_json::from_value(src.clone())?;
        if cfg.in_channels == default_in_channels()
            && let Some(lc) = vae
                .get("latent_channels")
                .and_then(serde_json::Value::as_u64)
        {
            cfg.in_channels = usize::try_from(lc).unwrap_or(cfg.in_channels);
        }
        if let Some(type_str) = vae
            .get("model_output_type")
            .and_then(serde_json::Value::as_str)
        {
            cfg.model_output_type = if type_str == "x0" {
                ModelOutputType::X0
            } else {
                ModelOutputType::V
            };
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Resolved stage-5 channel width.
    #[must_use]
    pub fn stage5_channels_resolved(&self) -> usize {
        self.stage5_channels
            .or_else(|| self.stage_channels.last().copied())
            .unwrap_or(128)
    }

    /// Validate structural invariants.
    ///
    /// # Errors
    ///
    /// Returns [`VaeDecoderError::InvalidConfig`] for malformed configs.
    pub fn validate(&self) -> Result<(), VaeDecoderError> {
        let n = self.stage_channels.len();
        if n < 2 {
            return Err(VaeDecoderError::InvalidConfig {
                detail: format!("stage_channels must have >= 2 entries, got {n}"),
            });
        }
        if self.stage_depths.len() != n {
            return Err(VaeDecoderError::InvalidConfig {
                detail: format!(
                    "stage_depths length {} != stage_channels length {n}",
                    self.stage_depths.len()
                ),
            });
        }
        if self.stage_kernels.len() != n {
            return Err(VaeDecoderError::InvalidConfig {
                detail: format!(
                    "stage_kernels length {} != stage_channels length {n}",
                    self.stage_kernels.len()
                ),
            });
        }
        let n_up = n.saturating_sub(1);
        if self.upsamples.len() != n_up {
            return Err(VaeDecoderError::InvalidConfig {
                detail: format!(
                    "upsamples length {} != stage_channels length - 1 = {n_up}",
                    self.upsamples.len()
                ),
            });
        }
        if self.patch_size < 1 {
            return Err(VaeDecoderError::InvalidConfig {
                detail: "patch_size must be >= 1".to_owned(),
            });
        }
        if self.head_dim < 1 {
            return Err(VaeDecoderError::InvalidConfig {
                detail: "head_dim must be >= 1".to_owned(),
            });
        }
        Ok(())
    }

    /// Resolved `RoPE` dim split, either from config or computed from `head_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`VaeDecoderError::InvalidConfig`] when the split is invalid.
    pub fn rope_dim_split_resolved(&self) -> Result<[usize; 3], VaeDecoderError> {
        if let Some(split) = self.rope_dim_split {
            let sum = split[0].saturating_add(split[1]).saturating_add(split[2]);
            if sum != self.head_dim {
                return Err(VaeDecoderError::InvalidConfig {
                    detail: format!("rope_dim_split sum {sum} != head_dim {}", self.head_dim),
                });
            }
            return Ok(split);
        }
        default_rope_dim_split(self.head_dim)
    }
}

fn default_rope_dim_split(head_dim: usize) -> Result<[usize; 3], VaeDecoderError> {
    let rem = head_dim.checked_rem(8).unwrap_or(1);
    if rem != 0 {
        return Err(VaeDecoderError::InvalidConfig {
            detail: format!("head_dim {head_dim} must be a multiple of 8 for default RoPE split"),
        });
    }
    let d_t = head_dim
        .checked_div(4)
        .and_then(|x| x.checked_div(2))
        .map_or(0, |x| x.saturating_mul(2));
    let d_hw = head_dim.saturating_sub(d_t).checked_div(2).unwrap_or(0);
    let (d_t_final, d_hw_final) = if d_hw.checked_rem(2).unwrap_or(0) != 0 {
        let d_t2 = d_t.saturating_sub(2);
        let d_hw2 = head_dim.saturating_sub(d_t2).checked_div(2).unwrap_or(0);
        (d_t2, d_hw2)
    } else {
        (d_t, d_hw)
    };
    if d_t_final == 0 || d_hw_final == 0 {
        return Err(VaeDecoderError::InvalidConfig {
            detail: format!("head_dim {head_dim} too small for default RoPE split"),
        });
    }
    Ok([d_t_final, d_hw_final, d_hw_final])
}

// ─── Defaults ────────────────────────────────────────────────────────────────

const fn default_in_channels() -> usize {
    128
}
const fn default_out_channels() -> usize {
    3
}
const fn default_patch_size() -> usize {
    4
}
const fn default_head_dim() -> usize {
    64
}
const fn default_t_emb_dim() -> usize {
    384
}
const fn default_num_inference_steps() -> usize {
    2
}
const fn default_timestep_scale() -> f32 {
    1.0
}

fn default_stage_channels() -> Vec<usize> {
    vec![1024, 512, 256, 256, 128]
}
fn default_stage_depths() -> Vec<usize> {
    vec![4, 6, 4, 2, 8]
}
fn default_stage_kernels() -> Vec<[usize; 3]> {
    vec![[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [3, 3, 3]]
}
fn default_upsamples() -> Vec<UpsampleSpec> {
    vec![
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
const fn default_stage5_kernel() -> [usize; 3] {
    [3, 7, 7]
}
