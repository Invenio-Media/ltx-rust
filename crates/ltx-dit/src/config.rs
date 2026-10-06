use serde::{Deserialize, Serialize};

use crate::error::DitError;

/// Which `RoPE` variant to use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeType {
    /// Each head's feature vector is split into two halves; pairs rotate together.
    #[default]
    Split,
    /// Features are interleaved in pairs (legacy; prefer [`RopeType::Split`]).
    Interleaved,
}

/// Feature flags collected from the transformer config booleans.
///
/// Groups all bool fields from `DiTConfig` to satisfy the `struct_excessive_bools`
/// lint constraint.  The struct intentionally carries more than three booleans
/// because it faithfully mirrors the Python config schema.
#[expect(
    clippy::struct_excessive_bools,
    reason = "faithful mirror of the Python transformer config; \
              each flag controls an independent architectural choice"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DiTFlags {
    /// Enable per-head sigmoid gating on attention output.
    pub apply_gated_attention: bool,
    /// Use `AdaLN` for the cross-attention query/key/value projections.
    pub cross_attention_adaln: bool,
    /// Include a second `AdaLN` MLP for prompt conditioning when
    /// `cross_attention_adaln` is enabled; ignored otherwise.
    pub use_prompt_adaln_single: bool,
    /// Whether the feed-forward layers have a bias term.
    pub ff_bias: bool,
    /// Attach a learned positional embedding to single-pixel-frame latent tokens.
    pub use_keyframes_abs_pos_embedding: bool,
    /// The caption projection runs in the text encoder (`true`) or inside this
    /// transformer (`false`).  LTX-2.5 22B sets `true`; no `caption_projection`
    /// module is present in the checkpoint.
    #[serde(default = "bool_true")]
    pub caption_proj_before_connector: bool,
}

/// Configuration for the LTX-2.5 video `DiT` transformer.
///
/// Fields mirror `config["transformer"]` in the safetensors checkpoint metadata,
/// as read by `LTXVideoOnlyModelConfigurator.from_metadata` in the reference.
///
/// ## LTX-2.5 22B defaults
///
/// `num_layers = 28`, `num_attention_heads = 32`, `attention_head_dim = 128`
/// (inner\_dim = 4096), `cross_attention_dim = 4096`, `ff_bias = false`,
/// `apply_gated_attention = false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiTConfig {
    /// Number of attention heads.
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    /// Per-head feature dimension.
    #[serde(default = "default_head_dim")]
    pub attention_head_dim: usize,
    /// VAE latent channel count (input to patch embedding).
    #[serde(default = "default_channels")]
    pub in_channels: usize,
    /// Denoiser output channels (same as latent channels).
    #[serde(default = "default_channels")]
    pub out_channels: usize,
    /// Number of transformer blocks.
    #[serde(default = "default_layers")]
    pub num_layers: usize,
    /// Text context dimension (output of the text encoder / connector).
    #[serde(default = "default_cross_attn_dim")]
    pub cross_attention_dim: usize,
    /// Epsilon for RMS norms and the output layer norm.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Base period for `RoPE` sinusoidal frequencies.
    #[serde(default = "default_rope_theta")]
    pub positional_embedding_theta: f64,
    /// Maximum position indices `[t_max, h_max, w_max]`.
    #[serde(default = "default_max_pos")]
    pub positional_embedding_max_pos: [usize; 3],
    /// Timestep scaling factor applied before the sinusoidal embedding.
    #[serde(default = "default_ts_scale")]
    pub timestep_scale_multiplier: u32,
    /// Use patch midpoint for `RoPE` instead of start index.
    #[serde(default = "bool_true")]
    pub use_middle_indices_grid: bool,
    /// `RoPE` variant.
    #[serde(default)]
    pub rope_type: RopeType,
    /// Frequency computation precision from the checkpoint config.
    ///
    /// The reference reads `config.get("frequencies_precision", False) == "float64"` to
    /// choose `numpy` `f64` `RoPE` freq computation. The Rust `DiT` always uses `f32`. Reject
    /// `"float64"` loudly so the user knows about the divergence.
    #[serde(default)]
    pub frequencies_precision: Option<String>,
    /// Feature flag set.
    #[serde(flatten)]
    pub flags: DiTFlags,
}

const fn default_heads() -> usize {
    32
}
const fn default_head_dim() -> usize {
    128
}
const fn default_channels() -> usize {
    128
}
const fn default_layers() -> usize {
    28
}
const fn default_cross_attn_dim() -> usize {
    4096
}
const fn default_norm_eps() -> f32 {
    1e-6_f32
}
const fn default_rope_theta() -> f64 {
    10_000.0
}
const fn default_max_pos() -> [usize; 3] {
    [20, 2048, 2048]
}
const fn default_ts_scale() -> u32 {
    1000
}
const fn bool_true() -> bool {
    true
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            num_attention_heads: default_heads(),
            attention_head_dim: default_head_dim(),
            in_channels: default_channels(),
            out_channels: default_channels(),
            num_layers: default_layers(),
            cross_attention_dim: default_cross_attn_dim(),
            norm_eps: default_norm_eps(),
            positional_embedding_theta: default_rope_theta(),
            positional_embedding_max_pos: default_max_pos(),
            timestep_scale_multiplier: default_ts_scale(),
            use_middle_indices_grid: true,
            rope_type: RopeType::Split,
            frequencies_precision: None,
            flags: DiTFlags {
                ff_bias: false,
                use_prompt_adaln_single: true,
                ..DiTFlags::default()
            },
        }
    }
}

impl DiTConfig {
    /// Inner (hidden) dimension: `num_attention_heads × attention_head_dim`.
    #[must_use]
    pub const fn inner_dim(&self) -> usize {
        self.num_attention_heads
            .saturating_mul(self.attention_head_dim)
    }

    /// Number of `AdaLN` modulation scalars per block (6 base + 3 if `cross_attention_adaln`).
    #[must_use]
    pub const fn adaln_coeff(&self) -> usize {
        if self.flags.cross_attention_adaln {
            9
        } else {
            6
        }
    }

    /// Parse from `config["transformer"]` JSON sub-object.
    ///
    /// # Errors
    ///
    /// Returns [`DitError::Json`] when the JSON is malformed.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, DitError> {
        let config: Self = serde_json::from_value(value.clone())?;
        config.validate()?;
        Ok(config)
    }

    /// Validate architectural constraints that would otherwise panic in tensor reshapes.
    ///
    /// # Errors
    ///
    /// Returns [`DitError::Config`] when unsupported flags or incompatible
    /// dimensions are present.
    pub fn validate(&self) -> Result<(), DitError> {
        if self.num_attention_heads == 0 {
            return Err(DitError::Config("num_attention_heads must be > 0".into()));
        }
        if !self.attention_head_dim.is_multiple_of(2) {
            return Err(DitError::Config(
                "attention_head_dim must be even for RoPE".into(),
            ));
        }
        let inner = self.inner_dim();
        let n_pos = self.positional_embedding_max_pos.len();
        let rope_pair_dims = n_pos
            .checked_mul(2)
            .ok_or_else(|| DitError::Config("positional embedding dimension overflow".into()))?;
        if inner < rope_pair_dims {
            return Err(DitError::Config(format!(
                "inner_dim {inner} must be at least 2 * positional dimensions {rope_pair_dims}"
            )));
        }
        let half = inner
            .checked_div(2)
            .ok_or_else(|| DitError::Config("inner_dim division overflow".into()))?;
        if !half.is_multiple_of(self.num_attention_heads) {
            return Err(DitError::Config(format!(
                "inner_dim / 2 ({half}) must be divisible by num_attention_heads {}",
                self.num_attention_heads
            )));
        }
        // `use_prompt_adaln_single` only allocates a model-level path when
        // cross-attention AdaLN is enabled. The default config keeps it true for
        // metadata parity, but with `cross_attention_adaln = false` it is inert.
        if self.flags.cross_attention_adaln && self.flags.use_prompt_adaln_single {
            return Err(DitError::Config(
                "use_prompt_adaln_single with cross_attention_adaln is not wired in this core"
                    .into(),
            ));
        }
        if self.flags.use_keyframes_abs_pos_embedding {
            return Err(DitError::Config(
                "use_keyframes_abs_pos_embedding is not wired in this core".into(),
            ));
        }
        if self.frequencies_precision.as_deref() == Some("float64") {
            return Err(DitError::Config(
                "frequencies_precision = \"float64\" (double-precision RoPE frequencies) is not \
                 implemented in ltx-dit. Add f64 frequency computation in rope.rs to support it."
                    .into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        DiTConfig::default().validate().unwrap();
    }

    #[test]
    fn validate_rejects_unsupported_model_level_flags() {
        let mut config = DiTConfig::default();
        config.flags.cross_attention_adaln = true;
        config.flags.use_prompt_adaln_single = true;
        assert!(config.validate().is_err());

        let mut config = DiTConfig::default();
        config.flags.use_keyframes_abs_pos_embedding = true;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_rope_shape_hazards() {
        let mut config = DiTConfig {
            num_attention_heads: 0,
            ..DiTConfig::default()
        };
        assert!(config.validate().is_err());

        config = DiTConfig {
            attention_head_dim: 127,
            ..DiTConfig::default()
        };
        assert!(config.validate().is_err());

        config = DiTConfig {
            num_attention_heads: 1,
            attention_head_dim: 4,
            ..DiTConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
