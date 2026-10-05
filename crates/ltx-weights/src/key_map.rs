//! Key transformation rules that map safetensors checkpoint keys to the names
//! the model modules use when calling `state_dict()`.
//!
//! [`KeyMap`] is a sequence of [`Rule`]s applied in order during
//! [`WeightStore::open`][crate::WeightStore::open].  Each rule can filter,
//! rename, or transform tensors.
//!
//! # Preset constructors
//!
//! | Constructor | Reference source |
//! |---|---|
//! | [`KeyMap::identity`] | — fixtures saved with `module.state_dict()` names |
//! | [`KeyMap::transformer`] | `LTXV_MODEL_COMFY_RENAMING_MAP` in `ltx_core/model/transformer/model_configurator.py`; called by `DiffusionStage.from_checkpoint` in `ltx_pipelines/utils/blocks.py` |
//! | [`KeyMap::video_encoder`] | `VAE_ENCODER_COMFY_KEYS_FILTER` in `ltx_core/model/video_vae/model_configurator.py`; used by `ImageConditioner` in `ltx_pipelines/utils/blocks.py` |
//! | [`KeyMap::video_decoder`] | `_build_diffusion_vae_decoder_sd_ops` in `ltx_core/model/video_vae/model_configurator.py`; called via `video_decoder_sd_ops_for_checkpoint(..., diffusion_vae=True, na_dsl_kernel=False)` |
//!
//! # Skipped ops
//!
//! - **`CHANNELS_LAST_3D_WEIGHTS`** (`ltx_core/model/video_vae/memory_efficient_decode.py`):
//!   reorders weight memory to channels-last for CUDA 3-D conv performance.
//!   Values are unchanged; layout is irrelevant for a CPU f32 copy.  Skipped.
//! - **`_emit_na_softmax_bound`** (DSL kernel–specific): computes a per-block
//!   softmax bound from `k_norm.weight`; only needed when `na_dsl_kernel=True`.
//!   Skipped (covered by [`KeyMap::video_decoder`] which uses `na_dsl_kernel=False`).
//! - **`_Stage4Hop`** context-proj fusion: fuses the stage-4 upsample projection
//!   into each block's `context_proj`; only needed for `na_dsl_kernel=True`.  Skipped.

/// A single rule in a [`KeyMap`].
#[derive(Clone, Debug)]
pub enum Rule {
    /// Key must match at least one of these prefixes (OR).  Empty → all pass.
    RequireAnyPrefix(Vec<String>),
    /// Substring replacement applied to the current key name.
    ///
    /// Mirrors Python `key.replace(find, replace)` (all occurrences).
    Replace {
        /// Substring to search for.
        find: String,
        /// Replacement string.
        replace: String,
    },
    /// Drop keys whose post-rename name starts with this prefix.
    DropPrefix(String),
    /// Drop keys whose post-rename name ends with this suffix.
    DropSuffix(String),
    /// Apply a prefix rename to post-filter keys.
    ///
    /// Used for `t_embedder.mlp.*` → `t_embedder.timestep_embedder.*` in the
    /// diffusion VAE decoder.
    RenameExact {
        /// Prefix to find in the already-renamed key.
        find: String,
        /// Replacement.
        replace: String,
    },
    /// Split `*.qkv.weight` / `*.qkv.bias` into `*.to_q`, `*.to_k`, `*.to_v`.
    ///
    /// Ported from `_split_fused_qkv_param` in
    /// `ltx_core/model/video_vae/model_configurator.py`.
    SplitQkv,
    /// Fold `gate_msa` / `gate_mlp` / `gate_ctx` scalars into neighbouring
    /// linear weights/biases, then drop the gate keys.
    ///
    /// Ported from `_build_diffusion_vae_decoder_sd_ops` and
    /// `_fold_gate_into_linear` in
    /// `ltx_core/model/video_vae/model_configurator.py`.
    FoldGates,
    /// Synthesise a zero `type_emb` tensor when the checkpoint does not carry
    /// one (pre-keyframe checkpoints).
    ///
    /// Shape is inferred from `conv_in.weight` (in-channels = `shape[1]`).
    ///
    /// Ported from `_emit_zero_type_emb` and `_checkpoint_has_type_emb` in
    /// `ltx_core/model/video_vae/model_configurator.py`.
    SynthZeroTypeEmb,
}

/// Ordered key transformation rules applied during
/// [`WeightStore::open`][crate::WeightStore::open].
#[derive(Clone, Debug, Default)]
pub struct KeyMap {
    pub rules: Vec<Rule>,
}

impl KeyMap {
    /// No-op map.  Keys pass through unchanged.
    ///
    /// Used for fixtures saved with `module.state_dict()` names directly.
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Alpha-gen transformer checkpoint layout.
    ///
    /// Filters to `model.diffusion_model.*` and strips the prefix.
    ///
    /// Reference: `LTXV_MODEL_COMFY_RENAMING_MAP` in
    /// `ltx_core/model/transformer/model_configurator.py`.
    #[must_use]
    pub fn transformer() -> Self {
        Self {
            rules: vec![
                Rule::RequireAnyPrefix(vec!["model.diffusion_model.".into()]),
                Rule::Replace {
                    find: "model.diffusion_model.".into(),
                    replace: String::new(),
                },
            ],
        }
    }

    /// Video VAE encoder (diffusion-VAE checkpoint layout).
    ///
    /// Accepts both monolithic `vae.encoder.*` / `vae.per_channel_statistics.*`
    /// prefixes and Comfy-split bare `encoder.*` / `per_channel_statistics.*`
    /// prefixes; strips all prefixes so the encoder module sees plain names.
    ///
    /// Reference: `VAE_ENCODER_COMFY_KEYS_FILTER` in
    /// `ltx_core/model/video_vae/model_configurator.py`; used by
    /// `ImageConditioner` in `ltx_pipelines/utils/blocks.py`.
    #[must_use]
    pub fn video_encoder() -> Self {
        Self {
            rules: vec![
                Rule::RequireAnyPrefix(vec![
                    "vae.encoder.".into(),
                    "vae.per_channel_statistics.".into(),
                    "encoder.".into(),
                    "per_channel_statistics.".into(),
                ]),
                Rule::Replace {
                    find: "vae.encoder.".into(),
                    replace: String::new(),
                },
                Rule::Replace {
                    find: "vae.per_channel_statistics.".into(),
                    replace: "per_channel_statistics.".into(),
                },
                Rule::Replace {
                    find: "encoder.".into(),
                    replace: String::new(),
                },
            ],
        }
    }

    /// Diffusion video VAE decoder (`na_dsl_kernel = false` path).
    ///
    /// Operations (in order):
    ///
    /// 1. Filter: `vae.decoder.*` or bare `decoder.*`; also
    ///    `vae.per_channel_statistics.*` / `per_channel_statistics.*`.
    /// 2. Strip `vae.decoder.` / `decoder.` prefix.
    /// 3. Rename `t_embedder.mlp.0.` → `t_embedder.timestep_embedder.linear_1.`
    ///    and `t_embedder.mlp.2.` → `t_embedder.timestep_embedder.linear_2.`.
    /// 4. Drop `coarse_*` parameters (preview head, not in the module).
    /// 5. Fold `gate_msa` / `gate_mlp` / `gate_ctx` scalars into neighbouring
    ///    weights/biases, then drop the gate keys ([`Rule::FoldGates`]).
    /// 6. Split `*.qkv.weight` / `*.qkv.bias` into `to_q`, `to_k`, `to_v`.
    /// 7. Synthesise zero `type_emb` for older checkpoints ([`Rule::SynthZeroTypeEmb`]).
    ///
    /// Reference: `_build_diffusion_vae_decoder_sd_ops` called via
    /// `video_decoder_sd_ops_for_checkpoint(..., diffusion_vae=True, na_dsl_kernel=False)`
    /// in `ltx_core/model/video_vae/model_configurator.py`.
    #[must_use]
    pub fn video_decoder() -> Self {
        Self {
            rules: vec![
                // 1. Filter
                Rule::RequireAnyPrefix(vec![
                    "vae.decoder.".into(),
                    "decoder.".into(),
                    "vae.per_channel_statistics.".into(),
                    "per_channel_statistics.".into(),
                ]),
                // 2. Strip prefixes
                Rule::Replace {
                    find: "vae.decoder.".into(),
                    replace: String::new(),
                },
                Rule::Replace {
                    find: "decoder.".into(),
                    replace: String::new(),
                },
                Rule::Replace {
                    find: "vae.per_channel_statistics.".into(),
                    replace: "per_channel_statistics.".into(),
                },
                // 3. t_embedder renames
                Rule::RenameExact {
                    find: "t_embedder.mlp.0.".into(),
                    replace: "t_embedder.timestep_embedder.linear_1.".into(),
                },
                Rule::RenameExact {
                    find: "t_embedder.mlp.2.".into(),
                    replace: "t_embedder.timestep_embedder.linear_2.".into(),
                },
                // 4. Drop coarse head
                Rule::DropPrefix("coarse_".into()),
                // 5. Fold gates then drop them
                Rule::FoldGates,
                // 6. QKV split
                Rule::SplitQkv,
                // 7. Synthesise missing type_emb
                Rule::SynthZeroTypeEmb,
            ],
        }
    }

    /// Whether this map includes a gate-fold rule.
    #[must_use]
    pub fn has_fold_gates(&self) -> bool {
        self.rules.iter().any(|r| matches!(r, Rule::FoldGates))
    }

    /// Whether this map includes a `QKV`-split rule.
    #[must_use]
    pub fn has_split_qkv(&self) -> bool {
        self.rules.iter().any(|r| matches!(r, Rule::SplitQkv))
    }

    /// Whether this map includes a zero `type_emb` synthesis rule.
    #[must_use]
    pub fn has_synth_type_emb(&self) -> bool {
        self.rules
            .iter()
            .any(|r| matches!(r, Rule::SynthZeroTypeEmb))
    }

    /// Apply filter + rename rules to a key; returns `None` if filtered out.
    ///
    /// Value-transform rules ([`Rule::SplitQkv`], [`Rule::FoldGates`],
    /// [`Rule::SynthZeroTypeEmb`], [`Rule::DropPrefix`], [`Rule::DropSuffix`])
    /// are handled by the store during `open()`.
    #[must_use]
    pub fn apply_key(&self, mut key: String) -> Option<String> {
        for rule in &self.rules {
            match rule {
                Rule::RequireAnyPrefix(prefixes) => {
                    if prefixes.is_empty() {
                        continue;
                    }
                    if !prefixes.iter().any(|p| key.starts_with(p.as_str())) {
                        return None;
                    }
                }
                Rule::Replace { find, replace } => {
                    if key.contains(find.as_str()) {
                        key = key.replace(find.as_str(), replace.as_str());
                    }
                }
                Rule::RenameExact { find, replace } => {
                    if let Some(suffix) = key.strip_prefix(find.as_str()) {
                        key = format!("{replace}{suffix}");
                    }
                }
                Rule::DropPrefix(p) => {
                    if key.starts_with(p.as_str()) {
                        return None;
                    }
                }
                Rule::DropSuffix(s) => {
                    if key.ends_with(s.as_str()) {
                        return None;
                    }
                }
                // Value-transform rules are deferred to the store.
                Rule::SplitQkv | Rule::FoldGates | Rule::SynthZeroTypeEmb => {}
            }
        }
        Some(key)
    }
}

// ─── Gate-fold helpers ────────────────────────────────────────────────────────

/// Suffix pairs: (param suffix, gate suffix) for gate folding.
///
/// Source: `_GATE_FOLD_TARGETS` in
/// `ltx_core/model/video_vae/model_configurator.py`.
pub const GATE_FOLD_TARGETS: &[(&str, &str)] = &[
    (".attn.proj.weight", ".gate_msa"),
    (".attn.proj.bias", ".gate_msa"),
    (".mlp.w_down.weight", ".gate_mlp"),
    (".mlp.w_down.bias", ".gate_mlp"),
    (".context_proj.weight", ".gate_ctx"),
    (".context_proj.bias", ".gate_ctx"),
];

/// Gate suffixes to drop after folding.
pub const GATE_DROP_SUFFIXES: &[&str] = &[".gate_msa", ".gate_mlp", ".gate_ctx"];

/// Returns `true` if `key` ends with a gate suffix.
pub fn is_gate_key(key: &str) -> bool {
    GATE_DROP_SUFFIXES.iter().any(|s| key.ends_with(s))
}

/// Derive the gate key for a given param key, or `None` if the param is not a
/// fold target.
pub fn gate_key_for_param(param_key: &str) -> Option<String> {
    for (leaf, gate_suffix) in GATE_FOLD_TARGETS {
        if param_key.ends_with(leaf) {
            let prefix_end = param_key.len().saturating_sub(leaf.len());
            let prefix = param_key.get(..prefix_end)?;
            return Some(format!("{prefix}{gate_suffix}"));
        }
    }
    None
}
