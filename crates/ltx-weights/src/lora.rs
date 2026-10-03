//! `LoRA` file loading and merging.
//!
//! The reference `LoRA` format stores `lora_A.weight` and `lora_B.weight` under
//! a common module prefix (e.g. `transformer_blocks.0.attn1.to_q`), with an
//! optional per-layer `alpha` scalar tensor.  The delta is:
//!
//! ```text
//! Δ = strength * (B @ A) * (alpha / rank)
//! ```
//!
//! where `rank` is `A.shape[0]` = `B.shape[1]`, and `alpha` defaults to
//! `rank` (i.e. `alpha/rank = 1`) when absent.
//!
//! Key matching follows `_products_for_sd_key` / `_affected_weight_keys` in
//! `ltx_core/loader/fuse_loras.py`.  A `LoRA` key `prefix.lora_A.weight` matches
//! base key `prefix.weight`.
//!
//! # IC-LoRA layout metadata
//!
//! A `LoRA` checkpoint can carry `reference_downscale_factor` and
//! `reference_temporal_scale_factor` in its safetensors `__metadata__`.
//! [`LoraFile::ic_layout`] reads those into an [`ltx_shape::IcLoraLayout`].

use std::{collections::HashMap, path::Path};

use ltx_shape::IcLoraLayout;
use memmap2::Mmap;
use safetensors::{Dtype as SfDtype, SafeTensors, tensor::TensorInfo};

use crate::{
    error::WeightError,
    fp8::{e4m3fn_slice_to_f32, e5m2_slice_to_f32},
};

// ─── LoraFile ─────────────────────────────────────────────────────────────────

/// An opened `LoRA` safetensors file ready for merging.
///
/// Tensor data is memory-mapped; dequantization to f32 happens on demand.
pub struct LoraFile {
    // Pinned to keep the mmap alive for the lifetime of the tensors map.
    mmap: Mmap,
    /// Absolute offset where tensor data starts in the mmap.
    data_start: usize,
    /// Per-tensor metadata (shape, dtype, `data_offsets`).
    tensors: HashMap<String, TensorInfo>,
    /// Raw `__metadata__` strings.
    metadata: HashMap<String, String>,
}

impl LoraFile {
    /// Open a `LoRA` safetensors file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::Io`] or [`WeightError::Safetensors`] on failure.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WeightError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file) }?;

        let bytes: &[u8] = mmap.as_ref();
        let bytes_static: &'static [u8] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) };

        let (header_len, meta_obj) = SafeTensors::read_metadata(bytes_static)?;
        let data_start = 8_usize.saturating_add(header_len);

        let metadata: HashMap<String, String> = meta_obj.metadata().clone().unwrap_or_default();

        let tensors: HashMap<String, TensorInfo> = meta_obj
            .tensors()
            .into_iter()
            .map(|(name, info)| (name, info.clone()))
            .collect();

        Ok(Self {
            mmap,
            data_start,
            tensors,
            metadata,
        })
    }

    /// Read a raw metadata value from `__metadata__`.
    #[must_use]
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Read `reference_downscale_factor` and `reference_temporal_scale_factor`
    /// from the `LoRA` metadata into an [`IcLoraLayout`].
    ///
    /// Both default to `1` when absent (i.e. full reference resolution).
    ///
    /// Reference: `read_lora_reference_downscale_factor` /
    /// `read_lora_reference_temporal_scale_factor` in
    /// `ltx_pipelines/iclora_utils.py`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::Shape`] when a stored factor is zero.
    pub fn ic_layout(&self) -> Result<IcLoraLayout, WeightError> {
        let downscale = parse_u32_meta(&self.metadata, "reference_downscale_factor").unwrap_or(1);
        let temporal =
            parse_u32_meta(&self.metadata, "reference_temporal_scale_factor").unwrap_or(1);
        IcLoraLayout::new(downscale, temporal).map_err(WeightError::Shape)
    }

    /// Iterate over all tensor names in this `LoRA` file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Read a tensor as f32 values, or `None` if the name is not present.
    #[must_use]
    pub fn read_f32(&self, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let info = self.tensors.get(name)?;
        let (rel_start, rel_end) = info.data_offsets;
        let start = self.data_start.saturating_add(rel_start);
        let end = self.data_start.saturating_add(rel_end);
        let bytes = self.mmap.get(start..end)?;
        let f32s = decode_lora_bytes(bytes, info.dtype).ok()?;
        Some((info.shape.clone(), f32s))
    }

    /// Read a tensor's shape and dtype info.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }
}

// ─── MergeReport ─────────────────────────────────────────────────────────────

/// Result of a [`WeightStore::merge_lora`][crate::WeightStore::merge_lora] call.
#[derive(Debug, Default)]
pub struct MergeReport {
    /// Base weight keys that had a matching `LoRA` delta applied.
    pub matched_keys: Vec<String>,
    /// `LoRA` `lora_A.weight` keys for which no matching base weight was found.
    ///
    /// These are recorded rather than raising an error, mirroring the reference
    /// behavior in `fuse_lora_weights` (`ltx_core/loader/fuse_loras.py`):
    /// missing base keys are silently skipped so partial `LoRA`s can be loaded
    /// without requiring the full model to be present.
    pub unmatched_lora_keys: Vec<String>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Decode `LoRA` tensor bytes to f32.
// `as_chunks` is nightly-only; `chunks_exact` is the stable equivalent here.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn decode_lora_bytes(bytes: &[u8], dtype: SfDtype) -> Result<Vec<f32>, WeightError> {
    match dtype {
        SfDtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| {
                let mut arr = [0u8; 4];
                arr.copy_from_slice(c);
                f32::from_le_bytes(arr)
            })
            .collect()),
        SfDtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| {
                let mut arr = [0u8; 2];
                arr.copy_from_slice(c);
                half::f16::from_le_bytes(arr).to_f32()
            })
            .collect()),
        SfDtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| {
                let mut arr = [0u8; 2];
                arr.copy_from_slice(c);
                half::bf16::from_le_bytes(arr).to_f32()
            })
            .collect()),
        SfDtype::F8_E4M3 => Ok(e4m3fn_slice_to_f32(bytes)),
        SfDtype::F8_E5M2 => Ok(e5m2_slice_to_f32(bytes)),
        other => Err(WeightError::UnsupportedDtype {
            key: String::new(),
            dtype: format!("{other:?}"),
        }),
    }
}

fn parse_u32_meta(meta: &HashMap<String, String>, key: &str) -> Option<u32> {
    meta.get(key)?.trim().parse::<u32>().ok()
}
