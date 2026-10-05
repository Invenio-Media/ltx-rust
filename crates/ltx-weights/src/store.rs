//! Memory-mapped safetensors weight store.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use bytemuck::cast_slice;
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype as SfDtype, SafeTensors};

use crate::{
    error::WeightError,
    fp8::{e4m3fn_slice_to_f32, e5m2_slice_to_f32},
    host_tensor::HostTensor,
    key_map::{KeyMap, gate_key_for_param, is_gate_key},
    lora::{LoraFile, MergeReport},
    scope::Scope,
};

// ─── Internal entry representation ────────────────────────────────────────────

/// Where a tensor's bytes come from.
enum EntrySource {
    /// Slice into the memory-mapped file (absolute byte range within the file).
    Mapped {
        file_idx: usize,
        /// Absolute start offset in the mmap.
        start: usize,
        /// Absolute end offset (exclusive) in the mmap.
        end: usize,
    },
    /// Heap-allocated (result of `QKV` split, gate fold, or `type_emb` synthesis).
    Owned(Vec<u8>),
}

struct Entry {
    shape: Vec<usize>,
    dtype: SfDtype,
    source: EntrySource,
}

impl Entry {
    fn read_raw(&self, key: &str, files: &[Mmap]) -> Result<Vec<u8>, WeightError> {
        match &self.source {
            EntrySource::Mapped {
                file_idx,
                start,
                end,
            } => {
                let file = files
                    .get(*file_idx)
                    .ok_or_else(|| WeightError::InvalidTensorData {
                        key: key.to_owned(),
                        message: "mapped file index is out of range".to_owned(),
                    })?;
                let bytes =
                    file.get(*start..*end)
                        .ok_or_else(|| WeightError::InvalidTensorData {
                            key: key.to_owned(),
                            message: "tensor byte range is outside the safetensors file".to_owned(),
                        })?;
                if bytes.len() != self.nbytes() {
                    return Err(WeightError::InvalidTensorData {
                        key: key.to_owned(),
                        message: format!(
                            "header expects {} bytes, range has {} bytes",
                            self.nbytes(),
                            bytes.len()
                        ),
                    });
                }
                Ok(bytes.to_vec())
            }
            EntrySource::Owned(v) => Ok(v.clone()),
        }
    }

    fn nbytes(&self) -> usize {
        self.shape
            .iter()
            .product::<usize>()
            .saturating_mul(dtype_elem_size(self.dtype))
    }
}

// ─── WeightStore ──────────────────────────────────────────────────────────────

/// Memory-mapped safetensors weight store.
///
/// Multiple files are merged; duplicate post-rename keys are rejected.  `FP8`,
/// `BF16`, `F16`, and `F32` tensors are dequantized to f32 on [`read`][Self::read].
/// Merged `LoRA` deltas (from [`merge_lora`][Self::merge_lora]) are added at
/// read time.
pub struct WeightStore {
    /// Memory-mapped file data.  Pinned for the lifetime of the store.
    files: Vec<Mmap>,
    /// Renamed key → tensor entry.  Includes scale keys (internal).
    entries: HashMap<String, Entry>,
    /// Keys that are `FP8` scale siblings (`{key}_scale`); excluded from `keys()`.
    scale_keys: HashSet<String>,
    /// `__metadata__` merged across all files.
    metadata: HashMap<String, String>,
    /// Cached `LoRA` deltas: key → f32 delta (same element count as weight).
    lora_deltas: HashMap<String, Vec<f32>>,
}

impl WeightStore {
    /// Open one or more safetensors files.
    ///
    /// Keys are renamed (and filtered) by `map`.  Duplicate post-rename keys
    /// across files are rejected with [`WeightError::DuplicateKey`].
    ///
    /// Value-transform rules in `map` (QKV split, gate fold) are applied
    /// eagerly.
    ///
    /// # Errors
    ///
    /// - [`WeightError::Io`] / [`WeightError::Safetensors`] on file read failure.
    /// - [`WeightError::DuplicateKey`] on key collision.
    /// - [`WeightError::QkvSplitDim`] if a QKV tensor's leading dim is not divisible by 3.
    pub fn open(paths: &[impl AsRef<Path>], map: &KeyMap) -> Result<Self, WeightError> {
        let mut mmaps: Vec<Mmap> = Vec::with_capacity(paths.len());
        let mut merged_meta: HashMap<String, String> = HashMap::new();
        let mut entries: HashMap<String, Entry> = HashMap::new();

        for (file_idx, path) in paths.iter().enumerate() {
            let file = std::fs::File::open(path.as_ref())?;
            // SAFETY: the mmap is read-only and kept alive in `self.files`.
            // Callers must not truncate or modify the safetensors file while
            // a `WeightStore` that maps it is alive.
            let mapped_file = unsafe { Mmap::map(&file) }?;

            let bytes: &[u8] = mapped_file.as_ref();
            let (header_len, metadata_obj) = SafeTensors::read_metadata(bytes)?;
            let data_start: usize = 8_usize.saturating_add(header_len);

            // Merge __metadata__ (first file wins on collision).
            if let Some(meta) = metadata_obj.metadata() {
                for (k, v) in meta {
                    merged_meta.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }

            for (name, info) in metadata_obj.tensors() {
                let Some(renamed) = map.apply_key(name) else {
                    continue;
                };
                let (rel_start, rel_end) = info.data_offsets;
                let start = data_start.saturating_add(rel_start);
                let end = data_start.saturating_add(rel_end);
                if entries.contains_key(&renamed) {
                    return Err(WeightError::DuplicateKey(renamed));
                }
                entries.insert(
                    renamed,
                    Entry {
                        shape: info.shape.clone(),
                        dtype: info.dtype,
                        source: EntrySource::Mapped {
                            file_idx,
                            start,
                            end,
                        },
                    },
                );
            }

            mmaps.push(mapped_file);
        }

        // Mark FP8 scale keys as internal.  A non-FP8 parameter named `foo`
        // with a real sibling `foo_scale` must stay visible and must not be
        // multiplied into `foo`.
        let all_keys: Vec<String> = entries.keys().cloned().collect();
        let mut scale_keys: HashSet<String> = HashSet::new();
        for k in &all_keys {
            if let Some(param_key) = k.strip_suffix("_scale")
                && entries
                    .get(param_key)
                    .is_some_and(|entry| is_fp8(entry.dtype))
            {
                scale_keys.insert(k.clone());
            }
        }

        let mut store = Self {
            files: mmaps,
            entries,
            scale_keys,
            metadata: merged_meta,
            lora_deltas: HashMap::new(),
        };

        if map.has_fold_gates() {
            store.apply_gate_fold()?;
        }
        if map.has_split_qkv() {
            store.apply_qkv_split()?;
        }
        if map.has_synth_type_emb() {
            store.apply_synth_type_emb();
        }

        Ok(store)
    }

    // ─── Gate fold ────────────────────────────────────────────────────────────

    /// Fold `gate_msa` / `gate_mlp` / `gate_ctx` scalars into the neighbouring
    /// linear weights/biases, then remove the gate entries.
    ///
    /// Reference: `_fold_gate_into_linear` in
    /// `ltx_core/model/video_vae/model_configurator.py`.
    fn apply_gate_fold(&mut self) -> Result<(), WeightError> {
        let gate_keys: Vec<String> = self
            .entries
            .keys()
            .filter(|k| is_gate_key(k.as_str()))
            .cloned()
            .collect();

        let mut gates: HashMap<String, f32> = HashMap::new();
        for gate_key in &gate_keys {
            let Some(entry) = self.entries.get(gate_key) else {
                continue;
            };
            let is_scalar = entry.shape.is_empty()
                || (entry.shape.len() == 1 && entry.shape.first().copied().unwrap_or(0) == 1);
            if !is_scalar {
                return Err(WeightError::GateNotScalar {
                    key: gate_key.clone(),
                    shape: entry.shape.clone(),
                });
            }
            let raw = entry.read_raw(gate_key, &self.files)?;
            let scalar = decode_scalar_f32(&raw, entry.dtype)?;
            gates.insert(gate_key.clone(), scalar);
        }

        let param_keys: Vec<String> = self.entries.keys().cloned().collect();
        for param_key in param_keys {
            let Some(gate_key) = gate_key_for_param(&param_key) else {
                continue;
            };
            let Some(&gate_val) = gates.get(&gate_key) else {
                continue;
            };

            let dtype = {
                let Some(entry) = self.entries.get(&param_key) else {
                    continue;
                };
                entry.dtype
            };
            let raw = self
                .entries
                .get(&param_key)
                .ok_or_else(|| WeightError::MissingKey(param_key.clone()))?
                .read_raw(&param_key, &self.files)?;
            let nbytes = raw.len();

            let mut f32_data = decode_to_f32(&raw, dtype, nbytes)?;
            if is_fp8(dtype) {
                let scale_key = format!("{param_key}_scale");
                if let Some(scale_entry) = self.entries.get(&scale_key) {
                    let scale_raw = scale_entry.read_raw(&scale_key, &self.files)?;
                    let scale = decode_scalar_f32(&scale_raw, scale_entry.dtype)?;
                    for value in &mut f32_data {
                        *value *= scale;
                    }
                }
                self.entries.remove(&scale_key);
                self.scale_keys.remove(&scale_key);
            }
            for val in &mut f32_data {
                *val *= gate_val;
            }
            let new_bytes = f32_slice_to_bytes(&f32_data);
            if let Some(e) = self.entries.get_mut(&param_key) {
                e.source = EntrySource::Owned(new_bytes);
                e.dtype = SfDtype::F32;
            }
        }

        for gate_key in gate_keys {
            self.entries.remove(&gate_key);
        }

        Ok(())
    }

    // ─── QKV split ────────────────────────────────────────────────────────────

    /// Split fused `*.qkv.weight` / `*.qkv.bias` into `*.to_q`, `*.to_k`,
    /// `*.to_v`.
    ///
    /// Reference: `_split_fused_qkv_param` in
    /// `ltx_core/model/video_vae/model_configurator.py`.
    fn apply_qkv_split(&mut self) -> Result<(), WeightError> {
        let qkv_keys: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.ends_with(".qkv.weight") || k.ends_with(".qkv.bias"))
            .cloned()
            .collect();

        for qkv_key in qkv_keys {
            let (shape, dtype, raw) = {
                let Some(entry) = self.entries.get(&qkv_key) else {
                    continue;
                };
                (
                    entry.shape.clone(),
                    entry.dtype,
                    entry.read_raw(&qkv_key, &self.files)?,
                )
            };

            let leading = shape.first().copied().unwrap_or(0);
            if leading % 3 != 0 {
                return Err(WeightError::QkvSplitDim {
                    key: qkv_key,
                    dim: leading,
                });
            }
            let d = leading / 3;

            let nbytes = raw.len();
            let mut f32_data = decode_to_f32(&raw, dtype, nbytes)?;
            if is_fp8(dtype) {
                let scale_key = format!("{qkv_key}_scale");
                if let Some(scale_entry) = self.entries.get(&scale_key) {
                    let scale_raw = scale_entry.read_raw(&scale_key, &self.files)?;
                    let scale = decode_scalar_f32(&scale_raw, scale_entry.dtype)?;
                    for value in &mut f32_data {
                        *value = scale.mul_add(*value, 0.0_f32);
                    }
                }
            }
            let elem_per_split = f32_data.len() / 3;

            let is_weight = qkv_key.ends_with(".weight");
            let leaf = if is_weight { "weight" } else { "bias" };
            let qkv_mark = ".qkv.";
            // `.find` returns byte offset into ASCII key; `.get()` is safe.
            let base_prefix = qkv_key
                .rfind(qkv_mark)
                .and_then(|p| qkv_key.get(..p))
                .unwrap_or(&qkv_key);

            let names = [
                format!("{base_prefix}.to_q.{leaf}"),
                format!("{base_prefix}.to_k.{leaf}"),
                format!("{base_prefix}.to_v.{leaf}"),
            ];

            let mut new_shape = shape.clone();
            if let Some(first) = new_shape.first_mut() {
                *first = d;
            }

            for (split_idx, name) in names.iter().enumerate() {
                let start = split_idx.saturating_mul(elem_per_split);
                let end = start.saturating_add(elem_per_split);
                let slice = f32_data.get(start..end).unwrap_or(&[]);
                let new_bytes = f32_slice_to_bytes(slice);
                if self.entries.contains_key(name) {
                    return Err(WeightError::DuplicateKey(name.clone()));
                }
                self.entries.insert(
                    name.clone(),
                    Entry {
                        shape: new_shape.clone(),
                        dtype: SfDtype::F32,
                        source: EntrySource::Owned(new_bytes),
                    },
                );
            }

            self.entries.remove(&qkv_key);
            let scale_key = format!("{qkv_key}_scale");
            self.entries.remove(&scale_key);
            self.scale_keys.remove(&scale_key);
        }

        Ok(())
    }

    // ─── Synthesise zero type_emb ──────────────────────────────────────────────

    /// Synthesise a zero `type_emb` tensor from `conv_in.weight` if absent.
    ///
    /// Reference: `_emit_zero_type_emb` / `_checkpoint_has_type_emb` in
    /// `ltx_core/model/video_vae/model_configurator.py`.
    fn apply_synth_type_emb(&mut self) {
        if self.entries.contains_key("type_emb") {
            return;
        }
        let Some(entry) = self.entries.get("conv_in.weight") else {
            return;
        };
        let in_channels = entry.shape.get(1).copied().unwrap_or(0);
        if in_channels == 0 {
            return;
        }
        let zeros = vec![0.0f32; in_channels];
        let bytes = f32_slice_to_bytes(&zeros);
        self.entries.insert(
            "type_emb".to_owned(),
            Entry {
                shape: vec![in_channels],
                dtype: SfDtype::F32,
                source: EntrySource::Owned(bytes),
            },
        );
    }

    // ─── Public accessors ─────────────────────────────────────────────────────

    /// Read a raw metadata value.
    #[must_use]
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Parse `__metadata__["config"]` as a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::MissingConfig`] when the key is absent, or
    /// [`WeightError::ConfigParse`] for invalid JSON.
    pub fn config(&self) -> Result<serde_json::Value, WeightError> {
        let s = self
            .metadata
            .get("config")
            .ok_or(WeightError::MissingConfig)?;
        serde_json::from_str(s).map_err(WeightError::ConfigParse)
    }

    /// Iterate over all visible (non-internal-scale) keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries
            .keys()
            .filter(|k| !self.scale_keys.contains(*k))
            .map(String::as_str)
    }

    /// Whether `key` is present and visible in the store.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key) && !self.scale_keys.contains(key)
    }

    /// Return the shape of `key`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::MissingKey`] when the key is absent.
    pub fn shape(&self, key: &str) -> Result<&[usize], WeightError> {
        self.entries
            .get(key)
            .map(|e| e.shape.as_slice())
            .ok_or_else(|| WeightError::MissingKey(key.to_owned()))
    }

    /// Read `key` as a host-side f32 tensor.
    ///
    /// All dtypes (F32, F16, BF16, FP8) are dequantized.  If a sibling
    /// `{key}_scale` entry exists, the scale is applied before returning.
    /// Any merged `LoRA` delta is added.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::MissingKey`] or [`WeightError::UnsupportedDtype`].
    pub fn read(&self, key: &str) -> Result<HostTensor, WeightError> {
        let entry = self
            .entries
            .get(key)
            .ok_or_else(|| WeightError::MissingKey(key.to_owned()))?;

        let shape = entry.shape.clone();
        let dtype = entry.dtype;
        let nbytes = entry.nbytes();
        let raw = entry.read_raw(key, &self.files)?;

        let mut f32_data = decode_to_f32(&raw, dtype, nbytes).map_err(|e| match e {
            WeightError::UnsupportedDtype { dtype: d, .. } => WeightError::UnsupportedDtype {
                key: key.to_owned(),
                dtype: d,
            },
            other => other,
        })?;

        // Apply per-tensor FP8 scale if present.
        let scale_key = format!("{key}_scale");
        if is_fp8(dtype)
            && let Some(scale_entry) = self.entries.get(&scale_key)
        {
            let scale_raw = scale_entry.read_raw(&scale_key, &self.files)?;
            let scale = decode_scalar_f32(&scale_raw, scale_entry.dtype)?;
            for v in &mut f32_data {
                *v = scale.mul_add(*v, 0.0_f32);
            }
        }

        // Apply cached LoRA delta.
        if let Some(delta) = self.lora_deltas.get(key) {
            if delta.len() != f32_data.len() {
                return Err(WeightError::LoraShapeMismatch(key.to_owned()));
            }
            for (v, d) in f32_data.iter_mut().zip(delta.iter()) {
                *v += *d;
            }
        }

        Ok(HostTensor {
            shape,
            data: f32_data,
        })
    }

    /// Merge a `LoRA` file into the store.
    ///
    /// For each base key `prefix.weight`, looks for `prefix.lora_A.weight`
    /// and `prefix.lora_B.weight` in the `LoRA` file.  If found, computes:
    ///
    /// ```text
    /// Δ = strength * (B @ A) * (alpha / rank)
    /// ```
    ///
    /// where `rank = A.shape[0]` and `alpha` defaults to `rank` when absent
    /// (making `alpha/rank = 1`).  Deltas accumulate across calls.
    ///
    /// Key matching follows `_products_for_sd_key` / `_affected_weight_keys`
    /// in `ltx_core/loader/fuse_loras.py`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::LoraShapeMismatch`] or [`WeightError::LoraRankMismatch`]
    /// when dimensions are inconsistent.
    pub fn merge_lora(
        &mut self,
        lora: &LoraFile,
        strength: f32,
    ) -> Result<MergeReport, WeightError> {
        let mut report = MergeReport::default();

        let lora_a_suffix = ".lora_A.weight";
        let lora_pairs: Vec<(String, String, String)> = lora
            .tensor_names()
            .filter(|k| k.ends_with(lora_a_suffix))
            .map(|k| {
                // `.strip_suffix` is always ASCII so the slice is char-safe.
                let prefix = k.strip_suffix(lora_a_suffix).unwrap_or(k);
                let base_key = format!("{prefix}.weight");
                let key_b = format!("{prefix}.lora_B.weight");
                (base_key, k.to_owned(), key_b)
            })
            .collect();

        let mut pending_deltas: Vec<(String, Vec<f32>)> = Vec::new();

        for (base_key, key_a, key_b) in lora_pairs {
            if !self.entries.contains_key(&base_key) {
                report.unmatched_lora_keys.push(key_a);
                continue;
            }

            let Some((shape_a, a_f32)) = lora.read_f32(&key_a)? else {
                report.unmatched_lora_keys.push(key_a);
                continue;
            };
            let Some((shape_b, b_f32)) = lora.read_f32(&key_b)? else {
                report.unmatched_lora_keys.push(key_a);
                continue;
            };
            let rank_a = shape_a.first().copied().unwrap_or(0);
            let rank_b = shape_b.get(1).copied().unwrap_or(0);
            if rank_a != rank_b || rank_a == 0 {
                return Err(WeightError::LoraRankMismatch {
                    key: base_key,
                    a_rank: rank_a,
                    b_rank: rank_b,
                });
            }
            let rank = rank_a;

            let (out_features, in_features) = {
                let e = self
                    .entries
                    .get(&base_key)
                    .ok_or_else(|| WeightError::MissingKey(base_key.clone()))?;
                (
                    e.shape.first().copied().unwrap_or(0),
                    e.shape.get(1).copied().unwrap_or(1),
                )
            };
            let b_out = shape_b.first().copied().unwrap_or(0);
            let a_in = shape_a.get(1).copied().unwrap_or(0);
            if b_out != out_features || a_in != in_features {
                return Err(WeightError::LoraShapeMismatch(base_key));
            }

            // Look for per-layer alpha; default = rank (alpha/rank = 1).
            let alpha_key = key_a
                .strip_suffix(lora_a_suffix)
                .map_or_else(String::new, |p| format!("{p}.alpha"));
            let alpha = lora
                .read_f32(&alpha_key)?
                .and_then(|(_, v)| v.first().copied())
                .unwrap_or_else(|| usize_to_f32(rank));

            // `usize as f32` safe for small rank values; see `usize_to_f32`.
            let coeff = strength.mul_add(alpha / usize_to_f32(rank), 0.0_f32);

            let delta = matmul_scaled(&b_f32, out_features, rank, &a_f32, in_features, coeff);

            pending_deltas.push((base_key.clone(), delta));
            report.matched_keys.push(base_key);
        }

        for (base_key, delta) in &pending_deltas {
            if let Some(existing) = self.lora_deltas.get(base_key)
                && existing.len() != delta.len()
            {
                return Err(WeightError::LoraShapeMismatch(base_key.clone()));
            }
        }

        if report.matched_keys.is_empty() && !report.unmatched_lora_keys.is_empty() {
            return Err(WeightError::NoLoraMatches);
        }

        for (base_key, delta) in pending_deltas {
            let entry_delta = self
                .lora_deltas
                .entry(base_key)
                .or_insert_with(|| vec![0.0f32; delta.len()]);
            for (acc, d) in entry_delta.iter_mut().zip(delta.iter()) {
                *acc += *d;
            }
        }

        Ok(report)
    }

    /// Return a scoped view of this store with a key prefix.
    #[must_use]
    pub fn scope(&self, prefix: &str) -> Scope<'_> {
        Scope::new(self, prefix.to_owned())
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Byte size of one element for a safetensors dtype.
const fn dtype_elem_size(dtype: SfDtype) -> usize {
    match dtype {
        SfDtype::F16 | SfDtype::BF16 | SfDtype::I16 | SfDtype::U16 => 2,
        SfDtype::F64 | SfDtype::I64 | SfDtype::U64 => 8,
        SfDtype::I8 | SfDtype::U8 | SfDtype::BOOL | SfDtype::F8_E4M3 | SfDtype::F8_E5M2 => 1,
        // F32/I32/U32 and any future dtypes default to 4.
        _ => 4,
    }
}
/// Whether `dtype` is one of the FP8 formats that uses a sibling scale tensor.
const fn is_fp8(dtype: SfDtype) -> bool {
    matches!(dtype, SfDtype::F8_E4M3 | SfDtype::F8_E5M2)
}

/// Decode raw bytes to `Vec<f32>`.
///
/// Uses `from_le_bytes` for alignment-safe reads (safetensors data sections
/// may not be aligned to the native element size).
// `as_chunks` is nightly-only; `chunks_exact` is the stable equivalent here.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn decode_to_f32(bytes: &[u8], dtype: SfDtype, _nbytes: usize) -> Result<Vec<f32>, WeightError> {
    match dtype {
        SfDtype::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(WeightError::UnsupportedDtype {
                    key: String::new(),
                    dtype: "F32 byte length not divisible by 4".to_owned(),
                });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(c);
                    f32::from_le_bytes(arr)
                })
                .collect())
        }
        SfDtype::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(WeightError::UnsupportedDtype {
                    key: String::new(),
                    dtype: "F16 byte length not divisible by 2".to_owned(),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    let mut arr = [0u8; 2];
                    arr.copy_from_slice(c);
                    f16::from_le_bytes(arr).to_f32()
                })
                .collect())
        }
        SfDtype::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(WeightError::UnsupportedDtype {
                    key: String::new(),
                    dtype: "BF16 byte length not divisible by 2".to_owned(),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    let mut arr = [0u8; 2];
                    arr.copy_from_slice(c);
                    bf16::from_le_bytes(arr).to_f32()
                })
                .collect())
        }
        SfDtype::F8_E4M3 => Ok(e4m3fn_slice_to_f32(bytes)),
        SfDtype::F8_E5M2 => Ok(e5m2_slice_to_f32(bytes)),
        other => Err(WeightError::UnsupportedDtype {
            key: String::new(),
            dtype: format!("{other:?}"),
        }),
    }
}

/// Decode a scalar (0-D or 1-element) tensor to f32.
fn decode_scalar_f32(bytes: &[u8], dtype: SfDtype) -> Result<f32, WeightError> {
    let v = decode_to_f32(bytes, dtype, bytes.len())?;
    Ok(v.first().copied().unwrap_or(1.0_f32))
}

/// Encode f32 slice to raw bytes.
fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    cast_slice::<f32, u8>(data).to_vec()
}

/// Convert `usize` to `f32`.  Safe for `LoRA` ranks which are always < 2^24.
fn usize_to_f32(val: usize) -> f32 {
    // LoRA rank << 2^24; u32::try_from avoids precision loss for pathological
    // values; u32→f32 cast may lose precision for values > 16_777_216 but
    // those are never valid LoRA ranks.
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    u32::try_from(val).map_or(f32::MAX, |v| v as f32)
}

/// Dense matrix multiply: `result[rows × cols] = scale * lhs[rows × inner] @ rhs[inner × cols]`.
/// Row-major layout.  `lhs` is `B` from the `LoRA` formula; `rhs` is `A`.
fn matmul_scaled(
    lhs: &[f32],
    rows: usize,
    inner: usize,
    rhs: &[f32],
    cols: usize,
    scale: f32,
) -> Vec<f32> {
    let total = rows.saturating_mul(cols);
    let mut out = vec![0.0f32; total];
    for row in 0..rows {
        for mid in 0..inner {
            let bv = lhs
                .get(row.saturating_mul(inner).saturating_add(mid))
                .copied()
                .unwrap_or(0.0_f32);
            for col in 0..cols {
                let av = rhs
                    .get(mid.saturating_mul(cols).saturating_add(col))
                    .copied()
                    .unwrap_or(0.0_f32);
                if let Some(slot) = out.get_mut(row.saturating_mul(cols).saturating_add(col)) {
                    *slot = bv.mul_add(av, *slot);
                }
            }
        }
    }
    for slot in &mut out {
        *slot = scale.mul_add(*slot, 0.0_f32);
    }
    out
}
