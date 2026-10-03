//! JSON cache for fitted memory models.
//!
//! The cache is stored in a single JSON file as an array of `{key, models}`
//! objects.  Reads are instantaneous (the file is loaded on construction).
//! Writes are atomic: the new content is written to a temporary file in the
//! same directory and then renamed over the target.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{BudgetError, MemoryModel};

// ── CacheKey ──────────────────────────────────────────────────────────────────

/// The key that identifies one calibrated model entry in the cache.
///
/// All fields should be stable across process restarts for cache hits to be
/// reliable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Name reported by [`DeviceMemory::device_name`].
    ///
    /// [`DeviceMemory::device_name`]: crate::DeviceMemory::device_name
    pub device_name: String,
    /// Data type used for inference (e.g. `"bf16"`, `"fp8"`).
    pub dtype: String,
    /// Offload mode (e.g. `"none"`, `"cpu"`, `"disk"`).
    pub offload_mode: String,
    /// Backend identifier (e.g. `"python"`, `"burn-cuda"`).
    pub backend_id: String,
    /// Model identifier (e.g. a safetensors hash or a model tag).
    pub model_id: String,
}

// ── CachedModels ─────────────────────────────────────────────────────────────

/// The payload stored under one [`CacheKey`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedModels {
    /// Quadratic `DiT` peak-memory model (x = sequence tokens).
    pub dit: MemoryModel,
    /// Optional linear VAE peak-memory model (x = tile pixels).
    pub vae: Option<MemoryModel>,
    /// Raw samples used to fit the `DiT` model: `(tokens, peak_bytes)`.
    pub dit_samples: Vec<(f64, u64)>,
    /// Raw samples used to fit the VAE model: `(pixels, peak_bytes)`.
    pub vae_samples: Vec<(f64, u64)>,
}

// ── On-disk format ────────────────────────────────────────────────────────────

/// One entry in the on-disk JSON array.
///
/// Wrapping key + models in an object avoids the `serde_json` restriction that
/// map keys must be JSON strings.
#[derive(Serialize, Deserialize)]
struct CacheEntry {
    key: CacheKey,
    models: CachedModels,
}

/// The on-disk JSON representation of the whole cache.
#[derive(Serialize, Deserialize)]
struct CacheFile {
    entries: Vec<CacheEntry>,
}

// ── Cache ─────────────────────────────────────────────────────────────────────

/// An in-memory view of the JSON cache file.
///
/// Construct with [`Cache::load`] to read an existing file, or use
/// [`Cache::default`] for an empty cache and call [`Cache::store`] to create
/// the file.
#[derive(Debug, Clone, Default)]
pub struct Cache {
    entries: HashMap<CacheKey, CachedModels>,
}

impl Cache {
    fn to_file(&self) -> CacheFile {
        CacheFile {
            entries: self
                .entries
                .iter()
                .map(|(k, v)| CacheEntry {
                    key: k.clone(),
                    models: v.clone(),
                })
                .collect(),
        }
    }

    fn from_file(f: CacheFile) -> Self {
        Self {
            entries: f.entries.into_iter().map(|e| (e.key, e.models)).collect(),
        }
    }

    /// Loads the cache from `path`.
    ///
    /// If the file does not exist, an empty cache is returned (not an error).
    ///
    /// # Errors
    /// Returns [`BudgetError::Io`] or [`BudgetError::Json`] when the file
    /// exists but cannot be read or parsed.
    pub fn load(path: &Path) -> Result<Self, BudgetError> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let f: CacheFile = serde_json::from_str(&text)?;
                Ok(Self::from_file(f))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(BudgetError::from(e)),
        }
    }

    /// Returns the entry for `key`, if present.
    #[must_use]
    pub fn get(&self, key: &CacheKey) -> Option<&CachedModels> {
        self.entries.get(key)
    }

    /// Inserts or replaces the entry for `key`.
    pub fn insert(&mut self, key: CacheKey, models: CachedModels) {
        self.entries.insert(key, models);
    }

    /// Removes the entry for `key`.  Returns whether a value was removed.
    pub fn remove(&mut self, key: &CacheKey) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Writes the cache to `path` atomically (temp file + rename).
    ///
    /// The parent directory is created if it does not exist.  If the file
    /// already exists it is replaced atomically so readers never see a partial
    /// write.
    ///
    /// # Errors
    /// Returns [`BudgetError::Io`] or [`BudgetError::Json`] on failure.
    pub fn store(&self, path: &Path) -> Result<(), BudgetError> {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(dir)?;

        let text = serde_json::to_string_pretty(&self.to_file())?;

        // Write to a temp file in the same directory, then persist (rename)
        // atomically.  `persist` cleans up the temp file if the rename fails,
        // unlike a manual `keep() + rename` pair.
        // Two concurrent writers can race here; concurrent use is a known
        // limitation documented in the module comment.
        let mut tmp = tempfile::Builder::new()
            .suffix(".tmp")
            .tempfile_in(dir)
            .map_err(BudgetError::Io)?;

        tmp.write_all(text.as_bytes())?;
        tmp.flush()?;
        tmp.persist(path).map_err(|e| BudgetError::Io(e.error))?;

        Ok(())
    }

    /// Convenience: load → update → store in one call.
    ///
    /// If the file does not exist, a new cache is created.
    ///
    /// # Errors
    /// See [`Cache::load`] and [`Cache::store`].
    pub fn upsert(path: &Path, key: CacheKey, models: CachedModels) -> Result<(), BudgetError> {
        let mut cache = Self::load(path)?;
        cache.insert(key, models);
        cache.store(path)
    }

    /// Returns the path to the default cache file inside `cache_dir`.
    #[must_use]
    pub fn default_path(cache_dir: &Path) -> PathBuf {
        cache_dir.join("ltx_budget_cache.json")
    }
}
