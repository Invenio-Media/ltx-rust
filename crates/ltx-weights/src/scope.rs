//! Namespaced view into a [`WeightStore`][crate::WeightStore].
//!
//! A `Scope` prepends a dot-joined prefix to every tensor lookup so model
//! modules can request weights by local name without knowing their absolute
//! key.

use burn::tensor::{Tensor, backend::Backend};

use crate::{error::WeightError, store::WeightStore};

/// A scoped view of a [`WeightStore`] with a dot-joined key prefix.
///
/// Created by [`WeightStore::scope`][crate::WeightStore::scope].
pub struct Scope<'a> {
    store: &'a WeightStore,
    /// Current prefix (may be empty for the root scope).
    prefix: String,
}

impl<'a> Scope<'a> {
    #[must_use]
    pub const fn new(store: &'a WeightStore, prefix: String) -> Self {
        Self { store, prefix }
    }

    /// Create a child scope by appending `child` with a `.` separator.
    ///
    /// If the current prefix is empty, the child prefix is just `child`.
    #[must_use]
    pub fn scope(&self, child: &str) -> Scope<'_> {
        let new_prefix = if self.prefix.is_empty() {
            child.to_owned()
        } else {
            format!("{}.{child}", self.prefix)
        };
        Scope::new(self.store, new_prefix)
    }

    /// Whether `name` (relative to this scope's prefix) is in the store.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.store.contains(&self.full_key(name))
    }

    /// Load the tensor at `name` (relative to this scope's prefix) onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::MissingKey`] or [`WeightError::RankMismatch`].
    pub fn tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, WeightError> {
        let key = self.full_key(name);
        let host = self.store.read(&key).map_err(|e| match e {
            WeightError::MissingKey(_) => WeightError::MissingKey(key.clone()),
            other => other,
        })?;
        host.into_tensor::<B, D>(device).map_err(|e| match e {
            WeightError::RankMismatch { got, expected, .. } => {
                WeightError::RankMismatch { key, got, expected }
            }
            other => other,
        })
    }

    /// Load the tensor at `name` if present, or return `None`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::RankMismatch`] if the key exists but has the
    /// wrong rank.
    pub fn optional<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Option<Tensor<B, D>>, WeightError> {
        if !self.contains(name) {
            return Ok(None);
        }
        self.tensor::<B, D>(name, device).map(Some)
    }

    fn full_key(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }
}
