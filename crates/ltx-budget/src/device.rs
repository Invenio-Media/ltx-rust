//! [`DeviceMemory`] trait and built-in implementations.

use crate::BudgetError;

/// Reports the memory capacity of a compute device.
///
/// The three values — device name, free bytes, total bytes — together provide
/// everything the solver needs.  Implement this trait to plug in any device
/// backend (NVML, Metal, simulated budget, …).
pub trait DeviceMemory {
    /// A stable human-readable name used as part of the cache key.
    fn device_name(&self) -> &str;

    /// Bytes not yet committed to any allocation.
    ///
    /// # Errors
    /// Returns [`BudgetError::Device`] when the underlying query fails.
    fn free_bytes(&self) -> Result<u64, BudgetError>;

    /// Total bytes on the device.
    ///
    /// # Errors
    /// Returns [`BudgetError::Device`] when the underlying query fails.
    fn total_bytes(&self) -> Result<u64, BudgetError>;
}

// ── NVML ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "nvml")]
mod nvml_impl;

#[cfg(feature = "nvml")]
pub use nvml_impl::NvmlDevice;

// ── Metal ────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_impl;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_impl::MetalDevice;

// ── Fixed budget ─────────────────────────────────────────────────────────────

/// A caller-supplied budget; always available regardless of platform or
/// feature flags.  Use this for the CLI's `--vram-gb` flag.
#[derive(Debug, Clone)]
pub struct FixedBudget {
    name: String,
    free: u64,
    total: u64,
}

impl FixedBudget {
    /// Creates a budget with the given free and total byte counts.
    #[must_use]
    pub fn new(name: impl Into<String>, free: u64, total: u64) -> Self {
        Self {
            name: name.into(),
            free,
            total,
        }
    }

    /// Creates a budget where both free and total equal `gib` gibibytes.
    ///
    /// Use this for the CLI's `--vram-gb` argument (pass integer or
    /// truncated value).
    #[must_use]
    pub fn from_gib(name: impl Into<String>, gib: u32) -> Self {
        const GIB: u64 = 1_073_741_824;
        let bytes = u64::from(gib).saturating_mul(GIB);
        Self::new(name, bytes, bytes)
    }
}

impl DeviceMemory for FixedBudget {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn free_bytes(&self) -> Result<u64, BudgetError> {
        Ok(self.free)
    }

    fn total_bytes(&self) -> Result<u64, BudgetError> {
        Ok(self.total)
    }
}
