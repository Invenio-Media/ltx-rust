//! Apple Silicon GPU memory via [`objc2_metal`].
//!
//! Enable the `metal` cargo feature on macOS to use this backend.
//! The free-memory estimate is:
//! `recommendedMaxWorkingSetSize − currentAllocatedSize`.
//!
//! ## Limitations
//!
//! `currentAllocatedSize` counts only the Metal allocations of the current
//! process.  On unified memory systems other processes and the OS kernel are
//! not reflected, so `free_bytes()` overestimates available memory when the
//! system is under pressure.  Pass a `FixedBudget` instead if you want a
//! conservative bound; consult `recommendedMaxWorkingSetSize` from Activity
//! Monitor for a reasonable ceiling.
//!
//! Concurrent writes to the same cache path are not protected by a lock.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use crate::{BudgetError, DeviceMemory};

/// Apple Silicon GPU memory queried through Metal.
///
/// Construct with [`MetalDevice::system_default`].
pub struct MetalDevice {
    /// Stable human-readable device name.
    name: String,
    /// Total recommended working-set size in bytes.
    total: u64,
    /// Retained reference to the Metal device.  Kept alive so that
    /// `current_allocated_size` always queries the same device, and to avoid
    /// the overhead of a new system-default lookup on every `free_bytes` call.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalDevice {
    /// Opens the system-default Metal device.
    ///
    /// # Errors
    /// Returns [`BudgetError::Device`] when no Metal device is available.
    pub fn system_default() -> Result<Self, BudgetError> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| BudgetError::Device("no system-default Metal device".into()))?;
        let name = device.name().to_string();
        let total = device.recommendedMaxWorkingSetSize();
        Ok(Self {
            name,
            total,
            device,
        })
    }
}

impl DeviceMemory for MetalDevice {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn free_bytes(&self) -> Result<u64, BudgetError> {
        // currentAllocatedSize returns NSUInteger = usize (64-bit on arm64).
        let used: usize = self.device.currentAllocatedSize();
        let used_u64 = u64::try_from(used).map_err(|e| BudgetError::Device(e.to_string()))?;
        Ok(self.total.saturating_sub(used_u64))
    }

    fn total_bytes(&self) -> Result<u64, BudgetError> {
        Ok(self.total)
    }
}
