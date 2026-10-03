//! Apple Silicon GPU memory via [`objc2_metal`].
//!
//! Enable the `metal` cargo feature on macOS to use this backend.
//! The free-memory estimate is:
//! `recommendedMaxWorkingSetSize − currentAllocatedSize`.
//!
//! On Apple Silicon the GPU and CPU share physical RAM.
//! `recommendedMaxWorkingSetSize` is the fraction Metal will use efficiently;
//! `currentAllocatedSize` is what has already been handed out.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use crate::{BudgetError, DeviceMemory};

/// Apple Silicon GPU memory queried through Metal.
///
/// Construct with [`MetalDevice::system_default`].
pub struct MetalDevice {
    name: String,
    total: u64,
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
        Ok(Self { name, total })
    }

    /// Returns the current byte count already allocated on the device.
    fn current_allocated() -> Result<u64, BudgetError> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| BudgetError::Device("no system-default Metal device".into()))?;
        // currentAllocatedSize returns NSUInteger = usize on arm64.
        let used: usize = device.currentAllocatedSize();
        // On arm64 usize is 64-bit; TryFrom always succeeds.
        u64::try_from(used).map_err(|e| BudgetError::Device(e.to_string()))
    }
}

impl DeviceMemory for MetalDevice {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn free_bytes(&self) -> Result<u64, BudgetError> {
        let used = Self::current_allocated()?;
        Ok(self.total.saturating_sub(used))
    }

    fn total_bytes(&self) -> Result<u64, BudgetError> {
        Ok(self.total)
    }
}
