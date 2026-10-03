//! NVIDIA device memory via [`nvml_wrapper`].
//!
//! Enable the `nvml` cargo feature to use this backend.  On systems without an
//! NVIDIA GPU, or when the `nvml` feature is not enabled, use [`FixedBudget`]
//! instead.
//!
//! [`FixedBudget`]: crate::FixedBudget

use nvml_wrapper::{Device, Nvml};

use crate::{BudgetError, DeviceMemory};

/// NVIDIA GPU memory queried through NVML.
///
/// `NvmlDevice` holds a shared NVML handle and queries the first NVIDIA GPU by
/// default.  Construct it with [`NvmlDevice::first`].
pub struct NvmlDevice {
    nvml: Nvml,
    index: u32,
    name: String,
}

impl NvmlDevice {
    /// Opens the first NVIDIA GPU.
    ///
    /// # Errors
    /// Returns [`BudgetError::Device`] when NVML initialisation fails or no
    /// GPU is available.
    pub fn first() -> Result<Self, BudgetError> {
        Self::at_index(0)
    }

    /// Opens the NVIDIA GPU at the given device index.
    ///
    /// # Errors
    /// Returns [`BudgetError::Device`] when NVML initialisation fails or the
    /// device index is out of range.
    pub fn at_index(index: u32) -> Result<Self, BudgetError> {
        let nvml = Nvml::init().map_err(|e| BudgetError::Device(e.to_string()))?;
        let device = nvml
            .device_by_index(index)
            .map_err(|e| BudgetError::Device(e.to_string()))?;
        let name = device
            .name()
            .map_err(|e| BudgetError::Device(e.to_string()))?;
        Ok(Self { nvml, index, name })
    }

    fn device(&self) -> Result<Device<'_>, BudgetError> {
        self.nvml
            .device_by_index(self.index)
            .map_err(|e| BudgetError::Device(e.to_string()))
    }
}

impl DeviceMemory for NvmlDevice {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn free_bytes(&self) -> Result<u64, BudgetError> {
        let info = self
            .device()?
            .memory_info()
            .map_err(|e| BudgetError::Device(e.to_string()))?;
        Ok(info.free)
    }

    fn total_bytes(&self) -> Result<u64, BudgetError> {
        let info = self
            .device()?
            .memory_info()
            .map_err(|e| BudgetError::Device(e.to_string()))?;
        Ok(info.total)
    }
}
