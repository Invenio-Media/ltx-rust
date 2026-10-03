//! GPU memory budget estimation and solver for LTX-2 video generation.
//!
//! The crate fits a quadratic peak-memory model for the `DiT` pass,
//! `peak(tokens) = resident + a·tokens + b·tokens²`, and a linear model for
//! the VAE decode pass, `peak(pixels) = resident + c·pixels`. Both use the
//! same [`MemoryModel`] type.
//!
//! Given a device memory budget, [`solve`] finds the largest `8k + 1` frame
//! count that keeps the predicted peak below a safety margin of the free
//! memory. If no frame count fits, it returns the best spatial tile instead.
//!
//! Calibration runs the caller-supplied probe closure at a few frame counts
//! (default 17, 33, 49) and fits the model by least squares. Because the
//! model is in sequence tokens rather than raw frame counts, one calibration
//! at the target resolution generalises to other resolutions and tile sizes.

pub mod cache;
pub mod device;
pub mod fit;
pub mod model;
pub mod solver;

pub use cache::{Cache, CacheKey, CachedModels};
pub use device::{DeviceMemory, FixedBudget};
pub use fit::FitError;
pub use model::{CalibrationError, MemoryModel, calibrate, calibrate_vae};
pub use solver::{SolveConfig, SolveResult, solve};

#[cfg(feature = "nvml")]
pub use device::NvmlDevice;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use device::MetalDevice;

use thiserror::Error;

/// Errors produced by this crate.
#[derive(Debug, Error)]
pub enum BudgetError {
    /// The device query failed.
    #[error("device query failed: {0}")]
    Device(String),
    /// A least-squares fit failed.
    #[error("fit failed: {0}")]
    Fit(#[from] FitError),
    /// A shape was invalid.
    #[error("shape error: {0}")]
    Shape(#[from] ltx_shape::ShapeError),
    /// File I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialisation or deserialisation failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// No valid frame count or tile fits in the given budget.
    #[error("no valid frame count or tile fits within the budget")]
    NoFit,
    /// Integer overflow in shape arithmetic.
    #[error("arithmetic overflow")]
    Overflow,
}
