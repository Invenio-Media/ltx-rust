//! [`MemoryModel`] and calibration helpers.
//!
//! The `DiT` model is quadratic in sequence tokens:
//! `peak(tokens) = resident + linear·tokens + quadratic·tokens²`.
//!
//! The VAE model is linear in tile pixels:
//! `peak(pixels) = resident + linear·pixels` (quadratic = 0).
//!
//! Both use the same [`MemoryModel`] type; the quadratic coefficient is zero
//! for the VAE model.  One `DiT` calibration at a fixed target resolution
//! generalises to all other resolutions because the model's x-axis is sequence
//! tokens, not raw frame count.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use ltx_shape::{IcLoraLayout, PixelShape, ScaleFactors, ShapeError};

pub use crate::fit::FitError;
use crate::fit::{self, FitResult, f64_of_u64};

// ── MemoryModel ───────────────────────────────────────────────────────────────

/// Parametric peak-memory model.
///
/// For the `DiT` pass: `peak(tokens) = resident + linear·tokens + quadratic·tokens²`.
/// For the VAE pass: `peak(pixels) = resident + linear·pixels` (quadratic = 0).
///
/// All values are in bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryModel {
    /// Baseline resident bytes when x = 0 (model weights, framework overhead).
    pub resident: f64,
    /// Bytes per unit x (token or pixel).
    pub linear: f64,
    /// Bytes per unit x².  Zero for linear fits (VAE model).
    pub quadratic: f64,
    /// Root-mean-square fit error in bytes.
    pub residual: f64,
}

impl MemoryModel {
    /// Predicted peak bytes at the given x (token count or pixel count).
    #[must_use]
    pub fn peak(&self, x: f64) -> f64 {
        self.quadratic
            .mul_add(x * x, self.linear.mul_add(x, self.resident))
    }

    /// Fits a quadratic model from `(x, peak_bytes)` samples.
    ///
    /// See [`fit::fit_quadratic`] for the constraint and refit rules.
    ///
    /// # Errors
    /// Returns [`FitError`] when the samples are degenerate or non-finite.
    pub fn fit_quadratic(samples: &[(f64, u64)]) -> Result<Self, FitError> {
        let FitResult {
            resident,
            linear,
            quadratic,
            residual,
        } = fit::fit_quadratic(samples)?;
        Ok(Self {
            resident,
            linear,
            quadratic,
            residual,
        })
    }

    /// Fits a linear model from `(x, peak_bytes)` samples.
    ///
    /// # Errors
    /// Returns [`FitError`] when the samples are degenerate or non-finite.
    pub fn fit_linear(samples: &[(f64, u64)]) -> Result<Self, FitError> {
        let FitResult {
            resident,
            linear,
            quadratic,
            residual,
        } = fit::fit_linear(samples)?;
        Ok(Self {
            resident,
            linear,
            quadratic,
            residual,
        })
    }
}

// ── CalibrationError ─────────────────────────────────────────────────────────

/// Errors produced by [`calibrate`] or [`calibrate_vae`].
#[derive(Debug, Error)]
pub enum CalibrationError<E> {
    /// The probe closure returned an error.
    #[error("probe returned an error")]
    Probe(E),
    /// An invalid shape was passed to the probe.
    #[error("invalid shape: {0}")]
    Shape(#[from] ShapeError),
    /// The least-squares fit failed.
    #[error("fit failed: {0}")]
    Fit(#[from] FitError),
}

// ── DiT calibration ───────────────────────────────────────────────────────────

/// Default `DiT` calibration frame counts.
const DEFAULT_FRAME_COUNTS: [u32; 3] = [17, 33, 49];

/// Calibrates the `DiT` peak-memory model by varying the frame count at a fixed
/// resolution.
///
/// The probe is called once per frame count in `frame_counts` (default
/// `[17, 33, 49]`).  Each call must return the measured peak GPU bytes for a
/// `DiT` forward pass at the given shape.
///
/// Because the model's x-axis is the total sequence-token count (including
/// both target and reference tokens from `layout`), one calibration at the
/// target resolution predicts memory use at any other resolution or tile size
/// without recalibration.
///
/// # Errors
/// Returns [`CalibrationError`] when the probe fails, the shapes are invalid,
/// or the least-squares fit is degenerate.
pub fn calibrate<E>(
    width: u32,
    height: u32,
    layout: IcLoraLayout,
    probe: &mut impl FnMut(PixelShape) -> Result<u64, E>,
    frame_counts: Option<&[u32]>,
) -> Result<MemoryModel, CalibrationError<E>> {
    let counts = frame_counts.unwrap_or(&DEFAULT_FRAME_COUNTS);
    let scale = ScaleFactors::LTX2;

    let mut samples: Vec<(f64, u64)> = Vec::with_capacity(counts.len());

    for &frames in counts {
        let shape =
            PixelShape::new(frames, height, width, scale).map_err(CalibrationError::Shape)?;
        let tokens = layout
            .sequence_tokens(shape)
            .map_err(CalibrationError::Shape)?
            .total()
            .map_err(CalibrationError::Shape)?;
        let peak_bytes = probe(shape).map_err(CalibrationError::Probe)?;
        samples.push((f64_of_u64(tokens), peak_bytes));
    }

    MemoryModel::fit_quadratic(&samples).map_err(CalibrationError::Fit)
}

// ── VAE calibration ───────────────────────────────────────────────────────────

/// Calibrates the VAE peak-memory model by varying the tile size at a fixed
/// frame count.
///
/// The probe receives different `(width, height)` tile shapes with the given
/// `frames` and must return the measured peak GPU bytes for one VAE decode
/// window.  At least three distinct tile sizes (distinct pixel counts) are
/// required.
///
/// The x-axis of the returned model is the tile pixel count `width × height`.
///
/// # Errors
/// Returns [`CalibrationError`] when the probe fails, the shapes are invalid,
/// or the least-squares fit is degenerate.
pub fn calibrate_vae<E>(
    frames: u32,
    tile_sizes: &[(u32, u32)],
    probe: &mut impl FnMut(PixelShape) -> Result<u64, E>,
) -> Result<MemoryModel, CalibrationError<E>> {
    let scale = ScaleFactors::LTX2;

    let mut samples: Vec<(f64, u64)> = Vec::with_capacity(tile_sizes.len());

    for &(width, height) in tile_sizes {
        let shape =
            PixelShape::new(frames, height, width, scale).map_err(CalibrationError::Shape)?;
        let pixel_count = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(ShapeError::Overflow)
            .map_err(CalibrationError::Shape)?;
        let peak_bytes = probe(shape).map_err(CalibrationError::Probe)?;
        samples.push((f64_of_u64(pixel_count), peak_bytes));
    }

    MemoryModel::fit_linear(&samples).map_err(CalibrationError::Fit)
}
