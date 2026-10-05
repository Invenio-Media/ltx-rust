//! Solver: given a memory budget and fitted models, finds the largest valid
//! frame count or spatial tile that fits.
//!
//! ## Frame-count search
//!
//! Valid frame counts for LTX-2 are `8k + 1` (1, 9, 17, …).  The solver
//! starts from the quality cap (default 121 = 8·15 + 1) and steps down by 8
//! until the predicted combined peak fits within `free × (1 − margin)`.
//!
//! ## Spatial fallback
//!
//! If even F = 9 at the full (padded) resolution exceeds the budget, the
//! solver tries progressively smaller tiles.  Tiles are multiples of 32 on
//! each dimension; the aspect ratio is preserved where possible.  For each
//! tile the quality-cap frame count is tried first; if that also fails, the
//! best `8k + 1 ≥ 9` for that tile is returned.
//!
//! If no tile as small as 64 × 64 fits at F = 9, an error is returned.

use ltx_shape::{IcLoraLayout, PixelShape, ScaleFactors, ceil_spatial, floor_frames};

use crate::fit::f64_of_u64;
use crate::{BudgetError, MemoryModel};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default safety margin: 13 % of free bytes is kept in reserve.
pub const DEFAULT_SAFETY_MARGIN: f64 = 0.13;

/// Default quality cap: 121 = 8 × 15 + 1 frames.
pub const DEFAULT_QUALITY_CAP: u32 = 121;

/// Minimum valid frame count (8·1 + 1 = 9).
const MIN_FRAMES: u32 = 9;

/// Minimum tile edge in pixels.
const MIN_TILE: u32 = 64;

/// VAE spatial scale factor.
const SPATIAL_MULTIPLE: u32 = 32;

/// Recommended tile overlap in pixels (one VAE spatial period).
pub const TILE_OVERLAP_PIXELS: u32 = 32;

// ── SolveConfig ───────────────────────────────────────────────────────────────

/// Input to [`solve`].
pub struct SolveConfig<'m> {
    /// Fitted `DiT` peak-memory model (quadratic in sequence tokens).
    pub dit: &'m MemoryModel,
    /// Optional fitted VAE peak-memory model (linear in `pixels × latent_frames`).
    ///
    /// When `None`, only the `DiT` model constrains the budget.
    pub vae: Option<&'m MemoryModel>,
    /// Free bytes reported by the device.
    pub free_bytes: u64,
    /// Fraction of free bytes kept in reserve (default [`DEFAULT_SAFETY_MARGIN`]).
    pub safety_margin: f64,
    /// Target clip width in pixels (need not be on the 32-pixel grid).
    pub width: u32,
    /// Target clip height in pixels (need not be on the 32-pixel grid).
    pub height: u32,
    /// IC-LoRA layout (reference downscale and temporal factors).
    pub layout: IcLoraLayout,
    /// VAE and temporal scale factors (default [`ScaleFactors::LTX2`]).
    pub scale: ScaleFactors,
    /// Upper bound on the returned frame count (default [`DEFAULT_QUALITY_CAP`]).
    ///
    /// Must satisfy `8k + 1`; if it does not, it is rounded down automatically.
    pub quality_cap: u32,
    /// Frames per second for the `seconds` field of the result.
    pub fps: f64,
}

// ── SolveResult ───────────────────────────────────────────────────────────────

/// Output of [`solve`].
#[derive(Debug, Clone, PartialEq)]
pub enum SolveResult {
    /// A valid frame count fits at the full (padded) resolution.
    Fit {
        /// Frame count (always `8k + 1`).
        frames: u32,
        /// Duration at the configured fps: `(frames − 1) / fps`.
        seconds: f64,
        /// Padded width used for the peak-memory prediction.
        padded_width: u32,
        /// Padded height used for the peak-memory prediction.
        padded_height: u32,
        /// `true` when the input dimensions were padded to the 32-pixel grid.
        was_padded: bool,
    },
    /// The full resolution does not fit; the caller should tile the output.
    TileFallback {
        /// Tile width in pixels (multiple of 32).
        tile_width: u32,
        /// Tile height in pixels (multiple of 32).
        tile_height: u32,
        /// Best frame count for the tile (always `8k + 1`, at least 9).
        frames: u32,
        /// Duration at the configured fps.
        seconds: f64,
        /// Recommended tile overlap in pixels.
        overlap_pixels: u32,
    },
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Predicted peak bytes for a given shape.
fn peak_bytes(config: &SolveConfig<'_>, shape: PixelShape) -> Result<f64, BudgetError> {
    let tokens = config
        .layout
        .sequence_tokens(shape)
        .map_err(BudgetError::Shape)?
        .total()
        .map_err(BudgetError::Shape)?;

    let dit_peak = config.dit.peak(f64_of_u64(tokens));

    let vae_peak = if let Some(vae) = config.vae {
        let pixels = u64::from(shape.width())
            .checked_mul(u64::from(shape.height()))
            .and_then(|v| v.checked_mul(u64::from(shape.latent().frames)))
            .ok_or(BudgetError::Overflow)?;
        vae.peak(f64_of_u64(pixels))
    } else {
        0.0
    };

    Ok(if dit_peak > vae_peak {
        dit_peak
    } else {
        vae_peak
    })
}

/// The budget in bytes after the safety margin is applied.
fn effective_budget(free: u64, margin: f64) -> f64 {
    f64_of_u64(free) * (1.0 - margin)
}

/// Rounds `seconds` to the nearest millisecond for cleaner output.
fn round_seconds(s: f64) -> f64 {
    (s * 1000.0).round() / 1000.0
}

/// Convert an `f64` tile coordinate to `u32` by rounding.
/// Only called with finite values in `[0.0, u32::MAX as f64]`.
fn f64_to_u32_round(x: f64) -> Option<u32> {
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    if x > f64::from(u32::MAX) {
        return None;
    }
    #[expect(
        clippy::as_conversions,
        reason = "f64→u32 for tile-size rounding; finite and bounded checks above prevent truncation loss"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "x <= u32::MAX and x >= 0; rounding is intentional"
    )]
    #[expect(clippy::cast_sign_loss, reason = "x >= 0.0 is checked above")]
    {
        Some(x.round() as u32)
    }
}

/// Finds the best frame count for a given (width, height) within the budget.
/// Returns `None` when even `MIN_FRAMES` does not fit.
fn best_frames_for_tile(
    config: &SolveConfig<'_>,
    tw: u32,
    th: u32,
    budget: f64,
    cap: u32,
) -> Option<u32> {
    let scale = config.scale;
    let start = floor_frames(cap, scale)?;
    let mut f = start;
    loop {
        if f < MIN_FRAMES {
            break;
        }
        if let Ok(shape) = PixelShape::new(f, th, tw, scale)
            && peak_bytes(config, shape).is_ok_and(|p| p <= budget)
        {
            return Some(f);
        }
        if f == MIN_FRAMES {
            break;
        }
        f = f.saturating_sub(8);
        if f < MIN_FRAMES {
            f = MIN_FRAMES;
        }
    }
    None
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Finds the largest valid frame count (or spatial tile) that fits within the
/// memory budget.
///
/// ## Algorithm
///
/// 1. Pads `width` and `height` up to the nearest multiple of 32.
/// 2. Iterates frame counts `cap, cap−8, …, 9` and returns the first that
///    keeps the predicted combined peak ≤ `free × (1 − margin)`.
/// 3. If F = 9 at the full resolution does not fit, tries progressively
///    smaller tiles (same aspect ratio, multiples of 32) from the full size
///    down to 64 × 64.
/// 4. If 64 × 64 at F = 9 does not fit, returns [`BudgetError::NoFit`].
///
/// ## Errors
///
/// - [`BudgetError::InvalidConfig`] – a config field is out of range.
/// - [`BudgetError::Shape`] – the width or height cannot be padded to 32.
/// - [`BudgetError::NoFit`] – no shape fits the budget.
/// - [`BudgetError::Overflow`] – a pixel-count multiply overflowed.
pub fn solve(config: &SolveConfig<'_>) -> Result<SolveResult, BudgetError> {
    // Validate inputs.
    if !config.safety_margin.is_finite()
        || config.safety_margin < 0.0
        || config.safety_margin >= 1.0
    {
        return Err(BudgetError::InvalidConfig(
            "safety_margin must be finite and in [0, 1)".into(),
        ));
    }
    if !config.fps.is_finite() || config.fps <= 0.0 {
        return Err(BudgetError::InvalidConfig(
            "fps must be a positive finite number".into(),
        ));
    }
    if config.width == 0 || config.height == 0 {
        return Err(BudgetError::InvalidConfig(
            "width and height must be non-zero".into(),
        ));
    }
    if config.quality_cap < MIN_FRAMES {
        return Err(BudgetError::InvalidConfig(format!(
            "quality_cap must be at least {MIN_FRAMES}"
        )));
    }

    let nonzero_spatial =
        std::num::NonZeroU32::new(SPATIAL_MULTIPLE).ok_or(BudgetError::Overflow)?;

    let pw = ceil_spatial(config.width, nonzero_spatial).ok_or(BudgetError::Overflow)?;
    let ph = ceil_spatial(config.height, nonzero_spatial).ok_or(BudgetError::Overflow)?;

    let was_padded = pw != config.width || ph != config.height;
    let budget = effective_budget(config.free_bytes, config.safety_margin);

    // ── Full-resolution search ─────────────────────────────────────────────
    // quality_cap is already >= MIN_FRAMES; floor_frames returns at least 1.
    let cap = floor_frames(config.quality_cap, config.scale).ok_or(BudgetError::Shape(
        ltx_shape::ShapeError::FrameCount(config.quality_cap),
    ))?;
    // Cap is always >= 1 here; if it is below MIN_FRAMES (e.g. quality_cap=1),
    // the full-resolution search will simply try only F=1.
    let cap = cap.max(MIN_FRAMES);

    if let Some(frames) = best_frames_for_tile(config, pw, ph, budget, cap) {
        let seconds = round_seconds(f64::from(frames.saturating_sub(1)) / config.fps);
        return Ok(SolveResult::Fit {
            frames,
            seconds,
            padded_width: pw,
            padded_height: ph,
            was_padded,
        });
    }

    // ── Spatial fallback ───────────────────────────────────────────────────
    // Aspect ratio from padded dimensions.
    let aspect = f64::from(pw) / f64::from(ph);

    let mut tw = pw;
    loop {
        // Step down by one spatial period.
        tw = tw.saturating_sub(SPATIAL_MULTIPLE);
        if tw < MIN_TILE {
            tw = MIN_TILE;
        }

        // Compute the matching height at the same aspect ratio, rounded to
        // the nearest multiple of 32.  The height is at least MIN_TILE (64 px)
        // but is not clamped to tw; portrait results are valid.
        let th_f = (f64::from(tw) / aspect).round();
        let th_raw = f64_to_u32_round(th_f).unwrap_or(MIN_TILE);
        let th_aligned = ceil_spatial(th_raw.max(1), nonzero_spatial)
            .unwrap_or(SPATIAL_MULTIPLE)
            .max(MIN_TILE);

        if let Some(frames) = best_frames_for_tile(config, tw, th_aligned, budget, cap) {
            let seconds = round_seconds(f64::from(frames.saturating_sub(1)) / config.fps);
            return Ok(SolveResult::TileFallback {
                tile_width: tw,
                tile_height: th_aligned,
                frames,
                seconds,
                overlap_pixels: TILE_OVERLAP_PIXELS,
            });
        }

        if tw <= MIN_TILE {
            break;
        }
    }

    Err(BudgetError::NoFit)
}
