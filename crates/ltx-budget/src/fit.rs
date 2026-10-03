//! Least-squares fitting for the [`MemoryModel`] coefficients.
//!
//! All arithmetic uses `f64`.  The normal equations are solved with Cramer's
//! rule via the `minor2` helper, which avoids any array indexing and satisfies
//! the `suboptimal_flops` lint throughout.
//!
//! [`MemoryModel`]: crate::MemoryModel

use thiserror::Error;

/// Errors that occur during a least-squares fit.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FitError {
    /// Fewer than 3 distinct x-values were provided.
    #[error("need at least 3 distinct x values, got {0}")]
    TooFewDistinct(usize),
    /// At least one sample contains a non-finite value.
    #[error("non-finite value in samples")]
    NonFinite,
    /// The normal-equations matrix is singular (degenerate input).
    #[error("degenerate system: normal-equations matrix is singular")]
    Singular,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Converts `u64` to `f64`.
///
/// Precision is exact up to 2^53 (~9 PiB). Values above that are rounded to
/// the nearest representable `f64`, which is immaterial for memory budgeting.
#[inline]
pub(crate) const fn f64_of_u64(n: u64) -> f64 {
    #[expect(
        clippy::as_conversions,
        reason = "u64→f64: values up to ~9 PiB are represented exactly; \
                  rounding above that is immaterial for memory budgeting"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "u64→f64: precision loss above 2^53 bytes (~9 PiB) is immaterial"
    )]
    {
        n as f64
    }
}

/// Converts `usize` to `f64`.
///
/// Used only for sample counts, which never exceed a few hundred.
#[inline]
fn f64_of_usize(n: usize) -> f64 {
    // Sample counts are always small; u32 is more than enough.
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

// ── Validation ───────────────────────────────────────────────────────────────

/// Checks that `samples` contains at least `min_distinct` distinct finite
/// x-values.
///
/// # Errors
/// - [`FitError::NonFinite`] if any x is not finite.
/// - [`FitError::TooFewDistinct`] if fewer than `min_distinct` distinct
///   x-values are present.
pub(crate) fn validate(samples: &[(f64, u64)], min_distinct: usize) -> Result<(), FitError> {
    let mut distinct: Vec<u64> = Vec::new();
    for &(x, _) in samples {
        if !x.is_finite() {
            return Err(FitError::NonFinite);
        }
        // Compare by bit pattern: for integer-valued f64 (token / pixel
        // counts) this is an exact equality test.
        let bits = x.to_bits();
        if !distinct.contains(&bits) {
            distinct.push(bits);
        }
    }
    if distinct.len() < min_distinct {
        return Err(FitError::TooFewDistinct(distinct.len()));
    }
    Ok(())
}

// ── 2×2 minor helper ─────────────────────────────────────────────────────────

/// Computes the 2×2 determinant `a·d − b·c` using a fused multiply-add.
#[inline]
fn minor2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    // a·d + b·(−c)
    b.mul_add(-c, a * d)
}

// ── Sums ─────────────────────────────────────────────────────────────────────

/// Accumulators for the normal equations.
struct Sums {
    n: f64,   // sample count
    s1: f64,  // Σ x
    s2: f64,  // Σ x²
    s3: f64,  // Σ x³
    s4: f64,  // Σ x⁴
    sy: f64,  // Σ y
    s1y: f64, // Σ x·y
    s2y: f64, // Σ x²·y
}

impl Sums {
    fn from_samples(samples: &[(f64, u64)]) -> Self {
        let mut n = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut s4 = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut s1y = 0.0_f64;
        let mut s2y = 0.0_f64;

        for &(x, y) in samples {
            let yf = f64_of_u64(y);
            let x2 = x * x;
            n += 1.0;
            s1 += x;
            s2 = x.mul_add(x, s2);
            s3 = x2.mul_add(x, s3);
            s4 = x2.mul_add(x2, s4);
            sy += yf;
            s1y = x.mul_add(yf, s1y);
            s2y = x2.mul_add(yf, s2y);
        }
        Self {
            n,
            s1,
            s2,
            s3,
            s4,
            sy,
            s1y,
            s2y,
        }
    }
}

// ── 3×3 Cramer solver ────────────────────────────────────────────────────────

/// Solves the 3×3 normal-equations system
/// ```text
/// [ n   s1  s2  ] [ r ]   [ sy  ]
/// [ s1  s2  s3  ] [ a ] = [ s1y ]
/// [ s2  s3  s4  ] [ b ]   [ s2y ]
/// ```
/// Returns `(resident, linear, quadratic)` or `None` when singular.
fn solve3(s: &Sums) -> Option<(f64, f64, f64)> {
    let (a00, a01, a02, d) = (s.n, s.s1, s.s2, s.sy);
    let (a10, a11, a12, e) = (s.s1, s.s2, s.s3, s.s1y);
    let (a20, a21, a22, f) = (s.s2, s.s3, s.s4, s.s2y);

    // 3×3 determinant by cofactor expansion along row 0.
    // det = a00·M₀₀ − a01·M₀₁ + a02·M₀₂
    let m00 = minor2(a11, a12, a21, a22);
    let m01 = minor2(a10, a12, a20, a22);
    let m02 = minor2(a10, a11, a20, a21);
    let det = a02.mul_add(m02, a01.mul_add(-m01, a00 * m00));

    if det.abs() < 1e-30 {
        return None;
    }

    // Cramer: replace column 0 with rhs → det0.
    let n00 = minor2(a11, a12, a21, a22);
    let n01 = minor2(e, a12, f, a22);
    let n02 = minor2(e, a11, f, a21);
    let det0 = a02.mul_add(n02, a01.mul_add(-n01, d * n00));

    // Cramer: replace column 1 with rhs → det1.
    let p00 = minor2(e, a12, f, a22);
    let p01 = minor2(a10, a12, a20, a22);
    let p02 = minor2(a10, e, a20, f);
    let det1 = a02.mul_add(p02, d.mul_add(-p01, a00 * p00));

    // Cramer: replace column 2 with rhs → det2.
    let q00 = minor2(a11, e, a21, f);
    let q01 = minor2(a10, e, a20, f);
    let q02 = minor2(a10, a11, a20, a21);
    let det2 = d.mul_add(q02, a01.mul_add(-q01, a00 * q00));

    Some((det0 / det, det1 / det, det2 / det))
}

// ── 2×2 Cramer solver ────────────────────────────────────────────────────────

/// Solves the 2×2 normal-equations system
/// ```text
/// [ n   sx  ] [ r ] = [ sy  ]
/// [ sx  sx2 ] [ c ]   [ sxy ]
/// ```
/// Returns `(resident, coeff)` or `None` when singular.
///
/// The `mul_add` form avoids `suboptimal_flops` while preserving the standard
/// 2×2 determinant formula n·Σx² − (Σx)².
fn solve2(n: f64, sx: f64, sx2: f64, sy: f64, sxy: f64) -> Option<(f64, f64)> {
    // det = n·sx2 − sx² = n·Σx² − (Σx)²
    let det = sx.mul_add(-sx, n * sx2);
    if det.abs() < 1e-30 {
        return None;
    }
    let r = sx.mul_add(-sxy, sy * sx2) / det;
    let c = sx.mul_add(-sy, n * sxy) / det;
    Some((r, c))
}

// ── Residual ─────────────────────────────────────────────────────────────────

/// Root-mean-square residual in bytes.
fn residual(samples: &[(f64, u64)], resident: f64, linear: f64, quadratic: f64) -> f64 {
    let n = f64_of_usize(samples.len());
    if n < 1.0 {
        return 0.0;
    }
    let sum_sq: f64 = samples
        .iter()
        .map(|&(x, y)| {
            // peak(x) = resident + linear·x + quadratic·x²
            let pred = quadratic.mul_add(x * x, linear.mul_add(x, resident));
            let err = pred - f64_of_u64(y);
            err * err
        })
        .sum();
    (sum_sq / n).sqrt()
}

// ── Public fitting API ───────────────────────────────────────────────────────

/// Result of a successful fit.
#[derive(Debug, Clone)]
pub(crate) struct FitResult {
    pub resident: f64,
    pub linear: f64,
    pub quadratic: f64,
    pub residual: f64,
}

/// Fits `y = resident + linear·x + quadratic·x²` from `samples`.
///
/// Requires at least 3 distinct x-values.  If the unconstrained fit yields
/// `quadratic < 0`, the model is refit with `quadratic` fixed at 0 (linear
/// regression on x).  If `linear < 0` in the unconstrained fit, the model is
/// refit with `linear` fixed at 0 (regression on x²).  When both are
/// negative, the quadratic term is dropped first (fused attention gives
/// `quadratic ≈ 0`).
///
/// # Errors
/// - [`FitError::TooFewDistinct`] – fewer than 3 distinct x-values.
/// - [`FitError::NonFinite`] – a non-finite value in `samples`.
/// - [`FitError::Singular`] – the normal-equations matrix is singular.
pub(crate) fn fit_quadratic(samples: &[(f64, u64)]) -> Result<FitResult, FitError> {
    validate(samples, 3)?;
    let s = Sums::from_samples(samples);

    let (r, a, b) = solve3(&s).ok_or(FitError::Singular)?;

    if b < 0.0 {
        // Refit as linear: y = r + a·x
        let (r2, a2) = solve2(s.n, s.s1, s.s2, s.sy, s.s1y).ok_or(FitError::Singular)?;
        let res = residual(samples, r2, a2, 0.0);
        return Ok(FitResult {
            resident: r2,
            linear: a2,
            quadratic: 0.0,
            residual: res,
        });
    }

    if a < 0.0 {
        // Refit as quadratic-only: y = r + b·x²  (use x² as predictor)
        let (r2, b2) = solve2(s.n, s.s2, s.s4, s.sy, s.s2y).ok_or(FitError::Singular)?;
        let res = residual(samples, r2, 0.0, b2);
        return Ok(FitResult {
            resident: r2,
            linear: 0.0,
            quadratic: b2,
            residual: res,
        });
    }

    let res = residual(samples, r, a, b);
    Ok(FitResult {
        resident: r,
        linear: a,
        quadratic: b,
        residual: res,
    })
}

/// Fits `y = resident + linear·x` from `samples`.
///
/// Requires at least 3 distinct x-values (same validation as the quadratic
/// variant, so the two can be compared on equal footing).
///
/// # Errors
/// See [`fit_quadratic`].
pub(crate) fn fit_linear(samples: &[(f64, u64)]) -> Result<FitResult, FitError> {
    validate(samples, 3)?;
    let s = Sums::from_samples(samples);
    let (r, c) = solve2(s.n, s.s1, s.s2, s.sy, s.s1y).ok_or(FitError::Singular)?;
    let res = residual(samples, r, c, 0.0);
    Ok(FitResult {
        resident: r,
        linear: c,
        quadratic: 0.0,
        residual: res,
    })
}
