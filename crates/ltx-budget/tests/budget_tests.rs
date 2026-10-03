//! Integration and unit tests for `ltx-budget`.
//!
//! Tests cover:
//! - Fit recovery from synthetic samples (exact and noisy).
//! - Negative-term refit.
//! - Solver boundaries (exact budget, cap clamp, margin, `8k+1` only,
//!   spatial fallback, nothing fits).
//! - Cache round-trip and key mismatch.
//! - 1920×1080 example from the plan.

use ltx_budget::{
    BudgetError, Cache, CacheKey, CachedModels, MemoryModel, SolveConfig, SolveResult, calibrate,
    calibrate_vae, solve,
    solver::{DEFAULT_QUALITY_CAP, DEFAULT_SAFETY_MARGIN, TILE_OVERLAP_PIXELS},
};
use ltx_shape::{IcLoraLayout, PixelShape, ScaleFactors};

// ── Test helpers ─────────────────────────────────────────────────────────────

const SCALE: ScaleFactors = ScaleFactors::LTX2;
const LAYOUT: IcLoraLayout = IcLoraLayout::FULL;

/// Evaluates the parametric model at x for sample generation.
const fn peak_at(resident: f64, linear: f64, quadratic: f64, x: f64) -> f64 {
    resident + linear * x + quadratic * x * x
}

/// Converts a positive finite `f64` to `u64` for synthetic samples.
#[expect(
    clippy::as_conversions,
    reason = "test helper: value is always positive and finite"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "test: truncation of fractional bytes is intentional"
)]
#[expect(clippy::cast_sign_loss, reason = "test: value is always >= 0")]
const fn to_bytes(x: f64) -> u64 {
    x as u64
}

/// Converts `u64` to `f64` for use in test assertions and linear-model setup.
#[expect(
    clippy::as_conversions,
    reason = "test helper: u64→f64 for arithmetic in test setup"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "test: values well within 2^53 precision"
)]
const fn u64_f64(n: u64) -> f64 {
    n as f64
}

/// Converts `f64` to `u64` for expected-budget arithmetic in solver tests.
#[expect(
    clippy::as_conversions,
    reason = "test: f64→u64 for budget calculation"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "test: intentional conversion"
)]
#[expect(clippy::cast_sign_loss, reason = "test: value is always >= 0")]
const fn budget_bytes(x: f64) -> u64 {
    x as u64
}

/// Converts `u64` to `u32` (for token counts known to fit in u32 in tests).
fn tokens_u32(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Builds exact synthetic samples from a known quadratic model.
fn make_samples_exact(resident: f64, linear: f64, quadratic: f64, xs: &[f64]) -> Vec<(f64, u64)> {
    xs.iter()
        .map(|&x| (x, to_bytes(peak_at(resident, linear, quadratic, x))))
        .collect()
}

// ── Fit tests ─────────────────────────────────────────────────────────────────

#[test]
fn fit_recovers_known_coefficients_exact() {
    let resident = 1_000_000_000.0_f64; // 1 GB
    let linear = 2_500.0_f64; // 2.5 KB / token
    let quadratic = 0.0001_f64;

    let xs = [10_000.0, 30_000.0, 60_000.0, 100_000.0];
    let samples = make_samples_exact(resident, linear, quadratic, &xs);

    let model = MemoryModel::fit_quadratic(&samples).expect("fit should succeed");

    let tol = 1e-3;
    assert!(
        (model.resident - resident).abs() < resident * tol,
        "resident: got {}, expected {}",
        model.resident,
        resident
    );
    assert!(
        (model.linear - linear).abs() < linear * tol,
        "linear: got {}, expected {}",
        model.linear,
        linear
    );
    assert!(
        (model.quadratic - quadratic).abs() < quadratic * tol,
        "quadratic: got {}, expected {}",
        model.quadratic,
        quadratic
    );
    // Exact samples → residual near zero.
    assert!(
        model.residual < 1.0,
        "residual should be near zero, got {}",
        model.residual
    );
}

#[test]
fn fit_recovers_known_coefficients_noisy() {
    // Model where the quadratic term contributes similarly to the linear term
    // at the midpoint (~50 000 tokens), so the normal equations are
    // well-conditioned and the fit is noise-robust.
    let resident = 1_000_000_000.0_f64; // 1 GB resident
    let linear = 1_000.0_f64; // 1 KB / token
    let quadratic = 0.01_f64; // 10 B / token²  (≈ 100 MB at 100 k tokens)

    // Tiny deterministic perturbation (±0.01 %).
    let xs = [5_000.0, 20_000.0, 50_000.0, 80_000.0, 120_000.0];
    let noise = [0.0001, -0.0001, 0.0001, -0.0001, 0.0];
    let samples: Vec<(f64, u64)> = xs
        .iter()
        .zip(noise.iter())
        .map(|(&x, &n)| {
            let y = peak_at(resident, linear, quadratic, x) * (1.0 + n);
            (x, to_bytes(y))
        })
        .collect();

    let model = MemoryModel::fit_quadratic(&samples).expect("noisy fit should succeed");

    // Allow 2 % error; with ±0.01 % input noise and a well-conditioned system
    // the coefficient errors stay well below this.
    let tol = 0.02;
    assert!(
        (model.resident - resident).abs() < resident * tol,
        "resident: got {}, expected {} (2% tol)",
        model.resident,
        resident
    );
    assert!(
        (model.linear - linear).abs() < linear * tol,
        "linear: got {}, expected {} (2% tol)",
        model.linear,
        linear
    );
    assert!(
        (model.quadratic - quadratic).abs() < quadratic * tol,
        "quadratic: got {}, expected {} (2% tol)",
        model.quadratic,
        quadratic
    );
}

#[test]
fn fit_negative_quadratic_refits_as_linear() {
    // Exact samples from a true concave-down model (b < 0).
    // The unconstrained quadratic fit recovers the negative b, triggering the
    // refit which clamps b to 0 and fits a linear model instead.
    let resident = 1_000_000_000.0_f64;
    let linear = 3_000.0_f64;
    let true_b = -0.5_f64; // concave down

    let xs = [100.0_f64, 500.0, 1_000.0, 2_000.0];
    let samples: Vec<(f64, u64)> = xs
        .iter()
        .map(|&x| (x, to_bytes(peak_at(resident, linear, true_b, x))))
        .collect();

    let model = MemoryModel::fit_quadratic(&samples).expect("refit should succeed");
    assert!(
        model.quadratic.to_bits() == 0.0_f64.to_bits(),
        "negative-b refit must set quadratic = 0, got {}",
        model.quadratic
    );
    assert!(model.linear >= 0.0, "linear must be >= 0 after refit");
}

#[test]
fn fit_negative_linear_refits_as_quadratic_only() {
    // Exact samples from a model with a negative linear coefficient.
    // The unconstrained fit recovers the negative linear term, which triggers
    // the refit that clamps linear to 0.
    let resident = 2_000_000_000.0_f64;
    let true_linear = -500.0_f64; // negative!
    let quadratic = 200.0_f64;

    let xs = [100.0_f64, 500.0, 1_000.0, 2_000.0];
    let samples: Vec<(f64, u64)> = xs
        .iter()
        .map(|&x| (x, to_bytes(peak_at(resident, true_linear, quadratic, x))))
        .collect();

    let model = MemoryModel::fit_quadratic(&samples).expect("refit should succeed");
    assert!(
        model.linear >= 0.0,
        "linear must be >= 0 after refit, got {}",
        model.linear
    );
}

#[test]
fn fit_too_few_distinct_x() {
    // All samples have the same x.
    let samples = vec![(10_000.0_f64, 2_000_000_000_u64); 5];
    let err = MemoryModel::fit_quadratic(&samples).unwrap_err();
    assert!(
        matches!(err, ltx_budget::FitError::TooFewDistinct(_)),
        "expected TooFewDistinct, got {err:?}"
    );
}

#[test]
fn fit_non_finite_x_rejected() {
    let samples = vec![
        (10_000.0, 1_000_000_000_u64),
        (f64::INFINITY, 2_000_000_000_u64),
        (60_000.0, 3_000_000_000_u64),
    ];
    let err = MemoryModel::fit_quadratic(&samples).unwrap_err();
    assert!(
        matches!(err, ltx_budget::FitError::NonFinite),
        "expected NonFinite, got {err:?}"
    );
}

#[test]
fn fit_linear_recovers_coefficients() {
    let resident = 800_000_000.0_f64;
    let linear = 5_000.0_f64;

    let pixel_sizes: &[u64] = &[100_000, 400_000, 900_000, 2_000_000];
    let samples: Vec<(f64, u64)> = pixel_sizes
        .iter()
        .map(|&p| {
            (
                u64_f64(p),
                to_bytes(peak_at(resident, linear, 0.0, u64_f64(p))),
            )
        })
        .collect();

    let model = MemoryModel::fit_linear(&samples).expect("linear fit should succeed");
    let tol = 1e-3;
    assert!((model.resident - resident).abs() < resident * tol);
    assert!((model.linear - linear).abs() < linear * tol);
    assert!(
        model.quadratic.to_bits() == 0.0_f64.to_bits(),
        "quadratic must be 0.0 for linear fit"
    );
}

// ── Solver tests ──────────────────────────────────────────────────────────────

/// Minimal model with no quadratic term; peak = resident + linear * tokens.
#[must_use]
const fn linear_dit(resident: f64, linear: f64) -> MemoryModel {
    MemoryModel {
        resident,
        linear,
        quadratic: 0.0,
        residual: 0.0,
    }
}

/// Builds a `SolveConfig` with the given free bytes and `DiT` model.
const fn basic_config(
    dit: &MemoryModel,
    free_bytes: u64,
    width: u32,
    height: u32,
) -> SolveConfig<'_> {
    SolveConfig {
        dit,
        vae: None,
        free_bytes,
        safety_margin: DEFAULT_SAFETY_MARGIN,
        width,
        height,
        layout: LAYOUT,
        scale: SCALE,
        quality_cap: DEFAULT_QUALITY_CAP,
        fps: 24.0,
    }
}

#[test]
fn solver_returns_8k_plus_1_frames_only() {
    let dit = linear_dit(1e9, 1.0);
    let cfg = basic_config(&dit, 100_000_000_000, 512, 512);
    let result = solve(&cfg).expect("should fit");
    if let SolveResult::Fit { frames, .. } = result {
        assert_eq!(
            (frames.saturating_sub(1)) % 8,
            0,
            "frames={frames} is not 8k+1"
        );
    } else {
        panic!("expected Fit variant");
    }
}

#[test]
fn solver_clamps_to_quality_cap() {
    let dit = linear_dit(0.0, 0.0);
    let cfg = SolveConfig {
        quality_cap: 33,
        ..basic_config(&dit, u64::MAX, 512, 512)
    };
    let result = solve(&cfg).expect("should fit");
    if let SolveResult::Fit { frames, .. } = result {
        assert!(frames <= 33, "frames={frames} exceeds cap=33");
        assert_eq!((frames.saturating_sub(1)) % 8, 0);
    } else {
        panic!("expected Fit");
    }
}

#[test]
fn solver_respects_margin() {
    let scale = SCALE;
    let layout = LAYOUT;
    let width = 512_u32;
    let height = 512_u32;

    // Compute tokens at 121 frames.
    let shape121 = PixelShape::new(121, height, width, scale).unwrap();
    let tokens121 = u64_f64(u64::from(tokens_u32(
        layout.sequence_tokens(shape121).unwrap().total().unwrap(),
    )));

    let free_bytes: u64 = 10_000_000_000; // 10 GB
    // Set linear so that peak(tokens121) == free_bytes exactly.
    let linear = u64_f64(free_bytes) / tokens121;
    let dit = MemoryModel {
        resident: 0.0,
        linear,
        quadratic: 0.0,
        residual: 0.0,
    };

    let cfg = basic_config(&dit, free_bytes, width, height);
    let result = solve(&cfg).expect("should fit at lower frame count");
    if let SolveResult::Fit { frames, .. } = result {
        assert!(
            frames < 121,
            "solver ignored margin: returned {frames} frames"
        );
        assert_eq!((frames.saturating_sub(1)) % 8, 0);
    } else {
        panic!("expected Fit, got spatial fallback");
    }
}

#[test]
fn solver_exact_fit_at_budget() {
    let scale = SCALE;
    let layout = LAYOUT;
    let width = 256_u32;
    let height = 256_u32;
    let frames = 17_u32;

    let shape = PixelShape::new(frames, height, width, scale).unwrap();
    let tokens = layout.sequence_tokens(shape).unwrap().total().unwrap();

    let peak: u64 = 5_000_000_000;
    let linear = u64_f64(peak) / u64_f64(tokens);
    let dit = MemoryModel {
        resident: 0.0,
        linear,
        quadratic: 0.0,
        residual: 0.0,
    };
    // free * (1 - margin) = peak  =>  free = peak / (1 - margin)
    let free_bytes = budget_bytes(u64_f64(peak) / (1.0 - DEFAULT_SAFETY_MARGIN)).saturating_add(1);

    let cfg = SolveConfig {
        quality_cap: 17,
        ..basic_config(&dit, free_bytes, width, height)
    };
    let result = solve(&cfg).expect("should fit exactly");
    if let SolveResult::Fit { frames: f, .. } = result {
        assert_eq!(f, 17);
    } else {
        panic!("expected Fit at 17 frames");
    }
}

#[test]
fn solver_spatial_fallback_when_f9_does_not_fit() {
    // Model: resident = 0, linear = 1 MB/token.
    // Tokens at 512×512, 9 frames: 2*(16*16)*2 = 1 024 (FULL layout).
    // Peak = 1M * 1024 = 1.024 GB > 870 MB budget → doesn't fit at full res.
    // At 448×448 (step down by 32 twice), tokens = 2*(14*14)*2 = 784.
    // Peak = 784 MB < 870 MB → fits.
    let dit = linear_dit(0.0, 1_000_000.0); // 1 MB / token
    let cfg = basic_config(&dit, 1_000_000_000, 512, 512); // 1 GB
    let result = solve(&cfg).expect("should return spatial fallback");
    assert!(
        matches!(result, SolveResult::TileFallback { .. }),
        "expected TileFallback, got {result:?}"
    );
    if let SolveResult::TileFallback {
        tile_width,
        tile_height,
        frames,
        overlap_pixels,
        ..
    } = result
    {
        assert_eq!(
            (frames.saturating_sub(1)) % 8,
            0,
            "fallback frames not 8k+1"
        );
        assert_eq!(tile_width % 32, 0, "tile width not multiple of 32");
        assert_eq!(tile_height % 32, 0, "tile height not multiple of 32");
        assert_eq!(overlap_pixels, TILE_OVERLAP_PIXELS);
    }
}

#[test]
fn solver_nothing_fits_returns_error() {
    let dit = MemoryModel {
        resident: f64::MAX,
        linear: 0.0,
        quadratic: 0.0,
        residual: 0.0,
    };
    let cfg = basic_config(&dit, 1_000_000_000, 64, 64);
    let err = solve(&cfg).unwrap_err();
    assert!(
        matches!(err, BudgetError::NoFit),
        "expected NoFit, got {err:?}"
    );
}

#[test]
fn solver_pads_spatial_dimensions() {
    let dit = linear_dit(1e9, 1.0);
    let cfg = basic_config(&dit, 100_000_000_000, 1920, 1080);
    let result = solve(&cfg).expect("should fit");
    if let SolveResult::Fit {
        padded_height,
        padded_width,
        was_padded,
        ..
    } = result
    {
        assert_eq!(padded_width, 1920, "width should not change");
        assert_eq!(padded_height, 1088, "height should be padded to 1088");
        assert!(was_padded, "was_padded should be true");
    } else {
        panic!("expected Fit");
    }
}

// ── 1920×1080 token-count test ────────────────────────────────────────────────

#[test]
fn token_count_1920x1080_121_frames() {
    // From the plan: 1920×1080 padded to 1920×1088; 121 frames.
    // With IcLoraLayout::FULL (reference at full target size):
    //   target latent: (121-1)/8 + 1 = 16 frames, 1088/32 = 34 h, 1920/32 = 60 w
    //   target tokens = 16 * 34 * 60 = 32 640
    //   reference = same shape → 32 640
    //   total = 65 280
    let pw = ltx_shape::ceil_spatial(1920, std::num::NonZeroU32::new(32).unwrap()).unwrap();
    let ph = ltx_shape::ceil_spatial(1080, std::num::NonZeroU32::new(32).unwrap()).unwrap();
    assert_eq!(pw, 1920);
    assert_eq!(ph, 1088);

    let shape = PixelShape::new(121, ph, pw, SCALE).expect("valid shape");
    let seq = LAYOUT.sequence_tokens(shape).expect("valid layout");
    let total = seq.total().expect("no overflow");
    assert_eq!(total, 65_280, "expected 65 280 tokens, got {total}");
}

// ── Calibrate tests ───────────────────────────────────────────────────────────

#[test]
fn calibrate_fits_dit_model() {
    let resident: u64 = 2_000_000_000;
    let linear_coeff: u64 = 2_000;

    let mut probe = |shape: PixelShape| -> Result<u64, std::convert::Infallible> {
        let tokens = LAYOUT.sequence_tokens(shape).unwrap().total().unwrap();
        Ok(resident.saturating_add(linear_coeff.saturating_mul(tokens)))
    };

    let model = calibrate(512, 512, LAYOUT, &mut probe, None).expect("calibrate should succeed");

    let tol = 1e-2; // 1 %
    assert!(
        (model.resident - u64_f64(resident)).abs() < u64_f64(resident) * tol,
        "resident off by more than 1%: {} vs {}",
        model.resident,
        resident
    );
    assert!(model.quadratic >= 0.0);
}

#[test]
fn calibrate_vae_fits_linear_model() {
    let resident: u64 = 500_000_000;
    let linear_coeff: u64 = 100; // 100 bytes per pixel

    let tile_sizes = [(256_u32, 256_u32), (512, 512), (768, 768)];
    let mut probe = |shape: PixelShape| -> Result<u64, std::convert::Infallible> {
        let pixels = u64::from(shape.width()) * u64::from(shape.height());
        Ok(resident.saturating_add(linear_coeff.saturating_mul(pixels)))
    };

    let model = calibrate_vae(9, &tile_sizes, &mut probe).expect("vae calibrate should succeed");

    assert!(
        model.quadratic.to_bits() == 0.0_f64.to_bits(),
        "VAE model must be linear"
    );
    let tol = 1e-2;
    assert!(
        (model.resident - u64_f64(resident)).abs() < u64_f64(resident) * tol,
        "VAE resident off: {} vs {}",
        model.resident,
        resident
    );
    assert!(
        (model.linear - u64_f64(linear_coeff)).abs() < u64_f64(linear_coeff) * tol,
        "VAE linear off: {} vs {}",
        model.linear,
        linear_coeff
    );
}

// ── Cache tests ───────────────────────────────────────────────────────────────

fn sample_key(suffix: &str) -> CacheKey {
    CacheKey {
        device_name: format!("TestDevice-{suffix}"),
        dtype: "bf16".into(),
        offload_mode: "none".into(),
        backend_id: "burn-ndarray".into(),
        model_id: "test-model-v1".into(),
    }
}

fn sample_models() -> CachedModels {
    CachedModels {
        dit: MemoryModel {
            resident: 1e9,
            linear: 2500.0,
            quadratic: 0.0001,
            residual: 100.0,
        },
        vae: Some(MemoryModel {
            resident: 5e8,
            linear: 100.0,
            quadratic: 0.0,
            residual: 50.0,
        }),
        dit_samples: vec![(10_000.0, 1_100_000_000), (50_000.0, 1_200_000_000)],
        vae_samples: vec![(100_000.0, 510_000_000), (500_000.0, 550_000_000)],
    }
}

#[test]
fn cache_round_trip() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = ltx_budget::Cache::default_path(tmp.path());

    let key = sample_key("A");
    let models = sample_models();

    Cache::upsert(&path, key.clone(), models.clone()).expect("upsert");

    let cache = Cache::load(&path).expect("load");
    let loaded = cache.get(&key).expect("key must be present");

    assert_eq!(loaded.dit, models.dit);
    assert_eq!(loaded.vae, models.vae);
}

#[test]
fn cache_missing_key_returns_none() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = ltx_budget::Cache::default_path(tmp.path());

    let key = sample_key("B");
    Cache::upsert(&path, key, sample_models()).expect("upsert");

    let cache = Cache::load(&path).expect("load");
    let other = sample_key("C"); // different suffix
    assert!(
        cache.get(&other).is_none(),
        "should not find a key that was not inserted"
    );
}

#[test]
fn cache_key_mismatch_returns_none() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = ltx_budget::Cache::default_path(tmp.path());

    let mut key = sample_key("D");
    Cache::upsert(&path, key.clone(), sample_models()).expect("upsert");

    let cache = Cache::load(&path).expect("load");
    key.dtype = "fp8".into(); // changed dtype
    assert!(
        cache.get(&key).is_none(),
        "dtype change should miss the cache"
    );
}

#[test]
fn cache_load_nonexistent_file_returns_empty() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = tmp.path().join("does_not_exist.json");
    let cache = Cache::load(&path).expect("should return empty cache, not error");
    assert!(cache.get(&sample_key("Z")).is_none());
}

#[test]
fn cache_atomic_write_creates_parent_dirs() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = tmp.path().join("nested").join("deep").join("cache.json");
    let mut cache = Cache::default();
    cache.insert(sample_key("E"), sample_models());
    cache.store(&path).expect("store should create parent dirs");
    assert!(path.exists());
}

// ── VAE + DiT combined solver ─────────────────────────────────────────────────

#[test]
fn solver_vae_limits_budget() {
    // `DiT` model is cheap; VAE model is expensive.
    let dit = linear_dit(0.0, 1.0);
    let vae = MemoryModel {
        resident: 0.0,
        linear: 100_000.0,
        quadratic: 0.0,
        residual: 0.0,
    };
    // At 512×512, pixels = 262 144, VAE peak ≈ 26 GB → should limit budget.
    let cfg = SolveConfig {
        vae: Some(&vae),
        ..basic_config(&dit, 2_000_000_000, 512, 512)
    };
    let result = solve(&cfg).expect("fallback expected");
    assert!(
        matches!(
            result,
            SolveResult::Fit { .. } | SolveResult::TileFallback { .. }
        ),
        "unexpected result: {result:?}"
    );
}
