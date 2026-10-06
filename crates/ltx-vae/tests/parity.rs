//! Parity test for `VideoEncoder::load`.
//!
//! Loads the safetensors fixture produced by `tools/parity/ltx-vae.py` (LTX-2
//! commit 9ec55f9, random-init, no real Lightricks weights), builds the Burn
//! encoder via `VideoEncoder::load`, encodes the same input video, and asserts
//! the output matches within tight tolerances.
//!
//! # Tolerances
//! Both the element-wise absolute criterion (`|a−e| ≤ atol`) and the combined
//! criterion (`|a−e| ≤ atol + rtol·|e|`) are checked.  The observed max abs
//! error is ≈1.5e-6 and the combined violation is negative (well inside both
//! limits).
//!
//! # Regenerating the fixture
//! ```text
//! python -W ignore tools/parity/ltx-vae.py
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::Tensor;
use ltx_vae::{VaeEncoderConfig, VideoEncoder};
use ltx_weights::{KeyMap, WeightStore};

type B = NdArray<f32>;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_vae_encoder_parity.safetensors"
);

/// Maximum allowed element-wise absolute error.
const ATOL: f32 = 1e-4;
/// Relative tolerance used in the combined criterion `|a−e| ≤ atol + rtol·|e|`.
const RTOL: f32 = 1e-3;

#[test]
fn vae_encoder_parity() {
    let device = NdArrayDevice::default();

    // ── Load fixture ──────────────────────────────────────────────────────────
    let store = WeightStore::open(&[FIXTURE], &KeyMap::identity()).expect("failed to open fixture");

    let config_val: serde_json::Value = store.config().expect("missing config metadata");
    let vae_config_val = config_val.get("vae").expect("config.vae missing");
    let cfg = VaeEncoderConfig::from_vae_json(vae_config_val).expect("config parse failed");

    // ── Build loaded encoder ──────────────────────────────────────────────────
    // Real LTX-2.5 path: WeightStore::open(..., &KeyMap::video_encoder()), root scope.
    // Fixture path (used here): KeyMap::identity(), root scope.
    let scope = store.scope("");
    let encoder: VideoEncoder<B> =
        VideoEncoder::load(&scope, &cfg, &device).expect("encoder load failed");

    // ── Load input and expected output ────────────────────────────────────────
    let input: Tensor<B, 5> = scope
        .tensor("input", &device)
        .expect("input tensor missing");
    let expected: Tensor<B, 5> = scope
        .tensor("output", &device)
        .expect("output tensor missing");

    // ── Run encode ────────────────────────────────────────────────────────────
    let actual = encoder.encode(input).expect("encode failed");

    // ── Compare ───────────────────────────────────────────────────────────────
    let abs_diff = actual.sub(expected.clone()).abs();

    // 1. Element-wise absolute check.
    let max_abs: f32 = abs_diff.clone().max().into_scalar();
    assert!(
        max_abs <= ATOL,
        "max abs error {max_abs:.2e} exceeds atol {ATOL:.0e}"
    );

    // 2. Combined criterion: |a−e| ≤ atol + rtol·|e|  (robust near zero).
    //    max(|a−e| − atol − rtol·|e|) ≤ 0 iff all elements satisfy the criterion.
    let combined_violation = abs_diff
        .sub(expected.abs().mul_scalar(RTOL).add_scalar(ATOL))
        .max()
        .into_scalar();
    assert!(
        combined_violation <= 0.0_f32,
        "combined tolerance exceeded: max(|a−e| − atol − rtol·|e|) = {combined_violation:.2e} \
         (atol = {ATOL:.0e}, rtol = {RTOL:.0e})"
    );
}
