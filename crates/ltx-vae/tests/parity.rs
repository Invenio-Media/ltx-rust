//! Parity test for `VideoEncoder::load`.
//!
//! Loads the safetensors fixture produced by `tools/parity/ltx-vae.py` (LTX-2
//! commit 9ec55f9, random-init, no real Lightricks weights), builds the Burn
//! encoder via `VideoEncoder::load`, encodes the same input video, and asserts
//! the output matches within tight tolerances.
//!
//! # Tolerances
//! Single combined criterion: `|a−e| ≤ atol + rtol·|e|`.  The observed max
//! `|a−e|` is ≈1.5e-6; `atol` is 1e-5 (10× margin), `rtol` 1e-3.
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

/// Absolute floor: `|a−e| ≤ atol + rtol·|e|`.  Tighter than required (observed ≈ 1.5e-6).
const ATOL: f32 = 1e-5;
/// Relative scale in the combined criterion.
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

    // ── Combined criterion: |a−e| ≤ atol + rtol·|e|  (robust near zero) ─────
    //    max(|a−e| − atol − rtol·|e|) ≤ 0  iff every element satisfies the bound.
    let abs_diff = actual.sub(expected.clone()).abs();
    let combined_violation = abs_diff
        .sub(expected.abs().mul_scalar(RTOL).add_scalar(ATOL))
        .max()
        .into_scalar();
    assert!(
        combined_violation <= 0.0_f32,
        "parity failed: max(|a−e| − atol − rtol·|e|) = {combined_violation:.2e} \
         (atol = {ATOL:.0e}, rtol = {RTOL:.0e})"
    );
}
