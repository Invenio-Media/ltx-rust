//! Integration tests for ltx-weights.
//!
//! Tests cover: basic round-trips, `KeyMap` presets, `FP8` dequantization with
//! scales, `LoRA` merging (including alpha/rank scaling), error cases, and [`Scope`].
//
// Integration tests: allow the patterns that clippy.toml exempts in #[test] fns.
// These only apply to this test-only compilation unit.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::approx_constant
)]

use std::path::PathBuf;

use ltx_weights::{KeyMap, LoraFile, WeightError, WeightStore};

// ─── Safetensors helpers ──────────────────────────────────────────────────────

/// Minimal safetensors serializer for test fixtures.
///
/// Format: `[8 LE u64 header_len]` `[header JSON]` `[tensor data]`.
fn make_safetensors(
    meta: Option<&[(&str, &str)]>,
    tensors: &[(&str, &[usize], &str, &[u8])],
) -> Vec<u8> {
    // Build data section first to compute offsets.
    let mut data: Vec<u8> = Vec::new();
    let mut offsets: Vec<(usize, usize)> = Vec::new();
    for (_, _, _, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        offsets.push((start, data.len()));
    }

    // Build JSON header.
    let mut header_map = serde_json::Map::new();
    if let Some(meta_kv) = meta {
        let mut meta_obj = serde_json::Map::new();
        for (k, v) in meta_kv {
            meta_obj.insert(k.to_string(), serde_json::Value::String(v.to_string()));
        }
        header_map.insert(
            "__metadata__".to_owned(),
            serde_json::Value::Object(meta_obj),
        );
    }
    for (idx, (name, shape, dtype, _)) in tensors.iter().enumerate() {
        let shape_arr: serde_json::Value = serde_json::json!(shape);
        let (start, end) = offsets[idx];
        header_map.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape_arr,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_json = serde_json::to_vec(&header_map).expect("valid JSON");

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&data);
    out
}

fn write_tmp(data: &[u8]) -> (tempfile::NamedTempFile, PathBuf) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), data).unwrap();
    let path = tmp.path().to_path_buf();
    (tmp, path)
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(vals).to_vec()
}

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect()
}

// ─── Basic round-trip tests ───────────────────────────────────────────────────

#[test]
fn identity_read_f32() {
    let vals = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(None, &[("weight", &[2, 2], "F32", &bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    assert!(store.contains("weight"));
    let ht = store.read("weight").unwrap();
    assert_eq!(ht.shape, [2, 2]);
    assert_eq!(ht.data, vals);
}

#[test]
fn identity_read_f16() {
    let vals = vec![0.5f32, 1.0, -1.0, 2.0];
    let bytes = f16_bytes(&vals);
    let sft = make_safetensors(None, &[("w", &[4], "F16", &bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let ht = store.read("w").unwrap();
    for (got, expected) in ht.data.iter().zip(vals.iter()) {
        assert!(
            (got - expected).abs() < 1e-3,
            "f16 conversion: {got} != {expected}"
        );
    }
}

#[test]
fn identity_read_bf16() {
    let vals = vec![3.14f32, -2.71, 0.0, 1.0];
    let bytes = bf16_bytes(&vals);
    let sft = make_safetensors(None, &[("w", &[4], "BF16", &bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let ht = store.read("w").unwrap();
    for (got, expected) in ht.data.iter().zip(vals.iter()) {
        assert!(
            (got - expected).abs() < 0.05,
            "bf16 conversion: {got} != {expected}"
        );
    }
}

// ─── KeyMap tests ─────────────────────────────────────────────────────────────

#[test]
fn transformer_key_map_strips_prefix() {
    let vals = vec![1.0f32, 2.0];
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(
        None,
        &[
            (
                "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
                &[2, 1],
                "F32",
                &bytes,
            ),
            ("unrelated_key", &[2], "F32", &bytes),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::transformer()).unwrap();
    assert!(store.contains("transformer_blocks.0.attn.to_q.weight"));
    // unrelated_key filtered out
    assert!(!store.contains("unrelated_key"));
    assert!(!store.contains("model.diffusion_model.transformer_blocks.0.attn.to_q.weight"));

    let ht = store.read("transformer_blocks.0.attn.to_q.weight").unwrap();
    assert_eq!(ht.data, vals);
}

#[test]
fn video_encoder_key_map() {
    let vals = vec![1.0f32, 2.0, 3.0];
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(
        None,
        &[
            ("vae.encoder.blocks.0.conv.weight", &[3, 1], "F32", &bytes),
            ("encoder.blocks.1.weight", &[3], "F32", &bytes),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_encoder()).unwrap();
    // Both strip their prefixes.
    assert!(store.contains("blocks.0.conv.weight"));
    assert!(store.contains("blocks.1.weight"));
}

#[test]
fn video_decoder_key_map_strips_prefix() {
    let vals = vec![0.5f32, 1.0];
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(
        None,
        &[("vae.decoder.conv_in.weight", &[2, 1], "F32", &bytes)],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_decoder()).unwrap();
    assert!(store.contains("conv_in.weight"));
}

#[test]
fn video_decoder_key_map_t_embedder_rename() {
    let vals = vec![1.0f32, 2.0];
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(
        None,
        &[
            (
                "vae.decoder.t_embedder.mlp.0.weight",
                &[2, 1],
                "F32",
                &bytes,
            ),
            ("vae.decoder.t_embedder.mlp.2.bias", &[2], "F32", &bytes),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_decoder()).unwrap();
    assert!(store.contains("t_embedder.timestep_embedder.linear_1.weight"));
    assert!(store.contains("t_embedder.timestep_embedder.linear_2.bias"));
}

#[test]
fn video_decoder_key_map_coarse_drop() {
    let vals = vec![1.0f32];
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(
        None,
        &[
            ("vae.decoder.coarse_up.weight", &[1], "F32", &bytes),
            ("vae.decoder.conv_in.weight", &[1, 1], "F32", &bytes),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_decoder()).unwrap();
    assert!(
        !store.contains("coarse_up.weight"),
        "coarse_ prefix should be dropped"
    );
    assert!(store.contains("conv_in.weight"));
}

#[test]
fn video_decoder_qkv_split() {
    // 3-head QKV: shape [3, 4] → split into [1, 4] each.
    let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(
        None,
        &[(
            "vae.decoder.blocks.0.attn.qkv.weight",
            &[3, 4],
            "F32",
            &bytes,
        )],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_decoder()).unwrap();
    assert!(store.contains("blocks.0.attn.to_q.weight"));
    assert!(store.contains("blocks.0.attn.to_k.weight"));
    assert!(store.contains("blocks.0.attn.to_v.weight"));
    assert!(!store.contains("blocks.0.attn.qkv.weight"));

    let q = store.read("blocks.0.attn.to_q.weight").unwrap();
    assert_eq!(q.shape, [1, 4]);
    assert_eq!(q.data, [0.0f32, 1.0, 2.0, 3.0]);

    let k = store.read("blocks.0.attn.to_k.weight").unwrap();
    assert_eq!(k.data, [4.0f32, 5.0, 6.0, 7.0]);

    let v = store.read("blocks.0.attn.to_v.weight").unwrap();
    assert_eq!(v.data, [8.0f32, 9.0, 10.0, 11.0]);
}

#[test]
fn qkv_split_applies_fp8_scale_before_splitting() {
    // e4m3 1.0, 2.0, 3.0 split into q/k/v and scaled by 4.
    let qkv = [0x38u8, 0x40u8, 0x44u8];
    let scale_bytes = f32_bytes(&[4.0]);
    let sft = make_safetensors(
        None,
        &[
            (
                "vae.decoder.blocks.0.attn.qkv.weight",
                &[3, 1],
                "F8_E4M3",
                &qkv,
            ),
            (
                "vae.decoder.blocks.0.attn.qkv.weight_scale",
                &[1],
                "F32",
                &scale_bytes,
            ),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_decoder()).unwrap();
    assert!(!store.contains("blocks.0.attn.qkv.weight_scale"));
    assert_eq!(store.read("blocks.0.attn.to_q.weight").unwrap().data, [4.0]);
    assert_eq!(store.read("blocks.0.attn.to_k.weight").unwrap().data, [8.0]);
    assert_eq!(
        store.read("blocks.0.attn.to_v.weight").unwrap().data,
        [12.0]
    );
}

// ─── Gate fold test ────────────────────────────────────────────────────────────

#[test]
fn video_decoder_gate_fold() {
    // gate_msa = 2.0, attn.proj.weight = [1.0, 2.0] → folded = [2.0, 4.0]
    let gate_bytes = f32_bytes(&[2.0f32]);
    let weight_bytes = f32_bytes(&[1.0f32, 2.0]);
    let bias_bytes = f32_bytes(&[0.5f32, 1.5]);
    let sft = make_safetensors(
        None,
        &[
            ("vae.decoder.blocks.0.gate_msa", &[1], "F32", &gate_bytes),
            (
                "vae.decoder.blocks.0.attn.proj.weight",
                &[2, 1],
                "F32",
                &weight_bytes,
            ),
            (
                "vae.decoder.blocks.0.attn.proj.bias",
                &[2],
                "F32",
                &bias_bytes,
            ),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_decoder()).unwrap();
    // Gate key removed.
    assert!(!store.contains("blocks.0.gate_msa"));

    let folded_w = store.read("blocks.0.attn.proj.weight").unwrap();
    assert!(
        (folded_w.data[0] - 2.0).abs() < 1e-6,
        "weight folded: {}",
        folded_w.data[0]
    );
    assert!((folded_w.data[1] - 4.0).abs() < 1e-6);

    let folded_b = store.read("blocks.0.attn.proj.bias").unwrap();
    assert!(
        (folded_b.data[0] - 1.0).abs() < 1e-6,
        "bias folded: {}",
        folded_b.data[0]
    );
    assert!((folded_b.data[1] - 3.0).abs() < 1e-6);
}

#[test]
fn gate_fold_applies_fp8_scale_before_folding() {
    let gate_bytes = f32_bytes(&[2.0f32]);
    let scale_bytes = f32_bytes(&[4.0f32]);
    let sft = make_safetensors(
        None,
        &[
            ("vae.decoder.blocks.0.gate_msa", &[1], "F32", &gate_bytes),
            (
                "vae.decoder.blocks.0.attn.proj.weight",
                &[1],
                "F8_E4M3",
                &[0x38u8],
            ),
            (
                "vae.decoder.blocks.0.attn.proj.weight_scale",
                &[1],
                "F32",
                &scale_bytes,
            ),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::video_decoder()).unwrap();
    assert!(!store.contains("blocks.0.attn.proj.weight_scale"));
    assert_eq!(store.read("blocks.0.attn.proj.weight").unwrap().data, [8.0]);
}

// ─── FP8 dequantization ───────────────────────────────────────────────────────

#[test]
fn fp8_e4m3fn_without_scale() {
    // 0x38 encodes 1.0 in e4m3fn.
    let sft = make_safetensors(None, &[("w", &[1], "F8_E4M3", &[0x38u8])]);
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let ht = store.read("w").unwrap();
    assert!(
        (ht.data[0] - 1.0).abs() < 1e-5,
        "e4m3fn 0x38 = {}",
        ht.data[0]
    );
}

#[test]
fn fp8_e4m3fn_with_scale() {
    // weight = 1.0 (0x38), scale = 3.0 → result = 3.0
    let scale_bytes = f32_bytes(&[3.0f32]);
    let sft = make_safetensors(
        None,
        &[
            ("w", &[1], "F8_E4M3", &[0x38u8]),
            ("w_scale", &[1], "F32", &scale_bytes),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    // Scale key should not be visible in keys()
    assert!(!store.contains("w_scale"), "scale key should be internal");
    let ht = store.read("w").unwrap();
    assert!(
        (ht.data[0] - 3.0).abs() < 1e-5,
        "scaled fp8: {}",
        ht.data[0]
    );
}

#[test]
fn non_fp8_scale_sibling_is_visible_and_not_applied() {
    let weight_bytes = f32_bytes(&[2.0]);
    let scale_bytes = f32_bytes(&[100.0]);
    let sft = make_safetensors(
        None,
        &[
            ("w", &[1], "F32", &weight_bytes),
            ("w_scale", &[1], "F32", &scale_bytes),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    assert!(store.contains("w_scale"));
    assert_eq!(store.read("w").unwrap().data, [2.0]);
    assert_eq!(store.read("w_scale").unwrap().data, [100.0]);
}

// ─── LoRA merge tests ─────────────────────────────────────────────────────────

#[test]
fn lora_merge_basic() {
    // Base weight: 2×2 identity matrix.
    // LoRA: A=[1,2] (2×1 vector treated as [rank=1, in=2]), B=[3,4] ([out=2, rank=1]).
    // delta = B @ A * strength = [[3,6],[4,8]] * 1.0 (alpha/rank = 1)
    // merged = [[1+3, 0+6],[0+4, 1+8]] = [[4,6],[4,9]]

    let base_vals = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
    let base_bytes = f32_bytes(&base_vals);
    let sft = make_safetensors(None, &[("layer.weight", &[2, 2], "F32", &base_bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let lora_a = vec![1.0f32, 2.0]; // shape [1, 2]
    let lora_b = vec![3.0f32, 4.0]; // shape [2, 1]
    let lora_a_bytes = f32_bytes(&lora_a);
    let lora_b_data = f32_bytes(&lora_b);
    let lora_sft = make_safetensors(
        None,
        &[
            ("layer.lora_A.weight", &[1, 2], "F32", &lora_a_bytes),
            ("layer.lora_B.weight", &[2, 1], "F32", &lora_b_data),
        ],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let mut store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let lora = LoraFile::open(&lora_path).unwrap();
    let report = store.merge_lora(&lora, 1.0).unwrap();

    assert_eq!(report.matched_keys, ["layer.weight"]);
    assert_eq!(report.unmatched_lora_keys, Vec::<String>::new());

    let ht = store.read("layer.weight").unwrap();
    // delta = B @ A = [[3*1+4*2]] wait...
    // B=[3,4] is [2,1], A=[1,2] is [1,2]
    // delta = B @ A = [[3*1, 3*2],[4*1, 4*2]] = [[3,6],[4,8]]
    // result = base + delta = [[1+3, 0+6],[0+4, 1+8]] = [[4,6],[4,9]]
    assert!((ht.data[0] - 4.0).abs() < 1e-5, "w[0,0]={}", ht.data[0]);
    assert!((ht.data[1] - 6.0).abs() < 1e-5, "w[0,1]={}", ht.data[1]);
    assert!((ht.data[2] - 4.0).abs() < 1e-5, "w[1,0]={}", ht.data[2]);
    assert!((ht.data[3] - 9.0).abs() < 1e-5, "w[1,1]={}", ht.data[3]);
}

#[test]
fn comfy_format_lora_matches_transformer_keys() {
    // Real checkpoints store `model.diffusion_model.*`; KeyMap::transformer()
    // strips that. ComfyUI LoRAs use `diffusion_model.*`, which the reference
    // renaming map removes before matching.
    let base_bytes = f32_bytes(&[1.0f32, 0.0, 0.0, 1.0]);
    let sft = make_safetensors(
        None,
        &[(
            "model.diffusion_model.blk.proj.weight",
            &[2, 2],
            "F32",
            &base_bytes,
        )],
    );
    let (_tmp, path) = write_tmp(&sft);
    let lora_sft = make_safetensors(
        None,
        &[
            (
                "diffusion_model.blk.proj.lora_A.weight",
                &[1, 2],
                "F32",
                &f32_bytes(&[1.0, 2.0]),
            ),
            (
                "diffusion_model.blk.proj.lora_B.weight",
                &[2, 1],
                "F32",
                &f32_bytes(&[3.0, 4.0]),
            ),
        ],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let mut store = WeightStore::open(&[&path], &KeyMap::transformer()).unwrap();
    let report = store
        .merge_lora(&LoraFile::open(&lora_path).unwrap(), 1.0)
        .unwrap();

    assert_eq!(report.matched_keys, ["blk.proj.weight"]);
    let ht = store.read("blk.proj.weight").unwrap();
    let expected = [4.0, 6.0, 4.0, 9.0];
    for (got, want) in ht.data.iter().zip(expected) {
        assert!((got - want).abs() < 1e-5, "{:?} != {expected:?}", ht.data);
    }
}

#[test]
fn lora_merge_with_alpha() {
    // Same base as above, but alpha=2, rank=1, so coeff = strength * alpha/rank = 1 * 2 = 2
    // delta = 2 * B @ A = [[6,12],[8,16]]
    let base_vals = vec![0.0f32; 4]; // 2x2 zeros
    let base_bytes = f32_bytes(&base_vals);
    let sft = make_safetensors(None, &[("w.weight", &[2, 2], "F32", &base_bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let lora_a = vec![1.0f32, 2.0];
    let lora_b = vec![3.0f32, 4.0];
    let alpha_val = vec![2.0f32];
    let lora_sft = make_safetensors(
        None,
        &[
            ("w.lora_A.weight", &[1, 2], "F32", &f32_bytes(&lora_a)),
            ("w.lora_B.weight", &[2, 1], "F32", &f32_bytes(&lora_b)),
            ("w.alpha", &[1], "F32", &f32_bytes(&alpha_val)),
        ],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let mut store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let lora = LoraFile::open(&lora_path).unwrap();
    store.merge_lora(&lora, 1.0).unwrap();

    let ht = store.read("w.weight").unwrap();
    // coeff = 1.0 * 2.0/1 = 2.0
    // delta = 2 * [[3,6],[4,8]] = [[6,12],[8,16]]
    assert!((ht.data[0] - 6.0).abs() < 1e-5);
    assert!((ht.data[1] - 12.0).abs() < 1e-5);
    assert!((ht.data[2] - 8.0).abs() < 1e-5);
    assert!((ht.data[3] - 16.0).abs() < 1e-5);
}

#[test]
fn lora_merges_stack_on_the_same_key() {
    // Two LoRAs on one 2x2 weight must add: W + 1.0*(B1@A1) + 0.5*(B2@A2).
    let base_bytes = f32_bytes(&[1.0f32, 0.0, 0.0, 1.0]);
    let sft = make_safetensors(None, &[("w.weight", &[2, 2], "F32", &base_bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    // B1@A1 = [[3,6],[4,8]]
    let first = make_safetensors(
        None,
        &[
            ("w.lora_A.weight", &[1, 2], "F32", &f32_bytes(&[1.0, 2.0])),
            ("w.lora_B.weight", &[2, 1], "F32", &f32_bytes(&[3.0, 4.0])),
        ],
    );
    // B2@A2 = [[2,0],[0,-2]] (rank 2), scaled by strength 0.5 -> [[1,0],[0,-1]]
    let second = make_safetensors(
        None,
        &[
            (
                "w.lora_A.weight",
                &[2, 2],
                "F32",
                &f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
            ),
            (
                "w.lora_B.weight",
                &[2, 2],
                "F32",
                &f32_bytes(&[2.0, 0.0, 0.0, -2.0]),
            ),
        ],
    );
    let (_first_tmp, first_path) = write_tmp(&first);
    let (_second_tmp, second_path) = write_tmp(&second);

    let mut store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    store
        .merge_lora(&LoraFile::open(&first_path).unwrap(), 1.0)
        .unwrap();
    store
        .merge_lora(&LoraFile::open(&second_path).unwrap(), 0.5)
        .unwrap();

    let ht = store.read("w.weight").unwrap();
    let expected = [1.0 + 3.0 + 1.0, 6.0, 4.0, 1.0 + 8.0 - 1.0];
    for (got, want) in ht.data.iter().zip(expected) {
        assert!((got - want).abs() < 1e-5, "{:?} != {expected:?}", ht.data);
    }
}

#[test]
fn lora_unmatched_keys_reported() {
    let base_bytes = f32_bytes(&[1.0f32]);
    let sft = make_safetensors(None, &[("present.weight", &[1], "F32", &base_bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    // LoRA targets "present" (matched) and "absent" (no base key).
    let lora_sft = make_safetensors(
        None,
        &[
            ("present.lora_A.weight", &[1, 1], "F32", &f32_bytes(&[1.0])),
            ("present.lora_B.weight", &[1, 1], "F32", &f32_bytes(&[1.0])),
            ("absent.lora_A.weight", &[1, 1], "F32", &f32_bytes(&[1.0])),
            ("absent.lora_B.weight", &[1, 1], "F32", &f32_bytes(&[1.0])),
        ],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let mut store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let lora = LoraFile::open(&lora_path).unwrap();
    let report = store.merge_lora(&lora, 1.0).unwrap();

    assert!(report.matched_keys.contains(&"present.weight".to_owned()));
    assert!(
        report
            .unmatched_lora_keys
            .iter()
            .any(|k| k.contains("absent"))
    );
}

#[test]
fn lora_all_unmatched_errors() {
    let base_bytes = f32_bytes(&[1.0f32]);
    let sft = make_safetensors(None, &[("present.weight", &[1], "F32", &base_bytes)]);
    let (_tmp, path) = write_tmp(&sft);
    let lora_sft = make_safetensors(
        None,
        &[
            ("absent.lora_A.weight", &[1, 1], "F32", &f32_bytes(&[1.0])),
            ("absent.lora_B.weight", &[1, 1], "F32", &f32_bytes(&[1.0])),
        ],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let mut store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let lora = LoraFile::open(&lora_path).unwrap();
    let err = store.merge_lora(&lora, 1.0).unwrap_err();
    assert!(
        matches!(err, WeightError::NoLoraMatches),
        "expected NoLoraMatches, got {err:?}"
    );
}

#[test]
fn lora_decode_error_is_not_reported_as_unmatched() {
    let base_bytes = f32_bytes(&[0.0f32]);
    let sft = make_safetensors(None, &[("layer.weight", &[1, 1], "F32", &base_bytes)]);
    let (_tmp, path) = write_tmp(&sft);
    let int_bytes = 7_i32.to_le_bytes();
    let lora_sft = make_safetensors(
        None,
        &[
            ("layer.lora_A.weight", &[1, 1], "I32", &int_bytes),
            ("layer.lora_B.weight", &[1, 1], "F32", &f32_bytes(&[1.0])),
        ],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let mut store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let lora = LoraFile::open(&lora_path).unwrap();
    let err = store.merge_lora(&lora, 1.0).unwrap_err();
    assert!(
        matches!(&err, WeightError::UnsupportedDtype { key, .. } if key == "layer.lora_A.weight"),
        "expected UnsupportedDtype for LoRA A, got {err:?}"
    );
}

// ─── LoRA ic_layout ──────────────────────────────────────────────────────────

#[test]
fn ic_layout_reads_metadata() {
    let lora_sft = make_safetensors(
        Some(&[
            ("reference_downscale_factor", "2"),
            ("reference_temporal_scale_factor", "4"),
        ]),
        &[("dummy.lora_A.weight", &[1, 1], "F32", &f32_bytes(&[1.0]))],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let lora = LoraFile::open(&lora_path).unwrap();
    let layout = lora.ic_layout().unwrap();
    assert_eq!(layout.reference_downscale(), 2);
    assert_eq!(layout.reference_temporal(), 4);
}

#[test]
fn ic_layout_defaults_to_one() {
    let lora_sft = make_safetensors(
        None,
        &[("dummy.lora_A.weight", &[1, 1], "F32", &f32_bytes(&[1.0]))],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let lora = LoraFile::open(&lora_path).unwrap();
    let layout = lora.ic_layout().unwrap();
    assert_eq!(layout.reference_downscale(), 1);
    assert_eq!(layout.reference_temporal(), 1);
}

// ─── Scope tests ──────────────────────────────────────────────────────────────

#[test]
fn scope_contains_and_read() {
    let vals = vec![1.0f32, 2.0];
    let bytes = f32_bytes(&vals);
    let sft = make_safetensors(
        None,
        &[
            ("transformer_blocks.0.attn.weight", &[2], "F32", &bytes),
            ("other.weight", &[2], "F32", &bytes),
        ],
    );
    let (_tmp, path) = write_tmp(&sft);
    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();

    let block_scope = store.scope("transformer_blocks.0");
    assert!(block_scope.contains("attn.weight"));
    assert!(!block_scope.contains("other.weight"));

    let attn_scope = block_scope.scope("attn");
    assert!(attn_scope.contains("weight"));

    let device = burn::backend::ndarray::NdArrayDevice::default();
    let tensor: burn::tensor::Tensor<burn::backend::NdArray, 1> =
        attn_scope.tensor("weight", &device).unwrap();
    let dims = tensor.dims();
    assert_eq!(dims, [2]);
}

#[test]
fn scope_optional_returns_none_for_missing() {
    let sft = make_safetensors(None, &[("a.weight", &[1], "F32", &f32_bytes(&[1.0]))]);
    let (_tmp, path) = write_tmp(&sft);
    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();

    let s = store.scope("a");
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let opt: Option<burn::tensor::Tensor<burn::backend::NdArray, 1>> =
        s.optional("missing", &device).unwrap();
    assert!(opt.is_none());
}

// ─── Error tests ──────────────────────────────────────────────────────────────

#[test]
fn missing_key_error() {
    let sft = make_safetensors(None, &[("x", &[1], "F32", &f32_bytes(&[0.0]))]);
    let (_tmp, path) = write_tmp(&sft);
    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();

    let result = store.read("does_not_exist");
    assert!(matches!(result, Err(WeightError::MissingKey(_))));
}

#[test]
fn rank_mismatch_error() {
    let sft = make_safetensors(
        None,
        &[("w", &[2, 2], "F32", &f32_bytes(&[1.0, 0.0, 0.0, 1.0]))],
    );
    let (_tmp, path) = write_tmp(&sft);
    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();

    let ht = store.read("w").unwrap();
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let result = ht.into_tensor::<burn::backend::NdArray, 1>(&device);
    assert!(
        matches!(
            result,
            Err(WeightError::RankMismatch {
                got: 2,
                expected: 1,
                ..
            })
        ),
        "expected RankMismatch, got: {result:?}"
    );
}

#[test]
fn bad_dtype_error() {
    // Build an I32 tensor (unsupported for f32 dequant).
    let int_bytes: Vec<u8> = bytemuck::cast_slice(&[42i32, -1i32]).to_vec();
    let sft = make_safetensors(None, &[("w", &[2], "I32", &int_bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let result = store.read("w");
    assert!(matches!(result, Err(WeightError::UnsupportedDtype { .. })));
}

#[test]
fn duplicate_keys_across_files_error() {
    let bytes = f32_bytes(&[1.0f32]);
    let sft1 = make_safetensors(None, &[("x", &[1], "F32", &bytes)]);
    let sft2 = make_safetensors(None, &[("x", &[1], "F32", &bytes)]);
    let (_tmp1, path1) = write_tmp(&sft1);
    let (_tmp2, path2) = write_tmp(&sft2);

    let result = WeightStore::open(&[&path1, &path2], &KeyMap::identity());
    assert!(matches!(result, Err(WeightError::DuplicateKey(_))));
}

#[test]
fn lora_shape_mismatch_error() {
    // Base weight is 2x3, but LoRA B is 4xrank (wrong out dim).
    let base_bytes = f32_bytes(&[0.0f32; 6]);
    let sft = make_safetensors(None, &[("layer.weight", &[2, 3], "F32", &base_bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let lora_sft = make_safetensors(
        None,
        &[
            (
                "layer.lora_A.weight",
                &[1, 3],
                "F32",
                &f32_bytes(&[1.0, 2.0, 3.0]),
            ),
            ("layer.lora_B.weight", &[4, 1], "F32", &f32_bytes(&[0.0; 4])), // wrong out
        ],
    );
    let (_lora_tmp, lora_path) = write_tmp(&lora_sft);

    let mut store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let lora = LoraFile::open(&lora_path).unwrap();
    let result = store.merge_lora(&lora, 1.0);
    assert!(matches!(result, Err(WeightError::LoraShapeMismatch(_))));
}

#[test]
fn qkv_split_bad_dim_error() {
    // 5×4 is not divisible by 3.
    let bytes = f32_bytes(&[0.0f32; 20]);
    let sft = make_safetensors(
        None,
        &[("decoder.blocks.0.attn.qkv.weight", &[5, 4], "F32", &bytes)],
    );
    let (_tmp, path) = write_tmp(&sft);

    let result = WeightStore::open(&[&path], &KeyMap::video_decoder());
    assert!(matches!(
        result,
        Err(WeightError::QkvSplitDim { dim: 5, .. })
    ));
}

// ─── Config test ──────────────────────────────────────────────────────────────

#[test]
fn config_parsed_from_metadata() {
    let config_json = r#"{"vae":{"_class_name":"CausalDiffusionVAE","latent_channels":128}}"#;
    let sft = make_safetensors(
        Some(&[("config", config_json)]),
        &[("w", &[1], "F32", &f32_bytes(&[0.0]))],
    );
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    let config = store.config().unwrap();
    assert_eq!(
        config["vae"]["_class_name"].as_str().unwrap(),
        "CausalDiffusionVAE"
    );
}

#[test]
fn config_missing_returns_error() {
    let sft = make_safetensors(None, &[("w", &[1], "F32", &f32_bytes(&[0.0]))]);
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    assert!(matches!(store.config(), Err(WeightError::MissingConfig)));
}

// ─── Shape accessor ───────────────────────────────────────────────────────────

#[test]
fn shape_returns_correct_dims() {
    let bytes = f32_bytes(&[0.0f32; 6]);
    let sft = make_safetensors(None, &[("w", &[2, 3], "F32", &bytes)]);
    let (_tmp, path) = write_tmp(&sft);

    let store = WeightStore::open(&[&path], &KeyMap::identity()).unwrap();
    assert_eq!(store.shape("w").unwrap(), &[2, 3]);
    assert!(matches!(
        store.shape("nope"),
        Err(WeightError::MissingKey(_))
    ));
}

// ─── Multiple files merged ────────────────────────────────────────────────────

#[test]
fn multiple_files_merged() {
    let bytes = f32_bytes(&[1.0f32]);
    let sft1 = make_safetensors(
        Some(&[("model_version", "2.5")]),
        &[("a.weight", &[1], "F32", &bytes)],
    );
    let sft2 = make_safetensors(None, &[("b.weight", &[1], "F32", &bytes)]);
    let (_tmp1, path1) = write_tmp(&sft1);
    let (_tmp2, path2) = write_tmp(&sft2);

    let store = WeightStore::open(&[&path1, &path2], &KeyMap::identity()).unwrap();
    assert!(store.contains("a.weight"));
    assert!(store.contains("b.weight"));
    // Metadata from first file.
    assert_eq!(store.metadata("model_version"), Some("2.5"));
}
