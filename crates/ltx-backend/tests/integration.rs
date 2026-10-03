//! Integration tests for `ltx-backend`.
//!
//! Tests against the real runner use a Python test double (`PROTOCOL_DOUBLE`)
//! that speaks the same JSON-lines protocol without loading any model.
//! The test that verifies the real runner's import/arg parsing runs it with
//! `--help` in the reference venv; it is skipped when the venv is absent.

use std::path::Path;
use std::process::Command;

use ltx_backend::types::VideoChunk;
use ltx_backend::{AlphaBackend, PythonBackend};
use ltx_shape::{PixelShape, ScaleFactors};

// ── helpers ────────────────────────────────────────────────────────────────────

/// Write the test-double Python script to a temp dir and return its path.
#[expect(
    clippy::expect_used,
    reason = "test helper — panics intentionally on failure"
)]
fn write_double(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("double.py");
    std::fs::write(&path, PROTOCOL_DOUBLE).expect("write double script");
    path
}

/// Minimal Python script that speaks the JSON-lines protocol without any model.
///
/// Probe returns a fixed `peak_bytes` of 1 GiB.  Run reads the RGB binary,
/// extracts Rec.709 luminance, and writes it as the alpha output binary.
const PROTOCOL_DOUBLE: &str = r#"#!/usr/bin/env python3
"""Protocol double: speaks the alphagen_runner JSON-lines protocol without any model."""
import json
import sys
import struct

# Force line-buffered mode so readline() returns immediately on newline.
try:
    sys.stdin.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # pre-3.7 fallback

while True:
    line = sys.stdin.readline()
    if not line:  # EOF
        break
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": f"bad JSON: {e}"}), flush=True)
        continue

    cmd = req.get("cmd")
    if cmd == "probe":
        print(json.dumps({"ok": True, "peak_bytes": 1073741824}), flush=True)

    elif cmd == "run":
        try:
            rgb_path = req["rgb_path"]
            alpha_out_path = req["alpha_out_path"]
            width = req["width"]
            height = req["height"]
            frames = req["frames"]
            with open(rgb_path, "rb") as f:
                rgb_bytes = f.read()
            n_floats = len(rgb_bytes) // 4
            rgb = struct.unpack(f"<{n_floats}f", rgb_bytes)
            pixel_count = frames * height * width
            alpha = []
            for i in range(pixel_count):
                base = i * 3
                r = rgb[base]
                g = rgb[base + 1]
                b = rgb[base + 2]
                y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                alpha.append(max(0.0, min(1.0, y)))
            with open(alpha_out_path, "wb") as f:
                f.write(struct.pack(f"<{pixel_count}f", *alpha))
            print(json.dumps({"ok": True}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)

    else:
        print(json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}), flush=True)
"#;

fn python3_present() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn reference_venv_python() -> std::path::PathBuf {
    std::path::PathBuf::from(
        "/Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2/.venv/bin/python",
    )
}

fn reference_runner() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Navigate from crates/ltx-backend to repo root, then python/alphagen_runner.py.
    manifest.ancestors().nth(2).map_or_else(
        || manifest.join("../../python/alphagen_runner.py"),
        |root| root.join("python").join("alphagen_runner.py"),
    )
}

// ── protocol double ────────────────────────────────────────────────────────────

#[test]
fn probe_via_double() {
    if !python3_present() {
        eprintln!("SKIP probe_via_double: python3 not on PATH");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let double = write_double(&dir);

    let backend = PythonBackend::spawn(Path::new("python3"), &double, &[]).expect("spawn double");

    let scale = ScaleFactors::default();
    let shape = PixelShape::new(25, 512, 512, scale).expect("valid shape");
    let mem = backend.probe(shape).expect("probe");

    assert_eq!(
        mem.peak_bytes, 1_073_741_824_u64,
        "expected 1 GiB from double"
    );
}

#[test]
fn run_chunk_via_double_luminance() {
    if !python3_present() {
        eprintln!("SKIP run_chunk_via_double_luminance: python3 not on PATH");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let double = write_double(&dir);

    let backend = PythonBackend::spawn(Path::new("python3"), &double, &[]).expect("spawn double");

    // All pixels are pure red (R=1.0, G=0.0, B=0.0) → Rec.709 luminance = 0.2126.
    let ww = 8u32;
    let hh = 6u32;
    let ff = 3u32;
    let pixel_count: usize = usize::try_from(ff)
        .expect("ff fits usize")
        .saturating_mul(usize::try_from(hh).expect("hh fits usize"))
        .saturating_mul(usize::try_from(ww).expect("ww fits usize"));

    let rgb: Vec<f32> = std::iter::repeat_n([1.0_f32, 0.0_f32, 0.0_f32], pixel_count)
        .flatten()
        .collect();

    let chunk = VideoChunk::new(0, ww, hh, ff, rgb).expect("valid chunk");
    let alpha_chunk = backend
        .run_chunk(&chunk, 42, None)
        .expect("run_chunk via double");

    assert_eq!(alpha_chunk.width, ww);
    assert_eq!(alpha_chunk.height, hh);
    assert_eq!(alpha_chunk.frame_count, ff);
    assert_eq!(alpha_chunk.data.len(), pixel_count);
    assert_eq!(alpha_chunk.start_frame, 0);

    for (i, &a) in alpha_chunk.data.iter().enumerate() {
        let diff = (a - 0.2126_f32).abs();
        assert!(
            diff < 1e-4_f32,
            "alpha[{i}] = {a:.6}, expected ~0.2126 (Rec.709 luminance of pure red)"
        );
    }
}

#[test]
fn runner_stays_alive_across_calls() {
    if !python3_present() {
        eprintln!("SKIP runner_stays_alive_across_calls: python3 not on PATH");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let double = write_double(&dir);
    let backend = PythonBackend::spawn(Path::new("python3"), &double, &[]).expect("spawn double");

    let scale = ScaleFactors::default();
    let shape = PixelShape::new(25, 512, 512, scale).expect("valid shape");

    let r1 = backend.probe(shape).expect("probe 1");
    let r2 = backend.probe(shape).expect("probe 2");
    let r3 = backend.probe(shape).expect("probe 3");
    assert_eq!(r1.peak_bytes, r2.peak_bytes);
    assert_eq!(r2.peak_bytes, r3.peak_bytes);
}

// ── real runner import/arg parsing ────────────────────────────────────────────

/// Verify that `alphagen_runner.py --help` exits 0 in the reference venv.
///
/// This proves the script loads cleanly (all imports succeed) and its argument
/// parser is functional.  A real generation requires the gated model weights
/// which are not on this machine.
#[test]
fn real_runner_help_exits_zero() {
    let venv_python = reference_venv_python();
    if !venv_python.exists() {
        eprintln!("SKIP real_runner_help_exits_zero: reference venv not found at {venv_python:?}");
        return;
    }

    let runner = reference_runner();
    if !runner.exists() {
        eprintln!("SKIP real_runner_help_exits_zero: runner not found at {runner:?}");
        return;
    }

    let output = Command::new(&venv_python)
        .arg("-W")
        .arg("ignore")
        .arg(&runner)
        .arg("--help")
        .output()
        .expect("spawn runner --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr_text = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "alphagen_runner.py --help exited {:?}\nstdout: {stdout}\nstderr: {stderr_text}",
        output.status.code()
    );
    assert!(
        stdout.contains("alphagen_runner"),
        "help text should mention the runner name\nstdout: {stdout}"
    );
}
