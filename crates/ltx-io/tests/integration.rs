//! Integration tests for `ltx-io`.
//!
//! Tests that require `ffmpeg`/`ffprobe` skip with an explanatory message when
//! the tools are absent.  The EXR round-trip test uses only the `exr` crate and
//! always runs.

use std::process::Command;

use ltx_io::matte::{MatteWriter, read_alpha_exr};

// ── helpers ────────────────────────────────────────────────────────────────────

fn ffmpeg_present() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn ffprobe_present() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Generate a small synthetic MP4 using ffmpeg's `testsrc` filter.
#[expect(
    clippy::expect_used,
    reason = "test helper — panics intentionally on failure"
)]
fn make_test_clip(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("testsrc.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-v",
            "quiet",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=24",
            "-frames:v",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            path.to_str().expect("valid path"),
        ])
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to generate test clip");
    path
}

// ── probe ──────────────────────────────────────────────────────────────────────

#[test]
fn probe_synthetic_clip() {
    if !ffprobe_present() {
        eprintln!("SKIP probe_synthetic_clip: ffprobe not on PATH");
        return;
    }
    if !ffmpeg_present() {
        eprintln!("SKIP probe_synthetic_clip: ffmpeg not on PATH");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let clip = make_test_clip(&dir);

    let meta = ltx_io::probe(&clip).expect("probe");
    assert_eq!(meta.width, 64);
    assert_eq!(meta.height, 48);
    assert_eq!(meta.fps.num, 24);
    assert_eq!(meta.fps.den, 1);
    assert!(meta.frame_count > 0, "frame_count must be positive");
    assert!(
        meta.frame_count >= 24 && meta.frame_count <= 26,
        "frame_count={} not in 24..26",
        meta.frame_count
    );
}

#[test]
fn probe_missing_file_returns_error() {
    if !ffprobe_present() {
        eprintln!("SKIP probe_missing_file_returns_error: ffprobe not on PATH");
        return;
    }
    let result = ltx_io::probe(std::path::Path::new("/nonexistent/path/clip.mp4"));
    assert!(result.is_err(), "expected error for missing file");
}

// ── frame stream ───────────────────────────────────────────────────────────────

#[test]
fn frame_stream_reads_correct_count() {
    if !ffmpeg_present() {
        eprintln!("SKIP frame_stream_reads_correct_count: ffmpeg not on PATH");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let clip = make_test_clip(&dir);

    let mut stream =
        ltx_io::FrameStream::open_with_size(&clip, 0, 10, 64, 48).expect("open stream");

    let expected_data_len: usize = 64_usize.saturating_mul(48).saturating_mul(3);

    let mut count = 0u32;
    while let Some(frame) = stream.next_frame() {
        let frame = frame.expect("read frame");
        assert_eq!(frame.index, count, "frame index mismatch");
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(
            frame.data.len(),
            expected_data_len,
            "data length mismatch for frame {count}",
        );
        for &v in &frame.data {
            assert!(
                (0.0_f32..=1.0_f32).contains(&v),
                "pixel value {v} out of [0,1]",
            );
        }
        count = count.saturating_add(1);
    }
    assert_eq!(count, 10, "expected 10 frames, got {count}");
}

#[test]
fn frame_stream_exact_frame_seek() {
    if !ffmpeg_present() {
        eprintln!("SKIP frame_stream_exact_frame_seek: ffmpeg not on PATH");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let clip = make_test_clip(&dir);

    let frames_0_10 = ltx_io::FrameStream::open_with_size(&clip, 0, 10, 64, 48)
        .expect("open 0..10")
        .collect_frames()
        .expect("collect 0..10");

    let frames_5_10 = ltx_io::FrameStream::open_with_size(&clip, 5, 10, 64, 48)
        .expect("open 5..10")
        .collect_frames()
        .expect("collect 5..10");

    let f5_from_start = frames_0_10
        .iter()
        .find(|f| f.index == 5)
        .expect("frame 5 in 0..10");
    let f5_direct = frames_5_10
        .iter()
        .find(|f| f.index == 5)
        .expect("frame 5 in 5..10");

    assert_eq!(
        f5_from_start.data, f5_direct.data,
        "frame 5 pixel data must match regardless of seek start"
    );
}

#[test]
fn frame_stream_empty_range_returns_error() {
    if !ffmpeg_present() {
        eprintln!("SKIP frame_stream_empty_range_returns_error: ffmpeg not on PATH");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let clip = make_test_clip(&dir);

    let result = ltx_io::FrameStream::open_with_size(&clip, 10, 5, 64, 48);
    assert!(
        matches!(
            result,
            Err(ltx_io::IoError::EmptyRange { start: 10, end: 5 })
        ),
        "expected EmptyRange, got {result:?}"
    );
}

// ── EXR alpha matte ─────────────────────────────────────────────────────────

#[test]
fn exr_roundtrip_single_channel_a() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = MatteWriter::new(dir.path().join("mattes"), "alpha").expect("MatteWriter::new");

    let width = 16u32;
    let height = 12u32;
    let pixel_count: usize = usize::try_from(width)
        .expect("width fits usize")
        .saturating_mul(usize::try_from(height).expect("height fits usize"));

    // Alternating 0 and 1 so the round-trip exercises both ends of the range.
    let alpha_in: Vec<f32> = (0..pixel_count)
        .map(|i| if i % 2 == 0 { 0.0_f32 } else { 1.0_f32 })
        .collect();

    writer
        .write_frame(0, width, height, &alpha_in)
        .expect("write_frame");

    let path = writer.frame_path(0);
    let (rw, rh, alpha_out) = read_alpha_exr(&path).expect("read_alpha_exr");

    assert_eq!(rw, width, "width mismatch");
    assert_eq!(rh, height, "height mismatch");
    assert_eq!(alpha_out.len(), pixel_count, "length mismatch");

    for (i, (&written, &read)) in alpha_in.iter().zip(alpha_out.iter()).enumerate() {
        let diff = (written - read).abs();
        assert!(
            diff < 1e-5_f32,
            "pixel {i}: written={written} read={read} diff={diff}"
        );
    }
}

#[test]
fn exr_write_frame_wrong_length_is_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = MatteWriter::new(dir.path(), "alpha").expect("MatteWriter::new");
    let result = writer.write_frame(0, 4, 4, &[0.5_f32; 8]);
    assert!(
        matches!(
            result,
            Err(ltx_io::IoError::DataLengthMismatch {
                expected: 16,
                got: 8
            })
        ),
        "expected DataLengthMismatch, got {result:?}"
    );
}

#[test]
fn extract_alpha_from_rgb_luminance() {
    // White → 1.0.
    let white_rgb = vec![1.0_f32; 3];
    let alpha = MatteWriter::extract_alpha_from_rgb(1, 1, &white_rgb).expect("extract white");
    let luma = alpha[0];
    assert!(
        (luma - 1.0_f32).abs() < 1e-5_f32,
        "white luma should be 1.0, got {luma}"
    );

    // Black → 0.0.
    let black_rgb = vec![0.0_f32; 3];
    let alpha = MatteWriter::extract_alpha_from_rgb(1, 1, &black_rgb).expect("extract black");
    assert!(
        alpha[0].abs() < 1e-5_f32,
        "black luma should be 0.0, got {}",
        alpha[0]
    );

    // Pure red R=1.0, G=0.0, B=0.0 → Y = 0.2126 (Rec.709).
    let red_rgb = vec![1.0_f32, 0.0_f32, 0.0_f32];
    let alpha = MatteWriter::extract_alpha_from_rgb(1, 1, &red_rgb).expect("extract red");
    let expected = 0.2126_f32;
    assert!(
        (alpha[0] - expected).abs() < 1e-5_f32,
        "red luma expected {expected}, got {}",
        alpha[0]
    );
}

#[test]
fn extract_alpha_from_rgb_wrong_length_is_error() {
    let result = MatteWriter::extract_alpha_from_rgb(2, 2, &[0.5_f32; 10]);
    assert!(
        matches!(
            result,
            Err(ltx_io::IoError::DataLengthMismatch {
                expected: 12,
                got: 10
            })
        ),
        "expected DataLengthMismatch, got {result:?}"
    );
}

// ── preview encode ─────────────────────────────────────────────────────────────

#[test]
fn preview_encode_creates_file() {
    if !ffmpeg_present() {
        eprintln!("SKIP preview_encode_creates_file: ffmpeg not on PATH");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("preview.mp4");

    let frame = vec![0.5_f32; 8 * 6 * 3];
    let frames = std::iter::repeat_n(frame, 5);

    ltx_io::encode_preview(frames, 24, 1, 8, 6, &out, 23).expect("encode_preview");

    assert!(out.exists(), "output file was not created");
    assert!(
        out.metadata().expect("metadata").len() > 0,
        "output file is empty"
    );
}
