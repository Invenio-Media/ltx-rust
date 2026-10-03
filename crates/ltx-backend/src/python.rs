//! Python backend: spawns and supervises `python/alphagen_runner.py`.
//!
//! # Protocol
//! The runner is a long-lived process that reads JSON-lines requests from
//! stdin and writes JSON-lines responses to stdout.  Every message is a single
//! UTF-8 line terminated by `\n`.
//!
//! ## Probe
//! Request:
//! ```json
//! {"cmd":"probe","width":1280,"height":720,"frames":25}
//! ```
//! Response (success):
//! ```json
//! {"ok":true,"peak_bytes":4294967296}
//! ```
//!
//! ## Run
//! Request:
//! ```json
//! {"cmd":"run","rgb_path":"/tmp/.../rgb.bin","alpha_out_path":"/tmp/.../alpha.bin",
//!  "width":1280,"height":720,"frames":25,"seed":42,
//!  "keyframes_path":"/tmp/.../kf.json"}
//! ```
//! `keyframes_path` is omitted when there is no seam conditioning.
//!
//! Response (success):
//! ```json
//! {"ok":true}
//! ```
//!
//! Error response (either command):
//! ```json
//! {"ok":false,"error":"<message>"}
//! ```
//!
//! # Binary exchange files
//! All exchange files are placed in a `tempfile::TempDir` owned by the backend
//! and live for the lifetime of the backend.
//!
//! - **`rgb.bin`**: little-endian f32, shape `[frames, height, width, 3]` (RGBRGB…).
//! - **`alpha.bin`**: little-endian f32, shape `[frames, height, width]`.
//!   The runner derives the Rec.709 luminance of the generated RGB output,
//!   clamps it to `[0, 1]`, and writes it here.
//! - **`kf_rgb.bin`**: f32 keyframe RGB, shape `[n, height, width, 3]`.
//! - **`kf.json`**: JSON descriptor for keyframe conditioning:
//!   `{"indices":[…],"strength":<f32>,"rgb_path":"<path to kf_rgb.bin>"}`.
//!
//! # stderr tail
//! A background thread collects the runner's stderr.  The last 50 lines are
//! included in [`BackendError::RunnerError`] when the runner reports failure.
//!
//! # Prompt context
//! The Gemma text encoder is loaded once by the runner at startup from a
//! precomputed `.safetensors` file (passed as `--prompt-context <path>`),
//! avoiding per-chunk Gemma calls.  If the reference pipeline API does not
//! support providing a precomputed context without forking reference code, the
//! runner falls back to loading Gemma once at startup and caching it in memory
//! for the lifetime of the process.  See `python/alphagen_runner.py` for
//! details.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use ltx_shape::PixelShape;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use crate::types::{AlphaChunk, Keyframes, MemSample, VideoChunk};
use crate::{AlphaBackend, BackendError};

/// Maximum stderr lines kept in the ring buffer.
const STDERR_LINES: usize = 50;

// ── JSON protocol types ────────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(tag = "cmd", rename_all = "lowercase")]
enum Request<'a> {
    Probe {
        width: u32,
        height: u32,
        frames: u32,
    },
    Run {
        rgb_path: &'a str,
        alpha_out_path: &'a str,
        width: u32,
        height: u32,
        frames: u32,
        seed: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        keyframes_path: Option<&'a str>,
    },
}

#[derive(Deserialize)]
struct Response {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    /// Probe response only.
    #[serde(default)]
    peak_bytes: Option<u64>,
}

#[derive(Serialize)]
struct KeyframesJson<'a> {
    indices: &'a [u32],
    strength: f32,
    rgb_path: &'a str,
}

// ── Inner (behind Mutex) ───────────────────────────────────────────────────────

struct Inner {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Inner {
    fn round_trip(
        &mut self,
        req: &Request<'_>,
        stderr_tail: &Arc<Mutex<Vec<String>>>,
    ) -> Result<Response, BackendError> {
        let line = serde_json::to_string(req)?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let mut resp_line = String::new();
        self.stdout.read_line(&mut resp_line)?;

        let resp: Response = serde_json::from_str(resp_line.trim_end())?;
        if resp.ok {
            Ok(resp)
        } else {
            let msg = resp.error.unwrap_or_else(|| "unknown runner error".into());
            let stderr = stderr_tail
                .lock()
                .map(|lines| lines.join("\n"))
                .unwrap_or_default();
            Err(BackendError::RunnerError { msg, stderr })
        }
    }
}

// ── PythonBackend ──────────────────────────────────────────────────────────────

/// Spawns and supervises the `alphagen_runner.py` process.
///
/// Callers use [`AlphaBackend::probe`] and [`AlphaBackend::run_chunk`].
/// The `Mutex` serialises I/O so the trait's `&self` methods can be called
/// from any thread without additional synchronisation.
pub struct PythonBackend {
    inner: Mutex<Inner>,
    stderr_tail: Arc<Mutex<Vec<String>>>,
    /// Temp dir holding RGB/alpha/keyframe binary exchange files.
    tmp: TempDir,
}

impl PythonBackend {
    /// Spawn `alphagen_runner.py`.
    ///
    /// # Parameters
    /// - `python`: Python interpreter (e.g. the reference venv at
    ///   `/Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2/.venv/bin/python`).
    /// - `script`: path to `python/alphagen_runner.py` in this repo.
    /// - `extra_args`: forwarded to the runner.  At minimum:
    ///   `["--transformer", "<path>", "--video-vae", "<path>",
    ///     "--lora", "<path>:<strength>"]`.
    ///   Run `alphagen_runner.py --help` for the full list.
    ///
    /// # Errors
    /// [`BackendError::SpawnFailed`] when the process cannot be started.
    pub fn spawn(python: &Path, script: &Path, extra_args: &[&str]) -> Result<Self, BackendError> {
        let tmp = TempDir::new()?;

        let mut child = Command::new(python)
            .arg(script)
            .args(extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| BackendError::SpawnFailed(e.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| BackendError::SpawnFailed("stdin pipe unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BackendError::SpawnFailed("stdout pipe unavailable".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| BackendError::SpawnFailed("stderr pipe unavailable".into()))?;

        let tail = Arc::new(Mutex::new(Vec::<String>::with_capacity(STDERR_LINES)));
        let tail_clone = Arc::clone(&tail);
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(mut buf) = tail_clone.lock() {
                    if buf.len() >= STDERR_LINES {
                        buf.remove(0);
                    }
                    buf.push(line);
                }
            }
        });

        Ok(Self {
            inner: Mutex::new(Inner {
                child,
                stdin: BufWriter::new(stdin),
                stdout: BufReader::new(stdout),
            }),
            stderr_tail: tail,
            tmp,
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, BackendError> {
        self.inner
            .lock()
            .map_err(|_| BackendError::SpawnFailed("runner Mutex poisoned".into()))
    }

    fn write_rgb_bin(tmp: &Path, chunk: &VideoChunk) -> Result<PathBuf, BackendError> {
        let path = tmp.join("rgb.bin");
        std::fs::write(&path, f32_slice_to_le_bytes(&chunk.frames))?;
        Ok(path)
    }

    fn write_keyframes_json(tmp: &Path, kf: &Keyframes) -> Result<PathBuf, BackendError> {
        let rgb_path = tmp.join("kf_rgb.bin");
        std::fs::write(&rgb_path, f32_slice_to_le_bytes(&kf.frames))?;

        let rgb_path_str = rgb_path.to_str().ok_or_else(|| {
            BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "non-UTF-8 keyframe rgb path",
            ))
        })?;
        let json = serde_json::to_string(&KeyframesJson {
            indices: &kf.indices,
            strength: kf.strength,
            rgb_path: rgb_path_str,
        })?;
        let json_path = tmp.join("kf.json");
        std::fs::write(&json_path, json.as_bytes())?;
        Ok(json_path)
    }

    fn read_alpha_bin(alpha_path: &Path, chunk: &VideoChunk) -> Result<AlphaChunk, BackendError> {
        let raw = std::fs::read(alpha_path)?;
        let data = le_bytes_to_f32_vec(&raw)?;

        let w = usize::try_from(chunk.width).map_err(|_| BackendError::DimensionOverflow)?;
        let h = usize::try_from(chunk.height).map_err(|_| BackendError::DimensionOverflow)?;
        let f = usize::try_from(chunk.frame_count).map_err(|_| BackendError::DimensionOverflow)?;
        let expected = f
            .checked_mul(h)
            .and_then(|n| n.checked_mul(w))
            .ok_or(BackendError::DimensionOverflow)?;

        if data.len() != expected {
            return Err(BackendError::DataLengthMismatch {
                expected,
                got: data.len(),
            });
        }

        Ok(AlphaChunk {
            start_frame: chunk.start_frame,
            width: chunk.width,
            height: chunk.height,
            data,
            frame_count: chunk.frame_count,
        })
    }
}

impl AlphaBackend for PythonBackend {
    fn probe(&self, shape: PixelShape) -> Result<MemSample, BackendError> {
        let req = Request::Probe {
            width: shape.width(),
            height: shape.height(),
            frames: shape.frames(),
        };
        let resp = self.lock()?.round_trip(&req, &self.stderr_tail)?;
        let peak_bytes = resp
            .peak_bytes
            .ok_or(BackendError::MissingField("peak_bytes"))?;
        Ok(MemSample { peak_bytes })
    }

    fn run_chunk(
        &self,
        rgb: &VideoChunk,
        seed: u64,
        cond: Option<&Keyframes>,
    ) -> Result<AlphaChunk, BackendError> {
        let tmp = self.tmp.path();

        let rgb_path = Self::write_rgb_bin(tmp, rgb)?;
        let rgb_path_str = rgb_path.to_str().ok_or_else(|| {
            BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "non-UTF-8 rgb path",
            ))
        })?;

        let alpha_out = tmp.join("alpha.bin");
        let alpha_out_str = alpha_out.to_str().ok_or_else(|| {
            BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "non-UTF-8 alpha_out path",
            ))
        })?;

        let kf_path = cond
            .map(|kf| Self::write_keyframes_json(tmp, kf))
            .transpose()?;
        let kf_path_str = kf_path.as_ref().and_then(|p| p.to_str());

        let req = Request::Run {
            rgb_path: rgb_path_str,
            alpha_out_path: alpha_out_str,
            width: rgb.width,
            height: rgb.height,
            frames: rgb.frame_count,
            seed,
            keyframes_path: kf_path_str,
        };

        self.lock()?.round_trip(&req, &self.stderr_tail)?;
        Self::read_alpha_bin(&alpha_out, rgb)
    }
}

impl Drop for PythonBackend {
    fn drop(&mut self) {
        // The runner is long-lived and blocks on stdin.  `Inner` still owns the
        // stdin pipe while we are inside `drop`, so waiting first can deadlock.
        // Kill is safe here: all requested work has already completed before a
        // `PythonBackend` is dropped.
        if let Ok(mut inner) = self.inner.lock() {
            let _ = inner.child.kill();
            let _ = inner.child.wait();
        }
    }
}

// ── binary helpers ─────────────────────────────────────────────────────────────

/// Encode `&[f32]` to little-endian bytes.
fn f32_slice_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let byte_len = data.len().saturating_mul(4);
    let mut out = Vec::with_capacity(byte_len);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode little-endian bytes to `Vec<f32>`.
///
/// Fails when `bytes.len()` is not a multiple of 4.
// `as_chunks` is nightly-only; `chunks_exact(4)` on a validated-length slice is safe.
#[expect(
    clippy::chunks_exact_to_as_chunks,
    reason = "std::slice::as_chunks is unstable on Rust 1.92"
)]
fn le_bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, BackendError> {
    // Check length without integer division (bitwise & is not arithmetic).
    if bytes.len() & 3 != 0 {
        // Round up to nearest multiple of 4 for the "expected" field.
        let expected = bytes.len().saturating_add(3) & !3_usize;
        return Err(BackendError::DataLengthMismatch {
            expected,
            got: bytes.len(),
        });
    }
    // Safe: bytes.len() is a multiple of 4; bit-shift replaces division.
    let count = bytes.len() >> 2;
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4) {
        // chunks_exact(4) guarantees exactly 4 bytes; first()/get() are always Some.
        let arr: [u8; 4] = [
            chunk.first().copied().unwrap_or(0),
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
            chunk.get(3).copied().unwrap_or(0),
        ];
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{f32_slice_to_le_bytes, le_bytes_to_f32_vec};

    #[test]
    fn f32_binary_round_trip() {
        let values = [0.0_f32, 0.5, 1.0, -1.0];
        let bytes = f32_slice_to_le_bytes(&values);
        assert_eq!(bytes.len(), 16);
        let decoded = le_bytes_to_f32_vec(&bytes).expect("decode");
        assert_eq!(decoded.len(), values.len());
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "bit-exact round-trip");
        }
    }

    #[test]
    fn non_multiple_of_four_is_error() {
        use crate::BackendError;
        let result = le_bytes_to_f32_vec(&[0u8, 1, 2]);
        assert!(
            matches!(result, Err(BackendError::DataLengthMismatch { got: 3, .. })),
            "expected DataLengthMismatch, got {result:?}"
        );
    }
}
