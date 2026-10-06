//! LTX-2.5 22B video `DiT` in Burn (video stream only).
//!
//! # Usage
//!
//! ```no_run
//! use ltx_dit::{DiTConfig, VideoTransformer, VideoInput, OffloadMode};
//! use burn::backend::NdArray;
//!
//! let device = Default::default();
//! let config = DiTConfig::default();
//! let model: VideoTransformer<NdArray> = VideoTransformer::new(&config, &device).unwrap();
//! ```
//!
//! # Crate features
//!
//! | Feature   | Enables |
//! |-----------|---------|
//! | `ndarray` | CPU/ndarray backend (default, used in tests) |
//! | `metal`   | Apple Metal GPU backend |
//! | `cuda`    | NVIDIA CUDA backend |

// All scalar integer arithmetic in this crate uses checked / saturating ops
// so no integer overflow is possible.  Burn tensor arithmetic delegates to
// the compute backend and cannot produce integer overflow.  clippy 0.1.98's
// `arithmetic-side-effects-allowed-binary` does not suppress this lint for
// generic const-parameter types (e.g. `Tensor<B, D>` where D is a generic
// const), so the affected functions carry narrow `#[expect]` attributes
// instead.  The `arithmetic-side-effects-allowed` entries in `clippy.toml`
// cover the remaining scalar f32/f64 and known-dim tensor operations.

pub mod adaln;
pub mod attention;
pub mod block;
pub mod config;
pub mod error;
pub mod feed_forward;
pub mod load;
pub mod model;
pub mod norm;
pub mod offload;
pub mod rope;

// Public re-exports ────────────────────────────────────────────────────────────

pub use config::{DiTConfig, DiTFlags, RopeType};
pub use error::DitError;
pub use model::{VideoInput, VideoTransformer};
pub use offload::{OffloadMode, block_param_count, head_tail_param_count, resident_bytes};
