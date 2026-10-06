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
// so no integer overflow is possible.  Burn tensor arithmetic runs on the
// compute backend and is allowed by the `arithmetic-side-effects-allowed`
// setting in `clippy.toml`.

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
