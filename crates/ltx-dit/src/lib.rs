//! LTX-2.5 22B video DiT in Burn (video stream only).
//!
//! # Usage
//!
//! ```no_run
//! use ltx_dit::{DiTConfig, VideoTransformer, VideoInput, OffloadMode};
//! use burn::backend::NdArray;
//!
//! let device = Default::default();
//! let config = DiTConfig::default();
//! let model: VideoTransformer<NdArray> = VideoTransformer::new(&config, &device);
//! ```
//!
//! # Crate features
//!
//! | Feature   | Enables |
//! |-----------|---------|
//! | `ndarray` | CPU/ndarray backend (default, used in tests) |
//! | `metal`   | Apple Metal GPU backend |
//! | `cuda`    | NVIDIA CUDA backend |

// burn::Tensor<B, D> implements +/-/*/÷ on the compute backend; there is no
// integer overflow risk.  clippy 0.1.98's `arithmetic-side-effects-allowed`
// does not suppress the lint for generic user-defined types (the list only
// works for known stdlib primitives like i32).  All scalar integer arithmetic
// in this crate uses .saturating_add()/.saturating_mul() etc.  We suppress
// the lint here so Tensor arithmetic compiles cleanly.
#![allow(clippy::arithmetic_side_effects)]
// Domain-specific acronyms (DiT, RoPE, AdaLN, IC-LoRA, K/V) appear throughout
// the docs.  They are not Rust identifiers and should not be in backticks.
#![allow(clippy::doc_markdown)]

pub mod adaln;
pub mod attention;
pub mod block;
pub mod config;
pub mod error;
pub mod feed_forward;
pub mod model;
pub mod norm;
pub mod offload;
pub mod rope;

// Public re-exports ────────────────────────────────────────────────────────────

pub use config::{DiTConfig, DiTFlags, RopeType};
pub use error::DitError;
pub use model::{VideoInput, VideoTransformer};
pub use offload::{OffloadMode, block_param_count, head_tail_param_count, resident_bytes};
