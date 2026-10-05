//! Memory-mapped safetensors weight store with `FP8` dequantization and `LoRA`
//! merging for the LTX-2 model family.
//!
//! # Quick start
//!
//! ```no_run
//! use ltx_weights::{KeyMap, WeightStore};
//!
//! let store = WeightStore::open(
//!     &["checkpoint.safetensors"],
//!     &KeyMap::transformer(),
//! )
//! .unwrap();
//!
//! let scope = store.scope("transformer_blocks.0");
//! ```
//!
//! # Dtype support
//!
//! | safetensors dtype | dequantization |
//! |---|---|
//! | `F32` | identity |
//! | `F16` | cast via [`half`](https://docs.rs/half) |
//! | `BF16` | cast via [`half`](https://docs.rs/half) |
//! | `F8_E4M3` | lookup table; scale from sibling `*_scale` key if present |
//! | `F8_E5M2` | lookup table; scale from sibling `*_scale` key if present |
//! | other | [`WeightError::UnsupportedDtype`] |

pub use error::WeightError;
pub use host_tensor::HostTensor;
pub use key_map::KeyMap;
pub use lora::{LoraFile, MergeReport};
pub use scope::Scope;
pub use store::WeightStore;

mod error;
pub(crate) mod fp8;
pub(crate) mod host_tensor;
pub(crate) mod key_map;
pub(crate) mod lora;
pub(crate) mod scope;
pub(crate) mod store;
