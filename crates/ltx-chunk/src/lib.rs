//! Temporal and spatial chunk planning, reflection padding, and crossfade
//! blending for the LTX-2 pipeline.
//!
//! ## Temporal chunking
//! [`plan::plan`] splits a long clip into overlapping windows of `chunk_len`
//! frames (`8k + 1`). [`reflect::pad_to_grid`] extends short clips to the
//! nearest valid frame count by mirror-without-edge reflection.
//! [`blend::Blender`] crossfades the overlap regions using smoothstep weights.
//! [`seam::seam_cond_plan`] produces the IC-LoRA keyframe conditioning plan
//! for each chunk seam.
//!
//! ## Spatial tiling
//! [`tile::plan_tiles`] computes an overlapping 2-D grid of tiles on the 32
//! pixel grid. [`tile::feather_weight_2d`] gives the separable smoothstep
//! weight at each pixel. [`tile::TileBlender`] accumulates tile outputs and
//! normalises them.

pub mod blend;
pub mod error;
pub mod plan;
pub mod reflect;
pub mod seam;
pub mod tile;

pub use blend::{Blender, smoothstep};
pub use error::ChunkError;
pub use plan::{Chunk, plan};
pub use reflect::{PadPlan, pad_to_grid, reflect_index};
pub use seam::{CondFrame, DEFAULT_KEYFRAME_STRENGTH, SeamCondPlan, seam_cond_plan};
pub use tile::{Tile, TileBlender, feather_weight_1d, feather_weight_2d, plan_tiles};
