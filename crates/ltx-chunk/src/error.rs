//! Error type for all chunk-planning and blending operations.

use thiserror::Error;

/// Errors returned by the chunking and blending API.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ChunkError {
    /// `chunk_len` is not of the form `8k + 1`.
    #[error("chunk_len {0} is not 8k+1")]
    ChunkLenNotOnGrid(u32),

    /// `overlap` is not a multiple of the temporal scale (8).
    #[error("overlap {0} is not a multiple of 8")]
    OverlapNotMultipleOf8(u32),

    /// `overlap` must be strictly less than `chunk_len`.
    #[error("overlap {overlap} must be less than chunk_len {chunk_len}")]
    OverlapNotSmallerThanChunkLen { overlap: u32, chunk_len: u32 },

    /// The stride (`chunk_len - overlap`) must be at least `overlap` so each
    /// pixel frame is covered by at most two chunks and blend weights sum to 1.
    #[error(
        "stride (chunk_len − overlap = {stride}) must be ≥ overlap {overlap}; \
         use a smaller overlap or a larger chunk_len"
    )]
    StrideSmallerThanOverlap { stride: u32, overlap: u32 },

    /// Total frame count must be at least 1.
    #[error("total frame count must be non-zero")]
    ZeroTotal,

    /// `width` is not a multiple of the spatial scale (32).
    #[error("width {0} is not a multiple of 32")]
    WidthNotOnGrid(u32),

    /// `height` is not a multiple of the spatial scale (32).
    #[error("height {0} is not a multiple of 32")]
    HeightNotOnGrid(u32),

    /// `tile_w` is not a multiple of 32 or is zero.
    #[error("tile_w {0} is not a positive multiple of 32")]
    TileWidthNotOnGrid(u32),

    /// `tile_h` is not a multiple of 32 or is zero.
    #[error("tile_h {0} is not a positive multiple of 32")]
    TileHeightNotOnGrid(u32),

    /// `tile_w` exceeds `width`.
    #[error("tile_w {tile_w} exceeds width {width}")]
    TileWidthTooLarge { tile_w: u32, width: u32 },

    /// `tile_h` exceeds `height`.
    #[error("tile_h {tile_h} exceeds height {height}")]
    TileHeightTooLarge { tile_h: u32, height: u32 },

    /// The spatial overlap must be a multiple of 32.
    #[error("spatial overlap {0} is not a multiple of 32")]
    SpatialOverlapNotOnGrid(u32),

    /// Spatial overlap must be strictly less than the tile dimension.
    #[error("spatial overlap {overlap} must be less than tile dimension {tile_dim}")]
    SpatialOverlapTooLarge { overlap: u32, tile_dim: u32 },

    /// The spatial stride (`tile_dim - overlap`) must be at least `overlap`.
    #[error(
        "spatial stride (tile_dim − overlap = {stride}) must be ≥ overlap {overlap}; \
         use a smaller overlap or a larger tile"
    )]
    SpatialStrideSmallerThanOverlap { stride: u32, overlap: u32 },

    /// Pixel data slice length does not match the expected frame size.
    #[error("pixel data length {got} does not match expected {expected}")]
    PixelsWrongLen { got: usize, expected: usize },

    /// Frame pushed to the blender is out of the expected order.
    #[error("frame at chunk-local index {idx} out of order; expected {expected}")]
    FrameOutOfOrder { idx: u32, expected: u32 },

    /// A chunk was finished with the wrong number of frames.
    #[error("chunk has {got} frames, expected {expected}")]
    ChunkFrameCount { got: u32, expected: u32 },

    /// The requested retained overlap exceeds the current chunk length.
    #[error("retained overlap {overlap} exceeds chunk frame count {frames}")]
    RetainedOverlapTooLarge { overlap: u32, frames: u32 },

    /// A frame was pushed after the current chunk was already full.
    #[error("current chunk already has {0} frames")]
    ChunkAlreadyFull(u32),

    /// `overlap` must be non-zero when requesting a seam conditioning plan.
    #[error("overlap must be non-zero for a seam conditioning plan")]
    ZeroOverlapForSeam,

    /// An index or size computation overflowed.
    #[error("arithmetic overflow in size or index computation")]
    Overflow,
}
