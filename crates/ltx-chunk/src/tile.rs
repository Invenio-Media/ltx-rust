//! Spatial tiling for the GPU-budget fallback.
//!
//! When the full frame is too large for the GPU, it is split into overlapping
//! tiles.  The tiling algorithm mirrors the temporal [`plan`] in one or two
//! spatial dimensions, with edge tiles anchored so the right/bottom edge is
//! exact.  All dimensions must be multiples of 32 (the LTX-2 spatial VAE
//! scale), and the same stride ≥ overlap constraint applies: the tile stride
//! in each axis must be at least the overlap to keep blend weights summing to
//! 1.
//!
//! ## Partition-of-unity weighting
//! At global pixel `x` in the overlap between tile A (left) and tile B (right):
//! let `k = x − B.x` (0-indexed position in the overlap) and `O` = actual
//! overlap length.
//!
//! - B's left-taper weight: `smoothstep(k / O)` (0 at the seam, 1 at the end).
//! - A's right-taper weight: `1 − smoothstep(k / O)`.
//!
//! Together they always sum to 1. Each [`Tile`] stores both left and right
//! overlap widths so [`feather_weight_1d`] can apply the correct formula to
//! each side.
//!
//! [`plan`]: crate::plan::plan

use crate::error::ChunkError;
use ltx_shape::ScaleFactors;

/// One spatial tile in the tiling plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Left pixel column (inclusive, global).
    pub x: u32,
    /// Top pixel row (inclusive, global).
    pub y: u32,
    /// Tile width in pixels.
    pub w: u32,
    /// Tile height in pixels.
    pub h: u32,
    /// Overlap with the LEFT neighbour in x (0 for the first column).
    pub real_overlap_x: u32,
    /// Overlap with the RIGHT neighbour in x (0 if this is the rightmost tile).
    pub real_overlap_right_x: u32,
    /// Overlap with the TOP neighbour in y (0 for the first row).
    pub real_overlap_y: u32,
    /// Overlap with the BOTTOM neighbour in y (0 if this is the bottommost tile).
    pub real_overlap_bottom_y: u32,
}

impl Tile {
    /// Number of pixels in this tile.
    ///
    /// # Errors
    /// Returns [`ChunkError::Overflow`] on multiplication overflow.
    pub fn pixel_count(self) -> Result<usize, ChunkError> {
        usize::try_from(self.w)
            .map_err(|_| ChunkError::Overflow)?
            .checked_mul(usize::try_from(self.h).map_err(|_| ChunkError::Overflow)?)
            .ok_or(ChunkError::Overflow)
    }
}

/// Plans a 2D grid of overlapping tiles.
///
/// - `width`, `height`: frame dimensions; must be positive multiples of 32.
/// - `tile_w`, `tile_h`: tile dimensions; must be positive multiples of 32
///   and ≤ the frame dimension they cover.
/// - `overlap`: pixels shared between adjacent tiles in both axes; must be a
///   multiple of 32, and the tile stride (`tile_dim − overlap`) must be ≥
///   `overlap` in both axes.
///
/// Edge tiles are anchored to the right/bottom boundary so no padding is
/// needed.  Tiles are returned in row-major order (left→right, top→bottom).
/// Each tile carries both its left/top and right/bottom overlap widths so
/// [`feather_weight_1d`] can produce partition-of-unity weights.
///
/// # Errors
/// Returns [`ChunkError`] when any constraint is violated.
pub fn plan_tiles(
    width: u32,
    height: u32,
    tile_w: u32,
    tile_h: u32,
    overlap: u32,
) -> Result<Vec<Tile>, ChunkError> {
    let spatial = ScaleFactors::LTX2.height.get(); // 32

    // Validate frame dimensions.
    if width == 0 || !width.is_multiple_of(spatial) {
        return Err(ChunkError::WidthNotOnGrid(width));
    }
    if height == 0 || !height.is_multiple_of(spatial) {
        return Err(ChunkError::HeightNotOnGrid(height));
    }

    // Validate tile dimensions.
    if tile_w == 0 || !tile_w.is_multiple_of(spatial) {
        return Err(ChunkError::TileWidthNotOnGrid(tile_w));
    }
    if tile_h == 0 || !tile_h.is_multiple_of(spatial) {
        return Err(ChunkError::TileHeightNotOnGrid(tile_h));
    }
    if tile_w > width {
        return Err(ChunkError::TileWidthTooLarge { tile_w, width });
    }
    if tile_h > height {
        return Err(ChunkError::TileHeightTooLarge { tile_h, height });
    }

    // Validate overlap.
    if !overlap.is_multiple_of(spatial) {
        return Err(ChunkError::SpatialOverlapNotOnGrid(overlap));
    }
    if overlap >= tile_w || overlap >= tile_h {
        let tile_dim = tile_w.min(tile_h);
        return Err(ChunkError::SpatialOverlapTooLarge { overlap, tile_dim });
    }

    let stride_x = tile_w.saturating_sub(overlap);
    let stride_y = tile_h.saturating_sub(overlap);
    if stride_x < overlap {
        return Err(ChunkError::SpatialStrideSmallerThanOverlap {
            stride: stride_x,
            overlap,
        });
    }
    if stride_y < overlap {
        return Err(ChunkError::SpatialStrideSmallerThanOverlap {
            stride: stride_y,
            overlap,
        });
    }

    // Build 1-D axis plans (mirrors temporal plan logic).
    let xs = axis_plan(width, tile_w, overlap, stride_x);
    let ys = axis_plan(height, tile_h, overlap, stride_y);
    let n_cols = xs.len();
    let n_rows = ys.len();

    // Cartesian product in row-major order, first pass (fill left/top overlaps).
    let mut tiles: Vec<Tile> = Vec::with_capacity(n_rows.saturating_mul(n_cols));
    for (row_idx, &(y, real_oy)) in ys.iter().enumerate() {
        for (col_idx, &(x, real_ox)) in xs.iter().enumerate() {
            tiles.push(Tile {
                x,
                y,
                w: tile_w,
                h: tile_h,
                real_overlap_x: if col_idx == 0 { 0 } else { real_ox },
                real_overlap_right_x: 0, // filled below
                real_overlap_y: if row_idx == 0 { 0 } else { real_oy },
                real_overlap_bottom_y: 0, // filled below
            });
        }
    }

    // Second pass: set right/bottom overlap from the next tile's left/top overlap.
    for row in 0..n_rows {
        for col in 0..n_cols {
            let idx = row.saturating_mul(n_cols).saturating_add(col);
            if col.saturating_add(1) < n_cols {
                let right_idx = row
                    .saturating_mul(n_cols)
                    .saturating_add(col.saturating_add(1));
                if let Some(right_tile) = tiles.get(right_idx) {
                    let right_overlap = right_tile.real_overlap_x;
                    if let Some(tile) = tiles.get_mut(idx) {
                        tile.real_overlap_right_x = right_overlap;
                    }
                }
            }
            if row.saturating_add(1) < n_rows {
                let bot_idx = row
                    .saturating_add(1)
                    .saturating_mul(n_cols)
                    .saturating_add(col);
                if let Some(bottom_tile) = tiles.get(bot_idx) {
                    let bottom_overlap = bottom_tile.real_overlap_y;
                    if let Some(tile) = tiles.get_mut(idx) {
                        tile.real_overlap_bottom_y = bottom_overlap;
                    }
                }
            }
        }
    }

    Ok(tiles)
}

/// 1-D axis plan: returns `(start, real_overlap)` pairs for each tile along
/// one spatial axis.  Mirrors [`plan`] but works in pixel units on the 32 grid.
///
/// [`plan`]: crate::plan::plan
fn axis_plan(total: u32, tile_dim: u32, overlap: u32, stride: u32) -> Vec<(u32, u32)> {
    if total <= tile_dim {
        return vec![(0, 0)];
    }
    let mut result: Vec<(u32, u32)> = Vec::new();
    let mut start = 0_u32;
    loop {
        let end = start.saturating_add(tile_dim);
        if end >= total {
            break;
        }
        result.push((start, overlap));
        start = start.saturating_add(stride);
    }
    // Anchored last tile.
    let last_start = total.saturating_sub(tile_dim);
    let real_overlap = result.last().map_or(0, |&(s, _)| {
        s.saturating_add(tile_dim).saturating_sub(last_start)
    });
    result.push((last_start, real_overlap));
    result
}

/// Returns the 1-D smoothstep feather weight for pixel `px` within a tile of
/// width `tile_dim`, given the actual overlap widths on each side.
///
/// - `overlap_left`: pixels shared with the left neighbour (0 = no taper).
/// - `overlap_right`: pixels shared with the right neighbour (0 = no taper).
///
/// **Partition-of-unity guarantee**: at any pixel covered by two adjacent
/// tiles, the sum of their weights is exactly 1. Tile A's right weight at
/// position `k` in the overlap is `1 − smoothstep(k / O)`, which is the
/// complement of tile B's left weight `smoothstep(k / O)`.
///
/// Returns values in `[0.0, 1.0]`.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "product of smoothstep values ∈ [0.0, 1.0]; narrowing f64→f32 loses ≤ 1 ULP; safe"
)]
pub fn feather_weight_1d(px: u32, tile_dim: u32, overlap_left: u32, overlap_right: u32) -> f32 {
    use crate::blend::smoothstep;

    // Rising edge: smoothstep(px / overlap_left), 0 at px=0, 1 at px=overlap.
    let w_left = if overlap_left > 0 && px < overlap_left {
        smoothstep(f64::from(px) / f64::from(overlap_left))
    } else {
        1.0_f64
    };

    // Falling edge: 1 − smoothstep(k / overlap_right) where k is the position
    // from the START of the right-overlap region.
    // Right overlap starts at tile_dim − overlap_right.
    let right_start = tile_dim.saturating_sub(overlap_right);
    let w_right = if overlap_right > 0 && px >= right_start {
        let k = px.saturating_sub(right_start);
        1.0_f64 - smoothstep(f64::from(k) / f64::from(overlap_right))
    } else {
        1.0_f64
    };

    (w_left * w_right) as f32
}

/// Returns the 2-D separable feather weight at tile-local pixel `(px, py)`.
///
/// `weight = feather_weight_1d(px, w, overlap_left, overlap_right)
///         × feather_weight_1d(py, h, overlap_top, overlap_bottom)`
///
/// All overlap widths are read from `tile`'s fields. Zero overlap means no
/// taper on that side (edge of frame).
#[must_use]
pub fn feather_weight_2d(px: u32, py: u32, tile: Tile) -> f32 {
    let w_x = feather_weight_1d(px, tile.w, tile.real_overlap_x, tile.real_overlap_right_x);
    let w_y = feather_weight_1d(py, tile.h, tile.real_overlap_y, tile.real_overlap_bottom_y);
    w_x * w_y
}

// ── TileBlender ───────────────────────────────────────────────────────────────

/// Weighted accumulator that blends overlapping spatial tiles into one frame.
///
/// Call [`add_tile`][`TileBlender::add_tile`] for each tile (in any order),
/// then [`finish`][`TileBlender::finish`] to normalise and retrieve the output.
///
/// Memory: `width × height × channel_count` f32 values (accumulator) +
/// `width × height` f32 values (weight sum).
pub struct TileBlender {
    width: u32,
    channel_count: usize,
    /// Weighted pixel sum: `accum[y * width * C + x * C + c]`.
    accum: Vec<f32>,
    /// Weight sum per pixel: `wsum[y * width + x]`.
    wsum: Vec<f32>,
}

impl TileBlender {
    /// Creates a new blender for a frame of the given dimensions.
    ///
    /// # Errors
    /// Returns [`ChunkError::Overflow`] when the accumulator size overflows
    /// `usize`.
    pub fn new(width: u32, height: u32, channel_count: usize) -> Result<Self, ChunkError> {
        let npix = usize::try_from(width)
            .map_err(|_| ChunkError::Overflow)?
            .checked_mul(usize::try_from(height).map_err(|_| ChunkError::Overflow)?)
            .ok_or(ChunkError::Overflow)?;
        let accum_len = npix
            .checked_mul(channel_count)
            .ok_or(ChunkError::Overflow)?;
        Ok(Self {
            width,
            channel_count,
            accum: vec![0.0_f32; accum_len],
            wsum: vec![0.0_f32; npix],
        })
    }

    /// Accumulates one tile's data into the blender.
    ///
    /// `pixels` must have length `tile.w × tile.h × channel_count`.
    /// Feather weights are computed from `tile`'s overlap fields.
    ///
    /// # Errors
    /// Returns [`ChunkError`] on length mismatch or overflow.
    pub fn add_tile(&mut self, tile: Tile, pixels: &[f32]) -> Result<(), ChunkError> {
        let expected = tile
            .pixel_count()?
            .checked_mul(self.channel_count)
            .ok_or(ChunkError::Overflow)?;
        if pixels.len() != expected {
            return Err(ChunkError::PixelsWrongLen {
                got: pixels.len(),
                expected,
            });
        }
        self.accumulate_tile(tile, pixels)
    }

    /// Normalises the accumulator by the weight sum and returns the blended
    /// output frame.
    ///
    /// Pixels with zero weight (uncovered) are left as 0.
    #[must_use]
    pub fn finish(self) -> Vec<f32> {
        let mut out = self.accum;
        let c = self.channel_count;
        for (px_idx, &w) in self.wsum.iter().enumerate() {
            if w > 0.0_f32 {
                let base = px_idx.saturating_mul(c);
                for ch in 0..c {
                    if let Some(v) = out.get_mut(base.saturating_add(ch)) {
                        *v /= w;
                    }
                }
            }
        }
        out
    }

    // ── private helpers ───────────────────────────────────────────────────

    fn accumulate_tile(&mut self, tile: Tile, pixels: &[f32]) -> Result<(), ChunkError> {
        let c = self.channel_count;
        let fw = usize::try_from(self.width).map_err(|_| ChunkError::Overflow)?;

        for py in 0..tile.h {
            for px in 0..tile.w {
                let w = feather_weight_2d(px, py, tile);

                // Global pixel position.
                let gx = tile.x.saturating_add(px);
                let gy = tile.y.saturating_add(py);
                let gpx = usize::try_from(gy)
                    .map_err(|_| ChunkError::Overflow)?
                    .saturating_mul(fw)
                    .saturating_add(usize::try_from(gx).map_err(|_| ChunkError::Overflow)?);

                // Tile-local pixel offset.
                let tw = usize::try_from(tile.w).map_err(|_| ChunkError::Overflow)?;
                let tpx = usize::try_from(py)
                    .map_err(|_| ChunkError::Overflow)?
                    .saturating_mul(tw)
                    .saturating_add(usize::try_from(px).map_err(|_| ChunkError::Overflow)?);

                // Accumulate weight sum.
                if let Some(ws) = self.wsum.get_mut(gpx) {
                    *ws += w;
                }

                // Accumulate weighted pixels.
                let accum_base = gpx.saturating_mul(c);
                let pixel_base = tpx.saturating_mul(c);
                for ch in 0..c {
                    let src = pixels
                        .get(pixel_base.saturating_add(ch))
                        .copied()
                        .unwrap_or(0.0);
                    if let Some(dst) = self.accum.get_mut(accum_base.saturating_add(ch)) {
                        *dst = w.mul_add(src, *dst);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────

    /// All pixels (0..width × 0..height) appear in at least one tile.
    fn assert_full_coverage(tiles: &[Tile], width: u32, height: u32) {
        for y in 0..height {
            for x in 0..width {
                assert!(
                    tiles.iter().any(|t| {
                        t.x <= x
                            && x < t.x.saturating_add(t.w)
                            && t.y <= y
                            && y < t.y.saturating_add(t.h)
                    }),
                    "pixel ({x},{y}) not covered by any tile"
                );
            }
        }
    }

    // ── plan_tiles ─────────────────────────────────────────────────────────

    #[test]
    fn single_tile_when_frame_fits() {
        let tiles = plan_tiles(64, 64, 64, 64, 0).unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].x, 0);
        assert_eq!(tiles[0].y, 0);
        assert_full_coverage(&tiles, 64, 64);
    }

    #[test]
    fn two_by_two_grid_full_coverage() {
        // 128×128 frame, 96×96 tiles, 32 overlap → stride=64; 2 tiles per axis
        let tiles = plan_tiles(128, 128, 96, 96, 32).unwrap();
        assert_eq!(tiles.len(), 4); // 2×2
        assert_full_coverage(&tiles, 128, 128);
    }

    #[test]
    fn right_bottom_overlaps_filled() {
        // Tile 0,0 should have right overlap = tile 1,0's left overlap.
        let tiles = plan_tiles(128, 128, 96, 96, 32).unwrap();
        // Tile order: (0,0), (1,0), (0,1), (1,1)
        let t00 = tiles[0];
        let t10 = tiles[1];
        assert_eq!(t00.real_overlap_right_x, t10.real_overlap_x);
        let t01 = tiles[2];
        assert_eq!(t00.real_overlap_bottom_y, t01.real_overlap_y);
        // Rightmost tiles have zero right overlap.
        assert_eq!(t10.real_overlap_right_x, 0);
    }

    #[test]
    fn last_tile_anchored_to_edge() {
        let tiles = plan_tiles(160, 160, 96, 96, 32).unwrap();
        assert_full_coverage(&tiles, 160, 160);
        let right_edge = tiles
            .iter()
            .map(|t| t.x.saturating_add(t.w))
            .max()
            .unwrap_or(0);
        assert_eq!(right_edge, 160);
        let bottom_edge = tiles
            .iter()
            .map(|t| t.y.saturating_add(t.h))
            .max()
            .unwrap_or(0);
        assert_eq!(bottom_edge, 160);
    }

    #[test]
    fn width_not_on_grid_is_error() {
        assert!(matches!(
            plan_tiles(100, 128, 64, 64, 0),
            Err(ChunkError::WidthNotOnGrid(100))
        ));
    }

    #[test]
    fn spatial_overlap_not_on_grid_is_error() {
        assert!(matches!(
            plan_tiles(128, 128, 96, 96, 16),
            Err(ChunkError::SpatialOverlapNotOnGrid(16))
        ));
    }

    #[test]
    fn spatial_stride_too_small_is_error() {
        // tile_w=96, overlap=64 → stride=32 < 64
        assert!(matches!(
            plan_tiles(128, 128, 96, 96, 64),
            Err(ChunkError::SpatialStrideSmallerThanOverlap { .. })
        ));
    }

    #[test]
    fn row_major_order() {
        let tiles = plan_tiles(128, 128, 96, 96, 32).unwrap();
        // y is non-decreasing.
        let ys: Vec<u32> = tiles.iter().map(|t| t.y).collect();
        assert!(ys.windows(2).all(|w| w[0] <= w[1]));
    }

    // ── feather_weight_1d ──────────────────────────────────────────────────

    #[test]
    fn feather_weight_no_overlap_is_one() {
        for px in [0_u32, 16, 31] {
            assert!(
                (feather_weight_1d(px, 32, 0, 0) - 1.0_f32).abs() < 1e-6,
                "px={px}"
            );
        }
    }

    #[test]
    fn feather_weight_left_overlap_rises() {
        // overlap_left=32: w rises from 0 at px=0 to 1 at px=32.
        let w0 = feather_weight_1d(0, 96, 32, 0);
        let w16 = feather_weight_1d(16, 96, 32, 0);
        let w32 = feather_weight_1d(32, 96, 32, 0);
        assert!(w0 < w16 && w16 < w32);
        assert!((w32 - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn feather_weight_right_complement_of_left() {
        // For a right-only tapered tile A and a left-only tapered tile B
        // sharing an overlap of O pixels, their weights must sum to 1.
        // Simulate: tile A (left), tile B (right), overlap=O.
        // w_A(px_A in right overlap) + w_B(k=px_A - stride) = 1.
        let overlap = 32_u32;
        let tile_dim = 96_u32;
        let stride = tile_dim.saturating_sub(overlap); // 64
        // At k in [0, overlap): left B = smoothstep(k/O), right A = 1 - smoothstep(k/O)
        for k in 0..overlap {
            let px_a = stride.saturating_add(k); // position in A
            let px_b = k; // position in B
            // A has no left overlap, right overlap = overlap.
            let w_a = feather_weight_1d(px_a, tile_dim, 0, overlap);
            // B has left overlap = overlap, no right overlap.
            let w_b = feather_weight_1d(px_b, tile_dim, overlap, 0);
            let sum = w_a + w_b;
            assert!(
                (sum - 1.0_f32).abs() < 1e-5_f32,
                "k={k}: w_a={w_a} + w_b={w_b} = {sum} ≠ 1"
            );
        }
    }

    // ── 2-D weight sum ─────────────────────────────────────────────────────

    #[test]
    fn weights_sum_to_one_at_each_pixel() {
        // For each global pixel, sum weights across all tiles that cover it.
        let (width, height) = (128_u32, 128_u32);
        let tiles = plan_tiles(width, height, 96, 96, 32).unwrap();

        for gy in 0..height {
            for gx in 0..width {
                let mut total_w = 0.0_f32;
                for &tile in &tiles {
                    if gx < tile.x || gx >= tile.x.saturating_add(tile.w) {
                        continue;
                    }
                    if gy < tile.y || gy >= tile.y.saturating_add(tile.h) {
                        continue;
                    }
                    let px = gx.saturating_sub(tile.x);
                    let py = gy.saturating_sub(tile.y);
                    total_w += feather_weight_2d(px, py, tile);
                }
                assert!(
                    (total_w - 1.0_f32).abs() < 1e-5_f32,
                    "pixel ({gx},{gy}): weights sum to {total_w}, expected 1.0"
                );
            }
        }
    }

    // ── TileBlender ────────────────────────────────────────────────────────

    #[test]
    fn tile_blender_constant_signal() {
        // All tiles filled with value 3.0 → output is uniformly 3.0.
        let (w, h) = (128_u32, 128_u32);
        let c = 1_usize;
        let tiles = plan_tiles(w, h, 96, 96, 32).unwrap();
        let tile_pixels = 96_usize.saturating_mul(96);
        let mut blender = TileBlender::new(w, h, c).unwrap();
        for tile in tiles {
            let data = vec![3.0_f32; tile_pixels.saturating_mul(c)];
            blender.add_tile(tile, &data).unwrap();
        }
        let out = blender.finish();
        for (idx, &v) in out.iter().enumerate() {
            assert!(
                (v - 3.0_f32).abs() < 1e-4_f32,
                "pixel {idx}: expected 3.0, got {v}"
            );
        }
    }

    #[test]
    fn tile_blender_output_size() {
        let (w, h) = (64_u32, 64_u32);
        let c = 3_usize;
        let blender = TileBlender::new(w, h, c).unwrap();
        let out = blender.finish();
        assert_eq!(out.len(), 64_usize.saturating_mul(64).saturating_mul(3));
    }
}
