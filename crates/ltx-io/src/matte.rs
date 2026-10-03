//! Per-frame alpha matte writer and reader.
//!
//! Writes single-channel f32 `OpenEXR` files with the VFX-standard channel name
//! `"A"`.  Each file holds exactly one frame.  Readers such as Nuke, `DaVinci`
//! Resolve, and `OpenImageIO` find the matte on the conventional alpha channel.
//!
//! # Channel name
//! The `OpenEXR` specification (v3.2) defines `R`, `G`, `B`, and `A` as the four
//! standard channel names.  A file that contains only `A` is a legal
//! single-channel matte and is the format used by this crate.  The Lightricks
//! reference pipeline writes three-channel `R`/`G`/`B` EXR frames for colour
//! output; for alpha mattes this crate writes `A` alone.
//!
//! # Alpha extraction
//! The alpha-gen IC-LoRA model is trained to output grayscale alpha values;
//! the VAE decoder produces three near-identical channels.  The runner (and
//! [`MatteWriter::extract_alpha_from_rgb`]) derives the matte via the Rec.709
//! luminance formula `Y = 0.2126·R + 0.7152·G + 0.0722·B`, clamped to
//! `[0, 1]`.
//!
//! Source: ITU-R BT.709-6, §3; ACES TB-2014-013.

use std::path::{Path, PathBuf};

use exr::prelude::{Image, SpecificChannels, WritableImage};

use crate::IoError;

/// Writes per-frame alpha EXR files into a directory.
///
/// File names: `<prefix><frame_number>.exr`, zero-padded to 8 digits
/// (e.g. `alpha00000000.exr`).
pub struct MatteWriter {
    dir: PathBuf,
    prefix: String,
}

impl MatteWriter {
    /// Create a writer that places files in `dir` with the given `prefix`.
    ///
    /// The directory is created if it does not exist.
    ///
    /// # Errors
    /// [`IoError::Io`] when the directory cannot be created.
    pub fn new(dir: impl Into<PathBuf>, prefix: impl Into<String>) -> Result<Self, IoError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            prefix: prefix.into(),
        })
    }

    /// File path for a given frame index.
    #[must_use]
    pub fn frame_path(&self, frame_index: u32) -> PathBuf {
        self.dir
            .join(format!("{}{:08}.exr", self.prefix, frame_index))
    }

    /// Write one alpha frame.
    ///
    /// `data` must have length `width * height`; values in `[0, 1]` are
    /// conventional but not enforced here.
    ///
    /// # Errors
    /// [`IoError::DataLengthMismatch`] when `data.len() != width * height`.
    /// [`IoError::DimensionOverflow`] when `width * height` overflows `usize`.
    /// [`IoError::Exr`] on EXR write failure.
    pub fn write_frame(
        &self,
        frame_index: u32,
        width: u32,
        height: u32,
        data: &[f32],
    ) -> Result<(), IoError> {
        let w = usize::try_from(width).map_err(|_| IoError::DimensionOverflow)?;
        let h = usize::try_from(height).map_err(|_| IoError::DimensionOverflow)?;
        let expected = w.checked_mul(h).ok_or(IoError::DimensionOverflow)?;

        if data.len() != expected {
            return Err(IoError::DataLengthMismatch {
                expected,
                got: data.len(),
            });
        }

        let path = self.frame_path(frame_index);

        // Build a single-channel "A" image via the exr typed-channels builder.
        // After the length check above, every position (x, y) with x < w and y < h
        // maps to a valid flat index, so `.unwrap_or(0.0)` is unreachable.
        let channels = SpecificChannels::build().with_channel("A").with_pixel_fn(
            |position: exr::math::Vec2<usize>| -> (f32,) {
                let idx = position.1.saturating_mul(w).saturating_add(position.0);
                (data.get(idx).copied().unwrap_or(0.0_f32),)
            },
        );

        let image = Image::from_channels((w, h), channels);
        image.write().to_file(&path).map_err(IoError::Exr)?;

        Ok(())
    }

    /// Extract the Rec.709 luminance from an interleaved RGB f32 slice and
    /// return a flat alpha buffer for [`write_frame`](MatteWriter::write_frame).
    ///
    /// Formula: `Y = 0.2126·R + 0.7152·G + 0.0722·B`, clamped to `[0, 1]`.
    /// Float arithmetic cannot overflow, so no checked ops are needed.
    ///
    /// # Errors
    /// [`IoError::DataLengthMismatch`] when `rgb.len() != width * height * 3`.
    /// [`IoError::DimensionOverflow`] when `width * height` overflows `usize`.
    // `as_chunks` is nightly-only; `chunks_exact(3)` on a validated-length slice is safe.
    #[expect(
        clippy::chunks_exact_to_as_chunks,
        reason = "std::slice::as_chunks is unstable on Rust 1.92"
    )]
    pub fn extract_alpha_from_rgb(
        width: u32,
        height: u32,
        rgb: &[f32],
    ) -> Result<Vec<f32>, IoError> {
        let pixel_w = usize::try_from(width).map_err(|_| IoError::DimensionOverflow)?;
        let pixel_h = usize::try_from(height).map_err(|_| IoError::DimensionOverflow)?;
        let pixel_count = pixel_w
            .checked_mul(pixel_h)
            .ok_or(IoError::DimensionOverflow)?;
        let expected_rgb = pixel_count
            .checked_mul(3)
            .ok_or(IoError::DimensionOverflow)?;

        if rgb.len() != expected_rgb {
            return Err(IoError::DataLengthMismatch {
                expected: expected_rgb,
                got: rgb.len(),
            });
        }

        let alpha: Vec<f32> = rgb
            .chunks_exact(3)
            .map(|px| {
                // chunks_exact(3) on a validated-length slice guarantees 3 elements.
                let red = px.first().copied().unwrap_or(0.0_f32);
                let green = px.get(1).copied().unwrap_or(0.0_f32);
                let blue = px.get(2).copied().unwrap_or(0.0_f32);
                // Rec.709 luminance via fused multiply-add for accuracy.
                // Float arithmetic: no panic risk; arithmetic_side_effects targets integers.
                let luma = 0.0722_f32.mul_add(blue, 0.7152_f32.mul_add(green, 0.2126_f32 * red));
                luma.clamp(0.0_f32, 1.0_f32)
            })
            .collect();

        Ok(alpha)
    }
}

/// Read the `A` channel from an EXR file written by [`MatteWriter`].
///
/// Returns `(width, height, alpha_data)` where `alpha_data` has length
/// `width * height`.
///
/// # Errors
/// [`IoError::Exr`] on parse failure.  [`IoError::MissingField`] when the file
/// contains no `A` channel.  [`IoError::DimensionOverflow`] on overflow.
pub fn read_alpha_exr(path: &Path) -> Result<(u32, u32, Vec<f32>), IoError> {
    use exr::prelude::read_first_flat_layer_from_file;

    // Read all channels as raw FlatSamples so we can find "A" by name.
    let image = read_first_flat_layer_from_file(path).map_err(IoError::Exr)?;

    let layer = &image.layer_data;
    // layer.size is Vec2<usize>: .0 = width, .1 = height.
    let w = layer.size.0;
    let h = layer.size.1;

    // FlatSamples::values_as_f32() is defined on the enum and iterates all samples
    // without indexing into the caller's code.  Text implements PartialEq<str>.
    let channel = layer
        .channel_data
        .list
        .iter()
        .find(|ch| ch.name == *"A")
        .ok_or(IoError::MissingField("A channel"))?;

    let flat: Vec<f32> = channel.sample_data.values_as_f32().collect();

    let width = u32::try_from(w).map_err(|_| IoError::DimensionOverflow)?;
    let height = u32::try_from(h).map_err(|_| IoError::DimensionOverflow)?;
    let expected = w.checked_mul(h).ok_or(IoError::DimensionOverflow)?;

    if flat.len() != expected {
        return Err(IoError::DataLengthMismatch {
            expected,
            got: flat.len(),
        });
    }

    Ok((width, height, flat))
}
