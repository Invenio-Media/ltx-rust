//! FP8 → f32 dequantization lookup tables.
//!
//! LTX-2 transformer checkpoints can be stored in `float8_e4m3fn` or
//! `float8_e5m2` dtype.  Both appear in safetensors headers as `"F8_E4M3"`
//! and `"F8_E5M2"` respectively.  Scalar per-tensor scales are stored
//! alongside as a sibling key with the suffix `"_scale"` (see
//! `ltx_core/quantization/fp8_cast.py` and `fp8_scaled_mm.py`).
//!
//! Dequantization:
//!
//! ```text
//! f32 = to_f32(fp8_byte) * scale   (if a sibling *_scale key exists)
//! f32 = to_f32(fp8_byte)            (otherwise)
//! ```
//!
//! The lookup tables are generated at const time.

/// Decode a `float8_e4m3fn` byte to `f32`.
#[must_use]
pub fn e4m3fn_to_f32(byte: u8) -> f32 {
    E4M3FN_LUT
        .get(usize::from(byte))
        .copied()
        .unwrap_or(f32::NAN)
}

/// Decode a `float8_e5m2` byte to `f32`.
#[must_use]
pub fn e5m2_to_f32(byte: u8) -> f32 {
    E5M2_LUT.get(usize::from(byte)).copied().unwrap_or(f32::NAN)
}

/// Convert a slice of `float8_e4m3fn` bytes to f32 values.
#[must_use]
pub fn e4m3fn_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| e4m3fn_to_f32(b)).collect()
}

/// Convert a slice of `float8_e5m2` bytes to f32 values.
#[must_use]
pub fn e5m2_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| e5m2_to_f32(b)).collect()
}

/// Lookup table for `float8_e4m3fn` → f32.
static E4M3FN_LUT: [f32; 256] = build_e4m3fn_lut();

/// Lookup table for `float8_e5m2` → f32.
static E5M2_LUT: [f32; 256] = build_e5m2_lut();

// `as u8` is safe (idx < 256); `u8::try_from` not const-stable.
// Direct indexing is safe; idx < 256 guaranteed by the while loop.
#[allow(clippy::as_conversions, clippy::indexing_slicing)]
const fn build_e4m3fn_lut() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut idx: usize = 0;
    while idx < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let byte = idx as u8;
        table[idx] = decode_e4m3fn(byte);
        idx = idx.saturating_add(1);
    }
    table
}

// `as u8` is safe (idx < 256); `u8::try_from` not const-stable.
// Direct indexing is safe; idx < 256 guaranteed by the while loop.
#[allow(clippy::as_conversions, clippy::indexing_slicing)]
const fn build_e5m2_lut() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut idx: usize = 0;
    while idx < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let byte = idx as u8;
        table[idx] = decode_e5m2(byte);
        idx = idx.saturating_add(1);
    }
    table
}

/// Decode `float8_e4m3fn`.
///
/// Encoding: S EEEE MMM. NaN: `0x7F` / `0xFF`. No infinities.
/// Sub: `exp == 0`, norm: `exp != 0`.
/// Decode `float8_e4m3fn`.
///
/// Encoding: S EEEE MMM. NaN: `0x7F` / `0xFF`. No infinities.
// `u8 as u32` is always safe widening; `u32::from` is not yet const-stable.
#[allow(clippy::as_conversions)]
const fn decode_e4m3fn(byte: u8) -> f32 {
    if byte == 0x7F || byte == 0xFF {
        return f32::NAN;
    }

    let sign_bit: u32 = if byte & 0x80 != 0 { 1 << 31 } else { 0 };
    let exp: u8 = (byte >> 3) & 0x0F;
    let mant: u8 = byte & 0x07;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign_bit); // ±0
        }
        // Subnormal: (−1)^S × mant × 2^(−9); build f32 bits from MSB of mant.
        let (exp_f32, mant_f32): (u32, u32) = match mant {
            1 => (118, 0),
            2 => (119, 0),
            3 => (119, 1 << 22),
            4 => (120, 0),
            5 => (120, 1 << 21),
            6 => (120, 1 << 22),
            _ => (120, 3 << 21), // mant == 7
        };
        return f32::from_bits(sign_bit | (exp_f32 << 23) | mant_f32);
    }

    // Normal: (−1)^S × 2^(E−7) × (1 + M/8)
    let exp_f32: u32 = (exp as u32).saturating_add(120); // u8→u32 safe; E + 127 - 7
    let mant_f32: u32 = (mant as u32) << 20; // u8→u32 safe
    f32::from_bits(sign_bit | (exp_f32 << 23) | mant_f32)
}

/// Decode `float8_e5m2`.
///
/// Encoding: S EEEEE MM. Standard IEEE with bias 15.
// `u8 as u32` is always safe widening; `u32::from` is not yet const-stable.
#[allow(clippy::as_conversions)]
const fn decode_e5m2(byte: u8) -> f32 {
    let sign: u32 = if byte & 0x80 != 0 { 1 << 31 } else { 0 };
    let exp: u8 = (byte >> 2) & 0x1F;
    let mant: u8 = byte & 0x03;

    match exp {
        31 => {
            if mant == 0 {
                f32::from_bits(sign | 0x7F80_0000) // ±inf
            } else {
                f32::NAN
            }
        }
        0 => {
            if mant == 0 {
                f32::from_bits(sign) // ±0
            } else {
                let (exp_f32, mant_f32): (u32, u32) = match mant {
                    1 => (111, 0),
                    2 => (112, 0),
                    _ => (112, 1 << 22), // mant == 3
                };
                f32::from_bits(sign | (exp_f32 << 23) | mant_f32)
            }
        }
        e => {
            let exp_f32: u32 = (e as u32).saturating_add(112); // u8→u32 safe; E + 127 - 15
            let mant_f32: u32 = (mant as u32) << 21; // u8→u32 safe
            f32::from_bits(sign | (exp_f32 << 23) | mant_f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_zeros() {
        assert_eq!(e4m3fn_to_f32(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(e4m3fn_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn e4m3fn_nans() {
        assert!(e4m3fn_to_f32(0x7F).is_nan());
        assert!(e4m3fn_to_f32(0xFF).is_nan());
    }

    #[test]
    fn e4m3fn_one() {
        // 1.0 = 0_0111_000 = 0x38; exp=7, mant=0 → 2^0 × 1 = 1.0
        assert!((e4m3fn_to_f32(0x38) - 1.0_f32).abs() < 1e-6_f32);
    }

    #[test]
    fn e4m3fn_max() {
        // Max normal: 0_1110_111 = 0x77; should be 448.0
        assert!(e4m3fn_to_f32(0x77) > 100.0_f32);
    }

    #[test]
    fn e5m2_zero() {
        assert_eq!(e5m2_to_f32(0x00).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn e5m2_one() {
        // 1.0 = 0_01111_00 = 0x3C
        assert!((e5m2_to_f32(0x3C) - 1.0_f32).abs() < 1e-6_f32);
    }

    #[test]
    fn e5m2_inf() {
        assert!(e5m2_to_f32(0x7C).is_infinite());
        assert!(e5m2_to_f32(0x7C).is_sign_positive());
        assert!(e5m2_to_f32(0xFC).is_infinite());
        assert!(e5m2_to_f32(0xFC).is_sign_negative());
    }
}
