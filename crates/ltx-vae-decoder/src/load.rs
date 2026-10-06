//! Safetensors weight loader for the diffusion VAE decoder.
//!
//! Reads a safetensors file produced by the Python parity fixture
//! (`module.state_dict()`) or by a full model checkpoint.
//!
//! ## Weight layout
//!
//! `PyTorch` `nn.Linear` stores weights as `[out_features, in_features]`.
//! Burn's `Linear` uses `[in_features, out_features]` (Row layout).
//! All 2-D linear weights are therefore **transposed** on load.

use std::collections::HashMap;

use burn::tensor::{Tensor, TensorData, backend::Backend};
use half::f16;
use safetensors::SafeTensors;

use crate::error::VaeDecoderError;

fn checked_element_count(key: &str, shape: &[usize]) -> Result<usize, VaeDecoderError> {
    shape.iter().try_fold(1_usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| VaeDecoderError::NumericOverflow {
                detail: format!("element count overflow for key {key} shape {shape:?}"),
            })
    })
}

fn check_byte_len(
    key: &str,
    shape: &[usize],
    raw_len: usize,
    element_size: usize,
) -> Result<(), VaeDecoderError> {
    let expected = checked_element_count(key, shape)?
        .checked_mul(element_size)
        .ok_or_else(|| VaeDecoderError::NumericOverflow {
            detail: format!(
                "byte count overflow for key {key} shape {shape:?} element_size {element_size}"
            ),
        })?;
    if raw_len != expected {
        return Err(VaeDecoderError::InvalidArgument {
            detail: format!(
                "byte length mismatch for {key}: shape {shape:?} with element size {element_size} needs {expected} bytes, got {raw_len}"
            ),
        });
    }
    Ok(())
}

fn f32_values_from_le_bytes(
    key: &str,
    shape: &[usize],
    raw: &[u8],
) -> Result<Vec<f32>, VaeDecoderError> {
    check_byte_len(key, shape, raw.len(), 4)?;
    let (chunks, remainder) = raw.as_chunks::<4>();
    let empty: &[u8] = &[];
    debug_assert_eq!(remainder, empty);
    Ok(chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn f16_values_from_le_bytes(
    key: &str,
    shape: &[usize],
    raw: &[u8],
) -> Result<Vec<f32>, VaeDecoderError> {
    check_byte_len(key, shape, raw.len(), 2)?;
    let (chunks, remainder) = raw.as_chunks::<2>();
    let empty: &[u8] = &[];
    debug_assert_eq!(remainder, empty);
    Ok(chunks
        .iter()
        .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)).to_f32())
        .collect())
}

fn bf16_values_from_le_bytes(
    key: &str,
    shape: &[usize],
    raw: &[u8],
) -> Result<Vec<f32>, VaeDecoderError> {
    check_byte_len(key, shape, raw.len(), 2)?;
    let (chunks, remainder) = raw.as_chunks::<2>();
    let empty: &[u8] = &[];
    debug_assert_eq!(remainder, empty);
    Ok(chunks
        .iter()
        .map(|bytes| half::bf16::from_bits(u16::from_le_bytes(*bytes)).to_f32())
        .collect())
}

/// Load a raw f32 1-D tensor from a safetensors view, returning the shape too.
///
/// Supports `bf16`, `f16`, and `f32` dtypes.
///
/// # Errors
///
/// Returns an error if the key is missing, the dtype is unsupported, or the
/// element count does not match the declared shape.
pub fn load_raw_f32<B: Backend>(
    st: &SafeTensors<'_>,
    key: &str,
    device: &B::Device,
) -> Result<(Tensor<B, 1>, Vec<usize>), VaeDecoderError> {
    let view = st.tensor(key).map_err(|_| VaeDecoderError::KeyNotFound {
        key: key.to_owned(),
    })?;

    let raw = view.data();
    let dtype = view.dtype();
    let shape: Vec<usize> = view.shape().to_vec();
    let n = checked_element_count(key, &shape)?;

    let data_f32: Vec<f32> = match dtype {
        safetensors::Dtype::F32 => f32_values_from_le_bytes(key, &shape, raw)?,
        safetensors::Dtype::BF16 => bf16_values_from_le_bytes(key, &shape, raw)?,
        safetensors::Dtype::F16 => f16_values_from_le_bytes(key, &shape, raw)?,
        other => {
            return Err(VaeDecoderError::InvalidArgument {
                detail: format!("unsupported dtype {other:?} for key {key}"),
            });
        }
    };

    let tensor = Tensor::<B, 1>::from_data(TensorData::new(data_f32, vec![n]), device);
    Ok((tensor, shape))
}

/// Load a 1-D tensor from safetensors.
///
/// # Errors
///
/// Returns an error if the key is missing or the tensor is not 1-D.
pub fn load_1d<B: Backend>(
    st: &SafeTensors<'_>,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, 1>, VaeDecoderError> {
    let (flat, shape) = load_raw_f32(st, key, device)?;
    if shape.len() != 1 {
        return Err(VaeDecoderError::ShapeMismatch {
            key: key.to_owned(),
            expected: vec![1],
            actual: vec![shape.len()],
        });
    }
    Ok(flat)
}

/// Load a 2-D tensor from safetensors **without** transposing.
///
/// # Errors
///
/// Returns an error if the key is missing or the tensor is not 2-D.
#[expect(
    clippy::indexing_slicing,
    reason = "bounds are verified by construction"
)]
pub fn load_2d_raw<B: Backend>(
    st: &SafeTensors<'_>,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, 2>, VaeDecoderError> {
    let (flat, shape) = load_raw_f32(st, key, device)?;
    if shape.len() != 2 {
        return Err(VaeDecoderError::ShapeMismatch {
            key: key.to_owned(),
            expected: vec![2],
            actual: vec![shape.len()],
        });
    }
    Ok(flat.reshape([shape[0], shape[1]]))
}

/// Load a 2-D linear weight from `PyTorch`'s `[out, in]` layout, transposing to
/// Burn's `[in, out]` layout.
///
/// # Errors
///
/// Returns an error if the key is missing or the tensor is not 2-D.
pub fn load_linear_weight<B: Backend>(
    st: &SafeTensors<'_>,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, 2>, VaeDecoderError> {
    let w = load_2d_raw(st, key, device)?;
    // PyTorch [d_out, d_in] → Burn Row layout [d_in, d_out]
    Ok(w.swap_dims(0, 1))
}

/// Read the JSON config from safetensors metadata.
///
/// # Errors
///
/// Returns [`VaeDecoderError::Json`] if the `"config"` entry is invalid JSON.
pub fn read_config<S: std::hash::BuildHasher>(
    meta: &HashMap<String, String, S>,
) -> Result<serde_json::Value, VaeDecoderError> {
    meta.get("config").map_or_else(
        || Ok(serde_json::Value::Object(serde_json::Map::new())),
        |s| serde_json::from_str(s).map_err(VaeDecoderError::Json),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_decode_accepts_unaligned_bytes() {
        let mut bytes = vec![0_u8];
        bytes.extend_from_slice(&1.25_f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.5_f32).to_le_bytes());

        let values = f32_values_from_le_bytes("w", &[2], &bytes[1..]).unwrap();

        assert_eq!(values, vec![1.25_f32, -2.5_f32]);
    }

    #[test]
    fn f16_decode_accepts_unaligned_bytes() {
        let mut bytes = vec![0_u8];
        bytes.extend_from_slice(&f16::from_f32(1.5).to_le_bytes());
        bytes.extend_from_slice(&f16::from_f32(-0.25).to_le_bytes());

        let values = f16_values_from_le_bytes("w", &[2], &bytes[1..]).unwrap();

        assert_eq!(values, vec![1.5_f32, -0.25_f32]);
    }

    #[test]
    fn byte_length_mismatch_errors_instead_of_panicking() {
        let err = f32_values_from_le_bytes("w", &[2], &[0_u8; 7]).unwrap_err();

        assert!(matches!(err, VaeDecoderError::InvalidArgument { .. }));
    }
}
