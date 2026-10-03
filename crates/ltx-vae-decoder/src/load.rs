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
    let n: usize = shape.iter().product();

    let data_f32: Vec<f32> = match dtype {
        safetensors::Dtype::F32 => bytemuck::cast_slice(raw).to_vec(),
        safetensors::Dtype::BF16 => {
            let u16s: &[u16] = bytemuck::cast_slice(raw);
            u16s.iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect()
        }
        safetensors::Dtype::F16 => {
            let u16s: &[u16] = bytemuck::cast_slice(raw);
            u16s.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
        }
        other => {
            return Err(VaeDecoderError::InvalidArgument {
                detail: format!("unsupported dtype {other:?} for key {key}"),
            });
        }
    };

    if data_f32.len() != n {
        return Err(VaeDecoderError::ShapeMismatch {
            key: key.to_owned(),
            expected: vec![n],
            actual: vec![data_f32.len()],
        });
    }

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
    let d0 = i32::try_from(shape[0]).unwrap_or(i32::MAX);
    let d1 = i32::try_from(shape[1]).unwrap_or(i32::MAX);
    Ok(flat.reshape([d0, d1]))
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
