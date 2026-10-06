//! Weight-loading helpers used by the per-module `load_weights_from_scope` methods.

use burn::module::Param;
use burn::tensor::{Tensor, backend::Backend};

use crate::error::VaeError;

/// Load a tensor from `scope` at `key`, validate its shape against `expected`,
/// and wrap it in a `Param` ready to replace a module weight.
///
/// The `Param` is created with `require_grad = true`.  For `NdArray` (the test
/// backend) this flag is a no-op; for autodiff backends the encoder is used
/// only in inference, so gradients are not needed.
///
/// # Errors
/// Returns [`VaeError::Load`] if the tensor is missing or has the wrong rank,
/// or [`VaeError::Config`] if its shape differs from `expected`.
pub fn load_param<B: Backend, const D: usize>(
    scope: &ltx_weights::Scope<'_>,
    key: &str,
    expected: &[usize],
    device: &B::Device,
) -> Result<Param<Tensor<B, D>>, VaeError> {
    let t: Tensor<B, D> = scope.tensor(key, device)?;
    check_dims(key, expected, &t.dims())?;
    Ok(Param::from_tensor(t))
}

/// Load a raw (non-parameter) tensor from `scope` at `key` and validate its
/// shape against `expected`.
///
/// Used for module buffers stored as plain `Tensor` fields (e.g.
/// `PerChannelStatistics::std_of_means`).
///
/// # Errors
/// Returns [`VaeError::Load`] if the tensor is missing or has the wrong rank,
/// or [`VaeError::Config`] if its shape differs from `expected`.
pub fn load_tensor<B: Backend, const D: usize>(
    scope: &ltx_weights::Scope<'_>,
    key: &str,
    expected: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, D>, VaeError> {
    let t: Tensor<B, D> = scope.tensor(key, device)?;
    check_dims(key, expected, &t.dims())?;
    Ok(t)
}

fn check_dims<const D: usize>(key: &str, expected: &[usize], got: &[usize; D]) -> Result<(), VaeError> {
    if got.as_slice() == expected {
        Ok(())
    } else {
        Err(VaeError::Config(format!(
            "`{key}`: expected shape {expected:?}, got {got:?}"
        )))
    }
}
