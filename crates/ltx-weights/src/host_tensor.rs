//! Host-side f32 tensor data returned by [`WeightStore::read`][crate::WeightStore::read].

use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::error::WeightError;

/// A host-side f32 copy of a weight tensor.
///
/// Created by [`WeightStore::read`][crate::WeightStore::read].  All dtypes
/// (`F32`, `F16`, `BF16`, `FP8`) are dequantized to f32 in the store; merged `LoRA`
/// deltas are already applied.
#[derive(Debug, Clone)]
pub struct HostTensor {
    /// Tensor shape (dimension lengths, outermost first).
    pub shape: Vec<usize>,
    /// Flat f32 data in row-major (C) order.
    pub data: Vec<f32>,
}

impl HostTensor {
    /// Move the host data onto a Burn device as a rank-`D` tensor.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError::RankMismatch`] when `self.shape.len() != D`.
    pub fn into_tensor<B: Backend, const D: usize>(
        self,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, WeightError> {
        let got = self.shape.len();
        if got != D {
            return Err(WeightError::RankMismatch {
                key: String::new(),
                got,
                expected: D,
            });
        }
        let shape_arr: [usize; D] =
            std::array::from_fn(|i| self.shape.get(i).copied().unwrap_or(0));
        Ok(Tensor::<B, D>::from_data(
            TensorData::new(self.data, shape_arr),
            device,
        ))
    }
}
