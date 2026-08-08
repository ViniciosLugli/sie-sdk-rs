//! Conversions into [`ndarray`] types, for callers whose numeric code already uses it.
//!
//! Enabled by the `ndarray` cargo feature. The core types stay `Vec`-based so the crate
//! does not force a numeric dependency on anyone who does not want one.

use half::f16;
use ndarray::{Array1, Array2};

use crate::error::{Error, Result};
use crate::types::{EncodeResult, Multivector, SparseVector};

/// The dense embedding as a 1-D array.
pub fn dense_array(result: &EncodeResult) -> Result<Array1<f32>> {
    Ok(Array1::from(result.require_dense()?.to_vec()))
}

/// A multivector as a `[tokens, dims]` array of `f32`.
pub fn multivector_f32(multivector: &Multivector) -> Result<Array2<f32>> {
    rows_to_array(multivector.to_f32())
}

/// A multivector as a `[tokens, dims]` array of `f16`, when the wire carried `f16`.
///
/// Returns `None` for an `f32` multivector rather than narrowing it, which would lose
/// precision the server chose to send.
pub fn multivector_f16(multivector: &Multivector) -> Option<Result<Array2<f16>>> {
    match multivector {
        Multivector::F16(rows) => Some(rows_to_array(rows.clone())),
        Multivector::F32(_) => None,
    }
}

/// A sparse embedding as a dense array of `dims` elements.
///
/// Every index must be inside `dims`; a term id past the end means the caller's `dims` does
/// not match the model's vocabulary.
pub fn sparse_to_dense(sparse: &SparseVector, dims: usize) -> Result<Array1<f32>> {
    let mut dense = Array1::zeros(dims);
    for (index, value) in sparse.indices.iter().zip(&sparse.values) {
        let index = *index as usize;
        if index >= dims {
            return Err(Error::decode(format!(
                "sparse index {index} is outside a {dims}-dimensional vector"
            )));
        }
        dense[index] = *value;
    }
    Ok(dense)
}

fn rows_to_array<T>(rows: Vec<Vec<T>>) -> Result<Array2<T>> {
    let height = rows.len();
    let width = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != width) {
        return Err(Error::decode("multivector rows have inconsistent widths"));
    }
    Array2::from_shape_vec((height, width), rows.into_iter().flatten().collect())
        .map_err(|err| Error::decode(format!("could not shape the multivector: {err}")))
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn dense_becomes_a_one_dimensional_array() {
        let result = EncodeResult {
            dense: Some(vec![0.5, -0.25, 1.0]),
            ..EncodeResult::default()
        };
        let array = dense_array(&result).unwrap();
        assert_eq!(array.len(), 3);
        assert_eq!(array[1], -0.25);
        assert!(dense_array(&EncodeResult::default()).is_err());
    }

    #[test]
    fn multivectors_keep_their_shape() {
        let multivector = Multivector::F32(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let array = multivector_f32(&multivector).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array[[1, 2]], 6.0);
    }

    #[test]
    fn f16_multivectors_are_offered_without_narrowing_f32_ones() {
        let narrow = Multivector::F16(vec![vec![f16::from_f32(1.5), f16::from_f32(0.5)]]);
        let array = multivector_f16(&narrow).unwrap().unwrap();
        assert_eq!(array.shape(), &[1, 2]);
        assert_eq!(array[[0, 0]], f16::from_f32(1.5));

        assert!(multivector_f16(&Multivector::F32(vec![vec![1.0]])).is_none());
    }

    #[test]
    fn ragged_multivectors_are_rejected_rather_than_reshaped() {
        let ragged = Multivector::F32(vec![vec![1.0, 2.0], vec![3.0]]);
        assert!(multivector_f32(&ragged).is_err());
    }

    #[test]
    fn sparse_expands_into_a_dense_vector() {
        let sparse = SparseVector {
            indices: vec![0, 3],
            values: vec![1.0, 0.5],
        };
        let dense = sparse_to_dense(&sparse, 4).unwrap();
        assert_eq!(dense.to_vec(), vec![1.0, 0.0, 0.0, 0.5]);
        assert!(sparse_to_dense(&sparse, 3).is_err());
    }
}
