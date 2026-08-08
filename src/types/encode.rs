//! Results of `/v1/encode`, `/v1/score` and `/v1/extract`.

// Wire-mirror types: field names are the API contract itself, and the ones whose
// meaning is not obvious carry their own doc comment.
#![allow(missing_docs)]

use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{RequestMetadata, TimingInfo};

/// A sparse embedding: term ids paired with their weights.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    /// The `{term_id: weight}` form most vector stores expect.
    pub fn to_map(&self) -> std::collections::HashMap<u32, f32> {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .collect()
    }

    /// How many terms the vector carries.
    pub fn len(&self) -> usize {
        self.indices.len().min(self.values.len())
    }

    /// Whether the vector carries no terms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A token-level embedding matrix, in the precision the model produced.
///
/// Late-interaction models emit `f16` by default; the SDK keeps that precision rather than
/// silently widening, because [`crate::scoring::maxsim`] has to control exactly where the
/// widening happens.
#[derive(Debug, Clone, PartialEq)]
pub enum Multivector {
    F16(Vec<Vec<f16>>),
    F32(Vec<Vec<f32>>),
}

impl Multivector {
    /// Number of token vectors.
    pub fn len(&self) -> usize {
        match self {
            Self::F16(rows) => rows.len(),
            Self::F32(rows) => rows.len(),
        }
    }

    /// Whether there are no token vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Width of each token vector.
    pub fn dims(&self) -> usize {
        match self {
            Self::F16(rows) => rows.first().map_or(0, Vec::len),
            Self::F32(rows) => rows.first().map_or(0, Vec::len),
        }
    }

    /// Widen to `f32` rows, copying when the wire precision was `f16`.
    pub fn to_f32(&self) -> Vec<Vec<f32>> {
        match self {
            Self::F16(rows) => rows
                .iter()
                .map(|row| row.iter().map(|value| value.to_f32()).collect())
                .collect(),
            Self::F32(rows) => rows.clone(),
        }
    }
}

/// One encoded item.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EncodeResult {
    pub model: Option<String>,
    /// Echoed from the request item, when it carried an id.
    pub id: Option<String>,
    pub dense: Option<Vec<f32>>,
    pub sparse: Option<SparseVector>,
    pub multivector: Option<Multivector>,
    pub timing: Option<TimingInfo>,
    pub request: Option<RequestMetadata>,
}

impl EncodeResult {
    /// The dense embedding, or an error naming what is missing.
    ///
    /// Use the `dense` field directly when its absence is expected.
    pub fn require_dense(&self) -> crate::error::Result<&[f32]> {
        self.dense
            .as_deref()
            .ok_or_else(|| crate::error::Error::decode("encode result has no dense embedding"))
    }

    /// The sparse embedding as a `{term_id: weight}` map, empty when there is none.
    pub fn sparse_map(&self) -> std::collections::HashMap<u32, f32> {
        self.sparse
            .as_ref()
            .map(SparseVector::to_map)
            .unwrap_or_default()
    }

    /// The multivector widened to `f32` rows, empty when there is none.
    pub fn multivector_f32(&self) -> Vec<Vec<f32>> {
        self.multivector
            .as_ref()
            .map(Multivector::to_f32)
            .unwrap_or_default()
    }
}

/// One scored candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreEntry {
    pub item_id: String,
    pub score: f64,
    /// Zero-based; `0` is the most relevant candidate.
    pub rank: u32,
}

/// Units consumed by a score call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreUsage {
    pub input_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<u64>,
}

/// The result of scoring one query against a candidate set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScoreResult {
    pub model: String,
    pub query_id: Option<String>,
    /// Ranked candidates, best first.
    pub scores: Vec<ScoreEntry>,
    pub usage: Option<ScoreUsage>,
    pub request: Option<RequestMetadata>,
}

/// A span or region the model recognised.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub score: f64,
    /// Character offset into the item text, for text models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<i64>,
    /// `[x, y, w, h]` in pixels, for visual models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Vec<i64>>,
}

/// A directed relation between two entities.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    pub head: String,
    pub tail: String,
    pub relation: String,
    pub score: f64,
}

/// A label with a confidence.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    pub label: String,
    pub score: f64,
}

/// A detected object with its bounding box.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DetectedObject {
    pub label: String,
    pub score: f64,
    /// `[x, y, w, h]` in pixels.
    pub bbox: Vec<i64>,
}

/// A per-item extraction failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractItemError {
    pub code: String,
    pub message: String,
}

/// One extracted item. Which fields are populated depends on the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtractResult {
    pub model: Option<String>,
    pub id: Option<String>,
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub classifications: Vec<Classification>,
    pub objects: Vec<DetectedObject>,
    /// Free-form structured output, for models driven by an output schema.
    pub data: Option<Value>,
    /// Set when this item failed while others in the batch succeeded.
    pub error: Option<ExtractItemError>,
    pub request: Option<RequestMetadata>,
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn sparse_converts_to_a_term_weight_map() {
        let sparse = SparseVector {
            indices: vec![3, 9],
            values: vec![0.5, 0.25],
        };
        let map = sparse.to_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&9], 0.25);
        assert_eq!(sparse.len(), 2);
        assert!(SparseVector::default().is_empty());
    }

    #[test]
    fn encode_result_accessors_are_lenient_except_where_they_promise_not_to_be() {
        let empty = EncodeResult::default();
        assert!(empty.require_dense().is_err());
        assert!(empty.sparse_map().is_empty());
        assert!(empty.multivector_f32().is_empty());

        let filled = EncodeResult {
            dense: Some(vec![0.5, 0.25]),
            sparse: Some(SparseVector {
                indices: vec![4],
                values: vec![1.0],
            }),
            ..EncodeResult::default()
        };
        assert_eq!(filled.require_dense().unwrap(), &[0.5, 0.25]);
        assert_eq!(filled.sparse_map()[&4], 1.0);
    }

    #[test]
    fn multivector_reports_its_shape_and_widens_on_request() {
        let mv = Multivector::F16(vec![
            vec![f16::from_f32(1.0), f16::from_f32(0.5)],
            vec![f16::from_f32(0.0), f16::from_f32(-1.0)],
        ]);
        assert_eq!(mv.len(), 2);
        assert_eq!(mv.dims(), 2);
        assert_eq!(mv.to_f32(), vec![vec![1.0, 0.5], vec![0.0, -1.0]]);
        assert!(Multivector::F32(Vec::new()).is_empty());
    }
}
