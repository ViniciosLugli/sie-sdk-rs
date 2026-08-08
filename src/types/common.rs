//! Scalar wire enums and the per-request metadata every endpoint can return.

// Wire-mirror types: field names are the API contract itself, and the ones whose
// meaning is not obvious carry their own doc comment.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

/// Compute dtype of a tensor on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Binary,
    UBinary,
}

/// Dtype a caller may request for returned embeddings.
///
/// `bfloat16` is compute-only and is never returned, so it has no variant here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputDType {
    #[default]
    Float32,
    Float16,
    Int8,
    UInt8,
    Binary,
    UBinary,
}

/// Which representations an encode call should return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Dense,
    Sparse,
    Multivector,
}

/// Lifecycle state of a model on a worker.
///
/// Pinned to `packages/wire-fixtures/model_state.json` in the SIE repo, which is the
/// cross-language source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelState {
    Available,
    Loading,
    Loaded,
    Unloading,
    Failed,
}

impl ModelState {
    /// Every variant, in fixture order.
    pub const ALL: [Self; 5] = [
        Self::Available,
        Self::Loading,
        Self::Loaded,
        Self::Unloading,
        Self::Failed,
    ];

    /// The wire string for this state.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Loading => "loading",
            Self::Loaded => "loaded",
            Self::Unloading => "unloading",
            Self::Failed => "failed",
        }
    }
}

/// Billable units consumed by one request.
///
/// Populated from the `X-SIE-Units-*` response headers, or from a response body `usage`
/// object when the server settled the charge inline.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pages: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits_charged: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_book_version: Option<String>,
}

impl RequestUsage {
    pub(crate) fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Server-reported metadata about a single request.
///
/// `retries` and `model_revision` are client-side observations; the rest come from the
/// server's response headers or body.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<RequestUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits_debited: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_book_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_identity_sha256: Option<String>,
    /// How many times the SDK retried before this response arrived.
    #[serde(default)]
    pub retries: u32,
    /// Value of `X-SIE-Model-Revision`, when the server sent one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
}

impl RequestMetadata {
    pub(crate) fn is_empty(&self) -> bool {
        self.id.is_none()
            && self.usage.is_none()
            && self.credits_debited.is_none()
            && self.rate_book_version.is_none()
            && self.execution_identity_sha256.is_none()
            && self.model_revision.is_none()
    }
}

/// Per-stage latency breakdown returned by encode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct TimingInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenization_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_strings_round_trip() {
        assert_eq!(
            serde_json::to_string(&DType::BFloat16).unwrap(),
            "\"bfloat16\""
        );
        assert_eq!(
            serde_json::to_string(&OutputType::Multivector).unwrap(),
            "\"multivector\""
        );
        assert_eq!(
            serde_json::to_string(&OutputDType::UBinary).unwrap(),
            "\"ubinary\""
        );
        for state in ModelState::ALL {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(json, format!("\"{}\"", state.as_str()));
            assert_eq!(serde_json::from_str::<ModelState>(&json).unwrap(), state);
        }
    }
}
