//! Files, batches, connections and pools.

// Wire-mirror types: field names are the API contract itself, and the ones whose
// meaning is not obvious carry their own doc comment.
#![allow(missing_docs)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An uploaded file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct File {
    #[serde(default)]
    pub id: String,
    #[serde(default, rename = "object")]
    pub object_kind: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub purpose: String,
    #[serde(default)]
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
}

/// One page of the file list.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileList {
    #[serde(default, rename = "object")]
    pub object_kind: String,
    #[serde(default)]
    pub data: Vec<File>,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    #[serde(default)]
    pub has_more: bool,
}

/// Confirmation that a file was deleted.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDeleted {
    #[serde(default)]
    pub id: String,
    #[serde(default, rename = "object")]
    pub object_kind: String,
    #[serde(default)]
    pub deleted: bool,
}

/// How many requests in a batch are in each state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub completed: u64,
    #[serde(default)]
    pub failed: u64,
}

/// An offline batch of requests.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    #[serde(default)]
    pub id: String,
    #[serde(default, rename = "object")]
    pub object_kind: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub input_file_id: String,
    #[serde(default)]
    pub completion_window: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub output_file_id: Option<String>,
    #[serde(default)]
    pub error_file_id: Option<String>,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub finished_at: Option<i64>,
    #[serde(default)]
    pub in_progress_at: Option<i64>,
    #[serde(default)]
    pub completed_at: Option<i64>,
    #[serde(default)]
    pub failed_at: Option<i64>,
    #[serde(default)]
    pub expired_at: Option<i64>,
    #[serde(default)]
    pub cancelled_at: Option<i64>,
    #[serde(default)]
    pub request_counts: BatchRequestCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errors: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// One page of the batch list.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchList {
    #[serde(default, rename = "object")]
    pub object_kind: String,
    #[serde(default)]
    pub data: Vec<Batch>,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    #[serde(default)]
    pub has_more: bool,
}

/// A stored credential for a data store a connector job can read or write.
///
/// The secret itself is never returned.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Connection {
    #[serde(default)]
    pub id: i64,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    /// Bumped whenever the secret is rotated.
    #[serde(default)]
    pub authorization_generation: i64,
    #[serde(default)]
    pub source_schema: Option<String>,
    #[serde(default)]
    pub sink_schema: Option<String>,
    #[serde(default)]
    pub created_at: f64,
}

/// A newly created connection, with the org it belongs to.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConnectionCreated {
    #[serde(default)]
    pub org: String,
    #[serde(default)]
    pub account_id: i64,
    #[serde(flatten)]
    pub connection: Connection,
}

/// Confirmation that a connection was revoked.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionRevoked {
    #[serde(default)]
    pub org: String,
    #[serde(default)]
    pub account_id: i64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub state: String,
}

/// A worker assigned to a pool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssignedWorkerInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub bundle: String,
}

/// A pool's live state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PoolStatusInfo {
    /// `pending`, `active` or `expired`.
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub assigned_workers: Vec<AssignedWorkerInfo>,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default)]
    pub last_renewed: f64,
}

/// A pool's requested shape, as the server echoes it back.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolSpecResponse {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub queue_pool: String,
    #[serde(default)]
    pub gpus: HashMap<String, u32>,
    /// Ceiling on assigned workers per GPU type.
    #[serde(default)]
    pub gpu_caps: HashMap<String, u32>,
    #[serde(default)]
    pub bundle: Option<String>,
    #[serde(default)]
    pub minimum_worker_count: u32,
    #[serde(default)]
    pub pinned_models: Vec<String>,
}

/// A pool, with its spec and its status.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PoolInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub spec: PoolSpecResponse,
    #[serde(default)]
    pub status: PoolStatusInfo,
}

/// A pool as it appears in a listing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolListItem {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub gpus: HashMap<String, u32>,
    #[serde(default)]
    pub worker_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn connection_created_flattens_the_connection_it_wraps() {
        let created: ConnectionCreated = serde_json::from_value(json!({
            "org": "acme", "account_id": 7, "id": 1, "type": "postgres", "name": "warehouse",
            "authorization_generation": 1, "created_at": 1.0
        }))
        .unwrap();
        assert_eq!(created.org, "acme");
        assert_eq!(created.connection.name, "warehouse");
        assert_eq!(created.connection.kind, "postgres");
        assert!(created.connection.source_schema.is_none());
    }

    #[test]
    fn batches_tolerate_the_null_timestamp_fields() {
        let batch: Batch = serde_json::from_value(json!({
            "id": "batch_1", "object": "batch", "endpoint": "/v1/embeddings",
            "input_file_id": "file_1", "completion_window": "24h", "status": "in_progress",
            "output_file_id": null, "failed_at": null, "created_at": 100,
            "request_counts": {"total": 10, "completed": 3, "failed": 0}
        }))
        .unwrap();
        assert_eq!(batch.status, "in_progress");
        assert_eq!(batch.output_file_id, None);
        assert_eq!(batch.request_counts.completed, 3);
    }

    #[test]
    fn a_file_without_the_optional_ttl_fields_decodes() {
        let file: File = serde_json::from_value(json!({
            "id": "file_1", "object": "file", "bytes": 42, "created_at": 1,
            "filename": "in.jsonl", "purpose": "batch", "status": "processed"
        }))
        .unwrap();
        assert_eq!(file.bytes, 42);
        assert!(file.expires_at.is_none());
    }
}
