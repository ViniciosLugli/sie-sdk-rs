//! Model catalogue, health and capacity payloads.

// Wire-mirror types: field names are the API contract itself, and the ones whose
// meaning is not obvious carry their own doc comment.
#![allow(missing_docs)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ModelState;

/// Dimensionality of each output a model can produce.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDims {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multivector: Option<u32>,
}

/// Optional generation features a model supports.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Grammar kinds accepted by `generate`: any of `json_schema`, `regex`, `ebnf`.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub grammar: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub tools: bool,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub lora_adapters: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub profile_lora_adapters: HashMap<String, Vec<String>>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub code: bool,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub sql: bool,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub guard: bool,
}

/// One entry of the model catalogue.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub name: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub loaded: bool,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub inputs: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub outputs: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub dims: ModelDims,
    /// Lifecycle state on the worker that holds it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ModelState>,
    /// Named profile variants, when the model advertises more than one.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub profiles: HashMap<String, Value>,
    /// Why the last load attempt failed, when one did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sequence_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub capabilities: ModelCapabilities,
}

/// Cluster-wide rollup of a model's availability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSummary {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ModelState>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub worker_count: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub gpu_types: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub total_queue_depth: u64,
}

/// One worker as the gateway sees it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInfo {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub name: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub url: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub gpu: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub gpu_count: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub ready_gpu_slots: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub healthy: bool,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub queue_depth: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub pending_cost: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub inflight_batches: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub loaded_models: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub memory_used_bytes: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub memory_total_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_config_hash: Option<String>,
}

/// Cluster totals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ClusterSummary {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub worker_count: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub gpu_count: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub models_loaded: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub total_qps: f64,
}

/// `GET /health` on a gateway or a worker.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HealthResponse {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub status: String,
    /// `gateway` or `worker`.
    #[serde(
        default,
        deserialize_with = "crate::types::null_as_default",
        rename = "type"
    )]
    pub kind: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub cluster: ClusterSummary,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub configured_gpu_types: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub live_gpu_types: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub workers: Vec<WorkerInfo>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub models: Vec<ModelSummary>,
}

/// Capacity as reported for the whole cluster, or for one GPU type.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CapacityInfo {
    /// `healthy`, `degraded` or `no_workers`.
    pub status: String,
    pub worker_count: u32,
    pub gpu_count: u32,
    pub models_loaded: u32,
    pub configured_gpu_types: Vec<String>,
    pub live_gpu_types: Vec<String>,
    pub workers: Vec<WorkerInfo>,
}

/// Process-level facts about a worker.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerInfo {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub version: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub uptime_seconds: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub user: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub working_dir: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub pid: i64,
}

/// One GPU's live utilization.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// Device handle, such as `cuda:0`.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub device: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub name: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub gpu_type: String,
    /// 0 to 100.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub utilization_pct: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub memory_used_bytes: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub memory_total_bytes: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub memory_threshold_pct: f64,
}

/// How a model was configured on the worker that loaded it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub hf_id: Option<String>,
    #[serde(default)]
    pub adapter: Option<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub inputs: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub dims: Option<Value>,
    #[serde(default)]
    pub max_sequence_length: Option<u32>,
    #[serde(default)]
    pub pooling: Option<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub normalize: bool,
    #[serde(default)]
    pub adapter_options_loadtime: Option<Value>,
    #[serde(default)]
    pub adapter_options_runtime: Option<Value>,
}

/// Adaptive batching state. Absent entirely when adaptive batching is off.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveBatchingStatus {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub calibrated: bool,
    #[serde(default)]
    pub target_p50_ms: Option<f64>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub wait_ms: f64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub batch_cost: u64,
    #[serde(default)]
    pub p50_ms: Option<f64>,
    #[serde(default)]
    pub headroom_ms: Option<f64>,
    #[serde(default)]
    pub fill_ratio: Option<f64>,
}

/// One model's state on one worker.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelStatus {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ModelState>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub memory_bytes: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub config: ModelConfig,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub queue_depth: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub queue_pending_items: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_batching: Option<AdaptiveBatchingStatus>,
}

/// A worker's periodic status broadcast.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerStatusMessage {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub timestamp: f64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub ready: bool,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub name: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub machine_profile: String,
    /// Empty when the worker is not running in queue mode.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub pool_name: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub gpu_count: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub bundle: String,
    /// Empty when the worker has not computed one.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub bundle_config_hash: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub loaded_models: Vec<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub server: ServerInfo,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub gpus: Vec<GpuMetrics>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub models: Vec<ModelStatus>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub max_batch_requests: u32,
    /// The worker owns this hysteresis; do not re-derive it from queue depth.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub saturated: bool,
}

/// A gateway's periodic status broadcast.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClusterStatusMessage {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub timestamp: f64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub cluster: ClusterSummary,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub workers: Vec<WorkerInfo>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub models: Vec<ModelSummary>,
}

/// One status broadcast, from whichever endpoint the client watched.
///
/// The two shapes share no required field, so which one a payload is comes from the
/// endpoint it arrived on rather than from the payload itself.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum StatusMessage {
    /// A gateway reporting on its whole cluster.
    Cluster(Box<ClusterStatusMessage>),
    /// A single worker reporting on itself.
    Worker(Box<WorkerStatusMessage>),
}

impl StatusMessage {
    /// The cluster summary, when this message came from a gateway.
    pub fn cluster(&self) -> Option<&ClusterStatusMessage> {
        match self {
            Self::Cluster(message) => Some(message),
            Self::Worker(_) => None,
        }
    }

    /// The worker report, when this message came from a worker.
    pub fn worker(&self) -> Option<&WorkerStatusMessage> {
        match self {
            Self::Worker(message) => Some(message),
            Self::Cluster(_) => None,
        }
    }
}
