//! Batch jobs, including the connector-driven ones.

// Wire-mirror types: field names are the API contract itself, and the ones whose
// meaning is not obvious carry their own doc comment.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Where a job is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Failed,
    Suspended,
    Cancelled,
}

impl JobState {
    /// Whether the job has stopped for good.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Suspended | Self::Cancelled
        )
    }
}

/// Which half of a connector job's two-phase execution to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobExecution {
    /// Compute a plan and price it, without touching the sink.
    Plan,
    /// Execute a plan that was already computed.
    Run,
}

/// How a connector source's rows map onto job items.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobFieldMap {
    /// Source column holding the row's stable identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_field: Option<String>,
    /// Source column holding the text or document to process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_field: Option<String>,
    /// Source columns copied through to the sink unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub carry: Vec<String>,
    /// How to interpret `input_field`: `text` or `document`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
}

impl JobFieldMap {
    pub(crate) fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// The cost the server estimated before starting the job.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobPreflight {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_credits: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimate_basis: Option<String>,
}

/// One completed slice of a job's output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JobChunk {
    #[serde(default)]
    pub seq: u64,
    #[serde(default)]
    pub items: u64,
    #[serde(default)]
    pub state: String,
    /// Where the chunk's msgpack payload is stored. Time-limited, and never the payload
    /// itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<u64>,
    /// Chunk charges sum exactly to the job's `settled_credits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits_charged: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_book_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

/// What a connector plan validated about its endpoints.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorJobValidation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sink: Option<String>,
}

/// What the connector pair can do, which decides how much of a re-run can be skipped.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorJobCapabilities {
    #[serde(default)]
    pub incremental_inference: bool,
    #[serde(default)]
    pub incremental_source_scan: bool,
    #[serde(default)]
    pub incremental_selection: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_scan: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_proof: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sink_targets: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordering: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deletion_handling: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publication: Option<String>,
}

/// The shape a connector plan will write.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorPlanOutputShape {
    #[serde(default)]
    pub result_kind: String,
    #[serde(default)]
    pub output_field: String,
    #[serde(default)]
    pub output_types: Vec<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// A priced, executable description of the work a connector job would do.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConnectorJobPlan {
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub expires_at: f64,
    #[serde(default)]
    pub executable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_available: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_availability: Option<String>,
    /// Why the plan cannot execute, when it cannot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocking_code: Option<String>,
    #[serde(default)]
    pub rows: u64,
    #[serde(default)]
    pub mapped_bytes: u64,
    #[serde(default)]
    pub input_bytes: u64,
    #[serde(default)]
    pub eligible_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eligible_count_quality: Option<String>,
    #[serde(default)]
    pub eligible_input_byte_count: u64,
    #[serde(default)]
    pub matched_checkpoint_count: u64,
    #[serde(default)]
    pub skipped_unchanged_count: u64,
    #[serde(default)]
    pub deleted_preserved_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dimensions: Option<u32>,
    #[serde(default)]
    pub output: ConnectorPlanOutputShape,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_basis: Option<String>,
    #[serde(default)]
    pub max_reservation_credits: u64,
    #[serde(default)]
    pub validation: ConnectorJobValidation,
    #[serde(default)]
    pub capabilities: ConnectorJobCapabilities,
}

/// Where the sink's incremental state stands.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorJobCheckpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(default)]
    pub profile_version: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default)]
    pub expected_generation: u64,
    #[serde(default)]
    pub generation: u64,
    #[serde(default)]
    pub published_revision: u64,
}

/// Per-item tallies for one attempt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorJobItemOutcomes {
    #[serde(default)]
    pub claimed: u64,
    #[serde(default)]
    pub dispatched: u64,
    #[serde(default)]
    pub inferred: u64,
    #[serde(default)]
    pub staged: u64,
    #[serde(default)]
    pub published: u64,
    #[serde(default)]
    pub failed: u64,
    #[serde(default)]
    pub reexecution_required: u64,
    #[serde(default)]
    pub skipped_unchanged: u64,
}

/// What one publication committed to the sink.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ConnectorJobPublication {
    #[serde(default)]
    pub attempt_ordinal: u64,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub published: u64,
    #[serde(default)]
    pub skipped_unchanged: u64,
    #[serde(default)]
    pub deleted: u64,
    #[serde(default)]
    pub failed: u64,
    #[serde(default)]
    pub reexecuted: u64,
    #[serde(default)]
    pub committed_at: f64,
}

/// The job currently holding a sink region another job wants.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorJobOverlapOwner {
    #[serde(default)]
    pub job_id: String,
    #[serde(default)]
    pub attempt_ordinal: u64,
}

/// One execution or repair pass over a plan.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConnectorJobAttempt {
    #[serde(default)]
    pub ordinal: u64,
    /// `execute` or `repair`.
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_attempt_ordinal: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(default)]
    pub replayed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billed_credits: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlap_owner: Option<ConnectorJobOverlapOwner>,
    #[serde(default)]
    pub item_outcomes: ConnectorJobItemOutcomes,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publication: Option<ConnectorJobPublication>,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<f64>,
}

/// How long a failed job may still be repaired.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ConnectorJobRepairWindow {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempts_used: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempts_remaining: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempts_max: Option<u32>,
}

/// Whether the job needs a repair pass, and what is left of its window.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConnectorJobRecovery {
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(default)]
    pub reexecution_required: bool,
    #[serde(default)]
    pub repair: ConnectorJobRepairWindow,
}

/// A job as the server reports it.
///
/// The same shape answers submit, get, cancel, execute and repair. Connector source and
/// sink URIs are deliberately absent: they can carry credentials.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JobStatus {
    #[serde(default)]
    pub id: String,
    #[serde(default, rename = "object")]
    pub object_kind: String,
    #[serde(default)]
    pub operation: String,
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<JobState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<JobExecution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_items: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_items: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunks: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preflight: Option<JobPreflight>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settled_credits: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_expires_at: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_expires_at: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<ConnectorJobPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<ConnectorJobCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt: Option<ConnectorJobAttempt>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attempts: Vec<ConnectorJobAttempt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publication: Option<ConnectorJobPublication>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<ConnectorJobRecovery>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<f64>,
    /// Holds `output.chunks`, the per-chunk result locations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
}

impl JobStatus {
    /// The chunks recorded under `output`.
    pub fn chunks(&self) -> Vec<JobChunk> {
        self.output
            .as_ref()
            .and_then(|output| output.get("chunks"))
            .and_then(|chunks| serde_json::from_value(chunks.clone()).ok())
            .unwrap_or_default()
    }

    /// Whether the job has stopped for good, or has produced a plan awaiting execution.
    pub fn is_settled(&self) -> bool {
        self.state.is_some_and(JobState::is_terminal) || self.phase.as_deref() == Some("planned")
    }
}

/// One decoded result row from a job chunk.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobResultItem {
    pub id: Option<String>,
    pub success: Option<bool>,
    /// Units billed for this row, as reported by the worker.
    pub units: Option<Value>,
    pub dims: Option<u32>,
    pub dense: Option<Vec<f32>>,
    /// The failure the worker reported, when this row failed.
    pub error: Option<String>,
}

/// A job's results, with every retrievable chunk decoded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobResults {
    pub job_id: String,
    pub state: Option<JobState>,
    pub total_items: Option<u64>,
    pub settled_credits: Option<u64>,
    pub chunks: Vec<JobChunk>,
    /// How many chunks were actually fetched; the rest had expired or had not succeeded.
    pub retrieved: usize,
    pub dims: Option<u32>,
    pub items: Vec<JobResultItem>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn terminal_states_are_the_ones_that_stop_a_poll() {
        assert!(!JobState::Queued.is_terminal());
        assert!(!JobState::Running.is_terminal());
        for state in [
            JobState::Succeeded,
            JobState::Failed,
            JobState::Suspended,
            JobState::Cancelled,
        ] {
            assert!(state.is_terminal(), "{state:?}");
        }
    }

    #[test]
    fn a_planned_connector_job_counts_as_settled() {
        let planned = JobStatus {
            state: Some(JobState::Running),
            phase: Some("planned".to_string()),
            ..JobStatus::default()
        };
        assert!(planned.is_settled());

        let running = JobStatus {
            state: Some(JobState::Running),
            ..JobStatus::default()
        };
        assert!(!running.is_settled());
    }

    #[test]
    fn chunks_are_read_out_of_the_output_envelope() {
        let status = JobStatus {
            output: Some(json!({"chunks": [
                {"seq": 0, "items": 10, "state": "succeeded", "ref": "https://store/chunk-0"},
                {"seq": 1, "items": 4, "state": "failed"}
            ]})),
            ..JobStatus::default()
        };
        let chunks = status.chunks();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].r#ref.as_deref(), Some("https://store/chunk-0"));
        assert_eq!(chunks[1].r#ref, None);
        assert!(JobStatus::default().chunks().is_empty());
    }

    #[test]
    fn job_status_tolerates_a_minimal_payload() {
        let status: JobStatus = serde_json::from_value(json!({
            "id": "job_1", "object": "job", "operation": "encode", "model": "m", "state": "queued"
        }))
        .unwrap();
        assert_eq!(status.state, Some(JobState::Queued));
        assert!(status.plan.is_none());
        assert!(status.attempts.is_empty());
    }
}
