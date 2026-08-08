//! Wire types for every SIE endpoint.

pub mod cluster;
pub mod common;
pub mod encode;
pub mod estimate;
pub mod generate;
pub mod item;
pub mod jobs;
pub mod resources;

pub use cluster::{
    AdaptiveBatchingStatus, CapacityInfo, ClusterStatusMessage, ClusterSummary, GpuMetrics,
    HealthResponse, ModelCapabilities, ModelConfig, ModelDims, ModelInfo, ModelStatus,
    ModelSummary, ServerInfo, StatusMessage, WorkerInfo, WorkerStatusMessage,
};
pub use common::{
    DType, ModelState, OutputDType, OutputType, RequestMetadata, RequestUsage, TimingInfo,
};
pub use encode::{
    Classification, DetectedObject, EncodeResult, Entity, ExtractItemError, ExtractResult,
    Multivector, Relation, ScoreEntry, ScoreResult, ScoreUsage, SparseVector,
};
pub use estimate::{AppliedRate, CostEstimate, RateIdentity};
pub use generate::{
    ChatChoice, ChatChunkChoice, ChatCompletion, ChatCompletionChunk, ChatContent, ChatContentPart,
    ChatDelta, ChatFinishReason, ChatImageUrl, ChatMessage, ChatRole, ChatUsage, FinishReason,
    GenerateChunk, GenerateResult, GenerationUsage, Grammar, ResponseInputMessage,
    ResponseOutputMessage, ResponseOutputText, ResponseResult, ResponseUsage,
};
pub use item::{AudioInput, BinaryInput, ImageInput, Item};
pub use jobs::{
    ConnectorJobAttempt, ConnectorJobCapabilities, ConnectorJobCheckpoint,
    ConnectorJobItemOutcomes, ConnectorJobOverlapOwner, ConnectorJobPlan, ConnectorJobPublication,
    ConnectorJobRecovery, ConnectorJobRepairWindow, ConnectorJobValidation,
    ConnectorPlanOutputShape, JobChunk, JobExecution, JobFieldMap, JobPreflight, JobResultItem,
    JobResults, JobState, JobStatus,
};
pub use resources::{
    AssignedWorkerInfo, Batch, BatchList, BatchRequestCounts, Connection, ConnectionCreated,
    ConnectionRevoked, File, FileDeleted, FileList, PoolInfo, PoolListItem, PoolSpecResponse,
    PoolStatusInfo,
};

/// Deserialize a field that the server may send as an explicit `null`.
///
/// `#[serde(default)]` only covers a *missing* key. SIE sends `null` for fields it has no
/// value for, which without this would be a hard decode failure rather than a default.
pub(crate) fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Default,
{
    use serde::Deserialize as _;

    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}
