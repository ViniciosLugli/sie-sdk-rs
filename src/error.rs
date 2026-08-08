//! The single error type returned by every fallible SDK operation.
//!
//! The Python SDK models failures as an exception hierarchy. Rust has no inheritance, so the
//! hierarchy collapses into one enum and the `isinstance` checks become the predicates on
//! [`Error`] ([`Error::is_server_error`], [`Error::status`], [`Error::code`], ...).

use std::time::Duration;

use crate::types::RequestMetadata;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Server error codes the SDK reacts to. Any other code is carried through as a string.
#[allow(missing_docs)]
pub mod codes {
    pub const LORA_LOADING: &str = "LORA_LOADING";
    pub const MODEL_LOADING: &str = "MODEL_LOADING";
    pub const PROVISIONING: &str = "PROVISIONING";
    pub const MODEL_LOAD_FAILED: &str = "MODEL_LOAD_FAILED";
    pub const INPUT_TOO_LONG: &str = "INPUT_TOO_LONG";
    pub const RESOURCE_EXHAUSTED: &str = "RESOURCE_EXHAUSTED";
    pub const QUEUE_UNAVAILABLE: &str = "QUEUE_UNAVAILABLE";
    pub const INTERNAL_ERROR: &str = "INTERNAL_ERROR";
    pub const ENCODE_RESULT_COUNT_MISMATCH: &str = "ENCODE_RESULT_COUNT_MISMATCH";
}

/// Why a model failed to load, as classified by the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLoadErrorClass {
    /// Repository requires accepting a licence or supplying a token.
    Gated,
    /// Out of device memory while loading.
    Oom,
    /// A required runtime dependency is missing.
    Dependency,
    /// The model id does not exist.
    NotFound,
    /// Transient network failure while fetching weights.
    Network,
    /// Anything else.
    Unknown,
}

impl ModelLoadErrorClass {
    fn from_wire(value: &str) -> Self {
        match value {
            "GATED" => Self::Gated,
            "OOM" => Self::Oom,
            "DEPENDENCY" => Self::Dependency,
            "NOT_FOUND" => Self::NotFound,
            "NETWORK" => Self::Network,
            _ => Self::Unknown,
        }
    }

    pub(crate) fn parse(value: Option<&str>) -> Self {
        value.map_or(Self::Unknown, Self::from_wire)
    }
}

/// How a transport-level failure occurred, which decides whether it may be retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportErrorKind {
    /// The connection was never established. Safe to retry for every endpoint.
    Connect,
    /// The request was in flight when the failure occurred. Only idempotent endpoints retry.
    MidFlight,
    /// The per-attempt timeout elapsed.
    Timeout,
}

/// Everything that can go wrong talking to a SIE server.
///
/// The variant fields are the failure's evidence: `message` is what the server said,
/// `code` is its error code, `status` the HTTP status, and `request` the metering metadata
/// that survived. Read them through the accessors when you only need one.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum Error {
    /// The request never produced an HTTP response.
    #[error("{message}")]
    Connection {
        message: String,
        kind: TransportErrorKind,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// A 4xx the SDK does not model more specifically.
    #[error("{message}")]
    Request {
        message: String,
        code: Option<String>,
        status: u16,
        request: Option<Box<RequestMetadata>>,
    },

    /// A 5xx the SDK does not model more specifically.
    #[error("{message}")]
    Server {
        message: String,
        code: Option<String>,
        status: u16,
        request: Option<Box<RequestMetadata>>,
    },

    /// The input exceeded the model's context window (400 `INPUT_TOO_LONG`).
    #[error("{message}")]
    InputTooLong {
        message: String,
        model: Option<String>,
        request: Option<Box<RequestMetadata>>,
    },

    /// The worker could not load the model (502 `MODEL_LOAD_FAILED`).
    #[error("{message}")]
    ModelLoadFailed {
        message: String,
        model: Option<String>,
        error_class: ModelLoadErrorClass,
        /// `false` only for cooldown-suppressed OOM/network failures, which may succeed later.
        permanent: bool,
        attempts: u32,
        request: Option<Box<RequestMetadata>>,
    },

    /// The server ran out of device memory and the SDK exhausted its retries.
    #[error("{message}")]
    ResourceExhausted {
        message: String,
        model: Option<String>,
        retries: u32,
        request: Option<Box<RequestMetadata>>,
    },

    /// `/v1/estimate` could not route the request to a rate identity.
    #[error("{message}")]
    EstimateUnroutable {
        message: String,
        code: Option<String>,
        request: Option<Box<RequestMetadata>>,
    },

    /// No capacity, and either the caller opted out of waiting or the budget ran out.
    #[error("{message}")]
    Provisioning {
        message: String,
        gpu: Option<String>,
        retry_after: Option<Duration>,
    },

    /// The model was still loading when the provision budget expired.
    #[error("{message}")]
    ModelLoading {
        message: String,
        model: Option<String>,
    },

    /// A `LoRA` adapter was still loading after the retry cap.
    #[error("{message}")]
    LoraLoading {
        message: String,
        lora: Option<String>,
        model: Option<String>,
    },

    /// A pool operation failed.
    #[error("{message}")]
    Pool {
        message: String,
        pool_name: Option<String>,
        state: Option<String>,
    },

    /// A response body could not be decoded as the documented shape.
    #[error("{0}")]
    Decode(String),

    /// The caller supplied arguments the SDK rejects before sending anything.
    #[error("{0}")]
    InvalidRequest(String),

    /// Local I/O failed (reading a file to upload, for example).
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    pub(crate) fn decode(message: impl Into<String>) -> Self {
        Self::Decode(message.into())
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    pub(crate) fn connection(
        kind: TransportErrorKind,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Connection {
            message: message.into(),
            kind,
            source: Some(Box::new(source)),
        }
    }

    /// HTTP status that produced this error, when there was a response.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Request { status, .. } | Self::Server { status, .. } => Some(*status),
            Self::InputTooLong { .. } => Some(400),
            Self::ModelLoadFailed { .. } => Some(502),
            Self::ResourceExhausted { .. }
            | Self::EstimateUnroutable { .. }
            | Self::Provisioning { .. }
            | Self::ModelLoading { .. } => Some(503),
            _ => None,
        }
    }

    /// Server error code (`X-SIE-Error-Code` or the body's `code` field).
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::Request { code, .. }
            | Self::Server { code, .. }
            | Self::EstimateUnroutable { code, .. } => code.as_deref(),
            Self::InputTooLong { .. } => Some(codes::INPUT_TOO_LONG),
            Self::ModelLoadFailed { .. } => Some(codes::MODEL_LOAD_FAILED),
            Self::ResourceExhausted { .. } => Some(codes::RESOURCE_EXHAUSTED),
            Self::Provisioning { .. } => Some(codes::PROVISIONING),
            Self::ModelLoading { .. } => Some(codes::MODEL_LOADING),
            Self::LoraLoading { .. } => Some(codes::LORA_LOADING),
            _ => None,
        }
    }

    /// Metadata the server attached to the failing request, when any survived parsing.
    pub fn request_metadata(&self) -> Option<&RequestMetadata> {
        #[allow(clippy::borrowed_box)]
        match self {
            Self::Request { request, .. }
            | Self::Server { request, .. }
            | Self::InputTooLong { request, .. }
            | Self::ModelLoadFailed { request, .. }
            | Self::ResourceExhausted { request, .. }
            | Self::EstimateUnroutable { request, .. } => request.as_deref(),
            _ => None,
        }
    }

    /// The server's `Retry-After` hint, when it sent one and the SDK gave up anyway.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Provisioning { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// True for every failure the server attributed to itself (5xx family).
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            Self::Server { .. }
                | Self::ModelLoadFailed { .. }
                | Self::ResourceExhausted { .. }
                | Self::EstimateUnroutable { .. }
        )
    }

    /// True for every failure the server attributed to the request (4xx family).
    pub fn is_request_error(&self) -> bool {
        matches!(self, Self::Request { .. } | Self::InputTooLong { .. })
    }

    /// True when the SDK gave up waiting for capacity rather than hitting a hard failure.
    pub fn is_capacity_error(&self) -> bool {
        matches!(
            self,
            Self::Provisioning { .. }
                | Self::ModelLoading { .. }
                | Self::LoraLoading { .. }
                | Self::ResourceExhausted { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicates_follow_the_python_hierarchy() {
        let exhausted = Error::ResourceExhausted {
            message: "boom".into(),
            model: None,
            retries: 3,
            request: None,
        };
        assert!(exhausted.is_server_error());
        assert!(exhausted.is_capacity_error());
        assert_eq!(exhausted.status(), Some(503));
        assert_eq!(exhausted.code(), Some(codes::RESOURCE_EXHAUSTED));

        let too_long = Error::InputTooLong {
            message: "too long".into(),
            model: Some("m".into()),
            request: None,
        };
        assert!(too_long.is_request_error());
        assert!(!too_long.is_server_error());
        assert_eq!(too_long.status(), Some(400));
    }

    #[test]
    fn model_load_error_class_defaults_to_unknown() {
        assert_eq!(
            ModelLoadErrorClass::parse(Some("GATED")),
            ModelLoadErrorClass::Gated
        );
        assert_eq!(
            ModelLoadErrorClass::parse(Some("nonsense")),
            ModelLoadErrorClass::Unknown
        );
        assert_eq!(
            ModelLoadErrorClass::parse(None),
            ModelLoadErrorClass::Unknown
        );
    }
}
