//! Rust client for the [SIE](https://github.com/superlinked/sie) inference server.
//!
//! ```no_run
//! use sie_sdk::{Client, Item, OutputType};
//!
//! # async fn example() -> sie_sdk::Result<()> {
//! let client = Client::new("http://localhost:8080")?;
//!
//! let result = client
//!     .encode("BAAI/bge-m3", [Item::text("Hello world")])
//!     .output_types([OutputType::Dense, OutputType::Sparse])
//!     .send_one()
//!     .await?;
//!
//! println!("{:?}", result.dense);
//! # Ok(())
//! # }
//! ```
//!
//! # Waiting for capacity
//!
//! SIE scales from zero and loads models on demand, so a request can legitimately be
//! answered with "not yet". The client absorbs that: `PROVISIONING`, `MODEL_LOADING`,
//! `LORA_LOADING` and `RESOURCE_EXHAUSTED` are retried inside a wall-clock budget
//! (15 minutes by default) with bounded, jittered backoff. Endpoints that are not
//! idempotent never replay a request that may already have reached a worker.
//!
//! Opt out per call when a failure is more useful than a wait:
//!
//! ```no_run
//! # use sie_sdk::{Client, Item};
//! # async fn example(client: Client) -> sie_sdk::Result<()> {
//! let result = client
//!     .encode("BAAI/bge-m3", [Item::text("Hello")])
//!     .wait_for_capacity(false)
//!     .max_oom_retries(0)
//!     .send_one()
//!     .await;
//!
//! if let Err(error) = &result
//!     && error.is_capacity_error()
//! {
//!     // Fall back to a smaller model, or shed the request.
//! }
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    // Fallibility is documented on the error type, not repeated on every method.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Builder setters are the dominant shape here; annotating each one adds noise.
    clippy::return_self_not_must_use,
    clippy::must_use_candidate,
    // Retry policy is a set of independent flags; a struct of bools is the point.
    clippy::struct_excessive_bools,
    clippy::module_name_repetitions,
    // Numeric widths are chosen deliberately at each conversion site.
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::needless_pass_by_value
)]

#[cfg(feature = "blocking")]
pub mod blocking;
pub mod client;
pub mod error;
pub mod media;
#[cfg(feature = "ndarray")]
pub mod ndarray;
pub mod redaction;
pub mod retry;
pub mod scoring;
pub mod types;
pub mod wire;

mod http;

#[cfg(feature = "watch")]
pub use client::WatchMode;
pub use client::jobs::{
    connection_name, require_connection_name, require_connector_idempotency_key,
};
pub use client::{ChunkStream, Client, ClientBuilder};
pub use error::{Error, ModelLoadErrorClass, Result, TransportErrorKind};
pub use media::Samples;
pub use retry::RequestOptions;
pub use scoring::{maxsim, maxsim_batch};
pub use types::{
    AudioInput, BinaryInput, CapacityInfo, Classification, DType, DetectedObject, EncodeResult,
    Entity, ExtractResult, HealthResponse, ImageInput, Item, ModelInfo, ModelState, Multivector,
    OutputDType, OutputType, RequestMetadata, RequestUsage, ScoreResult, SparseVector, TimingInfo,
};

/// The version this SDK reports to the server.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
