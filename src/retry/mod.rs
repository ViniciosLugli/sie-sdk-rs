//! The retry state machine shared by every endpoint.
//!
//! The Python SDK repeats a near-identical loop in each method, with small per-endpoint
//! differences buried inside it. Here those differences are a policy struct and the loop
//! exists once.

pub mod backoff;

use std::time::{Duration, Instant};

use crate::error::{Error, Result, TransportErrorKind, codes};
use crate::http::HttpResponse;
use crate::wire;

/// Per-call knobs every request builder exposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestOptions {
    /// Machine profile, optionally prefixed with a pool: `"pool/l4"` or `"l4"`.
    pub gpu: Option<String>,
    /// Whether to wait out provisioning and transport hiccups, or fail fast.
    pub wait_for_capacity: bool,
    /// Total wall-clock budget for the call, retries included.
    pub provision_timeout: Duration,
    /// Cap on `RESOURCE_EXHAUSTED` retries. Zero fails fast.
    pub max_oom_retries: u32,
}

impl Default for RequestOptions {
    fn default() -> Self {
        Self {
            gpu: None,
            wait_for_capacity: true,
            provision_timeout: backoff::DEFAULT_PROVISION_TIMEOUT,
            max_oom_retries: backoff::RESOURCE_EXHAUSTED_MAX_RETRIES,
        }
    }
}

/// Which retry branches an endpoint participates in.
///
/// The asymmetries are deliberate, not accidents: generation is non-idempotent, so it
/// never replays a request that may already have reached a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetryPolicy {
    /// Poll through `LORA_LOADING`. Only `/v1/encode` can receive it.
    pub lora_loading: bool,
    /// Whether `MODEL_LOADING` polling requires `wait_for_capacity`.
    pub model_loading_gated_on_wait: bool,
    /// Whether a 504 may be replayed. Only true for idempotent endpoints.
    pub retry_gateway_timeout: bool,
    /// Whether a failure before the connection was established may be replayed.
    pub retry_connect: bool,
    /// Whether a failure after the request was written may be replayed.
    pub retry_midflight_transport: bool,
    /// Whether `RESOURCE_EXHAUSTED` retries require `wait_for_capacity`.
    pub oom_gated_on_wait: bool,
    /// Whether a 503 capacity code means the request could not be priced at all.
    pub estimate_unroutable: bool,
}

impl RetryPolicy {
    /// `/v1/score`, `/v1/extract`: idempotent, so everything is replayable.
    pub(crate) const INFERENCE: Self = Self {
        lora_loading: false,
        model_loading_gated_on_wait: false,
        retry_gateway_timeout: true,
        retry_connect: true,
        retry_midflight_transport: true,
        oom_gated_on_wait: false,
        estimate_unroutable: false,
    };

    /// `/v1/encode`: as [`Self::INFERENCE`], plus `LoRA` adapter polling.
    pub(crate) const ENCODE: Self = Self {
        lora_loading: true,
        ..Self::INFERENCE
    };

    /// `/v1/generate` buffered: non-idempotent, and model loading is opt-in.
    pub(crate) const GENERATE: Self = Self {
        lora_loading: false,
        model_loading_gated_on_wait: true,
        retry_gateway_timeout: false,
        retry_connect: true,
        retry_midflight_transport: false,
        oom_gated_on_wait: false,
        estimate_unroutable: false,
    };

    /// `/v1/chat/completions`, `/v1/responses`, and every SSE stream.
    pub(crate) const STREAM: Self = Self {
        lora_loading: false,
        model_loading_gated_on_wait: false,
        retry_gateway_timeout: false,
        retry_connect: true,
        retry_midflight_transport: false,
        oom_gated_on_wait: true,
        estimate_unroutable: false,
    };

    /// Endpoints with no capacity semantics at all (models, health, files, jobs, pools).
    ///
    /// These are single-shot: a metadata call against a server that is down must fail in
    /// seconds, not sit inside a provisioning budget.
    pub(crate) const NONE: Self = Self {
        lora_loading: false,
        model_loading_gated_on_wait: false,
        retry_gateway_timeout: false,
        retry_connect: false,
        retry_midflight_transport: false,
        oom_gated_on_wait: false,
        estimate_unroutable: false,
    };

    /// `/v1/estimate`: a capacity code means the request cannot be priced, not that it
    /// should be retried.
    pub(crate) const ESTIMATE: Self = Self {
        estimate_unroutable: true,
        ..Self::NONE
    };
}

/// What the caller should do with a response the retry machine has inspected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// The response is terminal and successful; hand it to the parser.
    Accept,
    /// Sleep for this long, then replay the request.
    Retry(Duration),
}

/// Mutable state for one logical call, spanning all its attempts.
#[derive(Debug)]
pub(crate) struct RetryState {
    policy: RetryPolicy,
    wait_for_capacity: bool,
    max_oom_retries: u32,
    budget: Duration,
    start: Instant,
    gpu: Option<String>,
    model: Option<String>,
    retries: u32,
    oom_retries: u32,
    lora_retries: u32,
}

impl RetryState {
    pub(crate) fn new(policy: RetryPolicy, options: &RequestOptions, model: Option<&str>) -> Self {
        Self {
            policy,
            wait_for_capacity: options.wait_for_capacity,
            max_oom_retries: options.max_oom_retries,
            budget: options.provision_timeout,
            start: Instant::now(),
            gpu: options.gpu.clone(),
            model: model.map(str::to_string),
            retries: 0,
            oom_retries: 0,
            lora_retries: 0,
        }
    }

    /// Retries performed so far, reported back through [`crate::types::RequestMetadata`].
    pub(crate) fn retries(&self) -> u32 {
        self.retries
    }

    fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    fn remaining(&self) -> Duration {
        self.budget.saturating_sub(self.elapsed())
    }

    /// The timeout for the next attempt, or the terminal error when the budget is spent.
    pub(crate) fn attempt_timeout(&self, client_timeout: Duration) -> Result<Duration> {
        let remaining = self.remaining();
        if remaining.is_zero() {
            return Err(Error::Provisioning {
                message: format!(
                    "Provision timeout ({:.1}s) exceeded before request could be sent",
                    self.budget.as_secs_f64()
                ),
                gpu: self.gpu.clone(),
                retry_after: None,
            });
        }
        Ok(client_timeout.min(remaining))
    }

    /// Classify and act on a transport-level failure.
    pub(crate) fn on_transport_error(
        &mut self,
        error: &reqwest::Error,
        base_url: &str,
    ) -> Result<Duration> {
        let kind = classify_transport_error(error);
        let retryable = self.wait_for_capacity
            && match kind {
                TransportErrorKind::Connect => {
                    self.policy.retry_connect && is_transient_connect_error(error)
                }
                TransportErrorKind::Timeout | TransportErrorKind::MidFlight => {
                    self.policy.retry_midflight_transport
                }
            };

        if retryable && let Some(delay) = backoff::transport_delay(self.elapsed(), self.budget) {
            tracing::debug!(
                "{} retrying in {:.1}s (elapsed: {:.1}s, timeout: {:.1}s): {error}",
                transport_label(kind),
                delay.as_secs_f64(),
                self.elapsed().as_secs_f64(),
                self.budget.as_secs_f64(),
            );
            self.retries += 1;
            return Ok(delay);
        }

        let message = match kind {
            TransportErrorKind::Connect => format!("Failed to connect to {base_url}: {error}"),
            TransportErrorKind::Timeout => format!("Request timed out: {error}"),
            TransportErrorKind::MidFlight => format!(
                "Connection lost mid-request; the peer closed the connection before sending a \
                 complete response: {error}"
            ),
        };
        Err(Error::Connection {
            message,
            kind,
            source: None,
        })
    }

    /// Inspect a response and decide whether to accept, retry, or fail.
    ///
    /// Only `>= 400` responses reach the error branches; a 2xx or 3xx is accepted and left
    /// for the endpoint's parser.
    pub(crate) fn on_response(&mut self, response: &HttpResponse) -> Result<Decision> {
        // Terminal load failures short-circuit before any budget is engaged: retrying a
        // gated repo or a missing dependency wastes the whole provision window.
        wire::check_model_load_failed(response, self.model.as_deref(), self.retries)?;
        if self.policy.estimate_unroutable {
            wire::check_estimate_unroutable(response, self.retries)?;
        }

        if response.status == 503
            && let Some(delay) = self.on_service_unavailable(response)?
        {
            self.retries += 1;
            return Ok(Decision::Retry(delay));
        }

        if response.status == 504 {
            if let Some(delay) = self.on_gateway_timeout(response) {
                self.retries += 1;
                return Ok(Decision::Retry(delay));
            }
            if !self.policy.retry_gateway_timeout {
                return Err(Error::Server {
                    message: "Gateway timed out (504) after the request was published to the queue; \
                              a worker may already be generating. Not retried because generation is \
                              non-idempotent (retrying could double-bill)."
                        .to_string(),
                    code: wire::error_code(response),
                    status: 504,
                    request: crate::http::metadata::parse(&response.headers, None, self.retries).map(Box::new),
                });
            }
        }

        if response.status >= 400 {
            return Err(wire::handle_error(
                response,
                self.model.as_deref(),
                self.retries,
            ));
        }
        Ok(Decision::Accept)
    }

    /// The 503 branches. `Ok(None)` means "not a capacity code, fall through to the
    /// terminal handler".
    fn on_service_unavailable(&mut self, response: &HttpResponse) -> Result<Option<Duration>> {
        let Some(code) = wire::error_code(response) else {
            return Ok(None);
        };
        let hint = backoff::retry_after(&response.headers);

        match code.as_str() {
            codes::PROVISIONING => self.provisioning_delay(hint).map(Some),
            codes::LORA_LOADING if self.policy.lora_loading => self.lora_delay(hint).map(Some),
            codes::MODEL_LOADING => {
                if self.policy.model_loading_gated_on_wait && !self.wait_for_capacity {
                    return Ok(None);
                }
                self.model_loading_delay(hint).map(Some)
            }
            codes::RESOURCE_EXHAUSTED => {
                if self.policy.oom_gated_on_wait && !self.wait_for_capacity {
                    return Err(self.resource_exhausted(response));
                }
                self.oom_delay(response, hint).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn provisioning_delay(&self, hint: Option<Duration>) -> Result<Duration> {
        let gpu_label = self.gpu.as_deref().unwrap_or("default");
        if !self.wait_for_capacity {
            return Err(Error::Provisioning {
                message: format!(
                    "No capacity available for GPU '{gpu_label}'. Server is provisioning."
                ),
                gpu: self.gpu.clone(),
                retry_after: hint,
            });
        }
        let remaining = self.remaining();
        if remaining.is_zero() {
            return Err(Error::Provisioning {
                message: format!(
                    "Provisioning timeout after {:.1}s waiting for GPU '{gpu_label}'",
                    self.elapsed().as_secs_f64()
                ),
                gpu: self.gpu.clone(),
                retry_after: hint,
            });
        }
        // A server hint is honoured verbatim; only the SDK's own default is jittered.
        Ok(match hint {
            Some(hint) => hint.min(remaining),
            None => backoff::apply_jitter(backoff::DEFAULT_RETRY_DELAY.min(remaining)),
        })
    }

    fn model_loading_delay(&self, hint: Option<Duration>) -> Result<Duration> {
        let remaining = self.remaining();
        if remaining.is_zero() {
            return Err(Error::ModelLoading {
                message: format!(
                    "Model loading timeout after {:.1}s for '{}'",
                    self.elapsed().as_secs_f64(),
                    self.model.as_deref().unwrap_or("model")
                ),
                model: self.model.clone(),
            });
        }
        Ok(backoff::retry_after_or(hint, backoff::MODEL_LOADING_DELAY).min(remaining))
    }

    fn lora_delay(&mut self, hint: Option<Duration>) -> Result<Duration> {
        self.lora_retries += 1;
        let remaining = self.remaining();
        if self.lora_retries > backoff::LORA_LOADING_MAX_RETRIES || remaining.is_zero() {
            return Err(Error::LoraLoading {
                message: format!(
                    "LoRA loading timeout after {} retries",
                    self.lora_retries - 1
                ),
                lora: None,
                model: self.model.clone(),
            });
        }
        // Clamped to the remaining budget, unlike the Python SDK, which can overshoot
        // `provision_timeout_s` by up to ten seconds on this branch.
        Ok(backoff::retry_after_or(hint, backoff::LORA_LOADING_DELAY).min(remaining))
    }

    fn oom_delay(&mut self, response: &HttpResponse, hint: Option<Duration>) -> Result<Duration> {
        let remaining = self.remaining();
        if self.oom_retries >= self.max_oom_retries || remaining.is_zero() {
            return Err(self.resource_exhausted(response));
        }
        let delay = backoff::oom_backoff(hint, self.oom_retries);
        // Sleeping the whole remaining budget would surface a timeout instead of the root
        // cause, so report the exhaustion now.
        if delay >= remaining {
            return Err(self.resource_exhausted(response));
        }
        if self.oom_retries == 0 {
            tracing::warn!(
                "Server resource exhausted, retrying in {:.1}s (attempt 1/{}, elapsed: {:.1}s, timeout: {:.1}s)",
                delay.as_secs_f64(),
                self.max_oom_retries,
                self.elapsed().as_secs_f64(),
                self.budget.as_secs_f64(),
            );
        } else {
            tracing::info!(
                "Server resource exhausted, retrying in {:.1}s (attempt {}/{})",
                delay.as_secs_f64(),
                self.oom_retries + 1,
                self.max_oom_retries,
            );
        }
        self.oom_retries += 1;
        Ok(delay)
    }

    fn resource_exhausted(&self, response: &HttpResponse) -> Error {
        Error::ResourceExhausted {
            message: format!(
                "Server resource exhausted after {} retry attempt(s) for model '{}'",
                self.oom_retries,
                self.model.as_deref().unwrap_or("unknown")
            ),
            model: self.model.clone(),
            retries: self.oom_retries,
            request: crate::http::metadata::parse(&response.headers, None, self.retries)
                .map(Box::new),
        }
    }

    fn on_gateway_timeout(&self, response: &HttpResponse) -> Option<Duration> {
        if !self.policy.retry_gateway_timeout || !self.wait_for_capacity {
            return None;
        }
        let remaining = self.remaining();
        if remaining.is_zero() {
            return None;
        }
        let hint = backoff::retry_after(&response.headers);
        let delay = backoff::retry_after_or(hint, backoff::MODEL_LOADING_DELAY).min(remaining);
        tracing::info!(
            "Gateway timeout (504), retrying in {:.1}s (elapsed: {:.1}s, timeout: {:.1}s)",
            delay.as_secs_f64(),
            self.elapsed().as_secs_f64(),
            self.budget.as_secs_f64(),
        );
        Some(delay)
    }
}

fn transport_label(kind: TransportErrorKind) -> &'static str {
    match kind {
        TransportErrorKind::Connect => "Connect error,",
        TransportErrorKind::Timeout => "Request timeout,",
        TransportErrorKind::MidFlight => "Transient transport error,",
    }
}

pub(crate) fn classify_transport_error(error: &reqwest::Error) -> TransportErrorKind {
    if error.is_timeout() {
        TransportErrorKind::Timeout
    } else if error.is_connect() {
        TransportErrorKind::Connect
    } else {
        TransportErrorKind::MidFlight
    }
}

/// Errnos that describe a peer or network that may recover on its own.
const TRANSIENT_ERRNOS: &[i32] = &[
    libc_errno::ECONNREFUSED,
    libc_errno::ECONNRESET,
    libc_errno::ETIMEDOUT,
    libc_errno::EHOSTUNREACH,
    libc_errno::ENETUNREACH,
    libc_errno::ENETDOWN,
    libc_errno::EHOSTDOWN,
];

/// Whether a connect-time failure is worth replaying.
///
/// Walks the source chain looking for an OS errno. A TLS handshake failure surfaces as
/// `InvalidData` with no errno and is never transient: the peer's certificate or protocol
/// will not fix itself within a provision window. When nothing conclusive is found the
/// answer is "transient", matching the Python SDK, which defaults to retrying on platforms
/// that do not surface an errno.
pub(crate) fn is_transient_connect_error(error: &reqwest::Error) -> bool {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = source {
        if let Some(io_error) = current.downcast_ref::<std::io::Error>() {
            if let Some(errno) = io_error.raw_os_error() {
                return TRANSIENT_ERRNOS.contains(&errno);
            }
            if io_error.kind() == std::io::ErrorKind::InvalidData {
                return false;
            }
        }
        source = current.source();
    }
    true
}

/// The handful of errno values the SDK cares about, without pulling in a libc dependency.
#[allow(non_snake_case)]
mod libc_errno {
    pub const ECONNREFUSED: i32 = 111;
    pub const ECONNRESET: i32 = 104;
    pub const ETIMEDOUT: i32 = 110;
    pub const EHOSTUNREACH: i32 = 113;
    pub const ENETUNREACH: i32 = 101;
    pub const ENETDOWN: i32 = 100;
    pub const EHOSTDOWN: i32 = 112;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use reqwest::header::HeaderMap;

    fn response(status: u16, code: Option<&str>, retry_after: Option<&str>) -> HttpResponse {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(value) = retry_after {
            headers.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
        }
        let body = match code {
            Some(code) => format!(r#"{{"error": {{"code": "{code}", "message": "m"}}}}"#),
            None => "{}".to_string(),
        };
        HttpResponse {
            status,
            headers,
            body: Bytes::from(body),
        }
    }

    fn state(policy: RetryPolicy, options: RequestOptions) -> RetryState {
        RetryState::new(policy, &options, Some("bge-m3"))
    }

    #[test]
    fn success_is_accepted() {
        let mut state = state(RetryPolicy::ENCODE, RequestOptions::default());
        assert_eq!(
            state.on_response(&response(200, None, None)).unwrap(),
            Decision::Accept
        );
        assert_eq!(state.retries(), 0);
    }

    #[test]
    fn provisioning_retries_and_honours_the_hint() {
        let mut state = state(RetryPolicy::ENCODE, RequestOptions::default());
        let decision = state
            .on_response(&response(503, Some(codes::PROVISIONING), Some("7")))
            .unwrap();
        assert_eq!(decision, Decision::Retry(Duration::from_secs(7)));
        assert_eq!(state.retries(), 1);
    }

    #[test]
    fn provisioning_fails_fast_without_wait_for_capacity() {
        let mut state = state(
            RetryPolicy::ENCODE,
            RequestOptions {
                wait_for_capacity: false,
                ..Default::default()
            },
        );
        let err = state
            .on_response(&response(503, Some(codes::PROVISIONING), Some("5")))
            .unwrap_err();
        assert!(matches!(err, Error::Provisioning { .. }));
        assert_eq!(err.retry_after(), Some(Duration::from_secs(5)));
    }

    #[test]
    fn lora_loading_only_applies_to_encode() {
        let mut encode = state(RetryPolicy::ENCODE, RequestOptions::default());
        assert_eq!(
            encode
                .on_response(&response(503, Some(codes::LORA_LOADING), None))
                .unwrap(),
            Decision::Retry(backoff::LORA_LOADING_DELAY)
        );

        let mut score = state(RetryPolicy::INFERENCE, RequestOptions::default());
        let err = score
            .on_response(&response(503, Some(codes::LORA_LOADING), None))
            .unwrap_err();
        assert!(matches!(err, Error::Server { status: 503, .. }));
    }

    #[test]
    fn lora_loading_gives_up_after_ten_retries() {
        let mut state = state(RetryPolicy::ENCODE, RequestOptions::default());
        for _ in 0..backoff::LORA_LOADING_MAX_RETRIES {
            state
                .on_response(&response(503, Some(codes::LORA_LOADING), None))
                .unwrap();
        }
        let err = state
            .on_response(&response(503, Some(codes::LORA_LOADING), None))
            .unwrap_err();
        match err {
            Error::LoraLoading { message, .. } => {
                assert!(message.contains("after 10 retries"), "{message}");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn model_loading_gating_differs_between_generate_and_encode() {
        let fail_fast = RequestOptions {
            wait_for_capacity: false,
            ..Default::default()
        };

        let mut encode = state(RetryPolicy::ENCODE, fail_fast.clone());
        assert_eq!(
            encode
                .on_response(&response(503, Some(codes::MODEL_LOADING), None))
                .unwrap(),
            Decision::Retry(backoff::MODEL_LOADING_DELAY),
        );

        let mut generate = state(RetryPolicy::GENERATE, fail_fast);
        let err = generate
            .on_response(&response(503, Some(codes::MODEL_LOADING), None))
            .unwrap_err();
        assert!(matches!(err, Error::Server { status: 503, .. }));
    }

    #[test]
    fn oom_retries_are_capped_and_then_surface_the_root_cause() {
        let mut state = state(RetryPolicy::ENCODE, RequestOptions::default());
        for attempt in 0..backoff::RESOURCE_EXHAUSTED_MAX_RETRIES {
            let decision = state
                .on_response(&response(503, Some(codes::RESOURCE_EXHAUSTED), None))
                .unwrap();
            let Decision::Retry(delay) = decision else {
                panic!("expected a retry on attempt {attempt}")
            };
            assert!(delay <= backoff::RESOURCE_EXHAUSTED_MAX_DELAY);
        }
        let err = state
            .on_response(&response(503, Some(codes::RESOURCE_EXHAUSTED), None))
            .unwrap_err();
        match err {
            Error::ResourceExhausted {
                retries, message, ..
            } => {
                assert_eq!(retries, 3);
                assert!(message.contains("bge-m3"), "{message}");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn oom_is_disabled_by_zero_max_retries() {
        let mut state = state(
            RetryPolicy::ENCODE,
            RequestOptions {
                max_oom_retries: 0,
                ..Default::default()
            },
        );
        let err = state
            .on_response(&response(503, Some(codes::RESOURCE_EXHAUSTED), None))
            .unwrap_err();
        assert!(matches!(err, Error::ResourceExhausted { retries: 0, .. }));
    }

    #[test]
    fn oom_is_gated_on_wait_for_capacity_only_on_stream_paths() {
        let fail_fast = RequestOptions {
            wait_for_capacity: false,
            ..Default::default()
        };

        let mut buffered = state(RetryPolicy::ENCODE, fail_fast.clone());
        assert!(matches!(
            buffered.on_response(&response(503, Some(codes::RESOURCE_EXHAUSTED), None)),
            Ok(Decision::Retry(_))
        ));

        let mut streaming = state(RetryPolicy::STREAM, fail_fast);
        assert!(matches!(
            streaming.on_response(&response(503, Some(codes::RESOURCE_EXHAUSTED), None)),
            Err(Error::ResourceExhausted { .. })
        ));
    }

    #[test]
    fn gateway_timeout_is_replayed_only_for_idempotent_endpoints() {
        let mut encode = state(RetryPolicy::ENCODE, RequestOptions::default());
        assert_eq!(
            encode.on_response(&response(504, None, Some("3"))).unwrap(),
            Decision::Retry(Duration::from_secs(3))
        );

        let mut generate = state(RetryPolicy::GENERATE, RequestOptions::default());
        let err = generate
            .on_response(&response(504, None, None))
            .unwrap_err();
        match err {
            Error::Server {
                message, status, ..
            } => {
                assert_eq!(status, 504);
                assert!(message.contains("double-bill"), "{message}");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn gateway_timeout_is_terminal_without_wait_for_capacity() {
        let mut encode = state(
            RetryPolicy::ENCODE,
            RequestOptions {
                wait_for_capacity: false,
                ..Default::default()
            },
        );
        let err = encode.on_response(&response(504, None, None)).unwrap_err();
        assert!(matches!(err, Error::Server { status: 504, .. }));
    }

    #[test]
    fn model_load_failure_short_circuits_before_the_retry_budget() {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        let failed = HttpResponse {
            status: 502,
            headers,
            body: Bytes::from(
                r#"{"error": {"code": "MODEL_LOAD_FAILED", "message": "gated", "error_class": "GATED"}}"#,
            ),
        };
        let mut state = state(RetryPolicy::ENCODE, RequestOptions::default());
        assert!(matches!(
            state.on_response(&failed),
            Err(Error::ModelLoadFailed { .. })
        ));
        assert_eq!(state.retries(), 0);
    }

    #[test]
    fn exhausted_budget_refuses_to_send() {
        let state = state(
            RetryPolicy::ENCODE,
            RequestOptions {
                provision_timeout: Duration::ZERO,
                ..Default::default()
            },
        );
        let err = state.attempt_timeout(Duration::from_secs(30)).unwrap_err();
        match err {
            Error::Provisioning { message, .. } => {
                assert!(message.contains("before request could be sent"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn attempt_timeout_is_clamped_to_the_remaining_budget() {
        let state = state(
            RetryPolicy::ENCODE,
            RequestOptions {
                provision_timeout: Duration::from_secs(2),
                ..Default::default()
            },
        );
        assert!(state.attempt_timeout(Duration::from_secs(30)).unwrap() <= Duration::from_secs(2));
        assert_eq!(
            state.attempt_timeout(Duration::from_millis(500)).unwrap(),
            Duration::from_millis(500)
        );
    }
}
