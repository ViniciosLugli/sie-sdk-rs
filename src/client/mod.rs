//! The client, its builder, and the one request path every endpoint shares.

mod batches;
mod connections;
mod encode;
mod files;
pub(crate) mod generate;
pub mod jobs;
pub(crate) mod meta;
mod pools;
pub mod stream;
#[cfg(feature = "watch")]
mod watch;

pub use batches::{BatchCreate, BatchListRequest, Batches};
pub use connections::{ConnectionAdd, Connections};
pub use encode::{EncodeRequest, ExtractRequest, ScoreRequest};
pub use files::{FileListRequest, FileUpload, Files, SortOrder};
pub use generate::{ChatRequest, GenerateRequest, ResponseInput, ResponsesRequest};
pub use jobs::{
    JobItem, JobSink, JobSource, JobSubmit, Jobs, chunk_is_retrievable, connection_name,
    decode_chunk_payload, require_connection_name, require_connection_schema_policy,
    require_connector_idempotency_key,
};
pub use pools::{PoolCreate, Pools};
pub use stream::ChunkStream;
#[cfg(feature = "watch")]
pub use watch::WatchMode;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Method, Url};
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::error::{Error, Result, TransportErrorKind};
use crate::http::headers::Origin;
use crate::http::{HttpResponse, PreparedRequest, headers, version};
use crate::retry::{Decision, RequestOptions, RetryPolicy, RetryState};

/// Machine profile and pool a request should be routed to.
#[derive(Debug, Clone, Default)]
pub(crate) struct Routing {
    pub pool: Option<String>,
    pub profile: Option<String>,
}

pub(crate) struct Inner {
    http: reqwest::Client,
    /// Always ends in `/` so relative paths join predictably.
    base_url: Url,
    base_origin: Option<Origin>,
    control_plane_url: Option<Url>,
    org: Option<String>,
    timeout: Duration,
    defaults: RequestOptions,
    default_options: Option<Value>,
    edge_headers: Vec<(HeaderName, HeaderValue)>,
    /// Kept alongside the reqwest defaults for the WebSocket handshake, which builds its
    /// own request rather than going through reqwest.
    #[cfg(feature = "watch")]
    http_authorization: Option<HeaderValue>,
    concurrency: Option<Semaphore>,
    /// Background lease renewals, one per pool this client created.
    leases: Mutex<HashMap<String, tokio::task::JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Ok(leases) = self.leases.lock() {
            for handle in leases.values() {
                handle.abort();
            }
        }
    }
}

/// A handle to one SIE server.
///
/// Cloning is cheap and shares the underlying connection pool, so a `Client` can be stored
/// once and used from every task.
#[derive(Clone)]
pub struct Client {
    pub(crate) inner: Arc<Inner>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.inner.base_url.as_str())
            .field("timeout", &self.inner.timeout)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// A client with default settings.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self> {
        ClientBuilder::new(base_url).build()
    }

    /// Start configuring a client.
    pub fn builder(base_url: impl AsRef<str>) -> ClientBuilder {
        ClientBuilder::new(base_url)
    }

    /// The server root, always with a trailing slash.
    pub fn base_url(&self) -> &str {
        self.inner.base_url.as_str()
    }

    /// Per-call defaults this client applies when a request builder does not override them.
    pub fn default_options(&self) -> &RequestOptions {
        &self.inner.defaults
    }

    pub(crate) fn timeout(&self) -> Duration {
        self.inner.timeout
    }

    /// Resolve a path against the base URL.
    pub(crate) fn url(&self, path: &str) -> Result<Url> {
        self.inner
            .base_url
            .join(path.trim_start_matches('/'))
            .map_err(|err| Error::invalid(format!("invalid request path {path:?}: {err}")))
    }

    /// The control-plane base for the connections namespace.
    pub(crate) fn control_plane(&self) -> Result<(&Url, &str)> {
        let url = self.inner.control_plane_url.as_ref().ok_or_else(|| {
            Error::invalid("connections require a control_plane_url on the client builder")
        })?;
        let org =
            self.inner.org.as_deref().ok_or_else(|| {
                Error::invalid("connections require an org on the client builder")
            })?;
        Ok((url, org))
    }

    /// Split a `gpu` parameter into routing headers, falling back to the client default.
    pub(crate) fn routing(&self, gpu: Option<&str>) -> Routing {
        let resolved = gpu.or(self.inner.defaults.gpu.as_deref());
        match resolved {
            Some(value) => {
                let (pool, profile) = version::parse_gpu_param(value);
                Routing {
                    pool: pool.map(str::to_string),
                    profile: Some(profile.to_string()).filter(|p| !p.is_empty()),
                }
            }
            None => Routing::default(),
        }
    }

    /// Merge per-call runtime options over the client defaults.
    pub(crate) fn merge_options(&self, options: Option<&Value>) -> Option<Value> {
        match (self.inner.default_options.as_ref(), options) {
            (None, other) => other.cloned(),
            (Some(defaults), None) => Some(defaults.clone()),
            (Some(defaults), Some(overrides)) => {
                let mut merged = defaults.as_object().cloned().unwrap_or_default();
                if let Some(overrides) = overrides.as_object() {
                    for (key, value) in overrides {
                        merged.insert(key.clone(), value.clone());
                    }
                }
                Some(Value::Object(merged))
            }
        }
    }

    /// Build the per-call options for a request builder's defaults.
    pub(crate) fn request_options(&self) -> RequestOptions {
        self.inner.defaults.clone()
    }

    /// Options for the metadata endpoints, which never wait for capacity.
    pub(crate) fn metadata_options(&self) -> RequestOptions {
        RequestOptions {
            gpu: None,
            wait_for_capacity: false,
            provision_timeout: self.inner.timeout.max(Duration::from_secs(1)),
            max_oom_retries: 0,
        }
    }

    /// The `Authorization` value this client sends, when it has one.
    #[cfg(feature = "watch")]
    pub(crate) fn authorization_header(&self) -> Option<HeaderValue> {
        self.inner.http_authorization.clone()
    }

    /// The validated edge headers, for transports that cannot reuse the reqwest client.
    #[cfg(feature = "watch")]
    pub(crate) fn edge_headers(&self) -> &[(HeaderName, HeaderValue)] {
        &self.inner.edge_headers
    }

    /// Whether a `ws`/`wss` URL is the WebSocket counterpart of the base origin.
    #[cfg(feature = "watch")]
    pub(crate) fn websocket_matches_base_origin(&self, url: &Url) -> bool {
        !self.inner.edge_headers.is_empty()
            && self
                .inner
                .base_origin
                .as_ref()
                .is_some_and(|origin| origin.matches_websocket(url))
    }

    pub(crate) fn edge_headers_apply_to(&self, url: &Url) -> bool {
        !self.inner.edge_headers.is_empty()
            && self
                .inner
                .base_origin
                .as_ref()
                .is_some_and(|origin| Origin::parse(url).is_some_and(|target| target == *origin))
    }

    fn build_attempt(
        &self,
        request: &PreparedRequest,
        timeout: Duration,
        allow_edge: bool,
    ) -> reqwest::RequestBuilder {
        let mut builder = self
            .inner
            .http
            .request(request.method.clone(), request.url.clone())
            .timeout(timeout)
            .headers(request.headers.clone());

        if allow_edge && self.edge_headers_apply_to(&request.url) {
            for (name, value) in &self.inner.edge_headers {
                builder = builder.header(name.clone(), value.clone());
            }
        }
        match &request.body {
            Some(body) => builder.body(body.clone()),
            None => builder,
        }
    }

    async fn permit(&self) -> Option<tokio::sync::SemaphorePermit<'_>> {
        match &self.inner.concurrency {
            // The semaphore lives as long as the client, so acquire cannot fail.
            Some(semaphore) => semaphore.acquire().await.ok(),
            None => None,
        }
    }

    /// Send a request, retrying per the state's policy, and buffer the response.
    pub(crate) async fn send(
        &self,
        request: PreparedRequest,
        state: &mut RetryState,
    ) -> Result<HttpResponse> {
        loop {
            let timeout = state.attempt_timeout(self.inner.timeout)?;
            let outcome = {
                let _permit = self.permit().await;
                match self.build_attempt(&request, timeout, true).send().await {
                    Ok(response) => HttpResponse::read(response).await,
                    Err(error) => Err(error),
                }
            };

            let response = match outcome {
                Ok(response) => response,
                Err(error) => {
                    let delay = state.on_transport_error(&error, self.base_url())?;
                    tokio::time::sleep(delay).await;
                    continue;
                }
            };

            match state.on_response(&response)? {
                Decision::Accept => {
                    Self::check_version(&response);
                    return Ok(response);
                }
                Decision::Retry(delay) => tokio::time::sleep(delay).await,
            }
        }
    }

    /// Send a request whose successful response is consumed as a stream.
    ///
    /// Failures are buffered first so the shared error and retry handling sees a complete
    /// body, exactly as the buffered path does.
    pub(crate) async fn send_streaming(
        &self,
        request: &PreparedRequest,
        state: &mut RetryState,
    ) -> Result<StreamAttempt> {
        loop {
            let timeout = state.attempt_timeout(self.inner.timeout)?;
            // A stream is read after `send` returns, so the permit cannot be held for the
            // lifetime of the body without deadlocking a caller that reads them serially.
            let attempt = {
                let _permit = self.permit().await;
                self.build_attempt(request, timeout, true).send().await
            };

            let response = match attempt {
                Ok(response) => response,
                Err(error) => {
                    let delay = state.on_transport_error(&error, self.base_url())?;
                    tokio::time::sleep(delay).await;
                    continue;
                }
            };

            if response.status().is_success() {
                let buffered = HttpResponse {
                    status: response.status().as_u16(),
                    headers: response.headers().clone(),
                    body: bytes::Bytes::new(),
                };
                Self::check_version(&buffered);
                return Ok(StreamAttempt {
                    headers: buffered.headers,
                    response,
                });
            }

            let buffered = HttpResponse::read(response).await.map_err(|error| {
                Error::connection(TransportErrorKind::MidFlight, error.to_string(), error)
            })?;
            match state.on_response(&buffered)? {
                // A non-success status that the policy accepts is still an error here: the
                // caller asked for a stream and the server did not open one.
                Decision::Accept => {
                    return Err(crate::wire::handle_error(&buffered, None, state.retries()));
                }
                Decision::Retry(delay) => tokio::time::sleep(delay).await,
            }
        }
    }

    /// Warn once per process when the server and SDK versions have drifted.
    fn check_version(response: &HttpResponse) {
        if let Some(server_version) = response.header(headers::SERVER_VERSION) {
            version::warn_once(server_version);
        }
    }

    /// A single request with no retry policy, used by endpoints that own their own loop.
    pub(crate) async fn send_once(
        &self,
        request: PreparedRequest,
        policy: RetryPolicy,
    ) -> Result<HttpResponse> {
        let mut state = RetryState::new(policy, &self.metadata_options(), None);
        self.send(request, &mut state).await
    }

    /// A single request whose timeout is raised to at least `floor`.
    ///
    /// File and batch transfers move whole payloads, so the per-request timeout that suits
    /// an inference call is too tight for them.
    pub(crate) async fn send_with_timeout(
        &self,
        request: PreparedRequest,
        policy: RetryPolicy,
        floor: Duration,
    ) -> Result<HttpResponse> {
        let options = RequestOptions {
            provision_timeout: self.inner.timeout.max(floor),
            ..self.metadata_options()
        };
        let mut state = RetryState::new(policy, &options, None);
        self.send(request, &mut state).await
    }

    /// Build a request with the SDK's standard headers already applied.
    pub(crate) fn request(&self, method: Method, path: &str) -> Result<PreparedRequest> {
        Ok(PreparedRequest::new(method, self.url(path)?))
    }
}

/// A stream response that passed the retry gauntlet.
pub(crate) struct StreamAttempt {
    pub headers: HeaderMap,
    pub response: reqwest::Response,
}

/// Configures a [`Client`].
pub struct ClientBuilder {
    base_url: String,
    api_key: Option<String>,
    timeout: Duration,
    gpu: Option<String>,
    options: Option<Value>,
    max_connections: Option<usize>,
    max_concurrency: Option<usize>,
    control_plane_url: Option<String>,
    org: Option<String>,
    base_url_headers: HashMap<String, String>,
    wait_for_capacity: bool,
    provision_timeout: Duration,
    max_oom_retries: u32,
}

impl ClientBuilder {
    fn new(base_url: impl AsRef<str>) -> Self {
        let defaults = RequestOptions::default();
        Self {
            base_url: base_url.as_ref().to_string(),
            api_key: None,
            timeout: Duration::from_secs(30),
            gpu: None,
            options: None,
            max_connections: None,
            max_concurrency: None,
            control_plane_url: None,
            org: None,
            base_url_headers: HashMap::new(),
            wait_for_capacity: defaults.wait_for_capacity,
            provision_timeout: defaults.provision_timeout,
            max_oom_retries: defaults.max_oom_retries,
        }
    }

    /// Bearer token sent as `Authorization` on every request to the server.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Per-attempt HTTP timeout. Defaults to 30 seconds.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Default machine profile, optionally pool-qualified as `"pool/profile"`.
    pub fn gpu(mut self, gpu: impl Into<String>) -> Self {
        self.gpu = Some(gpu.into());
        self
    }

    /// Default runtime options, shallow-merged under any per-call options.
    pub fn options(mut self, options: Value) -> Self {
        self.options = Some(options);
        self
    }

    /// Cap on pooled connections.
    pub fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = Some(max);
        self
    }

    /// Cap on requests in flight at once.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Control-plane root, required by the connections namespace.
    pub fn control_plane_url(mut self, url: impl Into<String>) -> Self {
        self.control_plane_url = Some(url.into());
        self
    }

    /// Organisation slug, required by the connections namespace.
    pub fn org(mut self, org: impl Into<String>) -> Self {
        self.org = Some(org.into());
        self
    }

    /// Extra headers for an HTTP edge in front of the gateway.
    ///
    /// These are credentials: they are sent only to the exact origin of `base_url`, which
    /// must therefore be `https` without embedded userinfo.
    pub fn base_url_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.base_url_headers = headers;
        self
    }

    /// Whether calls wait out provisioning by default.
    pub fn wait_for_capacity(mut self, wait: bool) -> Self {
        self.wait_for_capacity = wait;
        self
    }

    /// Default wall-clock budget for a call including its retries.
    pub fn provision_timeout(mut self, timeout: Duration) -> Self {
        self.provision_timeout = timeout;
        self
    }

    /// Default cap on `RESOURCE_EXHAUSTED` retries.
    pub fn max_oom_retries(mut self, retries: u32) -> Self {
        self.max_oom_retries = retries;
        self
    }

    /// Validate the configuration and construct the client.
    pub fn build(self) -> Result<Client> {
        let trimmed = self.base_url.trim_end_matches('/');
        let base_url = Url::parse(&format!("{trimmed}/")).map_err(|err| {
            Error::invalid(format!("invalid base_url {:?}: {err}", self.base_url))
        })?;
        let base_origin = Origin::parse(&base_url);

        let edge_pairs = headers::validate_base_url_headers(&self.base_url_headers)?;
        if !edge_pairs.is_empty()
            && !base_origin
                .as_ref()
                .is_some_and(Origin::accepts_credentials)
        {
            return Err(Error::invalid(
                "base_url_headers require an absolute https base_url without embedded credentials",
            ));
        }
        let mut edge_headers = Vec::with_capacity(edge_pairs.len());
        for (name, value) in edge_pairs {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|err| Error::invalid(format!("invalid base_url_headers name: {err}")))?;
            let value = HeaderValue::from_str(&value)
                .map_err(|err| Error::invalid(format!("invalid base_url_headers value: {err}")))?;
            edge_headers.push((name, value));
        }

        let control_plane_url = self
            .control_plane_url
            .as_deref()
            .map(|url| {
                Url::parse(&format!("{}/", url.trim_end_matches('/'))).map_err(|err| {
                    Error::invalid(format!("invalid control_plane_url {url:?}: {err}"))
                })
            })
            .transpose()?;

        let mut default_headers = HeaderMap::new();
        default_headers.insert(
            HeaderName::from_static("x-sie-sdk-version"),
            HeaderValue::from_static(version::SDK_VERSION),
        );
        #[cfg(feature = "watch")]
        let mut http_authorization = None;
        if let Some(api_key) = &self.api_key {
            let mut value = HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|_| {
                Error::invalid("api_key contains characters that cannot be sent in a header")
            })?;
            value.set_sensitive(true);
            #[cfg(feature = "watch")]
            {
                http_authorization = Some(value.clone());
            }
            default_headers.insert(reqwest::header::AUTHORIZATION, value);
        }

        let mut http = reqwest::Client::builder()
            // Never follow redirects: a redirect off-origin would leak the Authorization
            // and edge headers to whatever host the response names.
            .redirect(reqwest::redirect::Policy::none())
            .default_headers(default_headers);
        if let Some(max) = self.max_connections {
            http = http.pool_max_idle_per_host(max);
        }
        let http = http
            .build()
            .map_err(|err| Error::invalid(format!("could not build the HTTP client: {err}")))?;

        Ok(Client {
            inner: Arc::new(Inner {
                http,
                base_url,
                base_origin,
                control_plane_url,
                org: self.org,
                timeout: self.timeout,
                defaults: RequestOptions {
                    gpu: self.gpu,
                    wait_for_capacity: self.wait_for_capacity,
                    provision_timeout: self.provision_timeout,
                    max_oom_retries: self.max_oom_retries,
                },
                default_options: self.options,
                edge_headers,
                #[cfg(feature = "watch")]
                http_authorization,
                concurrency: self.max_concurrency.map(Semaphore::new),
                leases: Mutex::new(HashMap::new()),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn client() -> Client {
        Client::new("https://sie.example.com").unwrap()
    }

    #[test]
    fn base_url_always_ends_in_a_slash() {
        assert_eq!(client().base_url(), "https://sie.example.com/");
        assert_eq!(
            Client::new("https://sie.example.com///")
                .unwrap()
                .base_url(),
            "https://sie.example.com/"
        );
    }

    #[test]
    fn paths_join_onto_the_base_url() {
        let client = Client::new("https://sie.example.com/prefix").unwrap();
        assert_eq!(
            client.url("/v1/models").unwrap().as_str(),
            "https://sie.example.com/prefix/v1/models"
        );
        assert_eq!(
            client.url("v1/models").unwrap().as_str(),
            "https://sie.example.com/prefix/v1/models"
        );
    }

    #[test]
    fn edge_headers_require_an_https_origin() {
        let headers = HashMap::from([("Modal-Key".to_string(), "k".to_string())]);
        assert!(
            Client::builder("https://sie.example.com")
                .base_url_headers(headers.clone())
                .build()
                .is_ok()
        );
        let err = Client::builder("http://localhost:8080")
            .base_url_headers(headers)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("https base_url"), "{err}");
    }

    #[test]
    fn edge_headers_are_scoped_to_the_base_origin() {
        let client = Client::builder("https://sie.example.com")
            .base_url_headers(HashMap::from([("Modal-Key".to_string(), "k".to_string())]))
            .build()
            .unwrap();
        assert!(
            client.edge_headers_apply_to(&Url::parse("https://sie.example.com/v1/models").unwrap())
        );
        assert!(
            !client
                .edge_headers_apply_to(&Url::parse("https://other.example.com/v1/models").unwrap())
        );
        assert!(
            !client.edge_headers_apply_to(&Url::parse("https://sie.example.com:8443/v1").unwrap())
        );
        assert!(!client.edge_headers_apply_to(&Url::parse("http://sie.example.com/v1").unwrap()));
    }

    #[test]
    fn routing_splits_pool_from_profile() {
        let client = Client::builder("https://sie.example.com")
            .gpu("prod/l4")
            .build()
            .unwrap();
        let routing = client.routing(None);
        assert_eq!(routing.pool.as_deref(), Some("prod"));
        assert_eq!(routing.profile.as_deref(), Some("l4"));

        let overridden = client.routing(Some("a100-80gb"));
        assert!(overridden.pool.is_none());
        assert_eq!(overridden.profile.as_deref(), Some("a100-80gb"));

        let bare = Client::new("https://sie.example.com")
            .unwrap()
            .routing(None);
        assert!(bare.pool.is_none() && bare.profile.is_none());
    }

    #[test]
    fn per_call_options_override_client_defaults() {
        let client = Client::builder("https://sie.example.com")
            .options(json!({"is_query": true, "keep": 1}))
            .build()
            .unwrap();
        let merged = client
            .merge_options(Some(&json!({"is_query": false})))
            .unwrap();
        assert_eq!(merged, json!({"is_query": false, "keep": 1}));
        assert_eq!(
            client.merge_options(None).unwrap(),
            json!({"is_query": true, "keep": 1})
        );
        assert!(
            Client::new("https://x.example.com")
                .unwrap()
                .merge_options(None)
                .is_none()
        );
    }

    #[test]
    fn connections_need_a_control_plane_and_org() {
        assert!(client().control_plane().is_err());
        let configured = Client::builder("https://sie.example.com")
            .control_plane_url("https://cp.example.com/")
            .org("acme")
            .build()
            .unwrap();
        let (url, org) = configured.control_plane().unwrap();
        assert_eq!(url.as_str(), "https://cp.example.com/");
        assert_eq!(org, "acme");
    }

    #[test]
    fn rejects_a_malformed_base_url() {
        assert!(Client::new("not a url").is_err());
    }
}
