//! A blocking facade over the async client.
//!
//! There is one implementation of every endpoint, and this module drives it on a runtime it
//! owns. Rather than mirroring every builder in a second, drift-prone surface, it hands you
//! the real [`crate::Client`] and runs the future it produces:
//!
//! ```no_run
//! use sie_sdk::{Item, blocking::Client};
//!
//! # fn main() -> sie_sdk::Result<()> {
//! let client = Client::new("http://localhost:8080")?;
//!
//! let result = client.call(|sie| sie.encode("BAAI/bge-m3", [Item::text("Hello")]).send_one())?;
//! println!("{:?}", result.dense);
//!
//! for model in client.call(|sie| sie.list_models())? {
//!     println!("{}", model.name);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Every builder, option and error is the same as on the async client.
//!
//! # Do not call this from async code
//!
//! Blocking a thread that is already inside a Tokio runtime deadlocks it. In an async
//! context, use [`crate::Client`] directly.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;

use crate::error::{Error, Result};

/// A blocking handle to one SIE server.
///
/// Cloning shares the runtime and the connection pool.
#[derive(Clone)]
pub struct Client {
    api: crate::Client,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("blocking::Client")
            .field("api", &self.api)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// A blocking client with default settings.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self> {
        Self::builder(base_url).build()
    }

    /// Start configuring a blocking client.
    pub fn builder(base_url: impl AsRef<str>) -> ClientBuilder {
        ClientBuilder {
            inner: crate::Client::builder(base_url),
            worker_threads: None,
        }
    }

    /// The async client underneath, for building requests.
    pub fn api(&self) -> &crate::Client {
        &self.api
    }

    /// The server root, always with a trailing slash.
    pub fn base_url(&self) -> &str {
        self.api.base_url()
    }

    /// Build a request from the async client and run it to completion.
    ///
    /// The closure's borrow is tied to this client, so a request that borrows it (the
    /// namespace accessors do) is as easy to write as one that owns its state.
    pub fn call<'a, F, Fut, T>(&'a self, request: F) -> Result<T>
    where
        F: FnOnce(&'a crate::Client) -> Fut,
        Fut: Future<Output = Result<T>> + 'a,
    {
        self.runtime.block_on(request(&self.api))
    }

    /// Run any future on this client's runtime.
    ///
    /// Useful for futures that are not a single request, such as joining several calls.
    pub fn block_on<Fut: Future>(&self, future: Fut) -> Fut::Output {
        self.runtime.block_on(future)
    }

    /// Drain a stream into a `Vec`, stopping at the first error.
    ///
    /// Streaming endpoints exist so a caller can act on each chunk as it lands; when the
    /// whole result is what you want, prefer the buffered endpoint instead of this.
    pub fn collect<S, T>(&self, stream: S) -> Result<Vec<T>>
    where
        S: futures_core::Stream<Item = Result<T>>,
    {
        self.runtime.block_on(async move {
            let mut stream = std::pin::pin!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Consume a stream chunk by chunk, without collecting it.
    ///
    /// The callback returns `false` to stop early, which closes the connection.
    pub fn for_each<S, T, F>(&self, stream: S, mut handler: F) -> Result<()>
    where
        S: futures_core::Stream<Item = Result<T>>,
        F: FnMut(T) -> bool,
    {
        self.runtime.block_on(async move {
            let mut stream = std::pin::pin!(stream);
            while let Some(item) = stream.next().await {
                if !handler(item?) {
                    break;
                }
            }
            Ok(())
        })
    }
}

/// Configures a blocking [`Client`].
pub struct ClientBuilder {
    inner: crate::ClientBuilder,
    worker_threads: Option<usize>,
}

impl ClientBuilder {
    /// How many threads the owned runtime uses. Defaults to one.
    ///
    /// One is right for a client used from a single thread. Raise it when the same client
    /// is shared across threads that make concurrent calls.
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = Some(threads);
        self
    }

    /// Bearer token sent as `Authorization` on every request to the server.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.inner = self.inner.api_key(api_key);
        self
    }

    /// Per-attempt HTTP timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.inner = self.inner.timeout(timeout);
        self
    }

    /// Default machine profile, optionally pool-qualified as `"pool/profile"`.
    pub fn gpu(mut self, gpu: impl Into<String>) -> Self {
        self.inner = self.inner.gpu(gpu);
        self
    }

    /// Default runtime options, shallow-merged under any per-call options.
    pub fn options(mut self, options: serde_json::Value) -> Self {
        self.inner = self.inner.options(options);
        self
    }

    /// Cap on pooled connections.
    pub fn max_connections(mut self, max: usize) -> Self {
        self.inner = self.inner.max_connections(max);
        self
    }

    /// Cap on requests in flight at once.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.inner = self.inner.max_concurrency(max);
        self
    }

    /// Control-plane root, required by the connections namespace.
    pub fn control_plane_url(mut self, url: impl Into<String>) -> Self {
        self.inner = self.inner.control_plane_url(url);
        self
    }

    /// Organisation slug, required by the connections namespace.
    pub fn org(mut self, org: impl Into<String>) -> Self {
        self.inner = self.inner.org(org);
        self
    }

    /// Extra headers for an HTTP edge in front of the gateway.
    pub fn base_url_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.inner = self.inner.base_url_headers(headers);
        self
    }

    /// Whether calls wait out provisioning by default.
    pub fn wait_for_capacity(mut self, wait: bool) -> Self {
        self.inner = self.inner.wait_for_capacity(wait);
        self
    }

    /// Default wall-clock budget for a call including its retries.
    pub fn provision_timeout(mut self, timeout: Duration) -> Self {
        self.inner = self.inner.provision_timeout(timeout);
        self
    }

    /// Default cap on `RESOURCE_EXHAUSTED` retries.
    pub fn max_oom_retries(mut self, retries: u32) -> Self {
        self.inner = self.inner.max_oom_retries(retries);
        self
    }

    /// Build the runtime and the client.
    pub fn build(self) -> Result<Client> {
        let mut runtime = tokio::runtime::Builder::new_multi_thread();
        runtime
            .worker_threads(self.worker_threads.unwrap_or(1))
            .enable_all();
        let runtime = runtime.build().map_err(|err| {
            Error::invalid(format!("could not start the blocking runtime: {err}"))
        })?;

        // The transport is constructed inside the runtime so its background resources are
        // registered with the reactor that will drive them.
        let api = runtime.block_on(async { self.inner.build() })?;
        Ok(Client {
            api,
            runtime: Arc::new(runtime),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Item;

    #[test]
    fn builder_options_reach_the_async_client() {
        let client = Client::builder("https://sie.example.com/")
            .timeout(Duration::from_secs(5))
            .max_oom_retries(0)
            .wait_for_capacity(false)
            .build()
            .unwrap();
        assert_eq!(client.base_url(), "https://sie.example.com/");
        assert!(!client.api().default_options().wait_for_capacity);
        assert_eq!(client.api().default_options().max_oom_retries, 0);
    }

    #[test]
    fn call_runs_a_request_and_returns_its_error() {
        let client = Client::builder("http://127.0.0.1:1")
            .timeout(Duration::from_millis(200))
            .wait_for_capacity(false)
            .build()
            .unwrap();
        let result = client.call(|sie| sie.encode("m", [Item::text("hi")]).send_one());
        assert!(
            matches!(result, Err(Error::Connection { .. })),
            "{result:?}"
        );
    }

    #[test]
    fn client_side_validation_still_applies() {
        let client = Client::new("https://sie.invalid").unwrap();
        let result = client.call(|sie| sie.encode("m", Vec::new()).send());
        assert!(
            matches!(result, Err(Error::InvalidRequest(_))),
            "{result:?}"
        );
    }

    #[test]
    fn block_on_drives_arbitrary_futures() {
        let client = Client::new("https://sie.invalid").unwrap();
        assert_eq!(client.block_on(async { 1 + 1 }), 2);
    }

    #[test]
    fn collect_stops_at_the_first_error() {
        let client = Client::new("https://sie.invalid").unwrap();
        let stream =
            futures_util::stream::iter(vec![Ok(1), Err(Error::invalid("stop here")), Ok(3)]);
        let result: Result<Vec<i32>> = client.collect(stream);
        assert!(result.is_err());

        let ok = futures_util::stream::iter(vec![Ok(1), Ok(2)]);
        assert_eq!(client.collect::<_, i32>(ok).unwrap(), vec![1, 2]);
    }

    #[test]
    fn for_each_can_stop_early() {
        let client = Client::new("https://sie.invalid").unwrap();
        let stream = futures_util::stream::iter(vec![Ok(1), Ok(2), Ok(3)]);
        let mut seen = Vec::new();
        client
            .for_each(stream, |value: i32| {
                seen.push(value);
                value < 2
            })
            .unwrap();
        assert_eq!(seen, vec![1, 2]);
    }
}
