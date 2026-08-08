//! `/v1/pools`: reserved worker capacity, and the lease that keeps it alive.
//!
//! A pool lease expires unless it is renewed, so creating one starts a background task that
//! renews it every minute. The task holds no reference to the [`Client`], so a dropped
//! client is collected normally and the task shuts down with it.

use std::collections::HashMap;
use std::time::Duration;

use reqwest::Method;
use reqwest::header::HeaderMap;
use serde_json::{Map, Value};

use crate::client::{Client, meta::parse_json};
use crate::error::{Error, Result};
use crate::http::headers;
use crate::retry::{RetryPolicy, backoff};
use crate::types::PoolInfo;

/// The pools namespace. Obtain one with [`Client::pools`].
#[derive(Debug, Clone)]
pub struct Pools {
    client: Client,
}

impl Client {
    /// Operations on worker pools.
    pub fn pools(&self) -> Pools {
        Pools {
            client: self.clone(),
        }
    }
}

impl Pools {
    /// Create a pool, or update the shape of one that already exists.
    ///
    /// The returned pool's lease is renewed in the background until the client is dropped.
    pub fn create(&self, name: impl Into<String>) -> PoolCreate {
        PoolCreate {
            client: self.client.clone(),
            name: name.into(),
            gpus: None,
            gpu_caps: None,
            queue_pool: None,
            bundle: None,
            minimum_worker_count: None,
            pinned_models: None,
        }
    }

    /// Fetch a pool, or `None` when it does not exist.
    pub async fn get(&self, name: &str) -> Result<Option<PoolInfo>> {
        let request = self
            .client
            .request(Method::GET, &format!("/v1/pools/{name}"))?
            .header("accept", headers::JSON_CONTENT_TYPE);
        match self.client.send_once(request, RetryPolicy::NONE).await {
            Ok(response) => Ok(Some(parse_json(&response, "pool")?)),
            Err(error) if error.status() == Some(404) => Ok(None),
            Err(error) => Err(pool_error(error, name)),
        }
    }

    /// Delete a pool. Returns `false` when there was nothing to delete.
    pub async fn delete(&self, name: &str) -> Result<bool> {
        self.client.stop_lease(name);
        let request = self
            .client
            .request(Method::DELETE, &format!("/v1/pools/{name}"))?
            .header("accept", headers::JSON_CONTENT_TYPE);
        match self.client.send_once(request, RetryPolicy::NONE).await {
            Ok(_) => Ok(true),
            Err(error) if error.status() == Some(404) => Ok(false),
            Err(error) => Err(pool_error(error, name)),
        }
    }
}

/// Restate a failure as a pool failure, keeping the server's message.
fn pool_error(error: Error, name: &str) -> Error {
    match error {
        Error::Request { message, .. }
        | Error::Server { message, .. }
        | Error::Connection { message, .. } => Error::Pool {
            message: format!("Pool '{name}': {message}"),
            pool_name: Some(name.to_string()),
            state: None,
        },
        other => other,
    }
}

/// Creates a pool. Build with [`Pools::create`].
#[derive(Debug, Clone)]
pub struct PoolCreate {
    client: Client,
    name: String,
    gpus: Option<HashMap<String, u32>>,
    gpu_caps: Option<HashMap<String, u32>>,
    queue_pool: Option<String>,
    bundle: Option<String>,
    minimum_worker_count: Option<u32>,
    pinned_models: Option<Vec<String>>,
}

impl PoolCreate {
    /// How many workers of each GPU type the pool reserves.
    pub fn gpus(mut self, gpus: impl IntoIterator<Item = (String, u32)>) -> Self {
        self.gpus = Some(gpus.into_iter().collect());
        self
    }

    /// Ceiling on assigned workers per GPU type.
    pub fn gpu_caps(mut self, caps: impl IntoIterator<Item = (String, u32)>) -> Self {
        self.gpu_caps = Some(caps.into_iter().collect());
        self
    }

    /// Which queue pool the workers draw from. Defaults to `default` server-side.
    pub fn queue_pool(mut self, queue_pool: impl Into<String>) -> Self {
        self.queue_pool = Some(queue_pool.into());
        self
    }

    /// Which model bundle the workers run.
    pub fn bundle(mut self, bundle: impl Into<String>) -> Self {
        self.bundle = Some(bundle.into());
        self
    }

    /// Workers to keep warm. Zero, the default, allows scaling to zero.
    pub fn minimum_worker_count(mut self, count: u32) -> Self {
        self.minimum_worker_count = Some(count);
        self
    }

    /// Models to keep resident, as `model` or `model:profile`.
    pub fn pinned_models(mut self, models: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.pinned_models = Some(models.into_iter().map(Into::into).collect());
        self
    }

    /// Send the request and start renewing the pool's lease.
    pub async fn send(self) -> Result<PoolInfo> {
        let mut body = Map::new();
        body.insert("name".to_string(), Value::String(self.name.clone()));
        if let Some(gpus) = &self.gpus {
            body.insert("gpus".to_string(), json_counts(gpus));
        }
        if let Some(caps) = &self.gpu_caps {
            body.insert("gpu_caps".to_string(), json_counts(caps));
        }
        if let Some(queue_pool) = self.queue_pool.filter(|value| !value.is_empty()) {
            body.insert("queue_pool".to_string(), Value::String(queue_pool));
        }
        if let Some(bundle) = self.bundle.filter(|value| !value.is_empty()) {
            body.insert("bundle".to_string(), Value::String(bundle));
        }
        if let Some(count) = self.minimum_worker_count {
            body.insert("minimum_worker_count".to_string(), Value::from(count));
        }
        if let Some(models) = &self.pinned_models {
            body.insert(
                "pinned_models".to_string(),
                Value::Array(
                    models
                        .iter()
                        .map(|model| Value::String(model.clone()))
                        .collect(),
                ),
            );
        }

        let request = self
            .client
            .request(Method::POST, "/v1/pools")?
            .json_headers()
            .body(serde_json::to_vec(&Value::Object(body)).unwrap_or_default());

        let response = self
            .client
            .send_once(request, RetryPolicy::NONE)
            .await
            .map_err(|error| pool_error(error, &self.name))?;

        let pool: PoolInfo = parse_json(&response, "pool")?;
        self.client.start_lease(&self.name)?;
        Ok(pool)
    }
}

fn json_counts(counts: &HashMap<String, u32>) -> Value {
    Value::Object(
        counts
            .iter()
            .map(|(key, value)| (key.clone(), Value::from(*value)))
            .collect(),
    )
}

/// Renews one pool's lease until it is cancelled.
///
/// It carries the transport and the built request rather than a [`Client`], so it cannot
/// keep the client alive by holding a reference back to it.
pub(crate) struct LeaseRenewer {
    http: reqwest::Client,
    url: reqwest::Url,
    edge_headers: Vec<(reqwest::header::HeaderName, reqwest::header::HeaderValue)>,
    name: String,
}

impl LeaseRenewer {
    pub(crate) fn new(
        http: reqwest::Client,
        url: reqwest::Url,
        edge_headers: Vec<(reqwest::header::HeaderName, reqwest::header::HeaderValue)>,
        name: String,
    ) -> Self {
        Self {
            http,
            url,
            edge_headers,
            name,
        }
    }

    /// Renew every minute, retrying a failed round with a bounded backoff.
    pub(crate) async fn run(self) {
        loop {
            tokio::time::sleep(backoff::LEASE_RENEWAL_INTERVAL).await;
            if !self.renew_round().await {
                tracing::error!(
                    "All {} lease renewal attempts failed for pool '{}'",
                    backoff::LEASE_RENEWAL_MAX_RETRIES,
                    self.name
                );
            }
        }
    }

    async fn renew_round(&self) -> bool {
        for attempt in 0..backoff::LEASE_RENEWAL_MAX_RETRIES {
            let mut headers = HeaderMap::new();
            headers.insert(
                reqwest::header::ACCEPT,
                headers::JSON_CONTENT_TYPE.parse().unwrap(),
            );
            for (name, value) in &self.edge_headers {
                headers.insert(name.clone(), value.clone());
            }

            let result = self
                .http
                .post(self.url.clone())
                .headers(headers)
                .send()
                .await
                .map(|response| response.status().is_success());

            match result {
                Ok(true) => return true,
                Ok(false) => tracing::warn!("Lease renewal for pool '{}' was rejected", self.name),
                Err(error) => {
                    tracing::warn!("Lease renewal for pool '{}' failed: {error}", self.name)
                }
            }
            // 1, 2, 4, 8, then 10 seconds.
            let backoff = Duration::from_secs(1 << attempt.min(3)).min(Duration::from_secs(10));
            tokio::time::sleep(backoff).await;
        }
        false
    }
}

impl Client {
    /// Start renewing a pool's lease, unless this client already renews it.
    fn start_lease(&self, name: &str) -> Result<()> {
        let mut leases = self
            .inner
            .leases
            .lock()
            .expect("lease registry is not poisoned");
        if leases.contains_key(name) {
            return Ok(());
        }
        let url = self.url(&format!("/v1/pools/{name}/renew"))?;
        let edge_headers = if self.edge_headers_apply_to(&url) {
            self.inner.edge_headers.clone()
        } else {
            Vec::new()
        };
        let renewer =
            LeaseRenewer::new(self.inner.http.clone(), url, edge_headers, name.to_string());
        leases.insert(name.to_string(), tokio::spawn(renewer.run()));
        Ok(())
    }

    fn stop_lease(&self, name: &str) {
        if let Some(handle) = self
            .inner
            .leases
            .lock()
            .expect("lease registry is not poisoned")
            .remove(name)
        {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> Client {
        Client::new("https://sie.invalid").unwrap()
    }

    #[test]
    fn the_renewal_url_is_derived_from_the_pool_name() {
        let url = client().url("/v1/pools/prod/renew").unwrap();
        assert_eq!(url.as_str(), "https://sie.invalid/v1/pools/prod/renew");
    }

    #[test]
    fn errors_are_restated_as_pool_errors() {
        let error = pool_error(
            Error::Request {
                message: "gpu type not configured".to_string(),
                code: None,
                status: 400,
                request: None,
            },
            "prod",
        );
        match error {
            Error::Pool {
                message, pool_name, ..
            } => {
                assert_eq!(pool_name.as_deref(), Some("prod"));
                assert!(message.contains("gpu type not configured"), "{message}");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn starting_the_same_lease_twice_spawns_one_task() {
        let client = client();
        client.start_lease("prod").unwrap();
        client.start_lease("prod").unwrap();
        assert_eq!(client.inner.leases.lock().unwrap().len(), 1);
        client.stop_lease("prod");
        assert!(client.inner.leases.lock().unwrap().is_empty());
    }
}
