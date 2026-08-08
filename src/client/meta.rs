//! Model catalogue, health, and capacity waiting.

use std::time::{Duration, Instant};

use reqwest::Method;
use serde_json::Value;

use crate::client::Client;
use crate::error::{Error, Result};
use crate::http::{HttpResponse, headers};
use crate::retry::{RetryPolicy, backoff};
use crate::types::{CapacityInfo, HealthResponse, ModelInfo, WorkerInfo};

/// Decode a successful JSON response, preserving the metering evidence on failure.
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(
    response: &HttpResponse,
    owner: &str,
) -> Result<T> {
    serde_json::from_slice(&response.body)
        .map_err(|err| Error::decode(format!("malformed {owner} response: {err}")))
}

impl Client {
    /// Every model the server knows about.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let request = self
            .request(Method::GET, "/v1/models")?
            .header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.send_once(request, RetryPolicy::NONE).await?;
        let body: Value = parse_json(&response, "models")?;
        let models = body
            .get("models")
            .cloned()
            .ok_or_else(|| Error::decode("models response is missing its `models` array"))?;
        serde_json::from_value(models)
            .map_err(|err| Error::decode(format!("malformed models response: {err}")))
    }

    /// Details for one model.
    pub async fn get_model(&self, model: &str) -> Result<ModelInfo> {
        let request = self
            .request(Method::GET, &format!("/v1/models/{model}"))?
            .header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "model")
    }

    /// Health, from either a gateway or a standalone worker.
    ///
    /// A gateway answers `/health` with the whole cluster's state. A worker running on its
    /// own has no such view: it serves the Kubernetes-style `/healthz` probe, which is
    /// plain text, so its reply is reported as a worker with that status and nothing else.
    pub async fn health(&self) -> Result<HealthResponse> {
        let request = self
            .request(Method::GET, "/health")?
            .header("accept", headers::JSON_CONTENT_TYPE);
        match self.send_once(request, RetryPolicy::NONE).await {
            Ok(response) => parse_json(&response, "health"),
            Err(error) if error.status() == Some(404) => self.worker_health().await,
            Err(error) => Err(error),
        }
    }

    /// The liveness probe a standalone worker serves in place of `/health`.
    async fn worker_health(&self) -> Result<HealthResponse> {
        let request = self.request(Method::GET, "/healthz")?;
        let response = self.send_once(request, RetryPolicy::NONE).await?;
        let status = response.text().trim().to_string();
        Ok(HealthResponse {
            status: if status.is_empty() {
                "ok".to_string()
            } else {
                status
            },
            kind: "worker".to_string(),
            ..HealthResponse::default()
        })
    }

    /// Cluster capacity, optionally narrowed to one GPU type.
    ///
    /// Requires a gateway: a worker's `/health` describes only itself.
    pub async fn get_capacity(&self, gpu: Option<&str>) -> Result<CapacityInfo> {
        let health = self.health().await?;
        if health.kind != "gateway" {
            return Err(Error::Request {
                message: "get_capacity() requires a gateway endpoint. This appears to be a worker."
                    .to_string(),
                code: Some("not_gateway".to_string()),
                status: 400,
                request: None,
            });
        }

        let workers: Vec<WorkerInfo> = match gpu {
            Some(gpu) => {
                let wanted = gpu.to_ascii_lowercase();
                health
                    .workers
                    .into_iter()
                    .filter(|worker| worker.gpu.eq_ignore_ascii_case(&wanted))
                    .collect()
            }
            None => health.workers,
        };

        Ok(CapacityInfo {
            status: health.status,
            // A filtered view reports the matching workers, not the cluster total.
            worker_count: if gpu.is_some() {
                workers.len() as u32
            } else {
                health.cluster.worker_count
            },
            gpu_count: health.cluster.gpu_count,
            models_loaded: health.cluster.models_loaded,
            configured_gpu_types: health.configured_gpu_types,
            live_gpu_types: health.live_gpu_types,
            workers,
        })
    }

    /// Block until a GPU type has capacity.
    ///
    /// With `model`, this warms the model as well: it issues one small encode with capacity
    /// waiting enabled, which triggers both scale-up and model loading, then reports the
    /// resulting capacity. Without it, `/health` is polled until a worker appears.
    pub async fn wait_for_capacity(
        &self,
        gpu: &str,
        model: Option<&str>,
        timeout: Option<Duration>,
        poll_interval: Duration,
    ) -> Result<CapacityInfo> {
        let budget = timeout.unwrap_or(backoff::DEFAULT_PROVISION_TIMEOUT);
        let start = Instant::now();

        if let Some(model) = model {
            self.encode(model, [crate::types::Item::text("warmup")])
                .gpu(gpu)
                .wait_for_capacity(true)
                .provision_timeout(budget)
                .send()
                .await?;
            return self.get_capacity(Some(gpu)).await;
        }

        loop {
            // A server that is still scaling up refuses connections and returns errors; both
            // are expected here and only the deadline ends the wait.
            if let Ok(capacity) = self.get_capacity(Some(gpu)).await
                && capacity.worker_count > 0
            {
                return Ok(capacity);
            }

            let elapsed = start.elapsed();
            if elapsed >= budget {
                return Err(Error::Provisioning {
                    message: format!(
                        "Timeout after {:.1}s waiting for GPU '{gpu}' capacity",
                        elapsed.as_secs_f64()
                    ),
                    gpu: Some(gpu.to_string()),
                    retry_after: None,
                });
            }
            tokio::time::sleep(poll_interval.min(budget.saturating_sub(elapsed))).await;
        }
    }
}
