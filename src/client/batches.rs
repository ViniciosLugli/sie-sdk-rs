//! `/v1/batches`: OpenAI-compatible offline batches.

use reqwest::Method;
use serde_json::{Value, json};

use crate::client::files::TRANSFER_TIMEOUT_FLOOR;
use crate::client::{Client, meta::parse_json};
use crate::error::Result;
use crate::http::{PreparedRequest, headers};
use crate::retry::RetryPolicy;
use crate::types::{Batch, BatchList};

/// The only endpoint batches can target today.
pub const DEFAULT_BATCH_ENDPOINT: &str = "/v1/embeddings";
/// The only completion window the server accepts today.
pub const DEFAULT_COMPLETION_WINDOW: &str = "24h";

/// The batches namespace. Obtain one with [`Client::batches`].
#[derive(Debug, Clone)]
pub struct Batches {
    client: Client,
}

impl Client {
    /// Operations on offline batches.
    pub fn batches(&self) -> Batches {
        Batches {
            client: self.clone(),
        }
    }
}

impl Batches {
    /// Queue a batch over a previously uploaded JSONL file.
    pub fn create(&self, input_file_id: impl Into<String>) -> BatchCreate {
        BatchCreate {
            client: self.client.clone(),
            input_file_id: input_file_id.into(),
            endpoint: DEFAULT_BATCH_ENDPOINT.to_string(),
            completion_window: DEFAULT_COMPLETION_WINDOW.to_string(),
            metadata: None,
        }
    }

    /// Fetch one batch.
    pub async fn retrieve(&self, batch_id: &str) -> Result<Batch> {
        let request = self
            .client
            .request(Method::GET, &format!("/v1/batches/{batch_id}"))?
            .header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "batch")
    }

    /// Ask the server to stop a batch.
    pub async fn cancel(&self, batch_id: &str) -> Result<Batch> {
        let request = self
            .client
            .request(Method::POST, &format!("/v1/batches/{batch_id}/cancel"))?
            .json_headers();
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "batch")
    }

    /// List batches.
    pub fn list(&self) -> BatchListRequest {
        BatchListRequest {
            client: self.client.clone(),
            after: None,
            limit: None,
        }
    }
}

/// Creates a batch. Build with [`Batches::create`].
#[derive(Debug, Clone)]
pub struct BatchCreate {
    client: Client,
    input_file_id: String,
    endpoint: String,
    completion_window: String,
    metadata: Option<Value>,
}

impl BatchCreate {
    /// Which endpoint the batch's lines target. Defaults to `/v1/embeddings`.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// How long the server may take. Defaults to `24h`.
    pub fn completion_window(mut self, window: impl Into<String>) -> Self {
        self.completion_window = window.into();
        self
    }

    /// Caller-defined metadata stored with the batch.
    pub fn metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Send the request.
    pub async fn send(self) -> Result<Batch> {
        let mut body = json!({
            "input_file_id": self.input_file_id,
            "endpoint": self.endpoint,
            "completion_window": self.completion_window,
        });
        if let Some(metadata) = self.metadata
            && let Some(object) = body.as_object_mut()
        {
            object.insert("metadata".to_string(), metadata);
        }

        let request = self
            .client
            .request(Method::POST, "/v1/batches")?
            .json_headers()
            .body(serde_json::to_vec(&body).unwrap_or_default());
        let response = self
            .client
            .send_with_timeout(request, RetryPolicy::NONE, TRANSFER_TIMEOUT_FLOOR)
            .await?;
        parse_json(&response, "batch")
    }
}

/// Lists batches. Build with [`Batches::list`].
#[derive(Debug, Clone)]
pub struct BatchListRequest {
    client: Client,
    after: Option<String>,
    limit: Option<u32>,
}

impl BatchListRequest {
    /// Start after this batch id.
    pub fn after(mut self, after: impl Into<String>) -> Self {
        self.after = Some(after.into());
        self
    }

    /// Cap on the page size.
    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Fetch one page, with its pagination cursors.
    pub async fn page(self) -> Result<BatchList> {
        let mut url = self.client.url("/v1/batches")?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(after) = &self.after {
                query.append_pair("after", after);
            }
            if let Some(limit) = self.limit {
                query.append_pair("limit", &limit.to_string());
            }
        }
        let request =
            PreparedRequest::new(Method::GET, url).header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "batch list")
    }

    /// Fetch one page and return only its batches.
    pub async fn send(self) -> Result<Vec<Batch>> {
        Ok(self.page().await?.data)
    }
}
