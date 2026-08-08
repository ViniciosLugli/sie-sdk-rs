//! `/v1/generate`, `/v1/chat/completions`, `/v1/responses` and `/v1/estimate`.

use reqwest::Method;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::client::encode::request_options;
use crate::client::{Client, meta::parse_json};
use crate::error::{Error, Result};
use crate::http::{HttpResponse, headers, metadata};
use crate::retry::{RequestOptions, RetryPolicy, RetryState};
use crate::types::{
    ChatCompletion, ChatMessage, CostEstimate, GenerateResult, Grammar, ImageInput,
    RequestMetadata, ResponseInputMessage, ResponseResult,
};

/// Path segments cannot carry a raw `/`, so the generate endpoints escape it.
pub(crate) fn escape_model(model: &str) -> String {
    model.replace('/', "__")
}

/// Turn a `Serialize` parameter block into a JSON object, dropping the unset fields.
fn params_object<T: Serialize>(params: &T) -> Result<Map<String, Value>> {
    match serde_json::to_value(params)
        .map_err(|err| Error::invalid(format!("could not encode request parameters: {err}")))?
    {
        Value::Object(map) => Ok(map),
        other => Err(Error::invalid(format!(
            "request parameters must serialize to an object, got {other}"
        ))),
    }
}

/// Encode images for a JSON endpoint. Msgpack carries raw bytes; JSON needs base64.
pub(crate) fn images_for_json(images: &[ImageInput]) -> Result<Vec<Value>> {
    use base64::Engine as _;

    images
        .iter()
        .map(|image| {
            let (data, format) = image.resolve()?;
            Ok(serde_json::json!({
                "data": base64::engine::general_purpose::STANDARD.encode(&data),
                "format": format,
            }))
        })
        .collect()
}

impl Client {
    /// Send a JSON body through the retry machine and decode the response.
    pub(crate) async fn send_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
        policy: RetryPolicy,
        model: Option<&str>,
        options: &RequestOptions,
        owner: &str,
    ) -> Result<(T, Option<RequestMetadata>)> {
        let response = self
            .send_json_raw(path, body, policy, model, options)
            .await?;
        let decoded: T = parse_json(&response.0, owner)?;
        let envelope: Value = parse_json(&response.0, owner)?;
        Ok((
            decoded,
            metadata::parse(&response.0.headers, Some(&envelope), response.1),
        ))
    }

    async fn send_json_raw(
        &self,
        path: &str,
        body: &Value,
        policy: RetryPolicy,
        model: Option<&str>,
        options: &RequestOptions,
    ) -> Result<(HttpResponse, u32)> {
        let routing = self.routing(options.gpu.as_deref());
        let encoded = serde_json::to_vec(body)
            .map_err(|err| Error::invalid(format!("could not encode the request body: {err}")))?;

        let request = self
            .request(Method::POST, path)?
            .json_headers()
            .maybe_header(headers::MACHINE_PROFILE, routing.profile.as_deref())
            .maybe_header(headers::POOL, routing.pool.as_deref())
            .body(encoded);

        let mut state = RetryState::new(policy, options, model);
        let response = self.send(request, &mut state).await?;
        let retries = state.retries();
        Ok((response, retries))
    }

    /// Estimate the cost of a request before sending it.
    ///
    /// `endpoint` is the absolute path the request would be sent to, and `request` is the
    /// body it would carry.
    pub async fn estimate(&self, endpoint: &str, request: Value) -> Result<CostEstimate> {
        if !endpoint.starts_with('/') {
            return Err(Error::invalid(format!(
                "estimate endpoint must be an absolute path, got {endpoint:?}"
            )));
        }
        if !request.is_object() {
            return Err(Error::invalid("estimate request must be a JSON object"));
        }

        let envelope = serde_json::json!({"endpoint": endpoint, "request": request});
        let encoded = serde_json::to_vec(&envelope).map_err(|err| {
            Error::invalid(format!("could not encode the estimate envelope: {err}"))
        })?;
        let prepared = self
            .request(Method::POST, "/v1/estimate")?
            .json_headers()
            .body(encoded);

        let options = self.metadata_options();
        let mut state = RetryState::new(RetryPolicy::ESTIMATE, &options, None);
        let response = self.send(prepared, &mut state).await?;
        parse_json(&response, "estimate")
    }

    /// Generate text with the native endpoint.
    pub fn generate(
        &self,
        model: impl Into<String>,
        prompt: impl Into<String>,
        max_new_tokens: u32,
    ) -> GenerateRequest {
        GenerateRequest {
            client: self.clone(),
            model: model.into(),
            prompt: prompt.into(),
            images: Vec::new(),
            params: GenerateParams {
                max_new_tokens,
                ..GenerateParams::default()
            },
            extra_body: None,
            options: self.request_options(),
        }
    }

    /// Generate a chat completion.
    pub fn chat(
        &self,
        model: impl Into<String>,
        messages: impl IntoIterator<Item = ChatMessage>,
    ) -> ChatRequest {
        ChatRequest {
            client: self.clone(),
            model: model.into(),
            messages: messages.into_iter().collect(),
            params: ChatParams::default(),
            extra_body: None,
            options: self.request_options(),
        }
    }

    /// Generate a response through the Responses API.
    pub fn responses(&self, model: impl Into<String>, input: ResponseInput) -> ResponsesRequest {
        ResponsesRequest {
            client: self.clone(),
            model: model.into(),
            input,
            params: ResponseParams::default(),
            options: self.request_options(),
        }
    }
}

/// Generate optional builder setters, each wrapping its value in `Some`.
macro_rules! setters {
    ($($(#[$meta:meta])* $name:ident: $ty:ty),* $(,)?) => {
        $(
            $(#[$meta])*
            pub fn $name(mut self, value: $ty) -> Self {
                self.params.$name = Some(value);
                self
            }
        )*
    };
}

/// Like [`setters!`], for fields whose value is worth accepting by conversion.
macro_rules! into_setters {
    ($($(#[$meta:meta])* $name:ident: $ty:ty),* $(,)?) => {
        $(
            $(#[$meta])*
            pub fn $name(mut self, value: impl Into<$ty>) -> Self {
                self.params.$name = Some(value.into());
                self
            }
        )*
    };
}

/// A `stop` setter that accepts any iterable of string-likes.
macro_rules! stop_setter {
    () => {
        /// Sequences that end the generation.
        pub fn stop(mut self, stop: impl IntoIterator<Item = impl Into<String>>) -> Self {
            self.params.stop = Some(stop.into_iter().map(Into::into).collect());
            self
        }
    };
}

/// Every optional field of a native generate request.
#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct GenerateParams {
    pub(crate) max_new_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) grammar: Option<Grammar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) routing_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) lora_adapter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_logprobs: Option<u32>,
}

/// Generates text with the native endpoint. Build with [`Client::generate`].
pub struct GenerateRequest {
    pub(crate) client: Client,
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) images: Vec<ImageInput>,
    pub(crate) params: GenerateParams,
    pub(crate) extra_body: Option<Value>,
    pub(crate) options: RequestOptions,
}

impl GenerateRequest {
    request_options!();

    setters! {
        /// Sampling temperature.
        temperature: f64,
        /// Nucleus sampling cutoff.
        top_p: f64,
        /// Penalty applied per prior occurrence of a token.
        frequency_penalty: f64,
        /// Penalty applied once a token has appeared at all.
        presence_penalty: f64,
        /// Constrain the output's shape.
        grammar: Grammar,
        /// Seed the sampler for reproducible output.
        seed: i64,
        /// Per-token logit adjustments, as a `{token_id: bias}` object.
        logit_bias: Value,
        /// Adapter-specific runtime options.
        options: Value,
    }

    into_setters! {
        /// Pin related requests to the same worker, to reuse its prefix cache.
        routing_key: String,
        /// Explicit prefix-cache key.
        prompt_cache_key: String,
        /// End-user identifier forwarded to safety tooling.
        safety_identifier: String,
        /// `LoRA` adapter to apply.
        lora_adapter: String,
    }

    stop_setter!();

    /// Attach images to a multimodal prompt.
    pub fn images(mut self, images: impl IntoIterator<Item = ImageInput>) -> Self {
        self.images = images.into_iter().collect();
        self
    }

    /// Merge extra fields into the request body, overriding anything the builder set.
    pub fn extra_body(mut self, extra: Value) -> Self {
        self.extra_body = Some(extra);
        self
    }

    pub(crate) fn body(&self, stream: bool) -> Result<Value> {
        let mut body = params_object(&self.params)?;
        body.insert("prompt".to_string(), Value::String(self.prompt.clone()));
        if !self.images.is_empty() {
            body.insert(
                "images".to_string(),
                Value::Array(images_for_json(&self.images)?),
            );
        }
        if stream {
            body.insert("stream".to_string(), Value::Bool(true));
        }
        // `top_logprobs` is meaningless without `logprobs`, and the gateway rejects the pair.
        if self.params.logprobs != Some(true) {
            body.remove("top_logprobs");
        }
        merge_extra_body(&mut body, self.extra_body.as_ref());
        // Re-assert the fields that define the request, so `extra_body` cannot rewrite them.
        body.insert("prompt".to_string(), Value::String(self.prompt.clone()));
        body.insert(
            "max_new_tokens".to_string(),
            Value::from(self.params.max_new_tokens),
        );
        if stream {
            body.insert("stream".to_string(), Value::Bool(true));
        }
        Ok(Value::Object(body))
    }

    pub(crate) fn path(&self) -> String {
        format!("/v1/generate/{}", escape_model(&self.model))
    }

    /// Send the request and wait for the whole generation.
    pub async fn send(self) -> Result<GenerateResult> {
        let body = self.body(false)?;
        let (mut result, request): (GenerateResult, _) = self
            .client
            .send_json(
                &self.path(),
                &body,
                RetryPolicy::GENERATE,
                Some(&self.model),
                &self.options,
                "generate",
            )
            .await?;
        result.request = request;
        Ok(result)
    }

    /// Estimate what this request would cost.
    pub async fn estimate(self) -> Result<CostEstimate> {
        let body = self.body(false)?;
        self.client.estimate(&self.path(), body).await
    }
}

/// Every optional field of a chat completion request.
#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct ChatParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) best_of: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) lora_adapter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream_options: Option<Value>,
}

/// Generates a chat completion. Build with [`Client::chat`].
pub struct ChatRequest {
    pub(crate) client: Client,
    pub(crate) model: String,
    pub(crate) messages: Vec<ChatMessage>,
    pub(crate) params: ChatParams,
    pub(crate) extra_body: Option<Value>,
    pub(crate) options: RequestOptions,
}

impl ChatRequest {
    request_options!();

    setters! {
        /// Cap on generated tokens. Prefer this over `max_tokens`.
        max_completion_tokens: u32,
        /// Legacy cap on generated tokens.
        max_tokens: u32,
        /// Sampling temperature.
        temperature: f64,
        /// Nucleus sampling cutoff.
        top_p: f64,
        /// Sample only from the k most likely tokens.
        top_k: u32,
        /// Penalty applied to repeated tokens.
        repetition_penalty: f64,
        /// Tool definitions the model may call.
        tools: Vec<Value>,
        /// Which tool the model must call, if any.
        tool_choice: Value,
        /// Whether several tool calls may be issued at once.
        parallel_tool_calls: bool,
        /// Structured output format.
        response_format: Value,
        /// Penalty applied per prior occurrence of a token.
        frequency_penalty: f64,
        /// Penalty applied once a token has appeared at all.
        presence_penalty: f64,
        /// Number of candidates to return.
        n: u32,
        /// Number of candidates to generate before picking. Incompatible with streaming.
        best_of: u32,
        /// Return per-token log probabilities.
        logprobs: bool,
        /// How many alternatives to report per token, from 0 to 20.
        top_logprobs: u32,
        /// Per-token logit adjustments, each from -100 to 100.
        logit_bias: Value,
        /// Seed the sampler for reproducible output.
        seed: i64,
        /// Stream tuning, such as `{"include_usage": true}`.
        stream_options: Value,
    }

    into_setters! {
        /// End-user identifier.
        user: String,
        /// End-user identifier forwarded to safety tooling.
        safety_identifier: String,
        /// `LoRA` adapter to apply.
        lora_adapter: String,
    }

    stop_setter!();

    /// Merge extra fields into the request body, overriding anything the builder set.
    ///
    /// The gateway rejects unknown fields, so this is for options newer than the SDK.
    pub fn extra_body(mut self, extra: Value) -> Self {
        self.extra_body = Some(extra);
        self
    }

    pub(crate) fn body(&self, stream: bool) -> Result<Value> {
        let mut params = self.params.clone();
        if stream {
            // `best_of` generates several candidates before choosing one, which cannot be
            // expressed incrementally.
            params.best_of = None;
        }
        let mut body = params_object(&params)?;
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert(
            "messages".to_string(),
            serde_json::to_value(&self.messages)
                .map_err(|err| Error::invalid(format!("could not encode chat messages: {err}")))?,
        );
        if stream {
            body.insert("stream".to_string(), Value::Bool(true));
        }
        merge_extra_body(&mut body, self.extra_body.as_ref());
        Ok(Value::Object(body))
    }

    /// Send the request and wait for the whole completion.
    pub async fn send(self) -> Result<ChatCompletion> {
        let body = self.body(false)?;
        let (mut result, request): (ChatCompletion, _) = self
            .client
            .send_json(
                "/v1/chat/completions",
                &body,
                RetryPolicy::STREAM,
                Some(&self.model),
                &self.options,
                "chat completion",
            )
            .await?;
        result.request = request;
        Ok(result)
    }

    /// Estimate what this request would cost.
    pub async fn estimate(self) -> Result<CostEstimate> {
        let body = self.body(false)?;
        self.client.estimate("/v1/chat/completions", body).await
    }
}

/// Input to the Responses API: a bare prompt or a message list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ResponseInput {
    /// A bare prompt.
    Text(String),
    /// An ordered conversation.
    Messages(Vec<ResponseInputMessage>),
}

impl From<&str> for ResponseInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for ResponseInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<ResponseInputMessage>> for ResponseInput {
    fn from(value: Vec<ResponseInputMessage>) -> Self {
        Self::Messages(value)
    }
}

/// Every optional field of a responses request. The endpoint takes no others.
#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct ResponseParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) seed: Option<i64>,
}

/// Generates a response. Build with [`Client::responses`].
pub struct ResponsesRequest {
    client: Client,
    model: String,
    input: ResponseInput,
    params: ResponseParams,
    options: RequestOptions,
}

impl ResponsesRequest {
    request_options!();

    setters! {
        /// Cap on generated tokens.
        max_output_tokens: u32,
        /// Sampling temperature.
        temperature: f64,
        /// Nucleus sampling cutoff.
        top_p: f64,
        /// Seed the sampler for reproducible output.
        seed: i64,
    }

    fn body(&self) -> Result<Value> {
        let mut body = params_object(&self.params)?;
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert(
            "input".to_string(),
            serde_json::to_value(&self.input).map_err(|err| {
                Error::invalid(format!("could not encode the response input: {err}"))
            })?,
        );
        Ok(Value::Object(body))
    }

    /// Send the request.
    pub async fn send(self) -> Result<ResponseResult> {
        let body = self.body()?;
        let (mut result, request): (ResponseResult, _) = self
            .client
            .send_json(
                "/v1/responses",
                &body,
                RetryPolicy::STREAM,
                Some(&self.model),
                &self.options,
                "response",
            )
            .await?;
        result.request = request;
        Ok(result)
    }

    /// Estimate what this request would cost.
    pub async fn estimate(self) -> Result<CostEstimate> {
        let body = self.body()?;
        self.client.estimate("/v1/responses", body).await
    }
}

fn merge_extra_body(body: &mut Map<String, Value>, extra: Option<&Value>) {
    if let Some(Value::Object(extra)) = extra {
        for (key, value) in extra {
            body.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn client() -> Client {
        Client::new("https://sie.invalid").unwrap()
    }

    #[test]
    fn generate_paths_escape_slashes_in_the_model_id() {
        assert_eq!(escape_model("BAAI/bge-m3"), "BAAI__bge-m3");
        assert_eq!(escape_model("qwen3"), "qwen3");
        let request = client().generate("org/model", "hi", 16);
        assert_eq!(request.path(), "/v1/generate/org__model");
    }

    #[test]
    fn generate_body_carries_only_what_was_set() {
        let body = client().generate("m", "Once upon", 32).body(false).unwrap();
        assert_eq!(body, json!({"prompt": "Once upon", "max_new_tokens": 32}));
    }

    #[test]
    fn generate_body_includes_every_optional_that_was_set() {
        let body = client()
            .generate("m", "p", 8)
            .temperature(0.7)
            .top_p(0.9)
            .stop(vec!["\n".to_string()])
            .grammar(Grammar::regex("[0-9]+"))
            .seed(42)
            .routing_key("tenant-1")
            .lora_adapter("sql")
            .body(false)
            .unwrap();
        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["top_p"], json!(0.9));
        assert_eq!(body["stop"], json!(["\n"]));
        assert_eq!(body["grammar"], json!({"regex": "[0-9]+"}));
        assert_eq!(body["seed"], json!(42));
        assert_eq!(body["routing_key"], json!("tenant-1"));
        assert_eq!(body["lora_adapter"], json!("sql"));
    }

    #[test]
    fn extra_body_cannot_rewrite_the_fields_that_define_the_request() {
        let body = client()
            .generate("m", "real prompt", 16)
            .extra_body(json!({"prompt": "hijacked", "max_new_tokens": 9999, "custom": true}))
            .body(true)
            .unwrap();
        assert_eq!(body["prompt"], json!("real prompt"));
        assert_eq!(body["max_new_tokens"], json!(16));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["custom"], json!(true));
    }

    #[test]
    fn top_logprobs_is_dropped_unless_logprobs_is_on() {
        let mut request = client().generate("m", "p", 4);
        request.params.top_logprobs = Some(5);
        assert!(request.body(true).unwrap().get("top_logprobs").is_none());

        let mut request = client().generate("m", "p", 4);
        request.params.logprobs = Some(true);
        request.params.top_logprobs = Some(5);
        assert_eq!(request.body(true).unwrap()["top_logprobs"], json!(5));
    }

    #[test]
    fn chat_body_always_names_the_model_and_messages() {
        let body = client()
            .chat("qwen3", [ChatMessage::user("hi")])
            .temperature(0.2)
            .body(false)
            .unwrap();
        assert_eq!(
            body,
            json!({"model": "qwen3", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2})
        );
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn chat_extra_body_is_merged_last() {
        let body = client()
            .chat("m", [ChatMessage::user("hi")])
            .temperature(0.2)
            .extra_body(json!({"temperature": 0.9, "future_field": 1}))
            .body(false)
            .unwrap();
        assert_eq!(body["temperature"], json!(0.9));
        assert_eq!(body["future_field"], json!(1));
    }

    #[test]
    fn best_of_is_dropped_when_streaming() {
        let request = client().chat("m", [ChatMessage::user("hi")]).best_of(4);
        assert_eq!(request.body(false).unwrap()["best_of"], json!(4));
        let streamed = request.body(true).unwrap();
        assert!(streamed.get("best_of").is_none());
        assert_eq!(streamed["stream"], json!(true));
    }

    #[test]
    fn responses_body_takes_text_or_messages() {
        let text = client()
            .responses("m", ResponseInput::from("Summarize this"))
            .max_output_tokens(64)
            .body()
            .unwrap();
        assert_eq!(
            text,
            json!({"model": "m", "input": "Summarize this", "max_output_tokens": 64})
        );

        let messages = client()
            .responses("m", vec![ResponseInputMessage::user("hi")].into())
            .body()
            .unwrap();
        assert_eq!(
            messages["input"],
            json!([{"role": "user", "content": "hi"}])
        );
    }

    #[tokio::test]
    async fn estimate_rejects_a_relative_endpoint_or_a_non_object_request() {
        let client = client();
        assert!(
            client
                .estimate("v1/chat/completions", json!({}))
                .await
                .is_err()
        );
        assert!(
            client
                .estimate("/v1/chat/completions", json!([]))
                .await
                .is_err()
        );
    }
}
