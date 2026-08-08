//! `/v1/jobs`: batch jobs over inline items or over a connector's data store.
//!
//! A job is either *inline* (items travel in the request, results come back) or
//! *connector-driven* (a `scheme://connection/...` URI names a store the server reads from
//! and writes to). The two shapes accept disjoint fields, and mixing them is rejected
//! before anything is sent.

use std::path::Path;
use std::time::Duration;

use reqwest::Method;
use rmpv::Value as MsgValue;
use serde_json::{Map, Value};

use crate::client::files::TRANSFER_TIMEOUT_FLOOR;
use crate::client::{Client, meta::parse_json};
use crate::error::{Error, Result};
use crate::http::{PreparedRequest, headers};
use crate::retry::RetryPolicy;
use crate::types::{
    JobChunk, JobExecution, JobFieldMap, JobResultItem, JobResults, JobStatus, OutputType,
};
use crate::wire::{msg, ndarray};

/// URI schemes the server resolves itself, without a caller-named connection.
const INTERNAL_SCHEMES: &[&str] = &["upload"];

/// One inline item.
#[derive(Debug, Clone, PartialEq)]
pub enum JobItem {
    /// Plain text; the server wraps it as `{"text": ...}`.
    Text(String),
    /// A pre-built item object.
    Object(Value),
}

impl From<&str> for JobItem {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for JobItem {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Value> for JobItem {
    fn from(value: Value) -> Self {
        Self::Object(value)
    }
}

impl JobItem {
    fn to_json(&self) -> Value {
        match self {
            Self::Text(text) => serde_json::json!({"text": text}),
            Self::Object(value) => value.clone(),
        }
    }
}

/// Where a job reads its input from.
#[derive(Debug, Clone, PartialEq)]
pub enum JobSource {
    /// Items carried in the request body.
    Items(Vec<JobItem>),
    /// A `scheme://connection/...` URI naming a data store.
    Connector(String),
}

impl JobSource {
    /// Inline items.
    pub fn items(items: impl IntoIterator<Item = impl Into<JobItem>>) -> Self {
        Self::Items(items.into_iter().map(Into::into).collect())
    }

    /// A connector URI.
    pub fn connector(uri: impl Into<String>) -> Self {
        Self::Connector(uri.into())
    }
}

/// Where a job writes its output.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum JobSink {
    /// Results come back through [`Jobs::results`]. The default.
    #[default]
    Return,
    /// Results are written back over the source rows.
    InPlace,
    /// Results are written to a second data store.
    Connector(String),
}

impl JobSink {
    /// A connector URI.
    pub fn connector(uri: impl Into<String>) -> Self {
        Self::Connector(uri.into())
    }
}

/// The connection a connector URI names.
///
/// Deliberately not a URL parse: control characters must reach the validator and be
/// rejected, rather than being normalized away into something that looks valid.
pub fn connection_name(uri: &str) -> Result<String> {
    let after_scheme = uri.split_once("://").map_or("", |(_, rest)| rest);
    let name = after_scheme
        .split(['/', '?', '#'])
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            Error::invalid(format!(
                "connector URI {uri:?} names no connection (expected 'scheme://<connection>/…')"
            ))
        })?;
    require_connection_name(name)
}

/// Validate a connection name and return its canonical form.
pub fn require_connection_name(name: &str) -> Result<String> {
    let valid = (1..=128).contains(&name.len())
        && name.is_ascii()
        && name
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-');
    if valid {
        Ok(name.to_string())
    } else {
        Err(Error::invalid(
            "connection name must be 1-128 ASCII letters, digits, '.', '_', or '-', \
             and start with a letter or digit",
        ))
    }
}

/// Validate the `PostgreSQL` schema pair, which is all-or-nothing.
pub fn require_connection_schema_policy(
    connection_type: &str,
    source_schema: Option<&str>,
    sink_schema: Option<&str>,
) -> Result<Option<(String, String)>> {
    match (source_schema, sink_schema) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(Error::invalid(
            "source_schema and sink_schema must be supplied together",
        )),
        (Some(source), Some(sink)) => {
            if connection_type != "postgres" {
                return Err(Error::invalid(format!(
                    "source_schema and sink_schema apply only to postgres connections, got {connection_type:?}"
                )));
            }
            Ok(Some((valid_schema(source)?, valid_schema(sink)?)))
        }
    }
}

fn valid_schema(schema: &str) -> Result<String> {
    let valid = (1..=63).contains(&schema.len())
        && schema.is_ascii()
        && schema
            .as_bytes()
            .first()
            .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
        && schema
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'$');
    if valid {
        Ok(schema.to_string())
    } else {
        Err(Error::invalid(format!(
            "schema {schema:?} must be 1-63 ASCII letters, digits, '_' or '$', and start with a letter or '_'"
        )))
    }
}

/// Validate a connector idempotency key.
pub fn require_connector_idempotency_key(key: &str) -> Result<String> {
    let bytes = key.as_bytes();
    if (1..=256).contains(&bytes.len()) && bytes.iter().all(|b| (0x20..=0x7e).contains(b)) {
        Ok(key.to_string())
    } else {
        Err(Error::invalid(
            "idempotency_key must be 1-256 printable ASCII bytes",
        ))
    }
}

fn is_connector_uri(value: &str) -> bool {
    value.contains("://")
}

fn is_internal_uri(uri: &str) -> bool {
    uri.split_once("://")
        .is_some_and(|(scheme, _)| INTERNAL_SCHEMES.contains(&scheme))
}

/// The jobs namespace. Obtain one with [`Client::jobs`].
#[derive(Debug, Clone)]
pub struct Jobs {
    client: Client,
}

impl Client {
    /// Operations on batch jobs.
    pub fn jobs(&self) -> Jobs {
        Jobs {
            client: self.clone(),
        }
    }
}

impl Jobs {
    /// Queue a job.
    pub fn submit(&self, source: JobSource, model: impl Into<String>) -> JobSubmit {
        JobSubmit {
            client: self.client.clone(),
            source,
            model: model.into(),
            operation: "encode".to_string(),
            sink: JobSink::Return,
            connection: None,
            sink_connection: None,
            field_map: JobFieldMap::default(),
            output_field: None,
            execution: None,
            output_types: None,
            options: None,
            idempotency_key: None,
        }
    }

    /// Fetch one job.
    pub async fn get(&self, job_id: &str) -> Result<JobStatus> {
        let response = self
            .client
            .send_once(
                self.job_request(Method::GET, job_id, "")?,
                RetryPolicy::NONE,
            )
            .await?;
        parse_json(&response, "job")
    }

    /// List this account's jobs.
    pub async fn list(&self) -> Result<Vec<JobStatus>> {
        let request = self
            .client
            .request(Method::GET, "/v1/jobs")?
            .header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        if let Ok(jobs) = serde_json::from_slice::<Vec<JobStatus>>(&response.body) {
            return Ok(jobs);
        }
        let envelope: Value = parse_json(&response, "job list")?;
        let data = envelope
            .get("data")
            .cloned()
            .ok_or_else(|| Error::decode("job list is missing its `data` array"))?;
        serde_json::from_value(data)
            .map_err(|err| Error::decode(format!("malformed job list: {err}")))
    }

    /// Ask the server to stop a job.
    pub async fn cancel(&self, job_id: &str) -> Result<JobStatus> {
        let request = self
            .job_request(Method::POST, job_id, "/cancel")?
            .json_headers();
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "job")
    }

    /// Execute a plan a previous `execution = plan` submit produced.
    pub async fn execute(
        &self,
        job_id: &str,
        plan_revision: u64,
        idempotency_key: &str,
    ) -> Result<JobStatus> {
        self.plan_action(
            job_id,
            "/execute",
            serde_json::json!({"plan_revision": plan_revision}),
            idempotency_key,
        )
        .await
    }

    /// Re-run the failed part of an attempt, inside the job's repair window.
    pub async fn repair(
        &self,
        job_id: &str,
        plan_revision: u64,
        recovery_attempt_ordinal: u64,
        idempotency_key: &str,
    ) -> Result<JobStatus> {
        self.plan_action(
            job_id,
            "/repair",
            serde_json::json!({
                "plan_revision": plan_revision,
                "recovery_attempt_ordinal": recovery_attempt_ordinal,
            }),
            idempotency_key,
        )
        .await
    }

    async fn plan_action(
        &self,
        job_id: &str,
        suffix: &str,
        body: Value,
        idempotency_key: &str,
    ) -> Result<JobStatus> {
        let key = require_connector_idempotency_key(idempotency_key)?;
        let request = self
            .job_request(Method::POST, job_id, suffix)?
            .json_headers()
            .header(headers::IDEMPOTENCY_KEY, &key)
            .body(serde_json::to_vec(&body).unwrap_or_default());
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "job")
    }

    /// Poll until the job settles, or the deadline passes.
    pub async fn wait(
        &self,
        job_id: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<JobStatus> {
        let start = std::time::Instant::now();
        loop {
            let status = self.get(job_id).await?;
            if status.is_settled() {
                return Ok(status);
            }
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(Error::Request {
                    message: format!(
                        "job {job_id} is still {:?} after {:.0}s",
                        status.state,
                        timeout.as_secs_f64()
                    ),
                    code: Some("job_wait_timeout".to_string()),
                    status: 504,
                    request: None,
                });
            }
            tokio::time::sleep(poll_interval.min(timeout.checked_sub(elapsed).unwrap())).await;
        }
    }

    /// Fetch a job's status and decode every chunk it can still retrieve.
    ///
    /// Chunks that failed, or whose payload has expired, are reported in `chunks` but
    /// contribute no items; `retrieved` says how many were actually read.
    pub async fn results(&self, job_id: &str) -> Result<JobResults> {
        let status = self.get(job_id).await?;
        let chunks = status.chunks();

        let mut items = Vec::new();
        let mut retrieved = 0;
        for chunk in &chunks {
            if !chunk_is_retrievable(chunk) {
                continue;
            }
            let Some(reference) = chunk.r#ref.as_deref() else {
                continue;
            };
            let raw = self.read_ref(reference).await?;
            items.extend(decode_chunk(&raw)?);
            retrieved += 1;
        }

        Ok(JobResults {
            job_id: status.id.clone(),
            state: status.state,
            total_items: status.total_items,
            settled_credits: status.settled_credits,
            dims: items.iter().find_map(|item| item.dims),
            chunks,
            retrieved,
            items,
        })
    }

    fn job_request(&self, method: Method, job_id: &str, suffix: &str) -> Result<PreparedRequest> {
        // A job id is one path segment, so every reserved character is escaped.
        let encoded: String =
            percent_encoding::utf8_percent_encode(job_id, percent_encoding::NON_ALPHANUMERIC)
                .collect();
        Ok(self
            .client
            .request(method, &format!("/v1/jobs/{encoded}{suffix}"))?
            .header("accept", headers::JSON_CONTENT_TYPE))
    }

    /// Fetch one chunk's payload.
    ///
    /// A payload reference points at a signed, time-limited object-store URL that is not
    /// the gateway. It is fetched without any credential: sending `Authorization` to a
    /// location the server named would hand the token to whatever host is behind it.
    async fn read_ref(&self, reference: &str) -> Result<bytes::Bytes> {
        if !reference.starts_with("http://") && !reference.starts_with("https://") {
            let path = Path::new(reference);
            return std::fs::read(path)
                .map(bytes::Bytes::from)
                .map_err(|_| Error::Request {
                    message: format!("job payload reference {reference:?} could not be resolved"),
                    code: Some("bad_ref".to_string()),
                    status: 400,
                    request: None,
                });
        }
        self.client.fetch_payload_ref(reference).await
    }
}

impl Client {
    /// Fetch a payload reference with a bare, unauthenticated request.
    async fn fetch_payload_ref(&self, reference: &str) -> Result<bytes::Bytes> {
        let url = reqwest::Url::parse(reference).map_err(|err| {
            Error::invalid(format!("invalid payload reference {reference:?}: {err}"))
        })?;

        let bare = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| {
                Error::invalid(format!("could not build the payload-ref client: {err}"))
            })?;

        let mut request = bare
            .get(url.clone())
            .header(reqwest::header::ACCEPT, headers::OCTET_STREAM_CONTENT_TYPE)
            .timeout(self.timeout().max(TRANSFER_TIMEOUT_FLOOR));
        // Edge credentials still apply when the reference happens to be the gateway itself.
        if self.edge_headers_apply_to(&url) {
            for (name, value) in &self.inner.edge_headers {
                request = request.header(name.clone(), value.clone());
            }
        }

        let response = request.send().await.map_err(|error| {
            Error::connection(
                crate::error::TransportErrorKind::Connect,
                format!("could not fetch payload reference {reference}: {error}"),
                error,
            )
        })?;
        let status = response.status().as_u16();
        let body = response.bytes().await.map_err(|error| {
            Error::connection(
                crate::error::TransportErrorKind::MidFlight,
                format!("could not read payload reference {reference}: {error}"),
                error,
            )
        })?;

        if status >= 400 {
            return Err(Error::Request {
                message: format!("payload reference {reference} returned HTTP {status}"),
                code: Some("bad_ref".to_string()),
                status,
                request: None,
            });
        }
        Ok(body)
    }
}

/// Decode a chunk's msgpack array of worker results.
fn decode_chunk(raw: &[u8]) -> Result<Vec<JobResultItem>> {
    let decoded: MsgValue = rmp_serde::from_slice(raw)
        .map_err(|err| Error::decode(format!("malformed job chunk: {err}")))?;
    let MsgValue::Array(entries) = decoded else {
        return Ok(Vec::new());
    };
    entries.iter().map(decode_result_item).collect()
}

fn decode_result_item(entry: &MsgValue) -> Result<JobResultItem> {
    let mut item = JobResultItem {
        id: msg::get_string(entry, "id"),
        success: msg::get(entry, "success").and_then(rmpv::Value::as_bool),
        units: msg::get(entry, "units").map(msg::to_json),
        error: msg::get_string(entry, "error"),
        ..JobResultItem::default()
    };

    // The worker's own result is an opaque msgpack blob the gateway never unpacks.
    let Some(MsgValue::Binary(payload)) = msg::get(entry, "result_msgpack") else {
        return Ok(item);
    };
    let Ok(inner) = rmp_serde::from_slice::<MsgValue>(payload) else {
        // A blob the SDK cannot read must not discard the row's outcome.
        return Ok(item);
    };
    if let Some(dense) = msg::get(&inner, "dense") {
        let (dims, values) = dense_info(dense)?;
        item.dims = dims;
        item.dense = values;
    }
    Ok(item)
}

/// Read a `dense` field, which may be a bare array or a `{dims, values}` envelope.
fn dense_info(dense: &MsgValue) -> Result<(Option<u32>, Option<Vec<f32>>)> {
    if ndarray::is_array(dense) {
        let array = ndarray::decode(dense)?;
        let values = array.to_f32();
        return Ok((Some(values.len() as u32), Some(values)));
    }
    if let MsgValue::Array(items) = dense {
        let values: Vec<f32> = items
            .iter()
            .filter_map(|value| value.as_f64().map(|v| v as f32))
            .collect();
        return Ok((Some(values.len() as u32), Some(values)));
    }
    if matches!(dense, MsgValue::Map(_)) {
        let declared = msg::get_u64(dense, "dims").map(|dims| dims as u32);
        for key in ["values", "vector", "dense"] {
            if let Some(node) = msg::get(dense, key) {
                let (derived, values) = dense_info(node)?;
                return Ok((declared.or(derived), values));
            }
        }
        return Ok((declared, None));
    }
    Ok((None, None))
}

/// Queues a job. Build with [`Jobs::submit`].
pub struct JobSubmit {
    client: Client,
    source: JobSource,
    model: String,
    operation: String,
    sink: JobSink,
    connection: Option<String>,
    sink_connection: Option<String>,
    field_map: JobFieldMap,
    output_field: Option<String>,
    execution: Option<JobExecution>,
    output_types: Option<Vec<OutputType>>,
    options: Option<Value>,
    idempotency_key: Option<String>,
}

impl JobSubmit {
    /// Which inference operation to run. Defaults to `encode`.
    pub fn operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = operation.into();
        self
    }

    /// Where results go. Defaults to [`JobSink::Return`].
    pub fn sink(mut self, sink: JobSink) -> Self {
        self.sink = sink;
        self
    }

    /// Override the connection the source URI names.
    pub fn connection(mut self, connection: impl Into<String>) -> Self {
        self.connection = Some(connection.into());
        self
    }

    /// Override the connection the sink URI names.
    pub fn sink_connection(mut self, connection: impl Into<String>) -> Self {
        self.sink_connection = Some(connection.into());
        self
    }

    /// How the source's columns map onto job items.
    pub fn field_map(mut self, field_map: JobFieldMap) -> Self {
        self.field_map = field_map;
        self
    }

    /// Which sink column receives the result.
    pub fn output_field(mut self, output_field: impl Into<String>) -> Self {
        self.output_field = Some(output_field.into());
        self
    }

    /// Which half of a connector job to run. Required for connector sources.
    pub fn execution(mut self, execution: JobExecution) -> Self {
        self.execution = Some(execution);
        self
    }

    /// Which representations to produce.
    pub fn output_types(mut self, types: impl IntoIterator<Item = OutputType>) -> Self {
        self.output_types = Some(types.into_iter().collect());
        self
    }

    /// Adapter-specific runtime options.
    pub fn options(mut self, options: Value) -> Self {
        self.options = Some(options);
        self
    }

    /// Deduplication key. Required for connector jobs, rejected for inline ones.
    pub fn idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }

    /// Assemble the request body, validating that the inline and connector shapes are not
    /// mixed. Only the fields that were set ride the wire.
    pub(crate) fn body(&self) -> Result<Value> {
        let mut body = Map::new();
        body.insert(
            "operation".to_string(),
            Value::String(self.operation.clone()),
        );
        body.insert("model".to_string(), Value::String(self.model.clone()));

        let source_connection = self.write_source(&mut body)?;
        let inline = body.contains_key("items");
        let sink_fields = self.sink_fields(source_connection.as_ref())?;

        if inline
            && (self.connection.is_some()
                || self.sink_connection.is_some()
                || !sink_fields.is_empty())
        {
            return Err(Error::invalid(
                "connection/sink/sink_connection apply only to connector-src jobs; inline items return results",
            ));
        }
        body.extend(sink_fields);

        self.write_execution(&mut body, inline)?;
        let mapping = self.mapping_fields()?;
        if !mapping.is_empty() {
            if inline {
                return Err(Error::invalid(
                    "field_map/output_field apply to connector-src jobs; an inline items job maps nothing",
                ));
            }
            body.extend(mapping);
        }

        if let Some(types) = &self.output_types
            && !types.is_empty()
        {
            body.insert(
                "output_types".to_string(),
                serde_json::to_value(types).unwrap_or(Value::Null),
            );
        }
        if let Some(options) = &self.options
            && options.as_object().is_some_and(|map| !map.is_empty())
        {
            body.insert("options".to_string(), options.clone());
        }

        Ok(Value::Object(body))
    }

    /// Write the source fields and return the connection the source resolved to.
    fn write_source(&self, body: &mut Map<String, Value>) -> Result<Option<String>> {
        match &self.source {
            JobSource::Items(items) => {
                if items.is_empty() {
                    return Err(Error::invalid("inline source has no items"));
                }
                body.insert(
                    "items".to_string(),
                    Value::Array(items.iter().map(JobItem::to_json).collect()),
                );
                Ok(None)
            }
            JobSource::Connector(uri) => {
                if !is_connector_uri(uri) {
                    return Err(Error::invalid(format!(
                        "connector source must be a 'scheme://<connection>/…' URI, got {uri:?}"
                    )));
                }
                body.insert("src".to_string(), Value::String(uri.clone()));
                let resolved = match (&self.connection, is_internal_uri(uri)) {
                    (Some(name), _) => Some(require_connection_name(name)?),
                    // An internal scheme is resolved by the server, so it names no connection.
                    (None, true) => None,
                    (None, false) => Some(connection_name(uri)?),
                };
                if let Some(name) = &resolved {
                    body.insert("connection".to_string(), Value::String(name.clone()));
                }
                Ok(resolved)
            }
        }
    }

    fn sink_fields(&self, source_connection: Option<&String>) -> Result<Map<String, Value>> {
        let mut fields = Map::new();
        match &self.sink {
            JobSink::Return => {}
            JobSink::InPlace => {
                fields.insert("sink".to_string(), Value::String("inplace".to_string()));
            }
            JobSink::Connector(uri) => {
                if !is_connector_uri(uri) {
                    return Err(Error::invalid(format!(
                        "sink must be 'return', 'inplace', or a connector URI, got {uri:?}"
                    )));
                }
                fields.insert("sink".to_string(), Value::String(uri.clone()));
                if is_internal_uri(uri) {
                    if let Some(name) = &self.sink_connection {
                        fields.insert(
                            "sink_connection".to_string(),
                            Value::String(require_connection_name(name)?),
                        );
                    }
                } else {
                    let resolved = match &self.sink_connection {
                        Some(name) => require_connection_name(name)?,
                        None => connection_name(uri)?,
                    };
                    // A sink in the same store as the source needs no second connection.
                    if self.sink_connection.is_some() || Some(&resolved) != source_connection {
                        fields.insert("sink_connection".to_string(), Value::String(resolved));
                    }
                }
            }
        }
        Ok(fields)
    }

    fn write_execution(&self, body: &mut Map<String, Value>, inline: bool) -> Result<()> {
        match (inline, self.execution) {
            (false, None) => Err(Error::invalid(
                "connector jobs require execution = Plan or execution = Run",
            )),
            (false, Some(execution)) => {
                let uses_internal = matches!(&self.source, JobSource::Connector(uri) if is_internal_uri(uri))
                    || matches!(&self.sink, JobSink::Connector(uri) if is_internal_uri(uri));
                if uses_internal && execution != JobExecution::Run {
                    return Err(Error::invalid(
                        "upload:// connector jobs are run-only; set execution = Run",
                    ));
                }
                body.insert(
                    "execution".to_string(),
                    serde_json::to_value(execution).unwrap_or(Value::Null),
                );
                Ok(())
            }
            (true, Some(_)) => Err(Error::invalid(
                "execution applies only to connector-src jobs; inline items must omit it",
            )),
            (true, None) => Ok(()),
        }
    }

    fn mapping_fields(&self) -> Result<Map<String, Value>> {
        let mut mapping = Map::new();
        if !self.field_map.is_empty() {
            if let Some(input_type) = &self.field_map.input_type
                && input_type != "text"
                && input_type != "document"
            {
                return Err(Error::invalid(format!(
                    "field_map.input_type must be 'text' or 'document', got {input_type:?}"
                )));
            }
            if self.field_map.carry.iter().any(String::is_empty) {
                return Err(Error::invalid(
                    "field_map.carry must not contain empty field names",
                ));
            }
            mapping.insert(
                "field_map".to_string(),
                serde_json::to_value(&self.field_map)
                    .map_err(|err| Error::invalid(format!("could not encode field_map: {err}")))?,
            );
        }
        if let Some(output_field) = &self.output_field {
            if output_field.is_empty() {
                return Err(Error::invalid("output_field must not be empty"));
            }
            mapping.insert(
                "output_field".to_string(),
                Value::String(output_field.clone()),
            );
        }
        Ok(mapping)
    }

    /// Send the request.
    pub async fn send(self) -> Result<JobStatus> {
        let body = self.body()?;
        let connector = body.get("src").is_some();

        let key = match (&self.idempotency_key, connector) {
            (Some(key), true) => Some(require_connector_idempotency_key(key)?),
            (None, true) => {
                return Err(Error::invalid(
                    "connector-src jobs require an idempotency_key so a retried submit cannot run twice",
                ));
            }
            (Some(_), false) => {
                return Err(Error::invalid(
                    "idempotency_key applies only to connector-src jobs; inline items must omit it",
                ));
            }
            (None, false) => None,
        };

        let request = self
            .client
            .request(Method::POST, "/v1/jobs")?
            .json_headers()
            .maybe_header(headers::IDEMPOTENCY_KEY, key.as_deref())
            .body(serde_json::to_vec(&body).unwrap_or_default());

        let response = self
            .client
            .send_with_timeout(request, RetryPolicy::NONE, TRANSFER_TIMEOUT_FLOOR)
            .await?;
        parse_json(&response, "job")
    }
}

/// Read a chunk's payload for callers holding one outside [`Jobs::results`].
pub fn decode_chunk_payload(raw: &[u8]) -> Result<Vec<JobResultItem>> {
    decode_chunk(raw)
}

/// Whether a chunk has a payload worth fetching.
pub fn chunk_is_retrievable(chunk: &JobChunk) -> bool {
    chunk.state == "succeeded" && chunk.r#ref.as_ref().is_some_and(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;
    use serde_json::json;

    fn client() -> Client {
        Client::new("https://sie.invalid").unwrap()
    }

    fn submit(source: JobSource) -> JobSubmit {
        client().jobs().submit(source, "BAAI/bge-m3")
    }

    #[test]
    fn an_inline_job_carries_only_operation_model_and_items() {
        let body = submit(JobSource::items(["a", "b"])).body().unwrap();
        assert_eq!(
            body,
            json!({"operation": "encode", "model": "BAAI/bge-m3",
                   "items": [{"text": "a"}, {"text": "b"}]})
        );
    }

    #[test]
    fn inline_object_items_pass_through_unchanged() {
        let body = submit(JobSource::items([json!({"id": "1", "text": "hi"})]))
            .body()
            .unwrap();
        assert_eq!(body["items"], json!([{"id": "1", "text": "hi"}]));
    }

    #[test]
    fn an_empty_inline_source_is_rejected() {
        let empty: Vec<&str> = Vec::new();
        assert!(submit(JobSource::items(empty)).body().is_err());
    }

    #[test]
    fn a_connector_source_derives_its_connection_from_the_uri() {
        let body = submit(JobSource::connector("postgres://warehouse?query=SELECT+1"))
            .execution(JobExecution::Plan)
            .body()
            .unwrap();
        assert_eq!(body["src"], json!("postgres://warehouse?query=SELECT+1"));
        assert_eq!(body["connection"], json!("warehouse"));
        assert_eq!(body["execution"], json!("plan"));
    }

    #[test]
    fn a_sink_in_the_same_store_needs_no_second_connection() {
        let body = submit(JobSource::connector("postgres://warehouse?query=SELECT+1"))
            .sink(JobSink::connector("postgres://warehouse?table=vecs"))
            .execution(JobExecution::Plan)
            .body()
            .unwrap();
        assert_eq!(body["sink"], json!("postgres://warehouse?table=vecs"));
        assert!(body.get("sink_connection").is_none());
    }

    #[test]
    fn a_sink_in_another_store_names_its_own_connection() {
        let body = submit(JobSource::connector("postgres://warehouse?query=SELECT+1"))
            .sink(JobSink::connector("s3://out-bucket/vecs"))
            .execution(JobExecution::Run)
            .body()
            .unwrap();
        assert_eq!(body["sink_connection"], json!("out-bucket"));
    }

    #[test]
    fn the_sink_variants_render_their_wire_forms() {
        let inplace = submit(JobSource::connector("postgres://wh?query=x"))
            .sink(JobSink::InPlace)
            .execution(JobExecution::Run)
            .body()
            .unwrap();
        assert_eq!(inplace["sink"], json!("inplace"));

        let returned = submit(JobSource::items(["a"])).body().unwrap();
        assert!(returned.get("sink").is_none());
    }

    #[test]
    fn upload_jobs_name_no_connection_and_are_run_only() {
        let body = submit(JobSource::connector("upload://file-abc?format=csv"))
            .sink(JobSink::connector("upload://file-out"))
            .field_map(JobFieldMap {
                id_field: Some("doc_id".to_string()),
                input_field: Some("text".to_string()),
                input_type: Some("text".to_string()),
                ..JobFieldMap::default()
            })
            .execution(JobExecution::Run)
            .body()
            .unwrap();
        assert!(body.get("connection").is_none());
        assert!(body.get("sink_connection").is_none());
        assert_eq!(body["field_map"]["id_field"], json!("doc_id"));

        let planned = submit(JobSource::connector("upload://file-abc"))
            .execution(JobExecution::Plan)
            .body()
            .unwrap_err();
        assert!(planned.to_string().contains("run-only"), "{planned}");
    }

    #[test]
    fn connector_and_inline_shapes_cannot_be_mixed() {
        let with_connection = submit(JobSource::items(["a"]))
            .connection("warehouse")
            .body()
            .unwrap_err();
        assert!(
            with_connection.to_string().contains("connector-src"),
            "{with_connection}"
        );

        let with_sink = submit(JobSource::items(["a"]))
            .sink(JobSink::connector("s3://out/vecs"))
            .body()
            .unwrap_err();
        assert!(
            with_sink.to_string().contains("connector-src"),
            "{with_sink}"
        );

        let with_mapping = submit(JobSource::items(["a"]))
            .output_field("embedding")
            .body()
            .unwrap_err();
        assert!(
            with_mapping.to_string().contains("maps nothing"),
            "{with_mapping}"
        );
    }

    #[test]
    fn execution_is_required_for_connectors_and_forbidden_inline() {
        let missing = submit(JobSource::connector("postgres://wh?query=x"))
            .body()
            .unwrap_err();
        assert!(
            missing.to_string().contains("require execution"),
            "{missing}"
        );

        let extra = submit(JobSource::items(["a"]))
            .execution(JobExecution::Run)
            .body()
            .unwrap_err();
        assert!(extra.to_string().contains("must omit it"), "{extra}");
    }

    #[test]
    fn optional_fields_only_appear_when_set() {
        let bare = submit(JobSource::items(["a"])).body().unwrap();
        assert!(bare.get("output_types").is_none());
        assert!(bare.get("options").is_none());

        let full = submit(JobSource::items(["a"]))
            .output_types([OutputType::Dense])
            .options(json!({"is_query": true}))
            .body()
            .unwrap();
        assert_eq!(full["output_types"], json!(["dense"]));
        assert_eq!(full["options"], json!({"is_query": true}));

        // An empty options object rides nothing.
        let empty = submit(JobSource::items(["a"]))
            .options(json!({}))
            .body()
            .unwrap();
        assert!(empty.get("options").is_none());
    }

    #[test]
    fn field_map_input_type_is_constrained() {
        let err = submit(JobSource::connector("postgres://wh?query=x"))
            .execution(JobExecution::Run)
            .field_map(JobFieldMap {
                input_type: Some("image".to_string()),
                ..JobFieldMap::default()
            })
            .body()
            .unwrap_err();
        assert!(err.to_string().contains("'text' or 'document'"), "{err}");
    }

    #[test]
    fn connection_names_reject_traversal_and_non_ascii() {
        assert_eq!(
            connection_name("postgres://warehouse?query=x").unwrap(),
            "warehouse"
        );
        assert_eq!(
            connection_name("s3://customer-bucket/in/").unwrap(),
            "customer-bucket"
        );
        assert_eq!(connection_name("gs://my-bucket").unwrap(), "my-bucket");

        for uri in [
            "postgres://../other",
            "postgres://warehouse\\name",
            "postgres://warehouse%2fname",
            "postgres://_leading",
            "postgres://café",
            "postgres://",
            "not-a-uri",
        ] {
            assert!(connection_name(uri).is_err(), "{uri:?} should be rejected");
        }
        // A control character must fail the validator rather than be normalized away.
        assert!(connection_name("postgres://warehouse\n").is_err());
        assert!(require_connection_name(&"a".repeat(129)).is_err());
        assert!(require_connection_name(&"a".repeat(128)).is_ok());
    }

    #[test]
    fn schema_policy_is_all_or_nothing_and_postgres_only() {
        assert!(
            require_connection_schema_policy("postgres", None, None)
                .unwrap()
                .is_none()
        );
        assert!(require_connection_schema_policy("postgres", Some("src"), None).is_err());
        assert!(require_connection_schema_policy("s3", Some("src"), Some("dst")).is_err());
        assert_eq!(
            require_connection_schema_policy("postgres", Some("src"), Some("dst")).unwrap(),
            Some(("src".to_string(), "dst".to_string()))
        );
        assert!(require_connection_schema_policy("postgres", Some("1bad"), Some("dst")).is_err());
    }

    #[test]
    fn idempotency_keys_must_be_printable_ascii() {
        assert!(require_connector_idempotency_key("run-2026-08-07").is_ok());
        assert!(require_connector_idempotency_key("").is_err());
        assert!(require_connector_idempotency_key(&"k".repeat(257)).is_err());
        assert!(require_connector_idempotency_key(&"k".repeat(256)).is_ok());
        assert!(require_connector_idempotency_key("with\nnewline").is_err());
        assert!(require_connector_idempotency_key("café").is_err());
    }

    #[tokio::test]
    async fn idempotency_keys_are_required_for_connectors_and_refused_inline() {
        let missing = submit(JobSource::connector("postgres://wh?query=x"))
            .execution(JobExecution::Run)
            .send()
            .await
            .unwrap_err();
        assert!(missing.to_string().contains("idempotency_key"), "{missing}");

        let extra = submit(JobSource::items(["a"]))
            .idempotency_key("k")
            .send()
            .await
            .unwrap_err();
        assert!(extra.to_string().contains("must omit it"), "{extra}");
    }

    /// Build a worker-result chunk the way the gateway packs one: the inner result is an
    /// opaque msgpack `bin`, not a nested structure.
    fn work_result_chunk(entries: Vec<Vec<(&str, MsgValue)>>) -> Vec<u8> {
        let array = MsgValue::Array(
            entries
                .into_iter()
                .map(|fields| {
                    MsgValue::Map(
                        fields
                            .into_iter()
                            .map(|(key, value)| (MsgValue::from(key), value))
                            .collect(),
                    )
                })
                .collect(),
        );
        rmp_serde::to_vec(&array).unwrap()
    }

    #[test]
    fn chunk_payloads_decode_into_result_items() {
        let inner = rmp_serde::to_vec(&MsgValue::Map(vec![(
            MsgValue::from("dense"),
            MsgValue::Map(vec![
                (MsgValue::from("dims"), MsgValue::from(3u64)),
                (
                    MsgValue::from("values"),
                    MsgValue::Array(vec![
                        MsgValue::F32(0.1),
                        MsgValue::F32(0.2),
                        MsgValue::F32(0.3),
                    ]),
                ),
            ]),
        )]))
        .unwrap();

        let chunk = work_result_chunk(vec![vec![
            ("success", MsgValue::Boolean(true)),
            ("id", MsgValue::from("0")),
            (
                "units",
                MsgValue::Map(vec![(MsgValue::from("input_tokens"), MsgValue::from(5u64))]),
            ),
            ("result_msgpack", MsgValue::Binary(inner)),
        ]]);

        let items = decode_chunk_payload(&chunk).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id.as_deref(), Some("0"));
        assert_eq!(items[0].success, Some(true));
        assert_eq!(items[0].dims, Some(3));
        assert_eq!(
            items[0].dense.as_deref(),
            Some([0.1f32, 0.2, 0.3].as_slice())
        );
        assert_eq!(items[0].units, Some(json!({"input_tokens": 5})));
    }

    #[test]
    fn an_unreadable_inner_payload_still_reports_the_rows_outcome() {
        // 0xc1 is the one byte msgpack never emits, so the inner decode must fail.
        let chunk = work_result_chunk(vec![vec![
            ("success", MsgValue::Boolean(false)),
            ("id", MsgValue::from("7")),
            ("error", MsgValue::from("worker crashed")),
            ("result_msgpack", MsgValue::Binary(vec![0xc1])),
        ]]);
        let items = decode_chunk_payload(&chunk).unwrap();
        assert_eq!(items[0].success, Some(false));
        assert_eq!(items[0].error.as_deref(), Some("worker crashed"));
        assert!(items[0].dense.is_none());
    }

    #[test]
    fn a_chunk_that_is_not_an_array_decodes_to_nothing() {
        let chunk = rmp_serde::to_vec(&MsgValue::Map(vec![(
            MsgValue::from("unexpected"),
            MsgValue::Boolean(true),
        )]))
        .unwrap();
        assert!(decode_chunk_payload(&chunk).unwrap().is_empty());
    }

    #[test]
    fn a_numpy_encoded_dense_result_decodes_too() {
        let inner = rmp_serde::to_vec(&MsgValue::Map(vec![(
            MsgValue::from("dense"),
            ndarray::fixtures::f32_array(&[2], &[1.5, -0.5]),
        )]))
        .unwrap();
        let chunk = work_result_chunk(vec![vec![
            ("id", MsgValue::from("0")),
            ("result_msgpack", MsgValue::Binary(inner)),
        ]]);
        let items = decode_chunk_payload(&chunk).unwrap();
        assert_eq!(items[0].dims, Some(2));
        assert_eq!(items[0].dense.as_deref(), Some([1.5f32, -0.5].as_slice()));
    }

    #[test]
    fn only_succeeded_chunks_with_a_ref_are_worth_fetching() {
        let succeeded = JobChunk {
            state: "succeeded".to_string(),
            r#ref: Some("https://store/0".to_string()),
            ..JobChunk::default()
        };
        assert!(chunk_is_retrievable(&succeeded));
        assert!(!chunk_is_retrievable(&JobChunk {
            state: "failed".to_string(),
            ..succeeded.clone()
        }));
        assert!(!chunk_is_retrievable(&JobChunk {
            r#ref: None,
            ..succeeded
        }));
    }
}
