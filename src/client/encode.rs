//! `/v1/encode`, `/v1/score` and `/v1/extract`: the msgpack inference endpoints.

use reqwest::Method;
use rmpv::Value as MsgValue;
use serde_json::Value;

use crate::client::Client;
use crate::error::{Error, Result, codes};
use crate::http::{HttpResponse, headers, metadata};
use crate::retry::{RequestOptions, RetryPolicy, RetryState};
use crate::types::item::json_to_msgpack;
use crate::types::{
    Classification, DetectedObject, EncodeResult, Entity, ExtractItemError, ExtractResult, Item,
    Multivector, OutputDType, OutputType, Relation, RequestMetadata, ScoreEntry, ScoreResult,
    ScoreUsage, SparseVector, TimingInfo,
};
use crate::wire::{msg, ndarray};

const MALFORMED_EXTRACT_ERROR_MESSAGE: &str = "Malformed extraction item error";

/// Options shared by every request builder in this module.
macro_rules! request_options {
    () => {
        /// Route to a machine profile, optionally pool-qualified as `"pool/profile"`.
        pub fn gpu(mut self, gpu: impl Into<String>) -> Self {
            self.options.gpu = Some(gpu.into());
            self
        }

        /// Whether to wait out provisioning and model loading, or fail fast.
        pub fn wait_for_capacity(mut self, wait: bool) -> Self {
            self.options.wait_for_capacity = wait;
            self
        }

        /// Total wall-clock budget for this call, retries included.
        pub fn provision_timeout(mut self, timeout: std::time::Duration) -> Self {
            self.options.provision_timeout = timeout;
            self
        }

        /// Cap on `RESOURCE_EXHAUSTED` retries. Zero fails fast.
        pub fn max_oom_retries(mut self, retries: u32) -> Self {
            self.options.max_oom_retries = retries;
            self
        }
    };
}

pub(crate) use request_options;

impl Client {
    /// Embed one or more items.
    pub fn encode(
        &self,
        model: impl Into<String>,
        items: impl IntoIterator<Item = Item>,
    ) -> EncodeRequest {
        EncodeRequest {
            client: self.clone(),
            model: model.into(),
            items: items.into_iter().collect(),
            output_types: None,
            instruction: None,
            output_dtype: None,
            is_query: None,
            runtime_options: None,
            options: self.request_options(),
        }
    }

    /// Score a query against a set of candidates.
    pub fn score(
        &self,
        model: impl Into<String>,
        query: Item,
        items: impl IntoIterator<Item = Item>,
    ) -> ScoreRequest {
        ScoreRequest {
            client: self.clone(),
            model: model.into(),
            query,
            items: items.into_iter().collect(),
            instruction: None,
            runtime_options: None,
            options: self.request_options(),
        }
    }

    /// Extract structure from one or more items.
    pub fn extract(
        &self,
        model: impl Into<String>,
        items: impl IntoIterator<Item = Item>,
    ) -> ExtractRequest {
        ExtractRequest {
            client: self.clone(),
            model: model.into(),
            items: items.into_iter().collect(),
            labels: None,
            output_schema: None,
            instruction: None,
            runtime_options: None,
            options: self.request_options(),
        }
    }

    /// Send a msgpack body and decode the response envelope.
    async fn send_msgpack(
        &self,
        path: &str,
        body: MsgValue,
        policy: RetryPolicy,
        model: &str,
        options: &RequestOptions,
    ) -> Result<(MsgValue, HttpResponse, u32)> {
        let routing = self.routing(options.gpu.as_deref());
        let encoded = rmp_serde::to_vec(&body)
            .map_err(|err| Error::invalid(format!("could not encode the request body: {err}")))?;

        let request = self
            .request(Method::POST, path)?
            .msgpack_headers()
            .maybe_header(headers::MACHINE_PROFILE, routing.profile.as_deref())
            .maybe_header(headers::POOL, routing.pool.as_deref())
            .body(encoded);

        let mut state = RetryState::new(policy, options, Some(model));
        let response = self.send(request, &mut state).await?;
        let decoded: MsgValue = rmp_serde::from_slice(&response.body)
            .map_err(|err| Error::decode(format!("malformed msgpack response: {err}")))?;
        let retries = state.retries();
        Ok((decoded, response, retries))
    }
}

/// Assemble the `params` sub-map, omitting it entirely when nothing was set.
fn params_map(entries: Vec<(&str, MsgValue)>) -> Option<MsgValue> {
    if entries.is_empty() {
        return None;
    }
    Some(MsgValue::Map(
        entries
            .into_iter()
            .map(|(key, value)| (MsgValue::from(key), value))
            .collect(),
    ))
}

fn items_to_msgpack(items: &[Item]) -> Result<MsgValue> {
    Ok(MsgValue::Array(
        items
            .iter()
            .map(Item::to_msgpack)
            .collect::<Result<Vec<_>>>()?,
    ))
}

/// The response envelope's own `usage` block, for the settled charge.
fn body_usage(body: &MsgValue) -> Option<Value> {
    Some(msg::to_json(&MsgValue::Map(vec![(
        MsgValue::from("usage"),
        msg::get(body, "usage")?.clone(),
    )])))
}

fn attach_metadata(
    response: &HttpResponse,
    body: &MsgValue,
    retries: u32,
) -> Option<RequestMetadata> {
    metadata::parse(&response.headers, body_usage(body).as_ref(), retries)
}

fn timing(body: &MsgValue) -> Option<TimingInfo> {
    let timing = msg::get(body, "timing")?;
    Some(TimingInfo {
        total_ms: msg::get_f64(timing, "total_ms"),
        queue_ms: msg::get_f64(timing, "queue_ms"),
        tokenization_ms: msg::get_f64(timing, "tokenization_ms"),
        inference_ms: msg::get_f64(timing, "inference_ms"),
    })
}

/// Embeds one or more items. Build with [`Client::encode`].
pub struct EncodeRequest {
    client: Client,
    model: String,
    items: Vec<Item>,
    output_types: Option<Vec<OutputType>>,
    instruction: Option<String>,
    output_dtype: Option<OutputDType>,
    is_query: Option<bool>,
    runtime_options: Option<Value>,
    options: RequestOptions,
}

impl EncodeRequest {
    request_options!();

    /// Which representations to return. Defaults to whatever the model produces.
    pub fn output_types(mut self, types: impl IntoIterator<Item = OutputType>) -> Self {
        self.output_types = Some(types.into_iter().collect());
        self
    }

    /// Task instruction prepended by instruction-tuned models.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    /// Quantize the returned embeddings.
    pub fn output_dtype(mut self, dtype: OutputDType) -> Self {
        self.output_dtype = Some(dtype);
        self
    }

    /// Encode as a query rather than a document, for asymmetric models.
    pub fn is_query(mut self, is_query: bool) -> Self {
        self.is_query = Some(is_query);
        self
    }

    /// Adapter-specific runtime options, merged over the client defaults.
    pub fn options(mut self, options: Value) -> Self {
        self.runtime_options = Some(options);
        self
    }

    fn body(&self) -> Result<MsgValue> {
        let mut options = self.client.merge_options(self.runtime_options.as_ref());
        // `is_query` rides inside `options`, not alongside it.
        if let Some(is_query) = self.is_query {
            let mut map = options
                .take()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            map.insert("is_query".to_string(), Value::Bool(is_query));
            options = Some(Value::Object(map));
        }

        let mut params = Vec::new();
        if let Some(types) = &self.output_types {
            params.push((
                "output_types",
                MsgValue::Array(
                    types
                        .iter()
                        .map(|kind| {
                            MsgValue::from(
                                serde_json::to_value(kind)
                                    .unwrap_or_default()
                                    .as_str()
                                    .unwrap_or(""),
                            )
                        })
                        .collect(),
                ),
            ));
        }
        if let Some(instruction) = &self.instruction {
            params.push(("instruction", MsgValue::from(instruction.as_str())));
        }
        if let Some(dtype) = self.output_dtype {
            params.push((
                "output_dtype",
                MsgValue::from(
                    serde_json::to_value(dtype)
                        .unwrap_or_default()
                        .as_str()
                        .unwrap_or(""),
                ),
            ));
        }
        if let Some(options) = &options {
            params.push(("options", json_to_msgpack(options)));
        }

        let mut fields = vec![(MsgValue::from("items"), items_to_msgpack(&self.items)?)];
        if let Some(params) = params_map(params) {
            fields.push((MsgValue::from("params"), params));
        }
        Ok(MsgValue::Map(fields))
    }

    /// Send the request.
    pub async fn send(self) -> Result<Vec<EncodeResult>> {
        if self.items.is_empty() {
            return Err(Error::invalid("encode requires at least one item"));
        }
        let expected = self.items.len();
        let body = self.body()?;
        let (decoded, response, retries) = self
            .client
            .send_msgpack(
                &format!("/v1/encode/{}", self.model),
                body,
                RetryPolicy::ENCODE,
                &self.model,
                &self.options,
            )
            .await?;

        let items = msg::get_array(&decoded, "items")
            .ok_or_else(|| Error::decode("encode response is missing its `items` array"))?;
        // The 1:1 input/output contract is load-bearing: callers reassemble batches
        // positionally, so a desynced response must never reach them.
        if items.len() != expected {
            return Err(Error::Server {
                message: format!(
                    "Server returned {} results for {expected} items (model '{}')",
                    items.len(),
                    self.model
                ),
                code: Some(codes::ENCODE_RESULT_COUNT_MISMATCH.to_string()),
                status: response.status,
                request: attach_metadata(&response, &decoded, retries).map(Box::new),
            });
        }

        let model = msg::get_string(&decoded, "model");
        let timing = timing(&decoded);
        let request = attach_metadata(&response, &decoded, retries);

        items
            .iter()
            .map(|item| {
                Ok(EncodeResult {
                    model: model.clone(),
                    id: msg::get_string(item, "id"),
                    dense: parse_dense(item)?,
                    sparse: parse_sparse(item)?,
                    multivector: parse_multivector(item)?,
                    timing,
                    request: request.clone(),
                })
            })
            .collect()
    }

    /// Send the request and return the single expected result.
    pub async fn send_one(self) -> Result<EncodeResult> {
        let mut results = self.send().await?;
        match results.len() {
            1 => Ok(results.remove(0)),
            other => Err(Error::invalid(format!(
                "send_one() expects exactly one item, but the request carried {other}"
            ))),
        }
    }
}

/// The wire nests each tensor under a `values` key; the SDK type does not.
fn tensor(item: &MsgValue, key: &str) -> Result<Option<ndarray::RawArray>> {
    let Some(node) = msg::get(item, key) else {
        return Ok(None);
    };
    let values = msg::get(node, "values").ok_or_else(|| {
        Error::decode(format!(
            "encode result `{key}` is missing its `values` array"
        ))
    })?;
    Ok(Some(ndarray::decode(values)?))
}

fn parse_dense(item: &MsgValue) -> Result<Option<Vec<f32>>> {
    Ok(tensor(item, "dense")?.map(|array| array.to_f32()))
}

fn parse_multivector(item: &MsgValue) -> Result<Option<Multivector>> {
    let Some(array) = tensor(item, "multivector")? else {
        return Ok(None);
    };
    Ok(Some(match array.element {
        ndarray::Element::F16 => Multivector::F16(array.rows_f16()?),
        _ => Multivector::F32(array.rows_f32()?),
    }))
}

fn parse_sparse(item: &MsgValue) -> Result<Option<SparseVector>> {
    let Some(node) = msg::get(item, "sparse") else {
        return Ok(None);
    };
    let indices = msg::get(node, "indices")
        .ok_or_else(|| Error::decode("encode result `sparse` is missing its `indices` array"))?;
    let values = msg::get(node, "values")
        .ok_or_else(|| Error::decode("encode result `sparse` is missing its `values` array"))?;
    Ok(Some(SparseVector {
        indices: ndarray::decode(indices)?.to_u32()?,
        values: ndarray::decode(values)?.to_f32(),
    }))
}

/// Scores a query against candidates. Build with [`Client::score`].
pub struct ScoreRequest {
    client: Client,
    model: String,
    query: Item,
    items: Vec<Item>,
    instruction: Option<String>,
    runtime_options: Option<Value>,
    options: RequestOptions,
}

impl ScoreRequest {
    request_options!();

    /// Task instruction prepended by instruction-tuned rerankers.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    /// Adapter-specific runtime options, merged over the client defaults.
    pub fn options(mut self, options: Value) -> Self {
        self.runtime_options = Some(options);
        self
    }

    /// Send the request.
    pub async fn send(self) -> Result<ScoreResult> {
        if self.items.is_empty() {
            return Err(Error::invalid("score requires at least one candidate item"));
        }

        // Score fields sit at the top level of the body, not inside `params`.
        let mut fields = vec![
            (MsgValue::from("query"), self.query.to_msgpack()?),
            (MsgValue::from("items"), items_to_msgpack(&self.items)?),
        ];
        if let Some(instruction) = &self.instruction {
            fields.push((
                MsgValue::from("instruction"),
                MsgValue::from(instruction.as_str()),
            ));
        }
        if let Some(options) = self.client.merge_options(self.runtime_options.as_ref()) {
            fields.push((MsgValue::from("options"), json_to_msgpack(&options)));
        }

        let (decoded, response, retries) = self
            .client
            .send_msgpack(
                &format!("/v1/score/{}", self.model),
                MsgValue::Map(fields),
                RetryPolicy::INFERENCE,
                &self.model,
                &self.options,
            )
            .await?;

        let model = msg::get_string(&decoded, "model")
            .ok_or_else(|| Error::decode("score response is missing its `model` field"))?;
        let entries = msg::get_array(&decoded, "scores")
            .ok_or_else(|| Error::decode("score response is missing its `scores` array"))?;

        let scores = entries
            .iter()
            .map(|entry| {
                Ok(ScoreEntry {
                    item_id: msg::get_string(entry, "item_id")
                        .ok_or_else(|| Error::decode("score entry is missing its `item_id`"))?,
                    score: msg::get_f64(entry, "score")
                        .ok_or_else(|| Error::decode("score entry is missing its `score`"))?,
                    rank: msg::get_u64(entry, "rank").unwrap_or(0) as u32,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ScoreResult {
            model,
            query_id: msg::get_string(&decoded, "query_id"),
            scores,
            usage: msg::get(&decoded, "usage").map(|usage| ScoreUsage {
                input_tokens: msg::get_u64(usage, "input_tokens").unwrap_or(0),
                images: msg::get_u64(usage, "images"),
            }),
            request: attach_metadata(&response, &decoded, retries),
        })
    }
}

/// Extracts structure from items. Build with [`Client::extract`].
pub struct ExtractRequest {
    client: Client,
    model: String,
    items: Vec<Item>,
    labels: Option<Vec<String>>,
    output_schema: Option<Value>,
    instruction: Option<String>,
    runtime_options: Option<Value>,
    options: RequestOptions,
}

impl ExtractRequest {
    request_options!();

    /// Candidate labels, for zero-shot classifiers and detectors.
    pub fn labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.labels = Some(labels.into_iter().map(Into::into).collect());
        self
    }

    /// JSON schema the model should populate.
    pub fn output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Task instruction.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    /// Adapter-specific runtime options, merged over the client defaults.
    pub fn options(mut self, options: Value) -> Self {
        self.runtime_options = Some(options);
        self
    }

    /// Send the request.
    pub async fn send(self) -> Result<Vec<ExtractResult>> {
        if self.items.is_empty() {
            return Err(Error::invalid("extract requires at least one item"));
        }

        let mut params = Vec::new();
        if let Some(labels) = &self.labels {
            params.push((
                "labels",
                MsgValue::Array(
                    labels
                        .iter()
                        .map(|label| MsgValue::from(label.as_str()))
                        .collect(),
                ),
            ));
        }
        if let Some(schema) = &self.output_schema {
            params.push(("output_schema", json_to_msgpack(schema)));
        }
        if let Some(instruction) = &self.instruction {
            params.push(("instruction", MsgValue::from(instruction.as_str())));
        }
        if let Some(options) = self.client.merge_options(self.runtime_options.as_ref()) {
            params.push(("options", json_to_msgpack(&options)));
        }

        let mut fields = vec![(MsgValue::from("items"), items_to_msgpack(&self.items)?)];
        if let Some(params) = params_map(params) {
            fields.push((MsgValue::from("params"), params));
        }

        let (decoded, response, retries) = self
            .client
            .send_msgpack(
                &format!("/v1/extract/{}", self.model),
                MsgValue::Map(fields),
                RetryPolicy::INFERENCE,
                &self.model,
                &self.options,
            )
            .await?;

        let items = msg::get_array(&decoded, "items")
            .ok_or_else(|| Error::decode("extract response is missing its `items` array"))?;
        let model = msg::get_string(&decoded, "model");
        let request = attach_metadata(&response, &decoded, retries);

        Ok(items
            .iter()
            .map(|item| ExtractResult {
                model: model.clone(),
                id: msg::get_string(item, "id"),
                entities: parse_list(item, "entities", parse_entity),
                relations: parse_list(item, "relations", parse_relation),
                classifications: parse_list(item, "classifications", parse_classification),
                objects: parse_list(item, "objects", parse_object),
                data: msg::get(item, "data")
                    .map(msg::to_json)
                    .filter(|data| !is_blank(data)),
                error: msg::get(item, "error").map(parse_extract_error),
                request: request.clone(),
            })
            .collect())
    }

    /// Send the request and return the single expected result.
    pub async fn send_one(self) -> Result<ExtractResult> {
        let mut results = self.send().await?;
        match results.len() {
            1 => Ok(results.remove(0)),
            other => Err(Error::invalid(format!(
                "send_one() expects exactly one item, but the response carried {other}"
            ))),
        }
    }
}

fn is_blank(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Object(map) => map.is_empty(),
        Value::Array(items) => items.is_empty(),
        _ => false,
    }
}

fn parse_list<T>(item: &MsgValue, key: &str, parse: fn(&MsgValue) -> T) -> Vec<T> {
    msg::get_array(item, key)
        .map(|entries| entries.iter().map(parse).collect())
        .unwrap_or_default()
}

fn parse_entity(value: &MsgValue) -> Entity {
    Entity {
        text: msg::get_string(value, "text").unwrap_or_default(),
        label: msg::get_string(value, "label").unwrap_or_default(),
        score: msg::get_f64(value, "score").unwrap_or_default(),
        start: msg::get_i64(value, "start"),
        end: msg::get_i64(value, "end"),
        bbox: parse_bbox(value),
    }
}

fn parse_relation(value: &MsgValue) -> Relation {
    Relation {
        head: msg::get_string(value, "head").unwrap_or_default(),
        tail: msg::get_string(value, "tail").unwrap_or_default(),
        relation: msg::get_string(value, "relation").unwrap_or_default(),
        score: msg::get_f64(value, "score").unwrap_or_default(),
    }
}

fn parse_classification(value: &MsgValue) -> Classification {
    Classification {
        label: msg::get_string(value, "label").unwrap_or_default(),
        score: msg::get_f64(value, "score").unwrap_or_default(),
    }
}

fn parse_object(value: &MsgValue) -> DetectedObject {
    DetectedObject {
        label: msg::get_string(value, "label").unwrap_or_default(),
        score: msg::get_f64(value, "score").unwrap_or_default(),
        bbox: parse_bbox(value).unwrap_or_default(),
    }
}

fn parse_bbox(value: &MsgValue) -> Option<Vec<i64>> {
    Some(
        msg::get_array(value, "bbox")?
            .iter()
            .filter_map(|dim| dim.as_i64().or_else(|| dim.as_f64().map(|v| v as i64)))
            .collect(),
    )
}

/// A per-item error is preserved only when both halves are present and non-blank;
/// anything else is normalized so callers never see a half-formed failure.
fn parse_extract_error(value: &MsgValue) -> ExtractItemError {
    let code = msg::get_text(value, "code")
        .map(str::trim)
        .unwrap_or_default();
    let message = msg::get_text(value, "message")
        .map(str::trim)
        .unwrap_or_default();
    if code.is_empty() || message.is_empty() {
        return ExtractItemError {
            code: codes::INTERNAL_ERROR.to_string(),
            message: MALFORMED_EXTRACT_ERROR_MESSAGE.to_string(),
        };
    }
    ExtractItemError {
        code: code.to_string(),
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::wire::ndarray::fixtures;

    fn map(entries: Vec<(&str, MsgValue)>) -> MsgValue {
        MsgValue::Map(
            entries
                .into_iter()
                .map(|(key, value)| (MsgValue::from(key), value))
                .collect(),
        )
    }

    #[test]
    fn unwraps_the_nested_tensor_envelopes() {
        let item = map(vec![
            ("id", MsgValue::from("doc-1")),
            (
                "dense",
                map(vec![(
                    "values",
                    fixtures::f32_array(&[3], &[0.1, 0.2, 0.3]),
                )]),
            ),
            (
                "sparse",
                map(vec![
                    ("indices", fixtures::i32_array(&[2], &[7, 42])),
                    ("values", fixtures::f32_array(&[2], &[0.5, 0.25])),
                ]),
            ),
            (
                "multivector",
                map(vec![(
                    "values",
                    fixtures::f16_array(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                )]),
            ),
        ]);

        assert_eq!(parse_dense(&item).unwrap().unwrap(), vec![0.1, 0.2, 0.3]);
        let sparse = parse_sparse(&item).unwrap().unwrap();
        assert_eq!(sparse.indices, vec![7, 42]);
        assert_eq!(sparse.values, vec![0.5, 0.25]);
        let mv = parse_multivector(&item).unwrap().unwrap();
        assert!(matches!(mv, Multivector::F16(_)));
        assert_eq!(mv.to_f32(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn absent_tensors_stay_absent() {
        let item = map(vec![("id", MsgValue::from("doc-1"))]);
        assert!(parse_dense(&item).unwrap().is_none());
        assert!(parse_sparse(&item).unwrap().is_none());
        assert!(parse_multivector(&item).unwrap().is_none());
    }

    #[test]
    fn a_tensor_without_its_values_key_is_a_decode_error() {
        let item = map(vec![("dense", map(vec![("dims", MsgValue::from(3u64))]))]);
        let err = parse_dense(&item).unwrap_err();
        assert!(err.to_string().contains("values"), "{err}");
    }

    #[test]
    fn well_formed_extract_errors_pass_through() {
        let error = parse_extract_error(&map(vec![
            ("code", MsgValue::from("INFERENCE_ERROR")),
            ("message", MsgValue::from("Document export failed")),
        ]));
        assert_eq!(error.code, "INFERENCE_ERROR");
        assert_eq!(error.message, "Document export failed");
    }

    #[test]
    fn malformed_extract_errors_are_normalized() {
        let cases = vec![
            map(vec![("code", MsgValue::from("INFERENCE_ERROR"))]),
            map(vec![("message", MsgValue::from("only a message"))]),
            map(vec![
                ("code", MsgValue::from(" ")),
                ("message", MsgValue::from("\t")),
            ]),
            map(vec![]),
        ];
        for case in cases {
            let error = parse_extract_error(&case);
            assert_eq!(error.code, codes::INTERNAL_ERROR);
            assert_eq!(error.message, MALFORMED_EXTRACT_ERROR_MESSAGE);
        }
    }

    #[test]
    fn extract_lists_default_to_empty_and_data_drops_when_blank() {
        let item = map(vec![("data", MsgValue::Map(Vec::new()))]);
        assert!(parse_list(&item, "entities", parse_entity).is_empty());
        assert!(
            msg::get(&item, "data")
                .map(msg::to_json)
                .as_ref()
                .is_none_or(is_blank)
        );
    }

    #[test]
    fn entity_and_object_shapes_decode() {
        let entity = parse_entity(&map(vec![
            ("text", MsgValue::from("Ada")),
            ("label", MsgValue::from("PERSON")),
            ("score", MsgValue::F64(0.91)),
            ("start", MsgValue::from(0i64)),
            ("end", MsgValue::from(3i64)),
        ]));
        assert_eq!(entity.text, "Ada");
        assert_eq!(entity.start, Some(0));
        assert!(entity.bbox.is_none());

        let object = parse_object(&map(vec![
            ("label", MsgValue::from("cat")),
            ("score", MsgValue::F64(0.7)),
            (
                "bbox",
                MsgValue::Array(vec![
                    MsgValue::from(1i64),
                    MsgValue::from(2i64),
                    MsgValue::from(30i64),
                    MsgValue::from(40i64),
                ]),
            ),
        ]));
        assert_eq!(object.bbox, vec![1, 2, 30, 40]);
    }

    #[tokio::test]
    async fn empty_input_is_rejected_before_any_request() {
        let client = Client::new("https://sie.invalid").unwrap();
        assert!(client.encode("m", Vec::new()).send().await.is_err());
        assert!(client.extract("m", Vec::new()).send().await.is_err());
        assert!(
            client
                .score("m", Item::text("q"), Vec::new())
                .send()
                .await
                .is_err()
        );
    }

    #[test]
    fn encode_body_nests_params_and_folds_is_query_into_options() {
        let client = Client::new("https://sie.invalid").unwrap();
        let body = client
            .encode("m", [Item::text("hello")])
            .output_types([OutputType::Dense, OutputType::Sparse])
            .instruction("query:")
            .output_dtype(OutputDType::Int8)
            .is_query(true)
            .body()
            .unwrap();

        let params = msg::get(&body, "params").unwrap();
        assert_eq!(msg::get_text(params, "instruction"), Some("query:"));
        assert_eq!(msg::get_text(params, "output_dtype"), Some("int8"));
        let types = msg::get_array(params, "output_types").unwrap();
        assert_eq!(types[0].as_str(), Some("dense"));
        assert_eq!(types[1].as_str(), Some("sparse"));
        let options = msg::get(params, "options").unwrap();
        assert_eq!(
            msg::get(options, "is_query").unwrap(),
            &MsgValue::Boolean(true)
        );
        assert!(msg::get_array(&body, "items").is_some());
    }

    #[test]
    fn encode_body_omits_params_when_nothing_was_set() {
        let client = Client::new("https://sie.invalid").unwrap();
        let body = client.encode("m", [Item::text("hello")]).body().unwrap();
        assert!(msg::get(&body, "params").is_none());
    }
}
