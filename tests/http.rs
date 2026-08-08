//! End-to-end tests over a mock HTTP server.
//!
//! These cover what unit tests cannot: the request the SDK actually puts on the wire, and
//! the retry loop driving real responses.

use std::time::Duration;

use rmpv::Value as MsgValue;
use serde_json::json;
use sie_sdk::{Client, Error, Item, OutputType};
use wiremock::matchers::{body_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// A client with the retry budget tightened so a failing test finishes in milliseconds.
async fn client(server: &MockServer) -> Client {
    Client::builder(server.uri())
        .api_key("test-key")
        .timeout(Duration::from_secs(2))
        .provision_timeout(Duration::from_secs(2))
        .build()
        .unwrap()
}

/// Build the msgpack-numpy encoding of an `f32` array, as the server sends one.
fn numpy_f32(shape: &[usize], values: &[f32]) -> MsgValue {
    MsgValue::Map(vec![
        (MsgValue::Binary(b"nd".to_vec()), MsgValue::Boolean(true)),
        (
            MsgValue::Binary(b"type".to_vec()),
            MsgValue::String("<f4".into()),
        ),
        (
            MsgValue::Binary(b"shape".to_vec()),
            MsgValue::Array(
                shape
                    .iter()
                    .map(|dim| MsgValue::from(*dim as u64))
                    .collect(),
            ),
        ),
        (
            MsgValue::Binary(b"data".to_vec()),
            MsgValue::Binary(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ),
    ])
}

fn encode_response(vectors: &[&[f32]]) -> Vec<u8> {
    let items = vectors
        .iter()
        .enumerate()
        .map(|(index, values)| {
            MsgValue::Map(vec![
                (MsgValue::from("id"), MsgValue::from(index.to_string())),
                (
                    MsgValue::from("dense"),
                    MsgValue::Map(vec![(
                        MsgValue::from("values"),
                        numpy_f32(&[values.len()], values),
                    )]),
                ),
            ])
        })
        .collect();
    let body = MsgValue::Map(vec![
        (MsgValue::from("model"), MsgValue::from("BAAI/bge-m3")),
        (MsgValue::from("items"), MsgValue::Array(items)),
    ]);
    rmp_serde::to_vec(&body).unwrap()
}

fn msgpack_response(body: Vec<u8>) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "application/msgpack")
}

fn capacity_response(code: &str) -> ResponseTemplate {
    ResponseTemplate::new(503)
        .insert_header("retry-after", "0")
        .set_body_json(json!({"error": {"code": code, "message": "not yet"}}))
}

#[tokio::test]
async fn encode_sends_msgpack_and_decodes_numpy_tensors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/BAAI/bge-m3"))
        .and(header("content-type", "application/msgpack"))
        .and(header("accept", "application/msgpack"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(msgpack_response(encode_response(&[&[0.25, -0.5]])))
        .mount(&server)
        .await;

    let result = client(&server)
        .await
        .encode("BAAI/bge-m3", [Item::text("hello")])
        .output_types([OutputType::Dense])
        .send_one()
        .await
        .unwrap();

    assert_eq!(result.model.as_deref(), Some("BAAI/bge-m3"));
    assert_eq!(result.require_dense().unwrap(), &[0.25, -0.5]);
}

#[tokio::test]
async fn routing_headers_carry_the_pool_and_profile() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/m"))
        .and(header("x-sie-machine-profile", "l4"))
        .and(header("x-sie-pool", "prod"))
        .respond_with(msgpack_response(encode_response(&[&[1.0]])))
        .mount(&server)
        .await;

    client(&server)
        .await
        .encode("m", [Item::text("hi")])
        .gpu("prod/l4")
        .send_one()
        .await
        .unwrap();
}

#[tokio::test]
async fn a_result_count_mismatch_is_refused() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(msgpack_response(encode_response(&[&[1.0]])))
        .mount(&server)
        .await;

    let err = client(&server)
        .await
        .encode("m", [Item::text("a"), Item::text("b")])
        .send()
        .await
        .unwrap_err();
    assert_eq!(err.code(), Some("ENCODE_RESULT_COUNT_MISMATCH"));
}

#[tokio::test]
async fn capacity_responses_are_retried_until_the_server_is_ready() {
    // A `Retry-After: 0` is honoured verbatim on the first OOM attempt but only floors the
    // exponential schedule afterwards, so that branch is exercised with a single failure.
    for (code, failures) in [
        ("PROVISIONING", 2),
        ("MODEL_LOADING", 2),
        ("LORA_LOADING", 2),
        ("RESOURCE_EXHAUSTED", 1),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(capacity_response(code))
            .up_to_n_times(failures)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(msgpack_response(encode_response(&[&[1.0]])))
            .mount(&server)
            .await;

        let result = client(&server)
            .await
            .encode("m", [Item::text("hi")])
            .send_one()
            .await
            .unwrap_or_else(|err| panic!("{code} should have been retried: {err}"));
        assert_eq!(result.request.unwrap().retries, failures as u32, "{code}");
    }
}

#[tokio::test]
async fn fail_fast_surfaces_the_capacity_error_instead_of_waiting() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(capacity_response("PROVISIONING"))
        .mount(&server)
        .await;

    let err = client(&server)
        .await
        .encode("m", [Item::text("hi")])
        .wait_for_capacity(false)
        .send_one()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Provisioning { .. }), "{err:?}");
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_model_load_failure_is_terminal() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(502).set_body_json(json!({
            "error": {"code": "MODEL_LOAD_FAILED", "message": "gated repo",
                      "error_class": "GATED", "permanent": true, "attempts": 2}
        })))
        .mount(&server)
        .await;

    let err = client(&server)
        .await
        .encode("m", [Item::text("hi")])
        .send_one()
        .await
        .unwrap_err();
    match err {
        Error::ModelLoadFailed {
            error_class,
            permanent,
            attempts,
            ..
        } => {
            assert_eq!(error_class, sie_sdk::ModelLoadErrorClass::Gated);
            assert!(permanent);
            assert_eq!(attempts, 2);
        }
        other => panic!("unexpected: {other:?}"),
    }
    // Terminal means one attempt, not one plus the retry budget.
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn generation_never_replays_a_gateway_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/generate/org__model"))
        .respond_with(
            ResponseTemplate::new(504).set_body_json(json!({"error": {"message": "timed out"}})),
        )
        .mount(&server)
        .await;

    let err = client(&server)
        .await
        .generate("org/model", "hi", 16)
        .send()
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(504));
    assert!(err.to_string().contains("double-bill"), "{err}");
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn request_metadata_is_read_off_the_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            msgpack_response(encode_response(&[&[1.0]]))
                .insert_header("x-sie-request-id", "req_123")
                .insert_header("x-sie-units-input-tokens", "42")
                .insert_header("x-sie-credits-debited", "7")
                .insert_header("x-sie-model-revision", "rev-9"),
        )
        .mount(&server)
        .await;

    let result = client(&server)
        .await
        .encode("m", [Item::text("hi")])
        .send_one()
        .await
        .unwrap();
    let metadata = result.request.unwrap();
    assert_eq!(metadata.id.as_deref(), Some("req_123"));
    assert_eq!(metadata.usage.unwrap().input_tokens, Some(42));
    assert_eq!(metadata.credits_debited, Some(7));
    assert_eq!(metadata.model_revision.as_deref(), Some("rev-9"));
}

#[tokio::test]
async fn chat_sends_json_and_returns_the_completion() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-1", "object": "chat.completion", "created": 1, "model": "qwen3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server)
        .await;

    let completion = client(&server)
        .await
        .chat("qwen3", [sie_sdk::types::ChatMessage::user("hi")])
        .temperature(0.2)
        .send()
        .await
        .unwrap();
    assert_eq!(completion.text(), Some("hello"));
}

#[tokio::test]
async fn chat_streams_chunks_and_stops_at_the_done_sentinel() {
    use futures_util::StreamExt;

    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"he\"}}]}\n\n",
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\"}}]}\n\n",
        "data: [DONE]\n\n",
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ignored\"}}]}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("accept", "text/event-stream"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let stream = client(&server)
        .await
        .chat("m", [sie_sdk::types::ChatMessage::user("hi")])
        .stream()
        .unwrap();
    let deltas: Vec<String> = stream
        .map(|chunk| chunk.unwrap().delta().unwrap_or_default().to_string())
        .collect()
        .await;
    assert_eq!(deltas, vec!["he", "llo"]);
}

#[tokio::test]
async fn a_capacity_error_as_the_first_stream_event_reconnects() {
    use futures_util::StreamExt;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("retry-after", "0")
                .set_body_raw(
                    "data: {\"error\":{\"code\":\"MODEL_LOADING\",\"message\":\"loading\"}}\n\n",
                    "text/event-stream",
                ),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\ndata: [DONE]\n",
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    let stream = client(&server)
        .await
        .chat("m", [sie_sdk::types::ChatMessage::user("hi")])
        .stream()
        .unwrap();
    let chunks: Vec<_> = stream.collect().await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].as_ref().unwrap().delta(), Some("ok"));
}

#[tokio::test]
async fn files_upload_raw_bytes_with_metadata_in_the_query() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .and(query_param("purpose", "batch"))
        .and(query_param("filename", "in.jsonl"))
        .and(header("content-type", "application/jsonl"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file_1", "object": "file", "bytes": 5, "created_at": 1,
            "filename": "in.jsonl", "purpose": "batch", "status": "processed"
        })))
        .mount(&server)
        .await;

    let file = client(&server)
        .await
        .files()
        .upload(b"{}\n{}".to_vec())
        .filename("in.jsonl")
        .send()
        .await
        .unwrap();
    assert_eq!(file.id, "file_1");

    let request: &Request = &server.received_requests().await.unwrap()[0];
    assert_eq!(request.body, b"{}\n{}");
}

#[tokio::test]
async fn a_connector_job_carries_its_idempotency_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .and(header("idempotency-key", "run-1"))
        .and(body_json(json!({
            "operation": "encode", "model": "m",
            "src": "postgres://warehouse?query=SELECT+1",
            "connection": "warehouse", "execution": "run"
        })))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "id": "job_1", "object": "job", "operation": "encode", "model": "m", "state": "queued"
        })))
        .mount(&server)
        .await;

    let job = client(&server)
        .await
        .jobs()
        .submit(
            sie_sdk::client::JobSource::connector("postgres://warehouse?query=SELECT+1"),
            "m",
        )
        .execution(sie_sdk::types::JobExecution::Run)
        .idempotency_key("run-1")
        .send()
        .await
        .unwrap();
    assert_eq!(job.id, "job_1");
}

#[tokio::test]
async fn capacity_filters_workers_by_gpu_type() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "healthy", "type": "gateway",
            "cluster": {"worker_count": 3, "gpu_count": 3, "models_loaded": 1, "total_qps": 0.0},
            "workers": [{"name": "a", "gpu": "l4"}, {"name": "b", "gpu": "a100-80gb"}],
            "configured_gpu_types": ["l4"], "live_gpu_types": ["l4"]
        })))
        .mount(&server)
        .await;

    let client = client(&server).await;
    assert_eq!(client.get_capacity(None).await.unwrap().worker_count, 3);
    let filtered = client.get_capacity(Some("L4")).await.unwrap();
    assert_eq!(filtered.worker_count, 1);
    assert_eq!(filtered.workers[0].name, "a");
}

#[tokio::test]
async fn capacity_refuses_a_worker_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"status": "healthy", "type": "worker"})),
        )
        .mount(&server)
        .await;

    let err = client(&server).await.get_capacity(None).await.unwrap_err();
    assert_eq!(err.code(), Some("not_gateway"));
}

#[tokio::test]
async fn metadata_endpoints_do_not_sit_inside_the_provisioning_budget() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(capacity_response("PROVISIONING"))
        .mount(&server)
        .await;

    let err = client(&server).await.list_models().await.unwrap_err();
    assert_eq!(err.status(), Some(503));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn edge_headers_never_leave_the_base_origin() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": []})))
        .mount(&server)
        .await;

    // The mock server is plain HTTP, so edge headers are refused at construction.
    let err = Client::builder(server.uri())
        .base_url_headers(std::collections::HashMap::from([(
            "Modal-Key".to_string(),
            "k".to_string(),
        )]))
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("https base_url"), "{err}");
}
