//! Server-sent event streaming for the generation endpoints.
//!
//! Reconnect semantics mirror the buffered path with one addition: an error that arrives as
//! the *first* SSE event is treated exactly like an HTTP 503, because nothing has been
//! delivered yet and replaying is safe. Once a chunk has been yielded, any error is
//! terminal: the caller has already consumed part of a generation that cannot be replayed.

use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::Method;
use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::client::generate::{ChatRequest, GenerateRequest};
use crate::client::{Client, StreamAttempt};
use crate::error::{Error, Result, TransportErrorKind};
use crate::http::{HttpResponse, PreparedRequest, headers};
use crate::retry::{Decision, RequestOptions, RetryPolicy, RetryState};
use crate::types::{ChatCompletionChunk, GenerateChunk};
use crate::wire::sse::{self, LineDecoder};

/// A stream of response chunks.
///
/// Boxed and pinned so it can be polled directly with [`futures_util::StreamExt`], without
/// the caller having to pin it first.
pub type ChunkStream<T> = Pin<Box<dyn Stream<Item = Result<T>> + Send>>;

impl GenerateRequest {
    /// Stream the generation token by token.
    pub fn stream(self) -> Result<ChunkStream<GenerateChunk>> {
        let body = self.body(true)?;
        let request = self
            .client
            .sse_request(&self.path(), &body, &self.options)?;
        Ok(Box::pin(sse_stream(
            self.client,
            request,
            RetryPolicy::STREAM,
            self.model,
            self.options,
        )))
    }
}

impl ChatRequest {
    /// Stream the completion token by token.
    pub fn stream(self) -> Result<ChunkStream<ChatCompletionChunk>> {
        let body = self.body(true)?;
        let request = self
            .client
            .sse_request("/v1/chat/completions", &body, &self.options)?;
        Ok(Box::pin(sse_stream(
            self.client,
            request,
            RetryPolicy::STREAM,
            self.model,
            self.options,
        )))
    }
}

impl Client {
    fn sse_request(
        &self,
        path: &str,
        body: &Value,
        options: &RequestOptions,
    ) -> Result<PreparedRequest> {
        let routing = self.routing(options.gpu.as_deref());
        let encoded = serde_json::to_vec(body)
            .map_err(|err| Error::invalid(format!("could not encode the request body: {err}")))?;
        Ok(self
            .request(Method::POST, path)?
            .sse_headers()
            .maybe_header(headers::MACHINE_PROFILE, routing.profile.as_deref())
            .maybe_header(headers::POOL, routing.pool.as_deref())
            .body(encoded))
    }
}

/// A mid-stream capacity signal, rendered as the HTTP response it stands in for.
///
/// Reusing the response path means the stream's first-event errors go through exactly the
/// same retry rules as a 503 received before the stream opened. The opening response's
/// headers are carried over, so a `Retry-After` the gateway set on the connection is
/// honoured rather than discarded.
fn as_capacity_response(code: &str, message: &str, headers: &HeaderMap) -> HttpResponse {
    let mut response_headers = headers.clone();
    response_headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    HttpResponse {
        status: 503,
        headers: response_headers,
        body: Bytes::from(
            serde_json::json!({"error": {"code": code, "message": message}}).to_string(),
        ),
    }
}

/// What one decoded SSE event means for the stream.
#[derive(Debug)]
enum Event<T> {
    Chunk(T),
    /// The stream ended cleanly.
    Done,
    /// The server reported a failure before anything was delivered.
    Capacity {
        code: String,
        message: String,
    },
}

fn decode_event<T: DeserializeOwned>(payload: &str) -> Result<Event<T>> {
    if payload == sse::DONE {
        return Ok(Event::Done);
    }
    let value: Value = serde_json::from_str(payload)
        .map_err(|err| Error::decode(format!("Malformed SSE chunk from server: {err}")))?;
    if let Some((code, message)) = sse::chunk_error(&value) {
        return Ok(Event::Capacity { code, message });
    }
    serde_json::from_value(value)
        .map(Event::Chunk)
        .map_err(|err| Error::decode(format!("Unexpected SSE chunk shape: {err}")))
}

fn sse_stream<T: DeserializeOwned + Send>(
    client: Client,
    request: PreparedRequest,
    policy: RetryPolicy,
    model: String,
    options: RequestOptions,
) -> impl Stream<Item = Result<T>> {
    async_stream::try_stream! {
        let mut state = RetryState::new(policy, &options, Some(&model));
        let mut yielded = false;

        loop {
            let StreamAttempt { response, headers } = client.send_streaming(&request, &mut state).await?;
            let mut body = response.bytes_stream();
            let mut decoder = LineDecoder::default();
            let mut reconnect_after = None;

            'attempt: while let Some(chunk) = body.next().await {
                // A transport failure part-way through a generation is never replayed.
                let chunk = chunk.map_err(|error| {
                    Error::connection(
                        TransportErrorKind::MidFlight,
                        format!("Connection lost during stream: {error}"),
                        error,
                    )
                })?;

                for line in decoder.push(&chunk) {
                    let Some(payload) = sse::data_payload(&line) else { continue };
                    match decode_event::<T>(payload)? {
                        Event::Chunk(value) => {
                            yielded = true;
                            yield value;
                        }
                        Event::Done => return,
                        Event::Capacity { code, message } => {
                            let terminal = || Error::Server {
                                message: message.clone(),
                                code: Some(code.clone()),
                                status: 503,
                                request: None,
                            };
                            if yielded {
                                Err(terminal())?;
                            }
                            match state.on_response(&as_capacity_response(&code, &message, &headers))? {
                                Decision::Retry(delay) => {
                                    reconnect_after = Some(delay);
                                    break 'attempt;
                                }
                                // Unreachable: a 503 the policy does not retry raises above.
                                Decision::Accept => Err(terminal())?,
                            }
                        }
                    }
                }
            }

            // A stream that ended without a trailing newline still has a final event.
            if reconnect_after.is_none()
                && let Some(line) = decoder.finish()
                && let Some(payload) = sse::data_payload(&line)
            {
                match decode_event::<T>(payload)? {
                    Event::Chunk(value) => yield value,
                    Event::Done | Event::Capacity { .. } => return,
                }
            }

            match reconnect_after {
                // Dropping the response body here closes the connection, which the gateway
                // sees as a client disconnect and stops generating for.
                Some(delay) => tokio::time::sleep(delay).await,
                // The stream ended without a `[DONE]` sentinel, which is still a clean end.
                None => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[test]
    fn decodes_payloads_sentinels_and_errors() {
        let chunk: Event<ChatCompletionChunk> =
            decode_event(r#"{"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}"#)
                .unwrap();
        match chunk {
            Event::Chunk(chunk) => assert_eq!(chunk.delta(), Some("hi")),
            _ => panic!("expected a chunk"),
        }

        assert!(matches!(
            decode_event::<ChatCompletionChunk>("[DONE]").unwrap(),
            Event::Done
        ));

        let capacity: Event<ChatCompletionChunk> =
            decode_event(r#"{"error": {"code": "RESOURCE_EXHAUSTED", "message": "oom"}}"#).unwrap();
        match capacity {
            Event::Capacity { code, message } => {
                assert_eq!(code, "RESOURCE_EXHAUSTED");
                assert_eq!(message, "oom");
            }
            _ => panic!("expected a capacity signal"),
        }
    }

    #[test]
    fn malformed_json_is_a_decode_error() {
        let err = decode_event::<ChatCompletionChunk>("{not json").unwrap_err();
        assert!(err.to_string().contains("Malformed SSE chunk"), "{err}");
    }

    #[test]
    fn the_synthesized_capacity_response_keeps_the_connections_retry_hint() {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "3".parse().unwrap());
        let response = as_capacity_response("MODEL_LOADING", "loading", &headers);
        assert_eq!(
            crate::retry::backoff::retry_after(&response.headers),
            Some(std::time::Duration::from_secs(3))
        );
    }

    #[test]
    fn the_synthesized_capacity_response_round_trips_through_the_error_reader() {
        let response = as_capacity_response("MODEL_LOADING", "still loading", &HeaderMap::new());
        assert_eq!(response.status, 503);
        let envelope = crate::wire::parse_envelope(&response);
        assert_eq!(envelope.code.as_deref(), Some("MODEL_LOADING"));
        assert_eq!(envelope.message.as_deref(), Some("still loading"));
    }

    #[tokio::test]
    async fn a_stream_against_an_unreachable_server_fails_rather_than_hanging() {
        // Port 1 is reserved and never listening, so this exercises the connect path.
        let client = Client::builder("http://127.0.0.1:1")
            .timeout(std::time::Duration::from_millis(200))
            .wait_for_capacity(false)
            .build()
            .unwrap();
        let mut stream = client
            .chat("m", [ChatMessage::user("hi")])
            .stream()
            .unwrap();
        let first = stream
            .next()
            .await
            .expect("the stream must yield a failure");
        assert!(matches!(first, Err(Error::Connection { .. })), "{first:?}");
    }
}
