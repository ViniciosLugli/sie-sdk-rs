//! Transport plumbing: the buffered response type every shared helper works on, and the
//! request description the retry driver replays.

pub mod headers;
pub mod metadata;
pub mod version;

use bytes::Bytes;
use reqwest::header::HeaderMap;
use reqwest::{Method, Url};
use serde_json::Value;

/// A fully-read HTTP response.
///
/// Both the buffered and the streaming paths funnel non-200 responses through this type so
/// error handling, metadata parsing and retry decisions have exactly one implementation.
#[derive(Debug, Clone)]
pub(crate) struct HttpResponse {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl HttpResponse {
    pub(crate) async fn read(response: reqwest::Response) -> reqwest::Result<Self> {
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response.bytes().await?;
        Ok(Self {
            status,
            headers,
            body,
        })
    }

    pub(crate) fn is_msgpack(&self) -> bool {
        self.headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.contains(headers::MSGPACK_CONTENT_TYPE))
    }

    /// The body decoded as a generic value, honouring the response content type.
    ///
    /// Returns `None` when the body is not a well-formed document, which callers treat the
    /// same way Python treats a decode exception: fall back to the raw text.
    pub(crate) fn decode_value(&self) -> Option<Value> {
        if self.is_msgpack() {
            rmp_serde::from_slice(&self.body).ok()
        } else {
            serde_json::from_slice(&self.body).ok()
        }
    }

    pub(crate) fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|value| value.to_str().ok())
    }
}

/// Everything needed to build one HTTP attempt, replayable across retries.
#[derive(Debug, Clone)]
pub(crate) struct PreparedRequest {
    pub method: Method,
    pub url: Url,
    pub headers: HeaderMap,
    pub body: Option<Bytes>,
}

impl PreparedRequest {
    pub(crate) fn new(method: Method, url: Url) -> Self {
        Self {
            method,
            url,
            headers: HeaderMap::new(),
            body: None,
        }
    }

    pub(crate) fn header(mut self, name: &str, value: &str) -> Self {
        if let (Ok(name), Ok(value)) = (
            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            self.headers.insert(name, value);
        }
        self
    }

    pub(crate) fn maybe_header(self, name: &str, value: Option<&str>) -> Self {
        match value {
            Some(value) => self.header(name, value),
            None => self,
        }
    }

    pub(crate) fn body(mut self, body: impl Into<Bytes>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Content-Type and Accept for a JSON endpoint.
    pub(crate) fn json_headers(self) -> Self {
        self.header("content-type", headers::JSON_CONTENT_TYPE)
            .header("accept", headers::JSON_CONTENT_TYPE)
    }

    /// Content-Type and Accept for a msgpack endpoint.
    pub(crate) fn msgpack_headers(self) -> Self {
        self.header("content-type", headers::MSGPACK_CONTENT_TYPE)
            .header("accept", headers::MSGPACK_CONTENT_TYPE)
    }

    /// Content-Type for a JSON request whose response is an SSE stream.
    pub(crate) fn sse_headers(self) -> Self {
        self.header("content-type", headers::JSON_CONTENT_TYPE)
            .header("accept", headers::SSE_CONTENT_TYPE)
    }
}
