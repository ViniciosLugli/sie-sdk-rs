//! Server error envelopes: decoding them, and mapping status + code onto [`Error`].

pub(crate) mod msg;
pub mod ndarray;
pub mod sse;

use serde_json::Value;

use crate::error::{Error, ModelLoadErrorClass, Result, codes};
use crate::http::{HttpResponse, headers, metadata};

/// The `{code, message}` pair carried by an error response.
#[derive(Debug, Clone, Default)]
pub(crate) struct ErrorEnvelope {
    pub code: Option<String>,
    pub message: Option<String>,
    /// The nested error object itself, for the fields only some errors carry.
    pub detail: Option<Value>,
}

/// The gateway spells `PROVISIONING` in lower case on one legacy path.
fn normalize_error_code(code: Option<String>) -> Option<String> {
    match code.as_deref() {
        Some("provisioning") => Some(codes::PROVISIONING.to_string()),
        _ => code,
    }
}

fn envelope_from_value(data: &Value) -> ErrorEnvelope {
    // `error` first, then `detail`; either may be an object or a bare string.
    for key in ["error", "detail"] {
        let Some(node) = data.get(key) else { continue };
        return match node {
            Value::Object(_) => ErrorEnvelope {
                code: node.get("code").and_then(Value::as_str).map(str::to_string),
                message: node
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                detail: Some(node.clone()),
            },
            Value::String(text) => ErrorEnvelope {
                code: None,
                message: Some(text.clone()),
                detail: None,
            },
            other => ErrorEnvelope {
                code: None,
                message: Some(other.to_string()),
                detail: None,
            },
        };
    }
    ErrorEnvelope::default()
}

/// Decode an error response into its envelope, falling back to raw text.
pub(crate) fn parse_envelope(response: &HttpResponse) -> ErrorEnvelope {
    let mut envelope = match response.decode_value() {
        Some(data) => envelope_from_value(&data),
        None => ErrorEnvelope {
            message: Some(response.text()).filter(|text| !text.is_empty()),
            ..Default::default()
        },
    };

    // The header is authoritative: it survives a body the SDK could not decode.
    if let Some(header_code) = response
        .header(headers::ERROR_CODE)
        .filter(|code| !code.is_empty())
    {
        envelope.code = Some(header_code.to_string());
    } else {
        envelope.code = normalize_error_code(envelope.code);
    }
    envelope
}

/// The error code for a response, header first.
pub(crate) fn error_code(response: &HttpResponse) -> Option<String> {
    parse_envelope(response).code
}

fn message_or_status(envelope: &ErrorEnvelope, status: u16) -> String {
    envelope
        .message
        .clone()
        .unwrap_or_else(|| format!("HTTP {status}"))
}

/// A 502 `MODEL_LOAD_FAILED` is terminal and must be surfaced before any retry budget.
pub(crate) fn check_model_load_failed(
    response: &HttpResponse,
    model: Option<&str>,
    retries: u32,
) -> Result<()> {
    if response.status != 502 {
        return Ok(());
    }
    let envelope = parse_envelope(response);
    if envelope.code.as_deref() != Some(codes::MODEL_LOAD_FAILED) {
        return Ok(());
    }
    let detail = envelope.detail.clone().unwrap_or(Value::Null);
    let attempts = detail
        .get("attempts")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_f64().map(|f| f as u64))
                .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(1)
        .max(1);

    Err(Error::ModelLoadFailed {
        message: message_or_status(&envelope, response.status),
        model: detail
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| model.map(str::to_string)),
        error_class: ModelLoadErrorClass::parse(detail.get("error_class").and_then(Value::as_str)),
        permanent: detail
            .get("permanent")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        attempts: attempts.min(u64::from(u32::MAX)) as u32,
        request: metadata::parse(&response.headers, None, retries).map(Box::new),
    })
}

/// `/v1/estimate` reports an unroutable identity as a 503 with a capacity code.
pub(crate) fn check_estimate_unroutable(response: &HttpResponse, retries: u32) -> Result<()> {
    if response.status != 503 {
        return Ok(());
    }
    let envelope = parse_envelope(response);
    let code = envelope.code.as_deref().unwrap_or_default();
    if code != codes::QUEUE_UNAVAILABLE && code != codes::PROVISIONING {
        return Ok(());
    }
    Err(Error::EstimateUnroutable {
        message: message_or_status(&envelope, response.status),
        code: envelope.code,
        request: metadata::parse(&response.headers, None, retries).map(Box::new),
    })
}

/// Terminal dispatcher: turn any `>= 400` response into the right [`Error`].
pub(crate) fn handle_error(response: &HttpResponse, model: Option<&str>, retries: u32) -> Error {
    let envelope = parse_envelope(response);
    let message = message_or_status(&envelope, response.status);
    let request = metadata::parse(&response.headers, None, retries).map(Box::new);
    let status = response.status;
    let code = envelope.code.as_deref();

    if status == 503 && code == Some(codes::PROVISIONING) {
        return Error::Provisioning {
            message,
            gpu: None,
            retry_after: crate::retry::backoff::retry_after(&response.headers),
        };
    }
    if status == 400 && code == Some(codes::INPUT_TOO_LONG) {
        return Error::InputTooLong {
            message,
            model: model.map(str::to_string),
            request,
        };
    }
    if status >= 500 {
        return Error::Server {
            message,
            code: envelope.code,
            status,
            request,
        };
    }
    Error::Request {
        message,
        code: envelope.code,
        status,
        request,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use reqwest::header::HeaderMap;

    fn response(
        status: u16,
        body: &str,
        content_type: &str,
        extra: &[(&str, &str)],
    ) -> HttpResponse {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, content_type.parse().unwrap());
        for (name, value) in extra {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        HttpResponse {
            status,
            headers,
            body: Bytes::copy_from_slice(body.as_bytes()),
        }
    }

    #[test]
    fn parses_error_and_detail_objects() {
        let envelope = parse_envelope(&response(
            500,
            r#"{"error": {"code": "INTERNAL_ERROR", "message": "boom"}}"#,
            "application/json",
            &[],
        ));
        assert_eq!(envelope.code.as_deref(), Some("INTERNAL_ERROR"));
        assert_eq!(envelope.message.as_deref(), Some("boom"));

        let envelope = parse_envelope(&response(
            400,
            r#"{"detail": {"code": "BAD", "message": "nope"}}"#,
            "application/json",
            &[],
        ));
        assert_eq!(envelope.code.as_deref(), Some("BAD"));
        assert_eq!(envelope.message.as_deref(), Some("nope"));
    }

    #[test]
    fn bare_string_error_becomes_the_message() {
        let envelope = parse_envelope(&response(
            400,
            r#"{"detail": "plain text"}"#,
            "application/json",
            &[],
        ));
        assert!(envelope.code.is_none());
        assert_eq!(envelope.message.as_deref(), Some("plain text"));
    }

    #[test]
    fn undecodable_body_falls_back_to_text() {
        let envelope = parse_envelope(&response(502, "<html>gateway</html>", "text/html", &[]));
        assert_eq!(envelope.message.as_deref(), Some("<html>gateway</html>"));
    }

    #[test]
    fn header_code_beats_body_code() {
        let envelope = parse_envelope(&response(
            503,
            r#"{"error": {"code": "SOMETHING_ELSE", "message": "m"}}"#,
            "application/json",
            &[(headers::ERROR_CODE, codes::RESOURCE_EXHAUSTED)],
        ));
        assert_eq!(envelope.code.as_deref(), Some(codes::RESOURCE_EXHAUSTED));
    }

    #[test]
    fn lowercase_provisioning_is_normalized() {
        let envelope = parse_envelope(&response(
            503,
            r#"{"error": {"code": "provisioning", "message": "m"}}"#,
            "application/json",
            &[],
        ));
        assert_eq!(envelope.code.as_deref(), Some(codes::PROVISIONING));
    }

    #[test]
    fn status_dispatch_matches_python() {
        let too_long = handle_error(
            &response(
                400,
                r#"{"error": {"code": "INPUT_TOO_LONG", "message": "m"}}"#,
                "application/json",
                &[],
            ),
            Some("bge"),
            0,
        );
        assert!(matches!(too_long, Error::InputTooLong { .. }));

        let server = handle_error(&response(500, "{}", "application/json", &[]), None, 0);
        assert!(matches!(server, Error::Server { status: 500, .. }));

        let request = handle_error(&response(404, "{}", "application/json", &[]), None, 0);
        assert!(matches!(request, Error::Request { status: 404, .. }));
        assert_eq!(request.to_string(), "HTTP 404");
    }

    #[test]
    fn model_load_failed_is_terminal_with_detail() {
        let err = check_model_load_failed(
            &response(
                502,
                r#"{"error": {"code": "MODEL_LOAD_FAILED", "message": "gated", "error_class": "GATED",
                    "permanent": true, "attempts": "3", "model": "org/m"}}"#,
                "application/json",
                &[],
            ),
            None,
            0,
        )
        .unwrap_err();
        match err {
            Error::ModelLoadFailed {
                error_class,
                permanent,
                attempts,
                model,
                ..
            } => {
                assert_eq!(error_class, ModelLoadErrorClass::Gated);
                assert!(permanent);
                assert_eq!(attempts, 3);
                assert_eq!(model.as_deref(), Some("org/m"));
            }
            other => panic!("unexpected: {other:?}"),
        }

        // A 502 with a different code is not this error.
        assert!(
            check_model_load_failed(&response(502, "{}", "application/json", &[]), None, 0).is_ok()
        );
    }

    #[test]
    fn msgpack_error_bodies_decode() {
        let body = rmp_serde::to_vec_named(&serde_json::json!({
            "error": {"code": "INTERNAL_ERROR", "message": "packed"}
        }))
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/msgpack".parse().unwrap(),
        );
        let response = HttpResponse {
            status: 500,
            headers,
            body: Bytes::from(body),
        };
        let envelope = parse_envelope(&response);
        assert_eq!(envelope.message.as_deref(), Some("packed"));
        assert_eq!(envelope.code.as_deref(), Some("INTERNAL_ERROR"));
    }
}
