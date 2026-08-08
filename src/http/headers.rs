//! SIE header constants and the `base_url_headers` origin-scoping policy.
//!
//! `base_url_headers` exist for an HTTP edge sitting in front of the gateway (Modal's
//! `Modal-Key` / `Modal-Secret` pair, for example). They are credentials, so they are
//! validated at construction and attached only to requests that resolve to the exact
//! origin of `base_url`.

use std::collections::HashMap;

use url::Url;

use crate::error::{Error, Result};

pub const SERVER_VERSION: &str = "X-SIE-Server-Version";
pub const MODEL_REVISION: &str = "X-SIE-Model-Revision";
pub const MACHINE_PROFILE: &str = "X-SIE-MACHINE-PROFILE";
pub const POOL: &str = "X-SIE-Pool";
pub const ERROR_CODE: &str = "X-SIE-Error-Code";
pub const REQUEST_ID: &str = "X-SIE-Request-ID";
pub const CREDITS_DEBITED: &str = "X-SIE-Credits-Debited";
pub const EXECUTION_IDENTITY: &str = "X-SIE-Execution-Identity-SHA256";
pub const IDEMPOTENCY_KEY: &str = "Idempotency-Key";

pub const UNITS_INPUT_TOKENS: &str = "X-SIE-Units-Input-Tokens";
pub const UNITS_PAIRS: &str = "X-SIE-Units-Pairs";
pub const UNITS_IMAGES: &str = "X-SIE-Units-Images";
pub const UNITS_PAGES: &str = "X-SIE-Units-Pages";
pub const UNITS_OUTPUT_TOKENS: &str = "X-SIE-Units-Output-Tokens";
pub const UNITS_AUDIO_MS: &str = "X-SIE-Units-Audio-Ms";

pub const MSGPACK_CONTENT_TYPE: &str = "application/msgpack";
pub const JSON_CONTENT_TYPE: &str = "application/json";
pub const JSONL_CONTENT_TYPE: &str = "application/jsonl";
pub const SSE_CONTENT_TYPE: &str = "text/event-stream";
pub const OCTET_STREAM_CONTENT_TYPE: &str = "application/octet-stream";

/// Headers the SDK owns and a caller may not override through `base_url_headers`.
const RESERVED: &[&str] = &[
    "accept",
    "authorization",
    "connection",
    "content-length",
    "content-type",
    "cookie",
    "host",
    "idempotency-key",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "set-cookie",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "x-sie-sdk-version",
];

fn is_valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

fn has_control_char(value: &str) -> bool {
    value
        .chars()
        .any(|c| (c < '\x20' && c != '\t') || c == '\x7f')
}

/// Validate and detach caller-supplied edge headers.
///
/// Detaching matters: it stops a caller mutating the credentials after the client is built.
pub fn validate_base_url_headers(
    headers: &HashMap<String, String>,
) -> Result<Vec<(String, String)>> {
    let mut copied = Vec::with_capacity(headers.len());
    let mut seen: Vec<String> = Vec::with_capacity(headers.len());

    // HashMap iteration order is arbitrary; sort so the error a caller sees is deterministic.
    let mut entries: Vec<(&String, &String)> = headers.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    for (name, value) in entries {
        if !is_valid_header_name(name) {
            return Err(Error::invalid(format!(
                "invalid base_url_headers name: {name:?}"
            )));
        }
        let normalized = name.to_ascii_lowercase();
        if seen.contains(&normalized) {
            return Err(Error::invalid(format!(
                "duplicate base_url_headers name (case-insensitive): {name:?}"
            )));
        }
        if RESERVED.contains(&normalized.as_str()) || normalized.starts_with("sec-websocket-") {
            return Err(Error::invalid(format!(
                "base_url_headers cannot override SDK-owned header {name:?}"
            )));
        }
        if has_control_char(value) {
            return Err(Error::invalid(format!(
                "invalid control character in base_url_headers value for {name:?}"
            )));
        }
        seen.push(normalized);
        copied.push((name.clone(), value.clone()));
    }
    Ok(copied)
}

/// A normalized HTTP origin: scheme, lowercased host, explicit port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    scheme: String,
    host: String,
    port: u16,
}

impl Origin {
    /// Parse an origin, rejecting non-HTTP schemes and any embedded userinfo.
    pub fn parse(url: &Url) -> Option<Self> {
        let scheme = url.scheme().to_ascii_lowercase();
        if scheme != "http" && scheme != "https" {
            return None;
        }
        if !url.username().is_empty() || url.password().is_some() {
            return None;
        }
        let host = url.host_str()?.to_ascii_lowercase();
        let port = url
            .port()
            .unwrap_or(if scheme == "https" { 443 } else { 80 });
        Some(Self { scheme, host, port })
    }

    /// Whether credentials may be transported to this origin at all.
    pub fn accepts_credentials(&self) -> bool {
        self.scheme == "https"
    }

    /// Whether a `ws`/`wss` URL is the WebSocket counterpart of this origin.
    #[cfg(any(feature = "watch", test))]
    pub fn matches_websocket(&self, ws_url: &Url) -> bool {
        let scheme = match ws_url.scheme() {
            "wss" => "https",
            "ws" => "http",
            _ => return false,
        };
        let mut http_url = ws_url.clone();
        if http_url.set_scheme(scheme).is_err() {
            return false;
        }
        Origin::parse(&http_url).is_some_and(|origin| origin == *self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn accepts_edge_credentials() {
        let out = validate_base_url_headers(&headers(&[("Modal-Key", "k"), ("Modal-Secret", "s")]))
            .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn rejects_reserved_and_websocket_headers() {
        for name in [
            "Authorization",
            "content-type",
            "X-SIE-SDK-Version",
            "Sec-WebSocket-Key",
        ] {
            let err = validate_base_url_headers(&headers(&[(name, "v")])).unwrap_err();
            assert!(err.to_string().contains("SDK-owned"), "{name}: {err}");
        }
    }

    #[test]
    fn rejects_bad_names_and_control_characters() {
        assert!(validate_base_url_headers(&headers(&[("bad name", "v")])).is_err());
        assert!(validate_base_url_headers(&headers(&[("X-Edge", "va\nlue")])).is_err());
        assert!(validate_base_url_headers(&headers(&[("X-Edge", "va\x7flue")])).is_err());
        // A tab is the one control character HTTP allows in a field value.
        assert!(validate_base_url_headers(&headers(&[("X-Edge", "va\tlue")])).is_ok());
    }

    #[test]
    fn origin_normalizes_default_ports_and_rejects_userinfo() {
        let a = Origin::parse(&Url::parse("https://gw.example.com/v1").unwrap()).unwrap();
        let b = Origin::parse(&Url::parse("https://GW.Example.com:443/other").unwrap()).unwrap();
        assert_eq!(a, b);
        assert!(a.accepts_credentials());

        assert!(Origin::parse(&Url::parse("https://user:pw@gw.example.com").unwrap()).is_none());
        assert!(Origin::parse(&Url::parse("ftp://gw.example.com").unwrap()).is_none());

        let plain = Origin::parse(&Url::parse("http://localhost:8080").unwrap()).unwrap();
        assert!(!plain.accepts_credentials());
        assert_ne!(plain, a);
    }

    #[test]
    fn websocket_counterpart_matching() {
        let origin = Origin::parse(&Url::parse("https://gw.example.com").unwrap()).unwrap();
        assert!(origin.matches_websocket(&Url::parse("wss://gw.example.com/ws/status").unwrap()));
        assert!(!origin.matches_websocket(&Url::parse("ws://gw.example.com/ws/status").unwrap()));
        assert!(
            !origin.matches_websocket(&Url::parse("wss://other.example.com/ws/status").unwrap())
        );
        assert!(
            !origin.matches_websocket(&Url::parse("https://gw.example.com/ws/status").unwrap())
        );
    }
}
