//! Parsing of the request-scoped metadata a terminal response carries.
//!
//! Every field is validated independently and dropped on the floor if malformed, so a
//! misbehaving proxy cannot inject junk into a caller's billing records.

use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::http::headers;
use crate::types::{RequestMetadata, RequestUsage};

const MAX_REQUEST_ID_LEN: usize = 256;

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

/// A meter header must be canonical unsigned decimal: ASCII digits, no leading zero
/// unless the value is exactly `"0"`, and within `u64`.
fn meter_header(headers: &HeaderMap, name: &str) -> Option<u64> {
    let raw = header_str(headers, name)?;
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if raw.len() > 1 && raw.starts_with('0') {
        return None;
    }
    raw.parse::<u64>().ok()
}

fn valid_request_id(raw: &str) -> bool {
    !raw.is_empty()
        && raw.len() <= MAX_REQUEST_ID_LEN
        && raw == raw.trim()
        && raw.bytes().all(|b| (0x20..=0x7e).contains(&b))
}

fn valid_execution_identity(raw: &str) -> bool {
    raw.len() == 64
        && raw
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The settled charge carried by one `usage` block.
///
/// Both halves are required: a charge with no rate-book version cannot be reconciled, and
/// a version with no charge describes nothing.
pub fn settled_charge_from_usage(usage: &Value) -> Option<(u64, String)> {
    let credits = usage.get("credits_charged")?;
    // `serde_json` parses `true` as a bool, never as an integer, so the Python bool guard
    // is structural here.
    let credits = credits.as_u64()?;
    let version = usage.get("rate_book_version")?.as_str()?;
    if version.is_empty() {
        return None;
    }
    Some((credits, version.to_string()))
}

fn settled_charge_from_body(body: Option<&Value>) -> Option<(u64, String)> {
    settled_charge_from_usage(body?.get("usage")?)
}

/// Parse metadata from a terminal response.
///
/// `body` is the decoded response envelope when the caller has one. The settled charge is
/// read from its `usage` block first and falls back to `X-SIE-Credits-Debited`, which is
/// the only source on a gateway that predates in-body surfacing.
pub fn parse(headers: &HeaderMap, body: Option<&Value>, retries: u32) -> Option<RequestMetadata> {
    let mut metadata = RequestMetadata {
        retries,
        ..Default::default()
    };

    if let Some(id) = header_str(headers, headers::REQUEST_ID).filter(|id| valid_request_id(id)) {
        metadata.id = Some(id.to_string());
    }

    let mut usage = RequestUsage {
        input_tokens: meter_header(headers, headers::UNITS_INPUT_TOKENS),
        pairs: meter_header(headers, headers::UNITS_PAIRS),
        images: meter_header(headers, headers::UNITS_IMAGES),
        pages: meter_header(headers, headers::UNITS_PAGES),
        output_tokens: meter_header(headers, headers::UNITS_OUTPUT_TOKENS),
        audio_ms: meter_header(headers, headers::UNITS_AUDIO_MS),
        ..Default::default()
    };

    match settled_charge_from_body(body) {
        Some((credits, version)) => {
            usage.credits_charged = Some(credits);
            usage.rate_book_version = Some(version.clone());
            metadata.credits_debited = Some(credits);
            metadata.rate_book_version = Some(version);
        }
        None => metadata.credits_debited = meter_header(headers, headers::CREDITS_DEBITED),
    }

    if !usage.is_empty() {
        metadata.usage = Some(usage);
    }

    if let Some(identity) = header_str(headers, headers::EXECUTION_IDENTITY)
        .filter(|value| valid_execution_identity(value))
    {
        metadata.execution_identity_sha256 = Some(identity.to_string());
    }

    if let Some(revision) =
        header_str(headers, headers::MODEL_REVISION).filter(|value| !value.is_empty())
    {
        metadata.model_revision = Some(revision.to_string());
    }

    if metadata.is_empty() && retries == 0 {
        None
    } else {
        Some(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn headers_from(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        map
    }

    #[test]
    fn meter_headers_must_be_canonical_decimal() {
        let cases = [
            ("12", Some(12)),
            ("0", Some(0)),
            ("007", None),
            ("01", None),
            ("-1", None),
            ("1.5", None),
            (" 4", None),
            ("", None),
            ("18446744073709551616", None), // u64::MAX + 1
            ("18446744073709551615", Some(u64::MAX)),
        ];
        for (raw, expected) in cases {
            let map = headers_from(&[(headers::UNITS_INPUT_TOKENS, raw)]);
            assert_eq!(
                meter_header(&map, headers::UNITS_INPUT_TOKENS),
                expected,
                "{raw:?}"
            );
        }
    }

    #[test]
    fn request_id_validation() {
        assert!(valid_request_id("req_abc-123"));
        assert!(!valid_request_id(""));
        assert!(!valid_request_id(" req"));
        assert!(!valid_request_id("req "));
        assert!(!valid_request_id(&"x".repeat(257)));
        assert!(valid_request_id(&"x".repeat(256)));
    }

    #[test]
    fn execution_identity_must_be_lowercase_hex_64() {
        assert!(valid_execution_identity(&"a1".repeat(32)));
        assert!(!valid_execution_identity(&"A1".repeat(32)));
        assert!(!valid_execution_identity(&"a1".repeat(31)));
    }

    #[test]
    fn body_settled_charge_wins_over_header() {
        let map = headers_from(&[(headers::CREDITS_DEBITED, "99")]);
        let body = json!({"usage": {"credits_charged": 7, "rate_book_version": "2026-01"}});
        let metadata = parse(&map, Some(&body), 0).unwrap();
        assert_eq!(metadata.credits_debited, Some(7));
        assert_eq!(metadata.rate_book_version.as_deref(), Some("2026-01"));
        let usage = metadata.usage.unwrap();
        assert_eq!(usage.credits_charged, Some(7));
        assert_eq!(usage.rate_book_version.as_deref(), Some("2026-01"));
    }

    #[test]
    fn incomplete_body_charge_falls_back_to_header() {
        let map = headers_from(&[(headers::CREDITS_DEBITED, "99")]);
        for body in [
            json!({"usage": {"credits_charged": 7}}),
            json!({"usage": {"rate_book_version": "v"}}),
        ] {
            let metadata = parse(&map, Some(&body), 0).unwrap();
            assert_eq!(metadata.credits_debited, Some(99));
            assert!(metadata.rate_book_version.is_none());
        }
    }

    #[test]
    fn nothing_valid_yields_no_metadata() {
        let map = headers_from(&[(headers::REQUEST_ID, " bad "), (headers::UNITS_PAIRS, "01")]);
        assert!(parse(&map, None, 0).is_none());
        // A retry count is itself worth reporting.
        assert_eq!(parse(&map, None, 2).unwrap().retries, 2);
    }
}
