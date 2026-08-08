//! Scrubbing secrets out of anything that reaches a log.
//!
//! These match the SIE gateway's own redaction byte for byte, so a token masked here and a
//! token masked server-side render identically in a correlated trace.

use url::Url;

/// Mask a bearer token, leaving only its last four characters.
///
/// A token of four characters or fewer is replaced wholesale: there is nothing left to
/// reveal, and a shorter mask would leak its length.
pub fn mask_token(token: &str) -> String {
    let len = token.chars().count();
    if len <= 4 {
        return "****".to_string();
    }
    let tail: String = token.chars().skip(len - 4).collect();
    format!("{}{tail}", "*".repeat(len - 4))
}

/// Reduce an endpoint URL to a credential-, path- and query-free origin.
///
/// Anything that is not a plain `http`/`https` URL with a host becomes `<redacted>`, since
/// a string the parser does not understand may be carrying a secret in a shape this
/// function cannot see.
pub fn endpoint_origin_for_log(endpoint: &str) -> String {
    const REDACTED: &str = "<redacted>";

    let Ok(url) = Url::parse(endpoint) else {
        return REDACTED.to_string();
    };
    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return REDACTED.to_string();
    }
    let Some(host) = url.host_str() else {
        return REDACTED.to_string();
    };

    // An IPv6 literal needs its brackets back; `host_str` strips them.
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    };

    match url.port() {
        Some(port) => format!("{scheme}://{host}:{port}"),
        None => format!("{scheme}://{host}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_keep_only_their_last_four_characters() {
        assert_eq!(mask_token("secret-token-123"), "************-123");
        assert_eq!(mask_token("abcde"), "*bcde");
    }

    #[test]
    fn short_tokens_are_replaced_wholesale() {
        for token in ["", "a", "abc", "abcd"] {
            assert_eq!(mask_token(token), "****", "{token:?}");
        }
    }

    #[test]
    fn origins_drop_credentials_paths_and_queries() {
        assert_eq!(
            endpoint_origin_for_log(
                "https://user:secret@collector.example:4318/v1/metrics?token=x"
            ),
            "https://collector.example:4318"
        );
        assert_eq!(
            endpoint_origin_for_log("http://127.0.0.1/v1/metrics"),
            "http://127.0.0.1"
        );
    }

    #[test]
    fn ipv6_hosts_keep_their_brackets() {
        assert_eq!(
            endpoint_origin_for_log("http://[::1]:4318/v1/metrics"),
            "http://[::1]:4318"
        );
    }

    #[test]
    fn anything_unparseable_or_off_scheme_is_redacted() {
        for endpoint in [
            "not a URL with secret",
            "ftp://host/path",
            "https://bad:port:99999",
            "",
        ] {
            assert_eq!(
                endpoint_origin_for_log(endpoint),
                "<redacted>",
                "{endpoint:?}"
            );
        }
    }
}
