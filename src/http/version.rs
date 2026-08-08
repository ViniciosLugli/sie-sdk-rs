//! SDK version reporting and the once-per-process server skew warning.

use std::sync::atomic::{AtomicBool, Ordering};

/// Version this SDK reports in `X-SIE-SDK-Version`.
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

static SKEW_WARNED: AtomicBool = AtomicBool::new(false);

fn major_minor(version: &str) -> Option<(u64, u64)> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// The warning to emit for a given SDK/server version pair, if any.
///
/// Non-semver inputs silently skip the check rather than failing a request.
pub fn skew_warning(sdk_version: &str, server_version: &str) -> Option<String> {
    let (sdk_major, sdk_minor) = major_minor(sdk_version)?;
    let (server_major, server_minor) = major_minor(server_version)?;

    if sdk_major != server_major {
        return Some(format!(
            "SDK version {sdk_version} has different major version than server {server_version}. Please upgrade."
        ));
    }
    if sdk_minor.abs_diff(server_minor) > 1 {
        let direction = if sdk_minor < server_minor {
            "behind"
        } else {
            "ahead of"
        };
        return Some(format!(
            "SDK version {sdk_version} is more than one minor version {direction} server {server_version}. \
             Consider upgrading."
        ));
    }
    None
}

/// Emit the skew warning at most once for the lifetime of the process.
pub fn warn_once(server_version: &str) {
    if SKEW_WARNED.load(Ordering::Relaxed) {
        return;
    }
    if let Some(message) = skew_warning(SDK_VERSION, server_version)
        && !SKEW_WARNED.swap(true, Ordering::Relaxed)
    {
        tracing::warn!("{message}");
    }
}

/// Split a `"pool/gpu"` parameter. A bare value names only the machine profile.
pub fn parse_gpu_param(gpu: &str) -> (Option<&str>, &str) {
    match gpu.split_once('/') {
        Some((pool, profile)) => (Some(pool), profile),
        None => (None, gpu),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn major_mismatch_warns() {
        let warning = skew_warning("1.2.0", "2.2.0").unwrap();
        assert!(warning.contains("different major version"));
    }

    #[test]
    fn minor_drift_warns_only_beyond_one() {
        assert!(skew_warning("0.6.30", "0.7.0").is_none());
        assert!(skew_warning("0.7.0", "0.6.30").is_none());
        assert!(skew_warning("0.6.30", "0.6.1").is_none());
        assert!(skew_warning("0.6.0", "0.8.0").unwrap().contains("behind"));
        assert!(skew_warning("0.9.0", "0.7.0").unwrap().contains("ahead of"));
    }

    #[test]
    fn non_semver_is_ignored() {
        assert!(skew_warning("unknown", "0.6.0").is_none());
        assert!(skew_warning("0.6.0", "dev").is_none());
        assert!(skew_warning("1", "1").is_none());
    }

    #[test]
    fn gpu_param_splits_on_first_slash_only() {
        assert_eq!(parse_gpu_param("eval-bench/l4"), (Some("eval-bench"), "l4"));
        assert_eq!(parse_gpu_param("l4"), (None, "l4"));
        assert_eq!(parse_gpu_param("a/b/c"), (Some("a"), "b/c"));
    }
}
