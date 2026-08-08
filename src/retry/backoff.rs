//! Delay arithmetic: `Retry-After` parsing, jitter, and the OOM backoff schedule.

use std::time::{Duration, SystemTime};

use reqwest::header::HeaderMap;

/// Base interval for provisioning and transport retries.
pub const DEFAULT_RETRY_DELAY: Duration = Duration::from_secs(5);
/// Total wall-clock budget a retrying call may consume, unless the caller overrides it.
pub const DEFAULT_PROVISION_TIMEOUT: Duration = Duration::from_mins(15);
/// Poll interval while a model loads.
pub const MODEL_LOADING_DELAY: Duration = Duration::from_secs(5);
/// Poll interval while a `LoRA` adapter loads.
pub const LORA_LOADING_DELAY: Duration = Duration::from_secs(1);
/// Retry cap for a `LoRA` adapter that never finishes loading.
pub const LORA_LOADING_MAX_RETRIES: u32 = 10;
/// Default retry cap for `RESOURCE_EXHAUSTED`.
pub const RESOURCE_EXHAUSTED_MAX_RETRIES: u32 = 3;
/// Base interval for the OOM backoff schedule.
pub const RESOURCE_EXHAUSTED_DELAY: Duration = Duration::from_secs(5);
/// Ceiling on any single OOM sleep.
pub const RESOURCE_EXHAUSTED_MAX_DELAY: Duration = Duration::from_secs(30);
/// How much of a delay downward jitter may remove.
pub const JITTER_FRACTION: f64 = 0.25;
/// How often a pool lease is renewed.
pub const LEASE_RENEWAL_INTERVAL: Duration = Duration::from_mins(1);
/// Attempts per lease-renewal round before giving up until the next round.
pub const LEASE_RENEWAL_MAX_RETRIES: u32 = 5;

/// Apply downward-only equal jitter given a uniform sample in `[0, 1]`.
///
/// The result stays within `[delay * 0.75, delay]`, so a jittered delay never exceeds the
/// cap its caller already applied.
pub(crate) fn jitter_with(delay: Duration, unit: f64) -> Duration {
    if delay.is_zero() {
        return delay;
    }
    let seconds = delay.as_secs_f64();
    let low = seconds * (1.0 - JITTER_FRACTION);
    Duration::from_secs_f64((low + unit.clamp(0.0, 1.0) * (seconds - low)).max(0.0))
}

/// De-correlate a fleet of clients that were all evicted by the same event.
pub fn apply_jitter(delay: Duration) -> Duration {
    jitter_with(delay, rand::random::<f64>())
}

/// Parse `Retry-After`, in either the delay-seconds or the HTTP-date form.
///
/// A malformed, non-finite or negative value means "no hint", which is distinct from a
/// hint of zero.
pub fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(seconds) = raw.parse::<f64>() {
        if !seconds.is_finite() || seconds < 0.0 {
            return None;
        }
        return Some(Duration::from_secs_f64(seconds));
    }
    let when = httpdate::parse_http_date(raw).ok()?;
    Some(
        when.duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO),
    )
}

/// A server hint if there is one, else the caller's default. Preserves an explicit zero.
pub fn retry_after_or(hint: Option<Duration>, default: Duration) -> Duration {
    hint.unwrap_or(default)
}

/// Bounded exponential backoff for `RESOURCE_EXHAUSTED`.
///
/// The first server hint is honoured verbatim: when the server says "wait N seconds" the
/// SDK obeys, and only de-correlates its own derived schedule. On later attempts the
/// exponential base is `max(base, hint)`, so `Retry-After: 0` cannot collapse the schedule
/// and a hint above the base keeps it non-decreasing.
pub(crate) fn oom_backoff_with(hint: Option<Duration>, attempt: u32, unit: f64) -> Duration {
    let max_delay = RESOURCE_EXHAUSTED_MAX_DELAY;
    if let Some(hint) = hint
        && attempt == 0
    {
        return hint.min(max_delay);
    }
    let base = hint.map_or(RESOURCE_EXHAUSTED_DELAY, |hint| {
        hint.max(RESOURCE_EXHAUSTED_DELAY)
    });
    let scaled = base
        .checked_mul(1u32.checked_shl(attempt).unwrap_or(u32::MAX))
        .unwrap_or(max_delay);
    jitter_with(scaled.min(max_delay), unit)
}

/// Bounded exponential backoff for `RESOURCE_EXHAUSTED`, jittered from the thread RNG.
pub fn oom_backoff(hint: Option<Duration>, attempt: u32) -> Duration {
    oom_backoff_with(hint, attempt, rand::random::<f64>())
}

/// Delay before replaying a request that failed at the transport layer.
///
/// `None` means the wall-clock budget is spent and the caller must surface the failure.
pub fn transport_delay(elapsed: Duration, budget: Duration) -> Option<Duration> {
    let remaining = budget.checked_sub(elapsed)?;
    if remaining.is_zero() {
        return None;
    }
    Some(apply_jitter(DEFAULT_RETRY_DELAY.min(remaining)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
        headers
    }

    #[test]
    fn jitter_never_exceeds_the_input_and_never_goes_below_three_quarters() {
        let delay = Duration::from_secs(8);
        assert_eq!(jitter_with(delay, 1.0), delay);
        assert_eq!(jitter_with(delay, 0.0), Duration::from_secs(6));
        for step in 0..=10 {
            let jittered = jitter_with(delay, f64::from(step) / 10.0);
            assert!(jittered <= delay && jittered >= Duration::from_secs(6));
        }
        assert!(jitter_with(Duration::ZERO, 0.5).is_zero());
    }

    #[test]
    fn retry_after_seconds_form() {
        assert_eq!(
            retry_after(&headers_with("5")),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            retry_after(&headers_with("2.5")),
            Some(Duration::from_millis(2500))
        );
        assert_eq!(retry_after(&headers_with("0")), Some(Duration::ZERO));
    }

    #[test]
    fn malformed_retry_after_means_no_hint() {
        for raw in ["-1", "nan", "inf", "-inf", "soon", ""] {
            assert_eq!(retry_after(&headers_with(raw)), None, "{raw:?}");
        }
        assert_eq!(retry_after(&HeaderMap::new()), None);
    }

    #[test]
    fn retry_after_http_date_form() {
        // A date in the past clamps to zero rather than going negative.
        let past = retry_after(&headers_with("Wed, 21 Oct 2015 07:28:00 GMT")).unwrap();
        assert_eq!(past, Duration::ZERO);
        let future = retry_after(&headers_with("Fri, 31 Dec 2100 23:59:59 GMT")).unwrap();
        assert!(future > Duration::from_mins(1));
    }

    #[test]
    fn retry_after_or_preserves_explicit_zero() {
        assert_eq!(
            retry_after_or(Some(Duration::ZERO), MODEL_LOADING_DELAY),
            Duration::ZERO
        );
        assert_eq!(
            retry_after_or(None, MODEL_LOADING_DELAY),
            MODEL_LOADING_DELAY
        );
    }

    #[test]
    fn oom_schedule_defaults_to_five_ten_twenty() {
        // unit = 1.0 removes jitter so the schedule itself is visible.
        assert_eq!(oom_backoff_with(None, 0, 1.0), Duration::from_secs(5));
        assert_eq!(oom_backoff_with(None, 1, 1.0), Duration::from_secs(10));
        assert_eq!(oom_backoff_with(None, 2, 1.0), Duration::from_secs(20));
        assert_eq!(oom_backoff_with(None, 3, 1.0), Duration::from_secs(30));
        assert_eq!(oom_backoff_with(None, 40, 1.0), Duration::from_secs(30));
    }

    #[test]
    fn first_oom_hint_is_honoured_verbatim_and_capped() {
        assert_eq!(
            oom_backoff_with(Some(Duration::from_secs(7)), 0, 0.0),
            Duration::from_secs(7)
        );
        assert_eq!(
            oom_backoff_with(Some(Duration::from_mins(10)), 0, 0.0),
            RESOURCE_EXHAUSTED_MAX_DELAY
        );
    }

    #[test]
    fn later_oom_attempts_keep_the_schedule_non_decreasing() {
        // A zero hint must not collapse the backoff.
        assert_eq!(
            oom_backoff_with(Some(Duration::ZERO), 1, 1.0),
            Duration::from_secs(10)
        );
        // A hint above the base raises it rather than lowering the next sleep.
        assert_eq!(
            oom_backoff_with(Some(Duration::from_secs(20)), 1, 1.0),
            RESOURCE_EXHAUSTED_MAX_DELAY
        );
    }

    #[test]
    fn transport_delay_respects_the_budget() {
        let budget = Duration::from_mins(15);
        let delay = transport_delay(Duration::from_secs(1), budget).unwrap();
        assert!(delay <= DEFAULT_RETRY_DELAY && delay >= Duration::from_secs_f64(3.75));

        let near_end = transport_delay(Duration::from_secs_f64(899.5), budget).unwrap();
        assert!(near_end <= Duration::from_millis(500));

        assert!(transport_delay(budget, budget).is_none());
        assert!(transport_delay(Duration::from_secs(901), budget).is_none());
    }
}
