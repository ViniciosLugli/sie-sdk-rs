//! Cost estimation for a request that has not been sent yet.

// Wire-mirror types: field names are the API contract itself, and the ones whose
// meaning is not obvious carry their own doc comment.
#![allow(missing_docs)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One priced unit and its exact rational rate.
///
/// Credits are integers and rates are exact rationals: multiply and round once, never per
/// unit, or the totals will not reconcile with the server's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedRate {
    pub unit: String,
    pub rate_numerator: i64,
    pub rate_denominator: i64,
}

/// What the rate book priced this request as.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateIdentity {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub model: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub profile: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub operation: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub region: String,
}

/// The estimated cost of a request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostEstimate {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub endpoint: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub identity: RateIdentity,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub estimated_credits: u64,
    /// Upper bound assumed for each priced unit.
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub unit_ceilings: HashMap<String, u64>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub applied_rates: Vec<AppliedRate>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub rate_book_version: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub rate_book_sha256: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub rounding_rule: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub estimate_basis: String,
    /// Meaningful only for duration-priced identities; `null` otherwise.
    #[serde(default)]
    pub minimum_billed_units: Option<HashMap<String, u64>>,
}
