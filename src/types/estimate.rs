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
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub profile: String,
    #[serde(default)]
    pub operation: String,
    #[serde(default)]
    pub region: String,
}

/// The estimated cost of a request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostEstimate {
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub identity: RateIdentity,
    #[serde(default)]
    pub estimated_credits: u64,
    /// Upper bound assumed for each priced unit.
    #[serde(default)]
    pub unit_ceilings: HashMap<String, u64>,
    #[serde(default)]
    pub applied_rates: Vec<AppliedRate>,
    #[serde(default)]
    pub rate_book_version: String,
    #[serde(default)]
    pub rate_book_sha256: String,
    #[serde(default)]
    pub rounding_rule: String,
    #[serde(default)]
    pub estimate_basis: String,
    /// Meaningful only for duration-priced identities; `null` otherwise.
    #[serde(default)]
    pub minimum_billed_units: Option<HashMap<String, u64>>,
}
