//! Client-side `MaxSim` scoring for late-interaction (ColBERT-style) models.
//!
//! For a query `Q` and a document `D`, both token-level matrices:
//!
//! ```text
//! score(Q, D) = Σ over query tokens i of ( max over document tokens j of Q[i] · D[j] )
//! ```
//!
//! There is no normalization and no division by the query length: the encoder is expected
//! to have L2-normalized its output already.
//!
//! # Precision
//!
//! Late-interaction corpora are usually stored as `f16` to halve their memory cost, but the
//! dot products must accumulate in `f32` or the scores drift. Both functions here widen to
//! `f32` immediately before the dot product and never accumulate at `f16`.
//! [`maxsim_batch`] additionally iterates documents in the outer loop, so an `f16` corpus is
//! never materialized as `f32` all at once.

use crate::types::Multivector;

/// Rows widened to `f32`, ready for the dot products.
fn rows(multivector: &Multivector) -> Vec<Vec<f32>> {
    multivector.to_f32()
}

fn score_against(query: &[Vec<f32>], document: &[Vec<f32>]) -> f32 {
    query
        .iter()
        .map(|query_token| {
            document
                .iter()
                .map(|document_token| dot(query_token, document_token))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        // A document with no tokens scores zero rather than negative infinity.
        .map(|best| if best.is_finite() { best } else { 0.0 })
        .sum()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Score one query against each document, in the order given.
///
/// The result is not sorted; ranking is the caller's decision.
pub fn maxsim(query: &Multivector, documents: &[Multivector]) -> Vec<f32> {
    let query = rows(query);
    documents
        .iter()
        // Widened per document, so only one document is held at f32 at a time.
        .map(|document| score_against(&query, &rows(document)))
        .collect()
}

/// Score every query against every document.
///
/// Returns one row per query, each holding one score per document, so
/// `maxsim_batch(queries, docs)[i][j] == maxsim(&queries[i], docs)[j]`.
pub fn maxsim_batch(queries: &[Multivector], documents: &[Multivector]) -> Vec<Vec<f32>> {
    let queries: Vec<Vec<Vec<f32>>> = queries.iter().map(rows).collect();
    let mut scores = vec![vec![0.0f32; documents.len()]; queries.len()];

    // Documents outer, queries inner: the corpus is the large side, and this widens one
    // document at a time rather than the whole corpus.
    for (document_index, document) in documents.iter().enumerate() {
        let document = rows(document);
        for (query_index, query) in queries.iter().enumerate() {
            scores[query_index][document_index] = score_against(query, &document);
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;
    use half::f16;

    fn f32_mv(rows: &[&[f32]]) -> Multivector {
        Multivector::F32(rows.iter().map(|row| row.to_vec()).collect())
    }

    fn f16_mv(rows: &[&[f32]]) -> Multivector {
        Multivector::F16(
            rows.iter()
                .map(|row| row.iter().map(|v| f16::from_f32(*v)).collect())
                .collect(),
        )
    }

    #[test]
    fn identical_orthonormal_matrices_score_the_query_length() {
        let query = f32_mv(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let scores = maxsim(&query, std::slice::from_ref(&query));
        assert_eq!(scores.len(), 1);
        approx::assert_relative_eq!(scores[0], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn scores_rank_by_similarity() {
        let query = f32_mv(&[&[1.0, 0.0]]);
        let documents = [
            f32_mv(&[&[1.0, 0.0]]),                           // identical
            f32_mv(&[&[std::f32::consts::FRAC_1_SQRT_2; 2]]), // 45 degrees
            f32_mv(&[&[0.0, 1.0]]),                           // orthogonal
        ];
        let scores = maxsim(&query, &documents);
        assert!(scores[0] > scores[1] && scores[1] > scores[2]);
        approx::assert_relative_eq!(scores[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn max_is_over_document_tokens_and_sum_is_over_query_tokens() {
        // Each query token matches a different document token exactly.
        let query = f32_mv(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let document = f32_mv(&[&[0.0, 1.0], &[1.0, 0.0], &[0.0, 0.0]]);
        approx::assert_relative_eq!(maxsim(&query, &[document])[0], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn f16_inputs_score_exactly_as_their_widened_selves() {
        let query = f16_mv(&[&[0.3, -0.7], &[0.1, 0.9]]);
        let documents = [f16_mv(&[&[0.2, 0.5], &[-0.4, 0.8]]), f16_mv(&[&[1.0, 0.0]])];

        let widened_query = Multivector::F32(query.to_f32());
        let widened_documents: Vec<Multivector> = documents
            .iter()
            .map(|d| Multivector::F32(d.to_f32()))
            .collect();

        // Bit-exact, not approximately equal: widening must happen before the dot product,
        // never after partial accumulation at f16.
        assert_eq!(
            maxsim(&query, &documents),
            maxsim(&widened_query, &widened_documents)
        );
    }

    #[test]
    fn batch_agrees_with_the_single_query_form() {
        let queries = [f32_mv(&[&[1.0, 0.0]]), f32_mv(&[&[0.0, 1.0], &[1.0, 0.0]])];
        let documents = [f32_mv(&[&[1.0, 0.0]]), f32_mv(&[&[0.6, 0.8]])];

        let batch = maxsim_batch(&queries, &documents);
        assert_eq!(batch.len(), 2);
        for (index, query) in queries.iter().enumerate() {
            let single = maxsim(query, &documents);
            for (document_index, score) in single.iter().enumerate() {
                approx::assert_relative_eq!(batch[index][document_index], score, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn variable_token_counts_all_produce_finite_scores() {
        let query = f32_mv(&[&[1.0, 0.0]]);
        for count in [1usize, 10, 100] {
            let document = Multivector::F32(
                (0..count)
                    .map(|i| vec![i as f32 / count as f32, 0.5])
                    .collect(),
            );
            assert!(maxsim(&query, &[document])[0].is_finite());
        }
    }

    #[test]
    fn empty_inputs_are_handled_rather_than_producing_infinities() {
        let query = f32_mv(&[&[1.0, 0.0]]);
        assert!(maxsim(&query, &[]).is_empty());
        assert_eq!(maxsim(&query, &[Multivector::F32(Vec::new())]), vec![0.0]);
        assert_eq!(maxsim(&Multivector::F32(Vec::new()), &[query]), vec![0.0]);
        assert!(maxsim_batch(&[], &[]).is_empty());
    }
}
