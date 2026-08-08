//! Integration tests against a running SIE server.
//!
//! Ignored by default, because they need a server and download model weights.
//!
//! ```text
//! docker run -d -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface \
//!   ghcr.io/superlinked/sie-server:latest-cpu-default
//!
//! SIE_BASE_URL=http://localhost:8080 cargo test --test live -- --ignored --test-threads=1
//! ```
//!
//! Models are discovered from the server's own catalogue by capability, so the suite runs
//! against whatever bundle a deployment happens to serve. A test whose capability is not
//! advertised, or whose endpoint the deployment does not expose, skips itself and says so.
//! Override any choice with `SIE_MODEL_DENSE`, `_SPARSE`, `_MULTIVECTOR`, `_SCORE`,
//! `_GENERATE`.

use std::sync::OnceLock;
use std::time::Duration;

use futures_util::StreamExt;
use sie_sdk::types::ChatMessage;
use sie_sdk::{Client, Item, ModelInfo, OutputType};

fn client() -> Client {
    let base_url =
        std::env::var("SIE_BASE_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let mut builder = Client::builder(base_url)
        .timeout(Duration::from_secs(300))
        .provision_timeout(Duration::from_secs(900));
    if let Ok(api_key) = std::env::var("SIE_API_KEY") {
        builder = builder.api_key(api_key);
    }
    builder.build().expect("SIE_BASE_URL must be a valid URL")
}

/// The catalogue, fetched once and shared by every test in the process.
async fn catalogue() -> &'static Vec<ModelInfo> {
    static CATALOGUE: OnceLock<Vec<ModelInfo>> = OnceLock::new();
    if let Some(models) = CATALOGUE.get() {
        return models;
    }
    let models = client()
        .list_models()
        .await
        .expect("the server must answer /v1/models");
    let _ = CATALOGUE.set(models);
    CATALOGUE.get().expect("catalogue was just set")
}

/// Pick the smallest model advertising an output, preferring any named in `preferred`.
///
/// "Smallest" is approximated by name length, which reliably favours a base model over its
/// profile variants (`model:profile`) and keeps the choice stable across runs.
async fn model_for(output: &str, env_override: &str, preferred: &[&str]) -> Option<String> {
    if let Ok(name) = std::env::var(env_override) {
        return Some(name);
    }
    let models = catalogue().await;
    let advertises = |model: &ModelInfo| model.outputs.iter().any(|kind| kind == output);

    for candidate in preferred {
        if let Some(model) = models
            .iter()
            .find(|model| model.name == *candidate && advertises(model))
        {
            return Some(model.name.clone());
        }
    }
    models
        .iter()
        .filter(|model| advertises(model) && !model.name.contains(':'))
        .min_by_key(|model| model.name.len())
        .map(|model| model.name.clone())
}

async fn dense_model() -> Option<String> {
    model_for(
        "dense",
        "SIE_MODEL_DENSE",
        &[
            "ibm-granite/granite-embedding-small-english-r2",
            "BAAI/bge-m3",
        ],
    )
    .await
}

async fn sparse_model() -> Option<String> {
    model_for(
        "sparse",
        "SIE_MODEL_SPARSE",
        &["ibm-granite/granite-embedding-30m-sparse", "BAAI/bge-m3"],
    )
    .await
}

async fn multivector_model() -> Option<String> {
    model_for(
        "multivector",
        "SIE_MODEL_MULTIVECTOR",
        &["mixedbread-ai/mxbai-edge-colbert-v0-32m", "BAAI/bge-m3"],
    )
    .await
}

async fn score_model() -> Option<String> {
    model_for(
        "score",
        "SIE_MODEL_SCORE",
        &[
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "BAAI/bge-reranker-v2-m3",
        ],
    )
    .await
}

async fn generate_model() -> Option<String> {
    model_for("text", "SIE_MODEL_GENERATE", &["Qwen/Qwen3-0.6B"]).await
}

/// Bind a model for a capability, or skip the test when the deployment has none.
macro_rules! model {
    ($chooser:ident, $capability:literal) => {
        match $chooser().await {
            Some(model) => model,
            None => {
                eprintln!("skipping: no model advertises {}", $capability);
                return;
            }
        }
    };
}

/// Skip when the deployment does not expose an endpoint at all.
macro_rules! skip_if_absent {
    ($result:expr, $what:literal) => {
        match $result {
            Ok(value) => value,
            Err(error) if error.status() == Some(404) => {
                eprintln!("skipping: this deployment has no {}", $what);
                return;
            }
            Err(error) => panic!("{} failed: {error}", $what),
        }
    };
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn lists_models_and_reports_health() {
    let client = client();

    let models = catalogue().await;
    assert!(!models.is_empty(), "server advertises no models");
    for model in models.iter().take(3) {
        println!(
            "{:<58} outputs={:?} dims={:?}",
            model.name, model.outputs, model.dims
        );
    }

    // The catalogue must decode fully, including the null-valued fields the server sends.
    assert!(models.iter().all(|model| !model.name.is_empty()));

    let health = client.health().await.unwrap();
    assert!(!health.status.is_empty());
    assert!(
        health.kind == "gateway" || health.kind == "worker",
        "unexpected endpoint type {:?}",
        health.kind
    );
    println!("health: {} ({})", health.status, health.kind);

    let first = &models[0].name;
    assert_eq!(&client.get_model(first).await.unwrap().name, first);
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn encodes_a_dense_embedding() {
    let client = client();
    let model = model!(dense_model, "dense");
    println!("dense model: {model}");

    let result = client
        .encode(&model, [Item::text("Hello world").with_id("doc-1")])
        .output_types([OutputType::Dense])
        .send_one()
        .await
        .unwrap();

    assert_eq!(result.id.as_deref(), Some("doc-1"));
    assert_eq!(result.model.as_deref(), Some(model.as_str()));

    let dense = result.require_dense().unwrap();
    let declared = client
        .get_model(&model)
        .await
        .unwrap()
        .dims
        .dense
        .expect("a dense model declares its dimensionality");
    assert_eq!(
        dense.len(),
        declared as usize,
        "embedding width disagrees with the catalogue"
    );
    assert!(dense.iter().all(|value| value.is_finite()));
    assert!(
        dense.iter().any(|value| *value != 0.0),
        "the embedding is all zeros"
    );

    println!("request metadata: {:?}", result.request);
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn encodes_a_sparse_embedding() {
    let client = client();
    let model = model!(sparse_model, "sparse");
    println!("sparse model: {model}");

    let result = client
        .encode(&model, [Item::text("vector databases store embeddings")])
        .output_types([OutputType::Sparse])
        .send_one()
        .await
        .unwrap();

    let sparse = result.sparse.as_ref().expect("sparse output was requested");
    assert!(!sparse.is_empty(), "the sparse vector carries no terms");
    assert_eq!(
        sparse.indices.len(),
        sparse.values.len(),
        "indices and values disagree in length"
    );
    assert!(sparse.values.iter().all(|weight| weight.is_finite()));
    assert_eq!(
        result.sparse_map().len(),
        sparse.indices.len(),
        "the term map lost entries, so indices were not unique"
    );
    println!("sparse terms: {}", sparse.len());
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn encodes_a_multivector() {
    let client = client();
    let model = model!(multivector_model, "multivector");
    println!("multivector model: {model}");

    let result = client
        .encode(&model, [Item::text("late interaction retrieval")])
        .output_types([OutputType::Multivector])
        .send_one()
        .await
        .unwrap();

    let multivector = result
        .multivector
        .as_ref()
        .expect("multivector was requested");
    assert!(!multivector.is_empty(), "no token vectors were returned");

    let declared = client.get_model(&model).await.unwrap().dims.multivector;
    if let Some(declared) = declared {
        assert_eq!(multivector.dims(), declared as usize);
    }
    // Every row must be the same width, or the decoder mis-sliced the buffer.
    let rows = multivector.to_f32();
    assert!(rows.iter().all(|row| row.len() == multivector.dims()));
    println!(
        "multivector: {} tokens x {} dims",
        multivector.len(),
        multivector.dims()
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn encodes_a_batch_and_keeps_the_order() {
    let client = client();
    let model = model!(dense_model, "dense");

    let items = ["first", "second", "third"]
        .into_iter()
        .enumerate()
        .map(|(index, text)| Item::text(text).with_id(index.to_string()));
    let results = client.encode(&model, items).send().await.unwrap();

    assert_eq!(results.len(), 3, "one result per input item");
    for (index, result) in results.iter().enumerate() {
        assert_eq!(result.id.as_deref(), Some(index.to_string().as_str()));
        assert!(!result.require_dense().unwrap().is_empty());
    }

    // Identical text must embed identically; unrelated text must not.
    let repeat = client
        .encode(&model, [Item::text("first")])
        .send_one()
        .await
        .unwrap();
    let same = cosine(
        results[0].require_dense().unwrap(),
        repeat.require_dense().unwrap(),
    );
    let different = cosine(
        results[0].require_dense().unwrap(),
        results[1].require_dense().unwrap(),
    );
    assert!(same > 0.999, "identical text drifted: {same}");
    assert!(
        different < same,
        "unrelated text scored as high as identical text"
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn encodes_with_an_instruction_and_a_query_flag() {
    let client = client();
    let model = model!(dense_model, "dense");

    // These ride in different places on the wire: `instruction` in `params`, `is_query`
    // folded into `params.options`. Both must be accepted.
    let result = client
        .encode(&model, [Item::text("What is a vector database?")])
        .instruction("Represent this query for retrieval:")
        .is_query(true)
        .send_one()
        .await
        .unwrap();
    assert!(!result.require_dense().unwrap().is_empty());
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn quantizes_the_output_dtype() {
    let client = client();
    let model = model!(dense_model, "dense");

    let float = client
        .encode(&model, [Item::text("quantize me")])
        .send_one()
        .await
        .unwrap();
    let quantized = match client
        .encode(&model, [Item::text("quantize me")])
        .output_dtype(sie_sdk::OutputDType::Int8)
        .send_one()
        .await
    {
        Ok(result) => result,
        Err(error) if error.is_request_error() => {
            eprintln!("skipping: {model} does not support int8 output ({error})");
            return;
        }
        Err(error) => panic!("int8 encode failed: {error}"),
    };

    let values = quantized.require_dense().unwrap();
    assert_eq!(values.len(), float.require_dense().unwrap().len());
    assert!(
        values.iter().all(|value| (-128.0..=127.0).contains(value)),
        "int8 values fell outside the signed byte range"
    );
    assert!(
        values.iter().all(|value| value.fract() == 0.0),
        "int8 values decoded with a fractional part"
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn scores_and_ranks_candidates() {
    let client = client();
    let model = model!(score_model, "score");
    println!("score model: {model}");

    let result = client
        .score(
            &model,
            Item::text("What is a vector database?"),
            [
                Item::text("A vector database stores and searches embeddings.").with_id("relevant"),
                Item::text("Bananas are a yellow fruit.").with_id("irrelevant"),
            ],
        )
        .send()
        .await
        .unwrap();

    assert_eq!(result.model, model);
    assert_eq!(result.scores.len(), 2, "one score per candidate");

    let best = result
        .scores
        .iter()
        .min_by_key(|entry| entry.rank)
        .expect("scores are non-empty");
    assert_eq!(best.rank, 0, "ranks are zero-based");
    assert_eq!(
        best.item_id, "relevant",
        "the reranker put the wrong candidate first"
    );
    // Ranks must be a permutation of 0..n, or the ordering is not usable.
    let mut ranks: Vec<u32> = result.scores.iter().map(|entry| entry.rank).collect();
    ranks.sort_unstable();
    assert_eq!(ranks, vec![0, 1]);
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn generates_text() {
    let client = client();
    let model = model!(generate_model, "text");

    let result = client
        .generate(&model, "Reply with exactly one word: hello", 32)
        .temperature(0.0)
        .send()
        .await
        .unwrap();

    assert_eq!(result.model, model);
    assert!(!result.text.is_empty());
    if let Some(usage) = result.usage {
        assert!(usage.completion_tokens > 0);
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn chats_buffered_and_streamed() {
    let client = client();
    let model = model!(generate_model, "text");

    let buffered = client
        .chat(&model, [ChatMessage::user("Say hello in one word.")])
        .max_completion_tokens(32)
        .temperature(0.0)
        .send()
        .await
        .unwrap();
    assert!(buffered.text().is_some_and(|text| !text.is_empty()));

    let mut stream = client
        .chat(&model, [ChatMessage::user("Say hello in one word.")])
        .max_completion_tokens(32)
        .temperature(0.0)
        .stream()
        .unwrap();

    let mut streamed = String::new();
    let mut chunks = 0;
    while let Some(chunk) = stream.next().await {
        if let Some(delta) = chunk.unwrap().delta() {
            streamed.push_str(delta);
        }
        chunks += 1;
    }
    assert!(chunks > 0, "the stream produced no chunks");
    assert!(!streamed.is_empty(), "the stream produced no text");
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn streams_the_native_generate_endpoint() {
    let client = client();
    let model = model!(generate_model, "text");

    let mut stream = client
        .generate(&model, "Count to three.", 48)
        .temperature(0.0)
        .stream()
        .unwrap();

    let mut text = String::new();
    let mut saw_terminal = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(delta) = &chunk.text_delta {
            text.push_str(delta);
        }
        saw_terminal |= chunk.done;
    }
    assert!(!text.is_empty(), "the stream produced no text");
    assert!(saw_terminal, "the stream never reported a terminal chunk");
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn extracts_structure() {
    let client = client();
    // Prefer a span extractor: several `json` models are document converters or VQA
    // heads, which answer a labelled-entity request with nothing at all.
    let model = match model_for(
        "json",
        "SIE_MODEL_EXTRACT",
        &[
            "gliner-community/gliner_small-v2.5",
            "numind/NuNER_Zero-span",
            "urchade/gliner_multi_pii-v1",
        ],
    )
    .await
    {
        Some(model) => model,
        None => {
            eprintln!("skipping: no model advertises json output");
            return;
        }
    };
    println!("extract model: {model}");

    let result = client
        .extract(&model, [Item::text("Ada Lovelace worked in London.")])
        .labels(["person", "location"])
        .send_one()
        .await;

    match result {
        Ok(result) => {
            println!(
                "entities={:?} classifications={} data={:?}",
                result
                    .entities
                    .iter()
                    .map(|entity| (&entity.text, &entity.label))
                    .collect::<Vec<_>>(),
                result.classifications.len(),
                result.data
            );
            // A per-item failure must never be half-formed.
            if let Some(error) = &result.error {
                assert!(!error.code.is_empty() && !error.message.is_empty());
            }
            // Every entity the model did return must be fully populated.
            for entity in &result.entities {
                assert!(!entity.text.is_empty(), "an entity carries no text");
                assert!(!entity.label.is_empty(), "an entity carries no label");
                assert!(entity.score.is_finite());
            }
        }
        Err(error) if error.is_request_error() => {
            eprintln!("skipping: {model} rejected this extraction shape ({error})");
        }
        Err(error) => panic!("extract failed: {error}"),
    }
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn reports_capacity_for_a_gateway() {
    let client = client();
    let health = client.health().await.unwrap();
    if health.kind != "gateway" {
        eprintln!(
            "skipping: capacity needs a gateway, this is a {}",
            health.kind
        );
        return;
    }

    let capacity = client.get_capacity(None).await.unwrap();
    assert!(!capacity.status.is_empty());
    println!(
        "{}: {} workers, {} GPUs, gpu types {:?}",
        capacity.status, capacity.worker_count, capacity.gpu_count, capacity.live_gpu_types
    );

    // A GPU type nothing runs on filters to nothing rather than erroring.
    let empty = client.get_capacity(Some("no-such-gpu")).await.unwrap();
    assert_eq!(empty.worker_count, 0);
    assert!(empty.workers.is_empty());
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn uploads_lists_reads_and_deletes_a_file() {
    let client = client();
    let body = b"{\"custom_id\":\"1\",\"body\":{\"input\":\"hello\"}}\n".to_vec();

    let uploaded = skip_if_absent!(
        client
            .files()
            .upload(body.clone())
            .filename("sie-sdk-rs-live.jsonl")
            .send()
            .await,
        "file store"
    );
    assert!(!uploaded.id.is_empty());
    assert_eq!(uploaded.filename, "sie-sdk-rs-live.jsonl");

    assert_eq!(
        client.files().retrieve(&uploaded.id).await.unwrap().id,
        uploaded.id
    );
    assert!(
        client
            .files()
            .list()
            .limit(20)
            .send()
            .await
            .unwrap()
            .iter()
            .any(|file| file.id == uploaded.id)
    );
    assert_eq!(
        client.files().content(&uploaded.id).await.unwrap().as_ref(),
        body.as_slice(),
        "round-tripped bytes differ"
    );
    assert!(client.files().delete(&uploaded.id).await.unwrap().deleted);
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn runs_an_inline_job_end_to_end() {
    let client = client();
    let model = model!(dense_model, "dense");

    let submitted = skip_if_absent!(
        client
            .jobs()
            .submit(sie_sdk::client::JobSource::items(["alpha", "beta"]), &model)
            .send()
            .await,
        "job queue"
    );
    assert!(!submitted.id.is_empty());

    let settled = client
        .jobs()
        .wait(
            &submitted.id,
            Duration::from_secs(600),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(
        settled.state,
        Some(sie_sdk::types::JobState::Succeeded),
        "job ended as {:?}",
        settled.state
    );

    let results = client.jobs().results(&submitted.id).await.unwrap();
    assert_eq!(results.items.len(), 2, "one result per input item");
    for item in &results.items {
        assert_eq!(item.success, Some(true));
        assert!(item.dense.as_ref().is_some_and(|values| !values.is_empty()));
    }
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn estimates_a_request() {
    let client = client();
    let model = model!(generate_model, "text");

    let estimate = skip_if_absent!(
        client
            .chat(&model, [ChatMessage::user("hello")])
            .max_completion_tokens(16)
            .estimate()
            .await,
        "cost estimation"
    );
    println!(
        "{} credits, basis {:?}, book {}",
        estimate.estimated_credits, estimate.estimate_basis, estimate.rate_book_version
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn surfaces_a_missing_model_as_a_typed_error() {
    let client = client();
    let error = client
        .encode("definitely/not-a-real-model", [Item::text("hi")])
        .wait_for_capacity(false)
        .send_one()
        .await
        .unwrap_err();

    println!("{error:?}");
    assert!(
        error.is_request_error(),
        "a missing model is a 4xx, got {error:?}"
    );
    assert_eq!(error.status(), Some(404));
    assert!(
        error.code().is_some(),
        "the server named no error code: {error:?}"
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn rejects_an_empty_request_before_sending() {
    let client = client();
    // Client-side validation must fire without a round trip.
    assert!(client.encode("any", Vec::new()).send().await.is_err());
    assert!(
        client
            .score("any", Item::text("q"), Vec::new())
            .send()
            .await
            .is_err()
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn scores_multivectors_locally() {
    let client = client();
    let model = model!(multivector_model, "multivector");

    let results = client
        .encode(
            &model,
            [
                Item::text("What is a vector database?"),
                Item::text("A vector database stores and searches embeddings."),
                Item::text("Bananas are a yellow fruit."),
            ],
        )
        .output_types([OutputType::Multivector])
        .send()
        .await
        .unwrap();

    let query = results[0].multivector.clone().expect("multivector output");
    let documents: Vec<_> = results[1..]
        .iter()
        .map(|result| result.multivector.clone().expect("multivector output"))
        .collect();

    let scores = sie_sdk::maxsim(&query, &documents);
    assert_eq!(scores.len(), 2);
    assert!(
        scores.iter().all(|score| score.is_finite()),
        "MaxSim produced a non-finite score: {scores:?}"
    );
    assert!(
        scores[0] > scores[1],
        "MaxSim ranked the unrelated document higher: {scores:?}"
    );

    // The batch form must agree with the single-query form.
    let batch = sie_sdk::maxsim_batch(std::slice::from_ref(&query), &documents);
    assert_eq!(batch[0].len(), 2);
    for (single, batched) in scores.iter().zip(&batch[0]) {
        assert!(
            (single - batched).abs() < 1e-4,
            "batch disagrees with single: {single} vs {batched}"
        );
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm: f32 = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let right_norm: f32 = right.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (left_norm * right_norm)
}
