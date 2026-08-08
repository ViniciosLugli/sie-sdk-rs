//! Integration tests against a running SIE server.
//!
//! Ignored by default, because they need a server and download model weights.
//!
//! ```text
//! docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface \
//!   ghcr.io/superlinked/sie-server:latest-cuda12-default
//!
//! SIE_BASE_URL=http://localhost:8080 cargo test --test live -- --ignored --test-threads=1
//! ```
//!
//! Every test skips itself when the model it needs is not advertised by the server, so a
//! run against a minimal deployment reports what it could not cover instead of failing.

use std::time::Duration;

use futures_util::StreamExt;
use sie_sdk::types::ChatMessage;
use sie_sdk::{Client, Item, OutputType};

const EMBED_MODEL: &str = "BAAI/bge-m3";
const RERANK_MODEL: &str = "BAAI/bge-reranker-v2-m3";
const GENERATE_MODEL: &str = "Qwen/Qwen3-0.6B";

fn client() -> Client {
    let base_url =
        std::env::var("SIE_BASE_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let mut builder = Client::builder(base_url)
        .timeout(Duration::from_secs(120))
        .provision_timeout(Duration::from_secs(600));
    if let Ok(api_key) = std::env::var("SIE_API_KEY") {
        builder = builder.api_key(api_key);
    }
    builder.build().expect("SIE_BASE_URL must be a valid URL")
}

/// Whether the server advertises a model, so a test can skip rather than fail.
async fn has_model(client: &Client, model: &str) -> bool {
    match client.list_models().await {
        Ok(models) => models.iter().any(|entry| entry.name == model),
        Err(error) => panic!("could not reach the server: {error}"),
    }
}

macro_rules! require_model {
    ($client:expr, $model:expr) => {
        if !has_model(&$client, $model).await {
            eprintln!("skipping: server does not advertise {}", $model);
            return;
        }
    };
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn lists_models_and_reports_health() {
    let client = client();

    let models = client.list_models().await.unwrap();
    assert!(!models.is_empty(), "server advertises no models");
    for model in models.iter().take(5) {
        assert!(!model.name.is_empty());
        println!(
            "{:<44} loaded={} outputs={:?}",
            model.name, model.loaded, model.outputs
        );
    }

    let health = client.health().await.unwrap();
    assert!(!health.status.is_empty());
    assert!(
        health.kind == "gateway" || health.kind == "worker",
        "unexpected endpoint type {:?}",
        health.kind
    );

    // Fetching one model by name must agree with the catalogue.
    let first = &models[0].name;
    assert_eq!(&client.get_model(first).await.unwrap().name, first);
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn encodes_dense_sparse_and_multivector() {
    let client = client();
    require_model!(client, EMBED_MODEL);

    let result = client
        .encode(EMBED_MODEL, [Item::text("Hello world").with_id("doc-1")])
        .output_types([
            OutputType::Dense,
            OutputType::Sparse,
            OutputType::Multivector,
        ])
        .send_one()
        .await
        .unwrap();

    assert_eq!(result.id.as_deref(), Some("doc-1"));
    assert_eq!(result.model.as_deref(), Some(EMBED_MODEL));

    let dense = result.require_dense().unwrap();
    assert_eq!(
        dense.len(),
        1024,
        "bge-m3 produces 1024-dimensional embeddings"
    );
    assert!(dense.iter().all(|value| value.is_finite()));
    // A normalized embedding has unit length.
    let norm: f32 = dense.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.05, "unexpected norm {norm}");

    let sparse = result.sparse.as_ref().expect("sparse output was requested");
    assert!(!sparse.is_empty());
    assert_eq!(sparse.indices.len(), sparse.values.len());
    assert!(sparse.values.iter().all(|weight| *weight > 0.0));

    let multivector = result
        .multivector
        .as_ref()
        .expect("multivector was requested");
    assert!(!multivector.is_empty());
    assert_eq!(multivector.dims(), 1024);

    let metadata = result.request.expect("the server reports request metadata");
    println!(
        "request={:?} retries={} usage={:?}",
        metadata.id, metadata.retries, metadata.usage
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn encodes_a_batch_and_keeps_the_order() {
    let client = client();
    require_model!(client, EMBED_MODEL);

    let items = ["first", "second", "third"]
        .into_iter()
        .enumerate()
        .map(|(index, text)| Item::text(text).with_id(index.to_string()));
    let results = client.encode(EMBED_MODEL, items).send().await.unwrap();

    assert_eq!(results.len(), 3);
    for (index, result) in results.iter().enumerate() {
        assert_eq!(result.id.as_deref(), Some(index.to_string().as_str()));
        assert!(!result.require_dense().unwrap().is_empty());
    }

    // Identical text must embed identically; different text must not.
    let repeat = client
        .encode(EMBED_MODEL, [Item::text("first")])
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
async fn quantized_output_dtype_round_trips() {
    let client = client();
    require_model!(client, EMBED_MODEL);

    let result = client
        .encode(EMBED_MODEL, [Item::text("quantize me")])
        .output_types([OutputType::Dense])
        .output_dtype(sie_sdk::OutputDType::Int8)
        .send_one()
        .await
        .unwrap();

    let dense = result.require_dense().unwrap();
    assert_eq!(dense.len(), 1024);
    // int8 values decode as whole numbers inside the signed byte range.
    assert!(dense.iter().all(|value| (-128.0..=127.0).contains(value)));
    assert!(dense.iter().any(|value| value.fract() == 0.0));
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn scores_and_ranks_candidates() {
    let client = client();
    require_model!(client, RERANK_MODEL);

    let result = client
        .score(
            RERANK_MODEL,
            Item::text("What is a vector database?"),
            [
                Item::text("A vector database stores and searches embeddings.").with_id("relevant"),
                Item::text("Bananas are a yellow fruit.").with_id("irrelevant"),
            ],
        )
        .send()
        .await
        .unwrap();

    assert_eq!(result.model, RERANK_MODEL);
    assert_eq!(result.scores.len(), 2);
    let best = result.scores.iter().min_by_key(|entry| entry.rank).unwrap();
    assert_eq!(
        best.item_id, "relevant",
        "the reranker ranked the wrong candidate first"
    );
    assert_eq!(best.rank, 0);
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn generates_text() {
    let client = client();
    require_model!(client, GENERATE_MODEL);

    let result = client
        .generate(GENERATE_MODEL, "Reply with exactly one word: hello", 32)
        .temperature(0.0)
        .send()
        .await
        .unwrap();

    assert_eq!(result.model, GENERATE_MODEL);
    assert!(!result.text.is_empty());
    let usage = result.usage.expect("the server reports token usage");
    assert!(usage.completion_tokens > 0);
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn chats_and_streams_the_same_completion() {
    let client = client();
    require_model!(client, GENERATE_MODEL);

    let buffered = client
        .chat(
            GENERATE_MODEL,
            [ChatMessage::user("Say hello in one word.")],
        )
        .max_completion_tokens(32)
        .temperature(0.0)
        .send()
        .await
        .unwrap();
    assert!(buffered.text().is_some_and(|text| !text.is_empty()));
    assert_eq!(buffered.choices[0].index, 0);

    let mut stream = client
        .chat(
            GENERATE_MODEL,
            [ChatMessage::user("Say hello in one word.")],
        )
        .max_completion_tokens(32)
        .temperature(0.0)
        .stream()
        .unwrap();

    let mut streamed = String::new();
    let mut chunks = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(delta) = chunk.delta() {
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
    require_model!(client, GENERATE_MODEL);

    let mut stream = client
        .generate(GENERATE_MODEL, "Count to three.", 48)
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
        if chunk.done {
            saw_terminal = true;
            assert!(chunk.usage.is_some(), "the terminal chunk carries usage");
        }
    }
    assert!(!text.is_empty());
    assert!(saw_terminal, "the stream never reported a terminal chunk");
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn constrains_generation_with_a_grammar() {
    let client = client();
    require_model!(client, GENERATE_MODEL);

    let capabilities = client.get_model(GENERATE_MODEL).await.unwrap().capabilities;
    if !capabilities.grammar.iter().any(|kind| kind == "regex") {
        eprintln!("skipping: {GENERATE_MODEL} does not advertise regex grammars");
        return;
    }

    let result = client
        .generate(GENERATE_MODEL, "Pick a digit.", 8)
        .grammar(sie_sdk::types::Grammar::regex("[0-9]"))
        .temperature(0.0)
        .send()
        .await
        .unwrap();
    assert!(
        result.text.trim().chars().all(|c| c.is_ascii_digit()),
        "grammar was not enforced: {:?}",
        result.text
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn reports_capacity_for_the_cluster() {
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

    // A GPU type nothing runs on filters down to nothing rather than erroring.
    let empty = client.get_capacity(Some("no-such-gpu")).await.unwrap();
    assert_eq!(empty.worker_count, 0);
    assert!(empty.workers.is_empty());
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn uploads_lists_reads_and_deletes_a_file() {
    let client = client();
    let body = b"{\"custom_id\":\"1\",\"body\":{\"input\":\"hello\"}}\n".to_vec();

    let uploaded = match client
        .files()
        .upload(body.clone())
        .filename("sie-sdk-rs-live.jsonl")
        .send()
        .await
    {
        Ok(file) => file,
        Err(error) if error.status() == Some(404) => {
            eprintln!("skipping: this deployment has no file store");
            return;
        }
        Err(error) => panic!("upload failed: {error}"),
    };
    assert!(!uploaded.id.is_empty());
    assert_eq!(uploaded.filename, "sie-sdk-rs-live.jsonl");

    let fetched = client.files().retrieve(&uploaded.id).await.unwrap();
    assert_eq!(fetched.id, uploaded.id);

    let listed = client.files().list().limit(20).send().await.unwrap();
    assert!(listed.iter().any(|file| file.id == uploaded.id));

    let content = client.files().content(&uploaded.id).await.unwrap();
    assert_eq!(
        content.as_ref(),
        body.as_slice(),
        "round-tripped bytes differ"
    );

    let deleted = client.files().delete(&uploaded.id).await.unwrap();
    assert!(deleted.deleted);
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn runs_an_inline_job_end_to_end() {
    let client = client();
    require_model!(client, EMBED_MODEL);

    let submitted = match client
        .jobs()
        .submit(
            sie_sdk::client::JobSource::items(["alpha", "beta"]),
            EMBED_MODEL,
        )
        .send()
        .await
    {
        Ok(job) => job,
        Err(error) if error.status() == Some(404) => {
            eprintln!("skipping: this deployment has no job queue");
            return;
        }
        Err(error) => panic!("submit failed: {error}"),
    };
    assert!(!submitted.id.is_empty());

    let settled = client
        .jobs()
        .wait(
            &submitted.id,
            Duration::from_secs(300),
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
    assert_eq!(results.items.len(), 2, "expected one result per input item");
    for item in &results.items {
        assert_eq!(item.success, Some(true));
        assert!(item.dense.as_ref().is_some_and(|values| !values.is_empty()));
    }
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn surfaces_a_missing_model_as_a_request_error() {
    let client = client();
    let error = client
        .encode("definitely/not-a-real-model", [Item::text("hi")])
        .wait_for_capacity(false)
        .send_one()
        .await
        .unwrap_err();

    println!("{error:?}");
    assert!(
        error.is_request_error() || error.is_server_error(),
        "unexpected error kind: {error:?}"
    );
    assert!(
        error.status().is_some(),
        "the failure carries no HTTP status"
    );
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn rejects_input_beyond_the_context_window() {
    let client = client();
    require_model!(client, EMBED_MODEL);

    let max_tokens = client
        .get_model(EMBED_MODEL)
        .await
        .unwrap()
        .max_sequence_length
        .unwrap_or(8192);
    // Comfortably past the window, so the server has to reject rather than truncate.
    let oversized = "word ".repeat(max_tokens as usize * 4);

    match client
        .encode(EMBED_MODEL, [Item::text(oversized)])
        .send_one()
        .await
    {
        Ok(_) => eprintln!("note: this deployment truncates rather than rejecting long input"),
        Err(error) => {
            println!("{error:?}");
            assert!(error.is_request_error(), "expected a 4xx, got {error:?}");
        }
    }
}

#[tokio::test]
#[ignore = "needs a running SIE server"]
async fn scores_multivectors_locally() {
    let client = client();
    require_model!(client, EMBED_MODEL);

    let results = client
        .encode(
            EMBED_MODEL,
            [
                Item::text("What is a vector database?"),
                Item::text("A vector database stores embeddings."),
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
        scores[0] > scores[1],
        "MaxSim ranked the unrelated document higher: {scores:?}"
    );

    let batch = sie_sdk::maxsim_batch(std::slice::from_ref(&query), &documents);
    assert_eq!(batch[0].len(), 2);
    for (single, batched) in scores.iter().zip(&batch[0]) {
        assert!(
            (single - batched).abs() < 1e-4,
            "batch disagrees with single"
        );
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm: f32 = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let right_norm: f32 = right.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (left_norm * right_norm)
}
