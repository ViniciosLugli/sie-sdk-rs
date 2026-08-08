# sie-sdk-rs

Rust client for the [SIE](https://github.com/superlinked/sie) inference server.

Embeddings, reranking, extraction and text generation, with the server's capacity
signals handled for you.

```toml
[dependencies]
sie-sdk-rs = "0.1"
```

## Quick start

```rust,no_run
use sie_sdk::{Client, Item, OutputType};

#[tokio::main]
async fn main() -> sie_sdk::Result<()> {
    let client = Client::new("http://localhost:8080")?;

    let result = client
        .encode("BAAI/bge-m3", [Item::text("Hello world")])
        .output_types([OutputType::Dense, OutputType::Sparse])
        .send_one()
        .await?;

    println!("{:?}", result.require_dense()?);
    Ok(())
}
```

## What it covers

| Area | API |
|---|---|
| Embeddings | `encode` |
| Reranking | `score` |
| Extraction | `extract` |
| Generation | `generate`, `chat`, `responses`, and `.stream()` on each |
| Cost | `estimate`, or `.estimate()` on a generation builder |
| Files & batches | `files()`, `batches()` |
| Batch jobs | `jobs()`, inline or connector-driven |
| Capacity | `pools()`, `get_capacity`, `wait_for_capacity` |
| Cluster status | `watch()` over WebSocket |
| Connections | `connections()` on the control plane |
| Local scoring | `maxsim`, `maxsim_batch` for late-interaction models |

## Waiting for capacity

SIE scales from zero and loads models on demand, so a request can legitimately be
answered with "not yet". The client absorbs that: `PROVISIONING`, `MODEL_LOADING`,
`LORA_LOADING` and `RESOURCE_EXHAUSTED` are retried inside a wall-clock budget
(15 minutes by default) with bounded, jittered backoff. Generation is not idempotent,
so it never replays a request that may already have reached a worker.

Opt out per call when a failure is more useful than a wait:

```rust,no_run
# use sie_sdk::{Client, Item};
# async fn example(client: Client) -> sie_sdk::Result<()> {
let result = client
    .encode("BAAI/bge-m3", [Item::text("Hello")])
    .wait_for_capacity(false)
    .max_oom_retries(0)
    .send_one()
    .await;

if let Err(error) = &result
    && error.is_capacity_error()
{
    // Fall back to a smaller model, or shed the request.
}
# Ok(())
# }
```

Every call also reports what it cost and how hard it worked, through
`result.request`: the request id, billed units, credits, and the number of retries.

## Streaming

```rust,no_run
# use futures_util::StreamExt;
# use sie_sdk::{Client, types::ChatMessage};
# async fn example(client: Client) -> sie_sdk::Result<()> {
let mut stream = client.chat("qwen3", [ChatMessage::user("Write a haiku")]).stream()?;

while let Some(chunk) = stream.next().await {
    if let Some(delta) = chunk?.delta() {
        print!("{delta}");
    }
}
# Ok(())
# }
```

## Features

| Feature | Default | Effect |
|---|---|---|
| `rustls-tls` | yes | TLS through rustls |
| `native-tls` | no | TLS through the platform library |
| `watch` | yes | `Client::watch` over WebSocket |
| `blocking` | no | `sie_sdk::blocking::Client`, for non-async callers |
| `ndarray` | no | Conversions into `ndarray` arrays |

The blocking client owns a runtime and runs the async one:

```rust,no_run
use sie_sdk::{Item, blocking::Client};

# fn main() -> sie_sdk::Result<()> {
let client = Client::new("http://localhost:8080")?;
let result = client.call(|sie| sie.encode("BAAI/bge-m3", [Item::text("Hello")]).send_one())?;
# Ok(())
# }
```

## Testing

`cargo test` runs unit tests plus an HTTP suite against a mock server. There is also a
suite that runs against a real deployment, ignored by default:

```bash
docker run -d -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cpu-default

SIE_BASE_URL=http://localhost:8080 cargo test --test live -- --ignored --test-threads=1
```

It discovers models from the server's own catalogue by capability, so it adapts to whatever
bundle a deployment serves; anything unavailable skips itself and says why.

## Relationship to the Python SDK

Feature parity with `sie_sdk` on the client side, designed for Rust rather than
transliterated:

- One async implementation, with `blocking` as a thin facade, instead of two mirrored clients.
- One `Error` enum with predicates in place of an exception hierarchy.
- Retry counts and model revisions arrive in `RequestMetadata` rather than in thread-local state.
- `LORA_LOADING` retries are clamped to the caller's budget, and a stream reconnect honours
  the connection's `Retry-After`; the Python SDK does neither.
- Metadata endpoints (`list_models`, `get_model`, `/health`) are single-shot rather than
  sitting inside the provisioning budget.
- The server-side helpers bundled into the Python package (object-store backends, the
  HuggingFace weight cache, bundle matching) are out of scope for a client.

## License

Apache-2.0
