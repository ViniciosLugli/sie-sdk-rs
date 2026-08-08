//! Encode, score and stream a chat completion against a running SIE server.
//!
//! ```text
//! SIE_BASE_URL=http://localhost:8080 cargo run --example quickstart
//! ```

use futures_util::StreamExt;
use sie_sdk::types::ChatMessage;
use sie_sdk::{Client, Item, OutputType};

#[tokio::main]
async fn main() -> sie_sdk::Result<()> {
    let base_url =
        std::env::var("SIE_BASE_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let mut builder = Client::builder(&base_url);
    if let Ok(api_key) = std::env::var("SIE_API_KEY") {
        builder = builder.api_key(api_key);
    }
    let client = builder.build()?;

    for model in client.list_models().await? {
        println!("{:<40} loaded={}", model.name, model.loaded);
    }

    let embedding = client
        .encode("BAAI/bge-m3", [Item::text("Hello world")])
        .output_types([OutputType::Dense, OutputType::Sparse])
        .send_one()
        .await?;
    println!("dense dims: {}", embedding.require_dense()?.len());
    println!("sparse terms: {}", embedding.sparse_map().len());

    let ranked = client
        .score(
            "BAAI/bge-reranker-v2-m3",
            Item::text("what is a vector database?"),
            [
                Item::text("A vector database stores embeddings.").with_id("a"),
                Item::text("Bananas are yellow.").with_id("b"),
            ],
        )
        .send()
        .await?;
    for entry in &ranked.scores {
        println!("#{} {} {:.4}", entry.rank, entry.item_id, entry.score);
    }

    let mut stream = client
        .chat(
            "qwen3",
            [ChatMessage::user("Write one sentence about vectors.")],
        )
        .max_completion_tokens(64)
        .stream()?;
    while let Some(chunk) = stream.next().await {
        if let Some(delta) = chunk?.delta() {
            print!("{delta}");
        }
    }
    println!();

    Ok(())
}
