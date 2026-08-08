//! Inspect cluster capacity and react to a server that is still scaling up.
//!
//! ```text
//! SIE_BASE_URL=http://localhost:8080 cargo run --example capacity
//! ```

use std::time::Duration;

use sie_sdk::{Client, Item};

#[tokio::main]
async fn main() -> sie_sdk::Result<()> {
    let base_url =
        std::env::var("SIE_BASE_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let client = Client::new(&base_url)?;

    let capacity = client.get_capacity(None).await?;
    println!(
        "{}: {} workers, {} GPUs, {} models loaded",
        capacity.status, capacity.worker_count, capacity.gpu_count, capacity.models_loaded
    );

    // Fail fast rather than waiting out a scale-up.
    let probe = client
        .encode("BAAI/bge-m3", [Item::text("probe")])
        .wait_for_capacity(false)
        .send_one()
        .await;

    match probe {
        Ok(result) => {
            let metadata = result.request.unwrap_or_default();
            println!(
                "ready; request {:?} after {} retries",
                metadata.id, metadata.retries
            );
        }
        Err(error) if error.is_capacity_error() => {
            println!("still provisioning: {error}");
            let capacity = client
                .wait_for_capacity(
                    "l4",
                    Some("BAAI/bge-m3"),
                    Some(Duration::from_secs(300)),
                    Duration::from_secs(5),
                )
                .await?;
            println!("warm: {} workers", capacity.worker_count);
        }
        Err(error) => return Err(error),
    }

    Ok(())
}
