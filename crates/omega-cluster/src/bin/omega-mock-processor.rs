use std::env;
use std::net::SocketAddr;

use omega_cluster::processor::MockProcessor;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let processor = env::var("OMEGA_PROCESSOR").unwrap_or_else(|_| "attention".to_string());
    let port: u16 = env::var("OMEGA_PROCESSOR_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_102);
    let service = match processor.as_str() {
        "executive" => MockProcessor::executive(),
        "attention" => MockProcessor::attention(),
        "memory" => MockProcessor::memory(),
        other => anyhow::bail!("unknown mock processor: {other}"),
    };
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    service.serve(addr).await
}
