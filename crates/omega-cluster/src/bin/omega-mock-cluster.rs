use std::env;

use omega_cluster::runtime::spawn_local_mock_cluster;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let base_port = env::var("OMEGA_CLUSTER_BASE_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_101);
    spawn_local_mock_cluster(base_port).await;

    println!(
        "mock cluster listening: executive={}, attention={}, memory={}",
        base_port,
        base_port + 1,
        base_port + 2
    );
    println!("Press Ctrl-C to stop.");
    tokio::signal::ctrl_c().await?;
    Ok(())
}
