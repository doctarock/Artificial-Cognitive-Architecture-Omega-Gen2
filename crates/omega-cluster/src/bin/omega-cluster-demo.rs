use omega_cluster::registry::ProcessorRegistry;
use omega_cluster::runtime::{OmegaRuntime, spawn_local_mock_cluster};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let base_port = std::env::var("OMEGA_CLUSTER_BASE_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_101);

    spawn_local_mock_cluster(base_port).await;
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    let registry = ProcessorRegistry::localhost(base_port);
    println!(
        "processor registry:\n{}",
        serde_json::to_string_pretty(&registry.as_json())?
    );

    let mut runtime = OmegaRuntime::new(registry);
    runtime.ingest_observation("Milestone 1 demo cognitive event")?;
    let responses = runtime.run_until_wait(10).await?;

    println!("cognitive responses:");
    for response in responses {
        println!(
            "- {} -> {:?} target={:?} confidence={:.2}",
            response.processor,
            response.result.operation,
            response.result.target_processor,
            response.confidence
        );
    }
    println!(
        "final state revision={} focus={:?} working_memory_items={} events={}",
        runtime.state.revision,
        runtime.state.focus,
        runtime.state.working_memory.len(),
        runtime.event_log.len()
    );

    Ok(())
}
