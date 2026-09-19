use std::env;
use std::path::PathBuf;

use omega_cluster::registry::ProcessorRegistry;
use omega_cluster::runtime::{OmegaRuntime, spawn_local_mock_cluster};
use omega_cluster::store::ClusterStore;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let base_port = env_u16("OMEGA_CLUSTER_BASE_PORT").unwrap_or(9_101);
    let spawn_mocks = env_bool("OMEGA_SPAWN_MOCKS").unwrap_or(true);
    let registry_path = env::var("OMEGA_REGISTRY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("config/omega-processors.local.json"));
    let state_dir = env::var("OMEGA_CLUSTER_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("dist/omega-cluster"));
    let input = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    let input = if input.is_empty() {
        "Omega runtime workbench event".to_string()
    } else {
        input
    };

    if spawn_mocks {
        spawn_local_mock_cluster(base_port).await;
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }

    let registry = if registry_path.exists() {
        ProcessorRegistry::from_json_file(&registry_path)?
    } else {
        let registry = ProcessorRegistry::localhost(base_port);
        if let Some(parent) = registry_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        registry.write_json_file(&registry_path)?;
        registry
    };

    let store = ClusterStore::open(&state_dir)?;
    let mut runtime = OmegaRuntime::with_store(registry, store.clone())?;
    runtime.ingest_observation(input)?;
    let responses = runtime.run_until_wait(16).await?;

    println!(
        "Omega runtime reached {:?}.",
        responses.last().map(|r| r.result.operation)
    );
    for response in responses {
        println!(
            "{} -> {:?} target={:?} confidence={:.2}",
            response.processor,
            response.result.operation,
            response.result.target_processor,
            response.confidence
        );
    }
    println!("state: {}", store.root().join("state.json").display());
    println!("trace: {}", store.root().join("events.jsonl").display());

    Ok(())
}

fn env_u16(name: &str) -> Option<u16> {
    env::var(name).ok()?.parse().ok()
}

fn env_bool(name: &str) -> Option<bool> {
    match env::var(name).ok()?.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
