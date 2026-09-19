use std::env;
use std::path::PathBuf;

use omega_cluster::registry::{ProcessorClient, ProcessorRegistry};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let registry_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config/omega-processors.local.json"));
    let registry = ProcessorRegistry::from_json_file(&registry_path)?;
    let client = ProcessorClient::new(registry.clone());

    println!("registry: {}", registry_path.display());
    for processor in registry.processors.keys() {
        match client.health(processor).await {
            Ok(health) => {
                print!("{:<10} health={}", processor, health.status);
                match client.capabilities(processor).await {
                    Ok(capabilities) => {
                        println!(" operations={:?}", capabilities.operations);
                    }
                    Err(err) => {
                        println!(" capabilities_error={err}");
                    }
                }
            }
            Err(err) => {
                println!("{:<10} unavailable error={err}", processor);
            }
        }
    }

    Ok(())
}
