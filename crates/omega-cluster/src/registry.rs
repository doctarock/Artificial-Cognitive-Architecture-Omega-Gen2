use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::{CapabilitiesResponse, CognitiveRequest, CognitiveResponse, HealthResponse};
use crate::state::CognitiveState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorRegistry {
    pub processors: BTreeMap<String, ProcessorRegistration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorRegistration {
    pub endpoint: String,
    pub timeout_ms: u64,
    pub retries: u8,
    pub fallback_processor: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProcessorClient {
    http: Client,
    registry: ProcessorRegistry,
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("unknown processor: {0}")]
    UnknownProcessor(String),
    #[error("processor call failed for {processor}: {source}")]
    CallFailed {
        processor: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("registry config error: {0}")]
    Config(String),
}

impl ProcessorClient {
    pub fn new(registry: ProcessorRegistry) -> Self {
        Self {
            http: Client::new(),
            registry,
        }
    }

    pub async fn health(&self, processor: &str) -> Result<HealthResponse, RegistryError> {
        let registration = self.registration(processor)?;
        self.http
            .get(format!("{}/health", registration.endpoint))
            .timeout(Duration::from_millis(registration.timeout_ms))
            .send()
            .await
            .map_err(|source| RegistryError::CallFailed {
                processor: processor.to_string(),
                source,
            })?
            .json()
            .await
            .map_err(|source| RegistryError::CallFailed {
                processor: processor.to_string(),
                source,
            })
    }

    pub async fn capabilities(
        &self,
        processor: &str,
    ) -> Result<CapabilitiesResponse, RegistryError> {
        let registration = self.registration(processor)?;
        self.http
            .get(format!("{}/capabilities", registration.endpoint))
            .timeout(Duration::from_millis(registration.timeout_ms))
            .send()
            .await
            .map_err(|source| RegistryError::CallFailed {
                processor: processor.to_string(),
                source,
            })?
            .json()
            .await
            .map_err(|source| RegistryError::CallFailed {
                processor: processor.to_string(),
                source,
            })
    }

    pub async fn process(
        &self,
        processor: &str,
        state: &CognitiveState,
    ) -> Result<CognitiveResponse, RegistryError> {
        let registration = self.registration(processor)?;
        let request = CognitiveRequest::new(
            processor,
            state.state_id,
            serde_json::json!({ "state": state }),
        );
        self.process_request(registration, processor, request).await
    }

    async fn process_request(
        &self,
        registration: &ProcessorRegistration,
        processor: &str,
        request: CognitiveRequest,
    ) -> Result<CognitiveResponse, RegistryError> {
        let attempts = registration.retries.saturating_add(1);
        let mut last_error = None;
        for _ in 0..attempts {
            match self
                .http
                .post(format!("{}/process", registration.endpoint))
                .timeout(Duration::from_millis(registration.timeout_ms))
                .json(&request)
                .send()
                .await
            {
                Ok(response) => {
                    return response
                        .json()
                        .await
                        .map_err(|source| RegistryError::CallFailed {
                            processor: processor.to_string(),
                            source,
                        });
                }
                Err(source) => last_error = Some(source),
            }
        }
        Err(RegistryError::CallFailed {
            processor: processor.to_string(),
            source: last_error.expect("at least one attempt is always made"),
        })
    }

    fn registration(&self, processor: &str) -> Result<&ProcessorRegistration, RegistryError> {
        self.registry
            .processors
            .get(processor)
            .ok_or_else(|| RegistryError::UnknownProcessor(processor.to_string()))
    }
}

impl ProcessorRegistry {
    pub fn from_json_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn write_json_file(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        fs::write(path, text)?;
        Ok(())
    }

    pub fn localhost(base_port: u16) -> Self {
        let mut processors = BTreeMap::new();
        processors.insert("executive".to_string(), registration(base_port, 1_000));
        processors.insert("attention".to_string(), registration(base_port + 1, 500));
        processors.insert("memory".to_string(), registration(base_port + 2, 1_000));
        Self { processors }
    }

    pub fn as_json(&self) -> Value {
        serde_json::to_value(self).expect("registry serializes")
    }
}

fn registration(port: u16, timeout_ms: u64) -> ProcessorRegistration {
    ProcessorRegistration {
        endpoint: format!("http://127.0.0.1:{port}"),
        timeout_ms,
        retries: 1,
        fallback_processor: None,
    }
}
