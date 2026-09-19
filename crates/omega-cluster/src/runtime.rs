use serde_json::json;

use crate::ocl::OclOperation;
use crate::processor::MockProcessor;
use crate::protocol::{CognitiveResponse, ProcessorStatus};
use crate::registry::{ProcessorClient, ProcessorRegistry, RegistryError};
use crate::state::{CognitiveEvent, CognitiveState, StateError, StateMutation, WorkspaceConfig};
use crate::store::{ClusterStore, TraceRecord};
use crate::workspace::WorkspaceCandidateBuilder;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    State(#[from] StateError),
    #[error("processor {processor} returned status {status:?}")]
    ProcessorStatus {
        processor: String,
        status: ProcessorStatus,
    },
    #[error("runtime exceeded max steps without WAIT")]
    MaxSteps,
    #[error(transparent)]
    Store(#[from] anyhow::Error),
}

pub struct OmegaRuntime {
    client: ProcessorClient,
    pub state: CognitiveState,
    pub event_log: Vec<CognitiveEvent>,
    workspace_config: WorkspaceConfig,
    store: Option<ClusterStore>,
}

impl OmegaRuntime {
    pub fn new(registry: ProcessorRegistry) -> Self {
        Self {
            client: ProcessorClient::new(registry),
            state: CognitiveState::default(),
            event_log: Vec::new(),
            workspace_config: WorkspaceConfig::default(),
            store: None,
        }
    }

    pub fn with_store(registry: ProcessorRegistry, store: ClusterStore) -> anyhow::Result<Self> {
        let state = store.load_state()?.unwrap_or_default();
        Ok(Self {
            client: ProcessorClient::new(registry),
            state,
            event_log: Vec::new(),
            workspace_config: WorkspaceConfig::default(),
            store: Some(store),
        })
    }

    pub fn ingest_observation(&mut self, text: impl Into<String>) -> Result<(), RuntimeError> {
        let text = text.into();
        let event = CognitiveEvent::new("external_observation", json!({ "text": text }));
        let candidate = WorkspaceCandidateBuilder::external_observation()
            .build(format!("observation_{}", event.id), event.payload.clone());
        self.commit(StateMutation::AddObservation {
            observation: event.payload.clone(),
        })?;
        self.commit(StateMutation::AddWorkspaceCandidate { candidate })?;
        self.commit(StateMutation::AddRecentEvent {
            event: event.clone(),
        })?;
        self.trace(TraceRecord::Event {
            event: event.clone(),
            state_revision: self.state.revision,
        })?;
        self.event_log.push(event);
        Ok(())
    }

    pub async fn run_until_wait(
        &mut self,
        max_steps: usize,
    ) -> Result<Vec<CognitiveResponse>, RuntimeError> {
        let mut responses = Vec::new();
        for _ in 0..max_steps {
            self.state.apply_workspace_physics(&self.workspace_config);
            let executive = self.call("executive").await?;
            let operation = executive.result.operation;
            let target = executive.result.target_processor.clone();
            self.apply_response(executive.clone())?;
            responses.push(executive);
            if operation == OclOperation::Wait {
                return Ok(responses);
            }
            let Some(target) = target else {
                return Ok(responses);
            };
            let specialist = match self.call(&target).await {
                Ok(response) => response,
                Err(err) => {
                    self.record_processor_unavailable(&target, &err)?;
                    continue;
                }
            };
            self.apply_response(specialist.clone())?;
            responses.push(specialist);
        }
        Err(RuntimeError::MaxSteps)
    }

    async fn call(&self, processor: &str) -> Result<CognitiveResponse, RuntimeError> {
        let response = self.client.process(processor, &self.state).await?;
        if response.status != ProcessorStatus::Success {
            return Err(RuntimeError::ProcessorStatus {
                processor: processor.to_string(),
                status: response.status,
            });
        }
        Ok(response)
    }

    fn apply_response(&mut self, response: CognitiveResponse) -> Result<(), RuntimeError> {
        for mutation in response.result.proposed_mutations.clone() {
            self.commit(mutation)?;
        }
        for event in response.result.emitted_events.clone() {
            self.trace(TraceRecord::Event {
                event: event.clone(),
                state_revision: self.state.revision,
            })?;
            self.event_log.push(event);
        }
        self.trace(TraceRecord::ProcessorResponse {
            response,
            state_revision: self.state.revision,
        })?;
        Ok(())
    }

    fn commit(&mut self, mutation: StateMutation) -> Result<(), RuntimeError> {
        self.state.apply_mutation(mutation.clone())?;
        self.trace(TraceRecord::Mutation {
            mutation,
            state_revision: self.state.revision,
        })?;
        if let Some(store) = &self.store {
            store.save_state(&self.state)?;
        }
        Ok(())
    }

    fn trace(&self, record: TraceRecord) -> Result<(), RuntimeError> {
        if let Some(store) = &self.store {
            store.append_trace(&record)?;
        }
        Ok(())
    }

    fn record_processor_unavailable(
        &mut self,
        processor: &str,
        err: &RuntimeError,
    ) -> Result<(), RuntimeError> {
        let event = CognitiveEvent::new(
            "processor_unavailable",
            json!({ "processor": processor, "error": err.to_string() }),
        );
        self.commit(StateMutation::AddRecentEvent {
            event: event.clone(),
        })?;
        self.trace(TraceRecord::Event {
            event: event.clone(),
            state_revision: self.state.revision,
        })?;
        self.event_log.push(event);
        Ok(())
    }
}

pub async fn spawn_local_mock_cluster(base_port: u16) {
    let services = [
        (MockProcessor::executive(), base_port),
        (MockProcessor::attention(), base_port + 1),
        (MockProcessor::memory(), base_port + 2),
    ];
    for (service, port) in services {
        let addr = ([127, 0, 0, 1], port).into();
        tokio::spawn(async move {
            if let Err(err) = service.serve(addr).await {
                tracing::error!(error = %err, "mock processor exited");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_mock_cluster_reaches_wait() {
        let base_port = 19_100;
        spawn_local_mock_cluster(base_port).await;
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let mut runtime = OmegaRuntime::new(ProcessorRegistry::localhost(base_port));
        runtime.ingest_observation("new user input").unwrap();
        let responses = runtime.run_until_wait(8).await.unwrap();

        assert!(
            responses
                .iter()
                .any(|response| response.processor == "attention")
        );
        assert!(
            responses
                .iter()
                .any(|response| response.processor == "memory")
        );
        assert_eq!(
            responses.last().unwrap().result.operation,
            OclOperation::Wait
        );
        assert!(runtime.state.focus.is_some());
        assert!(!runtime.state.working_memory.is_empty());
    }

    #[tokio::test]
    async fn runtime_persists_state_and_trace() {
        let base_port = 19_110;
        spawn_local_mock_cluster(base_port).await;
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let dir = std::env::temp_dir().join(format!("omega-cluster-test-{}", uuid::Uuid::now_v7()));
        let store = ClusterStore::open(&dir).unwrap();
        let mut runtime =
            OmegaRuntime::with_store(ProcessorRegistry::localhost(base_port), store.clone())
                .unwrap();
        runtime.ingest_observation("persistent input").unwrap();
        runtime.run_until_wait(8).await.unwrap();

        assert!(dir.join("state.json").exists());
        assert!(dir.join("events.jsonl").exists());
        let resumed =
            OmegaRuntime::with_store(ProcessorRegistry::localhost(base_port), store).unwrap();
        assert_eq!(resumed.state.revision, runtime.state.revision);
    }

    #[tokio::test]
    async fn unavailable_specialist_is_recorded_and_loop_reaches_wait() {
        let base_port = 19_120;
        for (service, port) in [
            (MockProcessor::executive(), base_port),
            (MockProcessor::attention(), base_port + 1),
        ] {
            let addr = ([127, 0, 0, 1], port).into();
            tokio::spawn(async move {
                if let Err(err) = service.serve(addr).await {
                    tracing::error!(error = %err, "mock processor exited");
                }
            });
        }
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let mut runtime = OmegaRuntime::new(ProcessorRegistry::localhost(base_port));
        runtime
            .ingest_observation("memory service is intentionally offline")
            .unwrap();
        let responses = runtime.run_until_wait(8).await.unwrap();

        assert_eq!(
            responses.last().unwrap().result.operation,
            OclOperation::Wait
        );
        assert!(
            runtime
                .event_log
                .iter()
                .any(|event| event.kind == "processor_unavailable")
        );
    }
}
