use std::net::SocketAddr;
use std::time::Instant;

use anyhow::Context;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde_json::{Value, json};

use crate::ocl::OclOperation;
use crate::protocol::{
    COGNITIVE_PROTOCOL, CapabilitiesResponse, CognitiveRequest, CognitiveResponse, HealthResponse,
    ProcessorResult, ProcessorStatus,
};
use crate::state::{CognitiveEvent, CognitiveOperationRecord, CognitiveState, StateMutation};
use crate::workspace::strongest_candidate;

#[derive(Debug, Clone, Copy)]
pub enum MockProcessorKind {
    Executive,
    Attention,
    Memory,
}

#[derive(Debug, Clone)]
pub struct MockProcessor {
    pub kind: MockProcessorKind,
    pub name: String,
    pub version: String,
    pub hardware_node: String,
}

impl MockProcessor {
    pub fn executive() -> Self {
        Self::new(MockProcessorKind::Executive, "executive")
    }

    pub fn attention() -> Self {
        Self::new(MockProcessorKind::Attention, "attention")
    }

    pub fn memory() -> Self {
        Self::new(MockProcessorKind::Memory, "memory")
    }

    fn new(kind: MockProcessorKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
            version: "0.1".to_string(),
            hardware_node: "localhost".to_string(),
        }
    }

    pub fn router(self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/capabilities", get(capabilities))
            .route("/process", post(process))
            .with_state(self)
    }

    pub async fn serve(self, addr: SocketAddr) -> anyhow::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!(processor = %self.name, addr = %listener.local_addr()?, "mock processor listening");
        axum::serve(listener, self.router())
            .await
            .context("processor server exited")
    }

    fn operations(&self) -> Vec<OclOperation> {
        match self.kind {
            MockProcessorKind::Executive => vec![
                OclOperation::Route,
                OclOperation::Wait,
                OclOperation::Continue,
            ],
            MockProcessorKind::Attention => vec![
                OclOperation::Attend,
                OclOperation::Maintain,
                OclOperation::Switch,
                OclOperation::Suppress,
                OclOperation::Ignore,
            ],
            MockProcessorKind::Memory => vec![
                OclOperation::Recall,
                OclOperation::Store,
                OclOperation::UpdateMemory,
                OclOperation::Forget,
            ],
        }
    }

    fn handle(&self, request: CognitiveRequest) -> CognitiveResponse {
        let started = Instant::now();
        let result = if request.protocol != COGNITIVE_PROTOCOL || request.processor != self.name {
            ProcessorResult {
                operation: OclOperation::Ignore,
                target_processor: None,
                reason_code: Some("INVALID_ENVELOPE".to_string()),
                proposed_mutations: Vec::new(),
                emitted_events: Vec::new(),
                data: Value::Null,
            }
        } else {
            match self.kind {
                MockProcessorKind::Executive => self.handle_executive(&request.task),
                MockProcessorKind::Attention => self.handle_attention(&request.task),
                MockProcessorKind::Memory => self.handle_memory(&request.task),
            }
        };
        let status = if result.reason_code.as_deref() == Some("INVALID_ENVELOPE") {
            ProcessorStatus::InvalidRequest
        } else {
            ProcessorStatus::Success
        };
        CognitiveResponse {
            protocol: COGNITIVE_PROTOCOL.to_string(),
            request_id: request.request_id,
            processor: self.name.clone(),
            status,
            confidence: confidence_for(&result.operation),
            latency_ms: started.elapsed().as_millis() as u64,
            result,
        }
    }

    fn handle_executive(&self, task: &Value) -> ProcessorResult {
        let Some(state) = parse_state(task) else {
            return ProcessorResult::wait();
        };
        let last_kind = state.recent_events.last().map(|event| event.kind.as_str());
        let pending_focus = state.focus.is_some();
        let target_processor = match last_kind {
            Some("external_observation") => Some("attention"),
            Some("attention_complete") => Some("memory"),
            Some("processor_unavailable") => None,
            Some("memory_complete") => None,
            _ if !pending_focus && !state.workspace_candidates.is_empty() => Some("attention"),
            _ => None,
        };
        match target_processor {
            Some(processor) => ProcessorResult {
                operation: OclOperation::Route,
                target_processor: Some(processor.to_string()),
                reason_code: Some("PENDING_COGNITIVE_WORK".to_string()),
                proposed_mutations: vec![StateMutation::RecordOperation {
                    record: operation_record(
                        &self.name,
                        OclOperation::Route,
                        0.86,
                        0,
                        &self.hardware_node,
                        json!({ "target_processor": processor }),
                    ),
                }],
                emitted_events: Vec::new(),
                data: json!({ "processor": processor, "priority": 0.8 }),
            },
            None => ProcessorResult::wait(),
        }
    }

    fn handle_attention(&self, task: &Value) -> ProcessorResult {
        let Some(state) = parse_state(task) else {
            return ProcessorResult::wait();
        };
        let best = strongest_candidate(&state.workspace_candidates);
        let Some(candidate) = best else {
            return ProcessorResult::wait();
        };
        let event = CognitiveEvent::new("attention_complete", json!({ "target": candidate.id }));
        ProcessorResult {
            operation: if state.focus.as_deref() == Some(candidate.id.as_str()) {
                OclOperation::Maintain
            } else {
                OclOperation::Switch
            },
            target_processor: None,
            reason_code: Some("HIGHEST_WORKSPACE_ACTIVATION".to_string()),
            proposed_mutations: vec![
                StateMutation::SetFocus {
                    focus_id: candidate.id.clone(),
                },
                StateMutation::AddRecentEvent {
                    event: event.clone(),
                },
                StateMutation::RecordOperation {
                    record: operation_record(
                        &self.name,
                        OclOperation::Switch,
                        0.9,
                        0,
                        &self.hardware_node,
                        json!({ "target": candidate.id }),
                    ),
                },
            ],
            emitted_events: vec![event],
            data: json!({ "target": candidate.id }),
        }
    }

    fn handle_memory(&self, task: &Value) -> ProcessorResult {
        let Some(state) = parse_state(task) else {
            return ProcessorResult::wait();
        };
        let Some(focus_id) = state.focus.as_deref() else {
            return ProcessorResult::wait();
        };
        let event = CognitiveEvent::new(
            "memory_complete",
            json!({ "focus": focus_id, "decision": "STORE" }),
        );
        ProcessorResult {
            operation: OclOperation::Store,
            target_processor: None,
            reason_code: Some("FOCUSED_ITEM_SHOULD_BE_TRACEABLE".to_string()),
            proposed_mutations: vec![
                StateMutation::AddWorkingMemory {
                    item: json!({ "source": "memory_mock", "focus": focus_id, "stored": true }),
                },
                StateMutation::AddRecentEvent {
                    event: event.clone(),
                },
                StateMutation::RecordOperation {
                    record: operation_record(
                        &self.name,
                        OclOperation::Store,
                        0.82,
                        0,
                        &self.hardware_node,
                        json!({ "focus": focus_id }),
                    ),
                },
            ],
            emitted_events: vec![event],
            data: json!({ "decision": "STORE", "focus": focus_id }),
        }
    }
}

fn parse_state(task: &Value) -> Option<CognitiveState> {
    serde_json::from_value(task.get("state")?.clone()).ok()
}

fn confidence_for(operation: &OclOperation) -> f32 {
    match operation {
        OclOperation::Wait => 0.75,
        OclOperation::Route | OclOperation::Switch | OclOperation::Store => 0.86,
        _ => 0.7,
    }
}

fn operation_record(
    processor: &str,
    operation: OclOperation,
    confidence: f32,
    latency_ms: u64,
    hardware_node: &str,
    details: Value,
) -> CognitiveOperationRecord {
    CognitiveOperationRecord {
        timestamp: Utc::now(),
        processor: processor.to_string(),
        operation: format!("{operation:?}"),
        confidence,
        latency_ms,
        hardware_node: hardware_node.to_string(),
        details,
    }
}

async fn health(State(processor): State<MockProcessor>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        processor: processor.name,
    })
}

async fn capabilities(State(processor): State<MockProcessor>) -> Json<CapabilitiesResponse> {
    let operations = processor.operations();
    Json(CapabilitiesResponse {
        processor: processor.name,
        version: processor.version,
        operations,
    })
}

async fn process(
    State(processor): State<MockProcessor>,
    Json(request): Json<CognitiveRequest>,
) -> Json<CognitiveResponse> {
    Json(processor.handle(request))
}
