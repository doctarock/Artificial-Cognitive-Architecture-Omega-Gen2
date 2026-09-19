use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::ocl::OclOperation;
use crate::state::StateMutation;

pub const COGNITIVE_PROTOCOL: &str = "omega-cognitive/v0.1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveRequest {
    pub protocol: String,
    pub request_id: Uuid,
    pub processor: String,
    pub state_ref: Uuid,
    pub timestamp: DateTime<Utc>,
    pub task: Value,
    pub constraints: RequestConstraints,
}

impl CognitiveRequest {
    pub fn new(processor: impl Into<String>, state_ref: Uuid, task: Value) -> Self {
        Self {
            protocol: COGNITIVE_PROTOCOL.to_string(),
            request_id: Uuid::now_v7(),
            processor: processor.into(),
            state_ref,
            timestamp: Utc::now(),
            task,
            constraints: RequestConstraints::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestConstraints {
    pub max_tokens: u32,
    pub deadline_ms: u64,
}

impl Default for RequestConstraints {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            deadline_ms: 1_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveResponse {
    pub protocol: String,
    pub request_id: Uuid,
    pub processor: String,
    pub status: ProcessorStatus,
    pub result: ProcessorResult,
    pub confidence: f32,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorStatus {
    Success,
    Unavailable,
    InvalidRequest,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorResult {
    pub operation: OclOperation,
    pub target_processor: Option<String>,
    pub reason_code: Option<String>,
    pub proposed_mutations: Vec<StateMutation>,
    pub emitted_events: Vec<crate::state::CognitiveEvent>,
    pub data: Value,
}

impl ProcessorResult {
    pub fn wait() -> Self {
        Self {
            operation: OclOperation::Wait,
            target_processor: None,
            reason_code: Some("NO_PENDING_COGNITIVE_WORK".to_string()),
            proposed_mutations: Vec::new(),
            emitted_events: Vec::new(),
            data: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub processor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitiesResponse {
    pub processor: String,
    pub version: String,
    pub operations: Vec<OclOperation>,
}
