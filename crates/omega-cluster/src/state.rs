use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

pub const COGNITIVE_STATE_PROTOCOL: &str = "omega-cognitive-state/v0.1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    pub protocol: String,
    pub state_id: Uuid,
    pub revision: u64,
    pub focus: Option<String>,
    pub workspace_candidates: Vec<WorkspaceCandidate>,
    pub working_memory: Vec<Value>,
    pub activated_memories: Vec<Value>,
    pub beliefs: Vec<Value>,
    pub goals: Vec<Value>,
    pub drives: Map<String, Value>,
    pub uncertainties: Vec<Value>,
    pub observations: Vec<Value>,
    pub recent_events: Vec<CognitiveEvent>,
    pub recent_cognitive_operations: Vec<CognitiveOperationRecord>,
    pub environment: Map<String, Value>,
    pub available_actions: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub working_memory_capacity: usize,
    pub activation_decay: f32,
    pub min_activation: f32,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            working_memory_capacity: 7,
            activation_decay: 0.92,
            min_activation: 0.05,
        }
    }
}

impl Default for CognitiveState {
    fn default() -> Self {
        Self {
            protocol: COGNITIVE_STATE_PROTOCOL.to_string(),
            state_id: Uuid::now_v7(),
            revision: 0,
            focus: None,
            workspace_candidates: Vec::new(),
            working_memory: Vec::new(),
            activated_memories: Vec::new(),
            beliefs: Vec::new(),
            goals: Vec::new(),
            drives: Map::new(),
            uncertainties: Vec::new(),
            observations: Vec::new(),
            recent_events: Vec::new(),
            recent_cognitive_operations: Vec::new(),
            environment: Map::new(),
            available_actions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceCandidate {
    pub id: String,
    pub source: String,
    pub content: Value,
    pub activation: f32,
    pub salience: f32,
    pub confidence: f32,
    pub goal_relevance: f32,
    pub novelty: f32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveEvent {
    pub id: Uuid,
    pub kind: String,
    pub payload: Value,
    pub timestamp: DateTime<Utc>,
}

impl CognitiveEvent {
    pub fn new(kind: impl Into<String>, payload: Value) -> Self {
        Self {
            id: Uuid::now_v7(),
            kind: kind.into(),
            payload,
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveOperationRecord {
    pub timestamp: DateTime<Utc>,
    pub processor: String,
    pub operation: String,
    pub confidence: f32,
    pub latency_ms: u64,
    pub hardware_node: String,
    pub details: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StateMutation {
    AddWorkspaceCandidate { candidate: WorkspaceCandidate },
    SetFocus { focus_id: String },
    AddWorkingMemory { item: Value },
    AddObservation { observation: Value },
    AddRecentEvent { event: CognitiveEvent },
    RecordOperation { record: CognitiveOperationRecord },
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("state protocol mismatch: {0}")]
    ProtocolMismatch(String),
    #[error("mutation references missing workspace candidate: {0}")]
    MissingCandidate(String),
}

impl CognitiveState {
    pub fn validate(&self) -> Result<(), StateError> {
        if self.protocol != COGNITIVE_STATE_PROTOCOL {
            return Err(StateError::ProtocolMismatch(self.protocol.clone()));
        }
        Ok(())
    }

    pub fn apply_mutation(&mut self, mutation: StateMutation) -> Result<(), StateError> {
        self.validate()?;
        match mutation {
            StateMutation::AddWorkspaceCandidate { candidate } => {
                self.workspace_candidates.push(candidate)
            }
            StateMutation::SetFocus { focus_id } => {
                if !self
                    .workspace_candidates
                    .iter()
                    .any(|candidate| candidate.id == focus_id)
                {
                    return Err(StateError::MissingCandidate(focus_id));
                }
                self.focus = Some(focus_id);
            }
            StateMutation::AddWorkingMemory { item } => self.working_memory.push(item),
            StateMutation::AddObservation { observation } => self.observations.push(observation),
            StateMutation::AddRecentEvent { event } => self.recent_events.push(event),
            StateMutation::RecordOperation { record } => {
                self.recent_cognitive_operations.push(record)
            }
        }
        self.revision += 1;
        self.state_id = Uuid::now_v7();
        Ok(())
    }

    pub fn apply_workspace_physics(&mut self, config: &WorkspaceConfig) {
        for candidate in &mut self.workspace_candidates {
            candidate.activation *= config.activation_decay;
        }
        self.workspace_candidates
            .retain(|candidate| candidate.activation >= config.min_activation);
        self.workspace_candidates.sort_by(|left, right| {
            workspace_score(right)
                .partial_cmp(&workspace_score(left))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.workspace_candidates
            .truncate(config.working_memory_capacity);
    }
}

pub fn workspace_score(candidate: &WorkspaceCandidate) -> f32 {
    candidate.activation
        + candidate.salience
        + candidate.goal_relevance
        + candidate.novelty
        + candidate.confidence
}
