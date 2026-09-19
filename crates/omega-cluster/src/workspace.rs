use chrono::Utc;
use serde_json::Value;

use crate::state::{WorkspaceCandidate, workspace_score};

#[derive(Debug, Clone)]
pub struct WorkspaceCandidateBuilder {
    source: String,
    activation: f32,
    salience: f32,
    confidence: f32,
    goal_relevance: f32,
    novelty: f32,
}

impl WorkspaceCandidateBuilder {
    pub fn from_source(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            activation: 0.5,
            salience: 0.5,
            confidence: 0.8,
            goal_relevance: 0.3,
            novelty: 0.5,
        }
    }

    pub fn external_observation() -> Self {
        Self {
            source: "environment".to_string(),
            activation: 0.95,
            salience: 0.88,
            confidence: 0.9,
            goal_relevance: 0.5,
            novelty: 0.8,
        }
    }

    pub fn build(self, id: impl Into<String>, content: Value) -> WorkspaceCandidate {
        WorkspaceCandidate {
            id: id.into(),
            source: self.source,
            content,
            activation: self.activation,
            salience: self.salience,
            confidence: self.confidence,
            goal_relevance: self.goal_relevance,
            novelty: self.novelty,
            created_at: Utc::now(),
        }
    }
}

pub fn strongest_candidate(candidates: &[WorkspaceCandidate]) -> Option<&WorkspaceCandidate> {
    candidates.iter().max_by(|left, right| {
        workspace_score(left)
            .partial_cmp(&workspace_score(right))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}
