use aca_types::Tier;
use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum TierError {
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("request to tier {tier:?} timed out after {elapsed_ms}ms")]
    Timeout { tier: Tier, elapsed_ms: u64 },
    #[error("malformed response from tier {tier:?}: {reason}")]
    MalformedResponse { tier: Tier, reason: String },
    #[error("no instance configured for tier {0:?}")]
    TierNotConfigured(Tier),
    #[error("all {attempted} candidate(s) for tier {tier:?} failed")]
    AllCandidatesFailed { tier: Tier, attempted: usize },
}

/// A request to a reasoning tier (Tier 1-4). `prompt` is already fully
/// assembled by the caller (Cognitive Core / Executive) — this crate has no
/// opinion on prompt construction, only transport.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub temperature: f32,
}

/// A tier's response: raw text plus a self-reported, clamped confidence.
/// `confidence` is always in `0.0..=1.0` by construction — parsing/clamping
/// happens once, at the transport boundary, per the plan's confidence
/// contract (missing/malformed input defaults to `0.5`, never fails the
/// turn).
#[derive(Debug, Clone)]
pub struct TierResponse {
    pub raw_text: String,
    pub confidence: f32,
    pub tier: Tier,
}

/// A reasoning-tier chat/generation endpoint. Implemented in Phase 5 by
/// Ollama-native and OpenAI-compatible adapters; the SOAR Executive and
/// Cognitive Core depend only on this trait, never on a concrete transport.
#[async_trait]
pub trait ChatClient: Send + Sync {
    async fn generate(&self, req: GenerateRequest) -> Result<TierResponse, TierError>;
}

/// The Tier-0-in-spirit embedding endpoint used by Predict/Observe/Compare.
/// Deliberately a separate trait from `ChatClient` (not one fat trait) —
/// Observe's async companion only ever needs this, keeping it structurally
/// decoupled from the reasoning tiers, mirroring specs.md's "embeddings are
/// substrate, decoupled from reasoning tiers" directly in the type system.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, TierError>;
}
