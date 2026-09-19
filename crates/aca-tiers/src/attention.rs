use aca_types::MentalObjectId;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::attention_response::parse_attention_response;

#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("malformed response: {reason}")]
    MalformedResponse { reason: String },
}

/// The Omega Attention model's trained operation vocabulary — a single
/// state transition applied to Working Memory, not a ranked admission list.
/// See `training/attention-v0/SCHEMA_V0_2.md` for the full contract this
/// mirrors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionOp {
    Attend,
    Maintain,
    Switch,
    Suppress,
    Ignore,
}

impl AttentionOp {
    /// Matches the exact uppercase tokens the model was trained to emit.
    /// Anything else is treated as unrecognized, not guessed at — a model
    /// that drifts from its trained vocabulary should fail closed into the
    /// deterministic fallback, not be silently coerced into some op.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "ATTEND" => Some(Self::Attend),
            "MAINTAIN" => Some(Self::Maintain),
            "SWITCH" => Some(Self::Switch),
            "SUPPRESS" => Some(Self::Suppress),
            "IGNORE" => Some(Self::Ignore),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionDecision {
    pub operation: AttentionOp,
    pub target: Option<MentalObjectId>,
    pub confidence: f32,
    pub reason_code: String,
}

/// A standalone, non-ladder model endpoint for the attention-arbitration
/// decision — deliberately not `ChatClient`: its `Tier` field has no honest
/// value here, and the `{"confidence","response"}` envelope every other
/// prompt in this codebase uses doesn't fit a model with its own fixed,
/// trained 4-key JSON contract (`operation`/`target`/`confidence`/`reason_code`).
#[async_trait]
pub trait AttentionClient: Send + Sync {
    /// `prompt` is already fully assembled by the caller (mirrors
    /// `ChatClient::generate`'s convention) — this trait has no opinion on
    /// prompt construction, only transport + response parsing.
    async fn suggest(&self, prompt: String) -> Result<AttentionDecision, AttentionError>;
}

/// Ollama's native `/api/generate` surface. Temperature is hardcoded to
/// `0.0` here (not caller-supplied) — the model's Ollama Modelfile already
/// bakes in its trained system prompt and `temperature 0`; no system field
/// is sent since it's already in the Modelfile.
pub struct OllamaAttentionClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaAttentionClient {
    pub fn new(http: reqwest::Client, base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
        }
    }
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[async_trait]
impl AttentionClient for OllamaAttentionClient {
    async fn suggest(&self, prompt: String) -> Result<AttentionDecision, AttentionError> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": { "temperature": 0.0 },
        });
        let response = self.http.post(&url).json(&body).send().await?.error_for_status()?;
        let parsed: OllamaGenerateResponse = response.json().await?;
        parse_attention_response(&parsed.response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::build_http_client;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn suggest_parses_a_well_formed_decision_through_the_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "response": r#"{"operation":"SWITCH","target":"550e8400-e29b-41d4-a716-446655440000","confidence":0.91,"reason_code":"HIGHER_PRIORITY_INTERRUPT"}"#
            })))
            .mount(&server)
            .await;

        let client = OllamaAttentionClient::new(build_http_client(), server.uri(), "omega-attention-v0.2");
        let decision = client.suggest("workspace json here".to_string()).await.unwrap();
        assert_eq!(decision.operation, AttentionOp::Switch);
        assert!((decision.confidence - 0.91).abs() < 1e-6);
        assert_eq!(decision.reason_code, "HIGHER_PRIORITY_INTERRUPT");
    }

    #[tokio::test]
    async fn suggest_surfaces_a_transport_error_on_a_dead_host() {
        let client = OllamaAttentionClient::new(build_http_client(), "http://127.0.0.1:1", "omega-attention-v0.2");
        let result = client.suggest("workspace json here".to_string()).await;
        assert!(matches!(result, Err(AttentionError::Transport(_))));
    }
}
