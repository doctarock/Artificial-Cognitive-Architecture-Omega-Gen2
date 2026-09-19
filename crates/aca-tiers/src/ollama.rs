use aca_types::Tier;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::client::{ChatClient, EmbeddingClient, GenerateRequest, TierError, TierResponse};
use crate::response::parse_tier_response;

/// Ollama's native (non-OpenAI-compatible) HTTP surface: `/api/generate`
/// and `/api/embed`.
pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    tier: Tier,
}

impl OllamaClient {
    pub fn new(http: reqwest::Client, base_url: impl Into<String>, model: impl Into<String>, tier: Tier) -> Self {
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            tier,
        }
    }
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl ChatClient for OllamaClient {
    async fn generate(&self, req: GenerateRequest) -> Result<TierResponse, TierError> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": self.model,
            "prompt": req.prompt,
            "stream": false,
            "options": { "temperature": req.temperature },
        });
        let response = self.http.post(&url).json(&body).send().await?.error_for_status()?;
        let parsed: OllamaGenerateResponse = response.json().await?;
        Ok(parse_tier_response(&parsed.response, self.tier))
    }
}

#[async_trait]
impl EmbeddingClient for OllamaClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, TierError> {
        let url = format!("{}/api/embed", self.base_url);
        let body = json!({ "model": self.model, "input": text });
        let response = self.http.post(&url).json(&body).send().await?.error_for_status()?;
        let parsed: OllamaEmbedResponse = response.json().await?;
        parsed
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| TierError::MalformedResponse {
                tier: self.tier,
                reason: "ollama embed response missing embeddings[0]".to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::build_http_client;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn generate_parses_the_response_field_through_the_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "response": r#"{"confidence": 0.75, "response": "hi there"}"#
            })))
            .mount(&server)
            .await;

        let client = OllamaClient::new(build_http_client(), server.uri(), "test-model", Tier::T3);
        let result = client
            .generate(GenerateRequest { prompt: "hello".into(), temperature: 0.3 })
            .await
            .unwrap();
        assert_eq!(result.raw_text, "hi there");
        assert!((result.confidence - 0.75).abs() < 1e-6);
        assert_eq!(result.tier, Tier::T3);
    }

    #[tokio::test]
    async fn embed_returns_the_first_embedding_vector() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[0.1, 0.2, 0.3]]
            })))
            .mount(&server)
            .await;

        let client = OllamaClient::new(build_http_client(), server.uri(), "embed-model", Tier::T0);
        let embedding = client.embed("hello").await.unwrap();
        assert_eq!(embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn embed_errors_cleanly_on_empty_embeddings_array() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": []
            })))
            .mount(&server)
            .await;

        let client = OllamaClient::new(build_http_client(), server.uri(), "embed-model", Tier::T0);
        let result = client.embed("hello").await;
        assert!(matches!(result, Err(TierError::MalformedResponse { .. })));
    }
}
