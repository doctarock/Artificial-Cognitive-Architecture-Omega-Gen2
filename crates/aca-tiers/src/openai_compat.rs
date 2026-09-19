use aca_types::Tier;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::client::{ChatClient, EmbeddingClient, GenerateRequest, TierError, TierResponse};
use crate::response::parse_tier_response;

/// The standard OpenAI-compatible HTTP surface (`/v1/chat/completions`,
/// `/v1/embeddings`) — what llama.cpp's server and most non-Ollama local
/// runtimes speak.
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    tier: Tier,
}

impl OpenAiCompatClient {
    pub fn new(
        http: reqwest::Client,
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        tier: Tier,
    ) -> Self {
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key,
            tier,
        }
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
}

#[async_trait]
impl ChatClient for OpenAiCompatClient {
    async fn generate(&self, req: GenerateRequest) -> Result<TierResponse, TierError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": req.prompt }],
            "temperature": req.temperature,
        });
        let response = self.apply_auth(self.http.post(&url).json(&body)).send().await?.error_for_status()?;
        let parsed: ChatCompletionResponse = response.json().await?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| TierError::MalformedResponse {
                tier: self.tier,
                reason: "openai-compatible response missing choices[0]".to_string(),
            })?;
        Ok(parse_tier_response(&content, self.tier))
    }
}

#[async_trait]
impl EmbeddingClient for OpenAiCompatClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, TierError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = json!({ "model": self.model, "input": text });
        let response = self.apply_auth(self.http.post(&url).json(&body)).send().await?.error_for_status()?;
        let parsed: EmbeddingResponse = response.json().await?;
        parsed
            .data
            .into_iter()
            .next()
            .map(|datum| datum.embedding)
            .ok_or_else(|| TierError::MalformedResponse {
                tier: self.tier,
                reason: "openai-compatible embeddings response missing data[0]".to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::build_http_client;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn generate_parses_choices_content_through_the_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{ "message": { "content": r#"{"confidence": 0.6, "response": "hi"}"# } }]
            })))
            .mount(&server)
            .await;

        let client = OpenAiCompatClient::new(build_http_client(), server.uri(), "test-model", None, Tier::T2);
        let result = client
            .generate(GenerateRequest { prompt: "hello".into(), temperature: 0.3 })
            .await
            .unwrap();
        assert_eq!(result.raw_text, "hi");
        assert!((result.confidence - 0.6).abs() < 1e-6);
    }

    #[tokio::test]
    async fn sends_bearer_auth_header_when_api_key_configured() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer secret-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{ "message": { "content": "ok" } }]
            })))
            .mount(&server)
            .await;

        let client = OpenAiCompatClient::new(
            build_http_client(),
            server.uri(),
            "test-model",
            Some("secret-key".to_string()),
            Tier::T2,
        );
        let result = client
            .generate(GenerateRequest { prompt: "hello".into(), temperature: 0.3 })
            .await;
        assert!(result.is_ok(), "request with matching auth header should succeed");
    }

    #[tokio::test]
    async fn embed_returns_the_first_embedding_datum() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "embedding": [0.4, 0.5] }]
            })))
            .mount(&server)
            .await;

        let client = OpenAiCompatClient::new(build_http_client(), server.uri(), "embed-model", None, Tier::T0);
        let embedding = client.embed("hello").await.unwrap();
        assert_eq!(embedding, vec![0.4, 0.5]);
    }
}
