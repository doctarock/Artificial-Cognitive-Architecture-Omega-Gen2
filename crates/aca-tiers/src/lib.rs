//! Access to the model-tier ladder (Tier 0-4): `ChatClient`/`EmbeddingClient`
//! traits, Ollama-native and OpenAI-compatible transport adapters, the
//! confidence-clamping response envelope, and per-tier concurrency/timeout
//! pools. The SOAR Executive and Cognitive Core (in `aca-engine`) depend
//! only on the trait objects here, never on a concrete transport.

mod attention;
mod attention_response;
mod client;
mod ollama;
mod openai_compat;
mod pool;
mod response;
pub mod testing;
mod transport;

pub use attention::{AttentionClient, AttentionDecision, AttentionError, AttentionOp, OllamaAttentionClient};
pub use client::{ChatClient, EmbeddingClient, GenerateRequest, TierError, TierResponse};
pub use ollama::OllamaClient;
pub use openai_compat::OpenAiCompatClient;
pub use pool::{DivergentPool, TierPool, TimeoutEmbeddingClient};
pub use response::parse_tier_response;
pub use transport::build_http_client;
