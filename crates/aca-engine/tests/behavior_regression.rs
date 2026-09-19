//! Behavioral regression tests for invariants optimization work must not
//! erode. These stay intentionally end-to-end: the point is to catch the
//! system drifting back toward a conventional prompt harness while internals
//! are being tuned.

use std::sync::Arc;
use std::time::Duration;

use aca_engine::{CognitiveLoopActor, CycleEventKind, LoopConfig, ToolRegistry};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::Tier;
use aca_util::SystemClock;
use async_trait::async_trait;

struct ReflectingChatClient;

#[async_trait]
impl ChatClient for ReflectingChatClient {
    async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
        Ok(TierResponse {
            raw_text: "I hear the contact and can orient to it.".to_string(),
            confidence: 0.92,
            tier: Tier::T3,
        })
    }
}

#[tokio::test]
async fn conversation_input_is_reflected_before_it_is_spoken() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let (mut actor, mut handles) = CognitiveLoopActor::new(
        LoopConfig::default(),
        Arc::new(FakeEmbeddingClient::default()),
        Arc::new(ReflectingChatClient),
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![]),
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![]),
        TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
        TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
        Arc::new(ReflectingChatClient),
        store.clone(),
        store,
        ToolRegistry::empty(),
        Arc::new(SystemClock),
        Vec::new(),
    );

    let user_text = "hello Omega, are you there?";
    handles.input_tx.send(user_text.to_string()).await.unwrap();

    let mut spoken = Vec::new();
    for _ in 0..20 {
        actor.tick().await;
        while let Ok(event) = handles.events_rx.try_recv() {
            assert_ne!(event.event_type, CycleEventKind::Error, "ordinary conversation should not error: {event:?}");
            if event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                spoken.push(event.payload.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string());
            }
        }
        if !spoken.is_empty() {
            break;
        }
    }

    assert_eq!(spoken, vec!["I hear the contact and can orient to it."]);
    assert_ne!(spoken[0], user_text, "Omega should not collapse back to immediate input echoing");
}
