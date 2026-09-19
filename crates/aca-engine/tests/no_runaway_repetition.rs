//! Regression test for a real bug found via a live smoke test: without
//! state-aware operator proposal, an object that isn't decaying out of
//! Working Memory within a few ticks got the *same* operator (Speak)
//! re-selected and re-executed every single tick indefinitely - the same
//! reply spoken hundreds of times over a few seconds of continuous
//! ticking. Fixed by having `propose_operators` check what's already been
//! done (`produced_by_operator` / `MemoryRole::Episodic`) before proposing
//! Speak/Ask/Remember again.

use std::sync::Arc;
use std::time::Duration;

use aca_engine::{CognitiveLoopActor, CycleEventKind, LoopConfig};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::Tier;
use aca_util::SystemClock;
use async_trait::async_trait;

struct FixedChatClient;

#[async_trait]
impl ChatClient for FixedChatClient {
    async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
        Ok(TierResponse { raw_text: "reply".into(), confidence: 0.9, tier: Tier::T3 })
    }
}

#[tokio::test]
async fn speak_fires_once_not_once_per_tick_for_static_working_memory() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let (mut actor, mut handles) = CognitiveLoopActor::new(
        LoopConfig::default(),
        Arc::new(FakeEmbeddingClient::default()),
        Arc::new(FixedChatClient),
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![]),
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![]),
        TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
        TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
        Arc::new(FixedChatClient),
        store.clone(),
        store,
        aca_engine::ToolRegistry::empty(),
        Arc::new(SystemClock),
        Vec::new(),
    );

    handles.input_tx.send("the sky is blue today".to_string()).await.unwrap();

    let mut speak_count = 0;
    for _ in 0..200 {
        actor.tick().await;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                speak_count += 1;
            }
            // A settled, fully-handled object should never produce a new
            // impasse on every subsequent idle tick either.
            assert_ne!(event.event_type, CycleEventKind::Error, "no tick should error for a static, already-resolved object");
        }
    }

    assert_eq!(speak_count, 1, "Speak should fire exactly once for one static object across 200 ticks, not once per tick");
}
