//! Phase 5's crowding slice (see `docs/cognitive-capability-audit.md`'s
//! "Phase 5, revisited" section): divisive normalization -
//! `steps::coalition::apply_crowding_normalization` - makes how *crowded*
//! this tick's competing field is a real, measurable factor in what can
//! ignite, distinct from that candidate's own ACT-R activation. This file
//! proves the live-actor signature directly: the identical fresh turn
//! ignites when it arrives alone, and fails to ignite when several other
//! genuinely strong candidates are simultaneously competing - with
//! `working_memory_capacity` set high enough that ordinary capacity-limited
//! eviction (already proven elsewhere in this suite) cannot be what's
//! actually responsible for the difference.

use std::sync::Arc;
use std::time::Duration;

use aca_engine::{CognitiveLoopActor, LoopConfig, ToolRegistry};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::Tier;
use aca_util::{EpochMillis, ManualClock};
use async_trait::async_trait;

struct FixedChatClient;

#[async_trait]
impl ChatClient for FixedChatClient {
    async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
        Ok(TierResponse { raw_text: "a reply".to_string(), confidence: 0.9, tier: Tier::T3 })
    }
}

fn empty_pool(tier: Tier) -> DivergentPool {
    DivergentPool::new(tier, Duration::from_secs(5), Vec::new())
}

fn build_actor(config: LoopConfig, clock: Arc<ManualClock>) -> (CognitiveLoopActor, aca_engine::LoopHandles) {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    CognitiveLoopActor::new(
        config,
        Arc::new(FakeEmbeddingClient::default()),
        Arc::new(FixedChatClient),
        empty_pool(Tier::T1),
        empty_pool(Tier::T2),
        TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
        TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
        Arc::new(FixedChatClient),
        store.clone(),
        store,
        ToolRegistry::empty(),
        clock,
        Vec::new(),
    )
}

async fn settle(actor: &mut CognitiveLoopActor) {
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;
}

/// A deliberately unambiguous `crowding_strength` for this test - the point
/// under test is "does crowding change the outcome at all," not "does the
/// production default's exact magnitude." Calibrated directly (measured,
/// not guessed - the same workflow `ignition_threshold`'s own production
/// default was calibrated with): strong enough that three established
/// competitors measurably suppress a fourth, fresh arrival, but not so
/// strong that mutual suppression during the crowd's own establishment
/// phase chokes the crowd itself out before it can become a real, stable
/// resident presence - confirmed live that a much larger crowding_strength
/// (2.0) and a bigger crowd (6) does exactly that: everything suppresses
/// everything else into oblivion, leaving only the single earliest arrival
/// standing, which would prove nothing about a *fourth* thing failing
/// against an *established* crowd. `working_memory_capacity` is set high
/// enough that ordinary top-N capacity eviction (already covered by
/// `displacement_causality.rs`) cannot be what explains the difference
/// between the two scenarios below.
fn test_config(crowding_strength: f32) -> LoopConfig {
    let mut config = LoopConfig::default();
    config.crowding_strength = crowding_strength;
    config.working_memory_capacity = 20;
    // The production default (`-1.8`) is deliberately permissive (see its
    // own doc comment) - permissive enough that even a crowded score often
    // still clears it. This test needs a bar an *uncrowded* real score
    // (~3-4, measured directly) clears easily but a crowded one does not -
    // the same kind of controlled, directly-measured override
    // `attention_ignition_dissociation.rs` already uses rather than
    // fighting ACT-R precision blindly.
    config.ignition_threshold = 1.0;
    config
}

/// **The crowding signature, through the live actor.** The identical target
/// turn: alone, it ignites (the quiet-tick case - `others_positive` is
/// zero, so `apply_crowding_normalization` is an identity regardless of
/// `crowding_strength`). Preceded by several other genuinely strong, still-
/// resident turns competing on the very same tick, it fails to ignite - not
/// because of capacity (there's room for all of them at once), but because
/// their combined positive score suppresses the target's effective score
/// below `ignition_threshold`.
#[tokio::test]
async fn the_identical_turn_ignites_alone_but_not_when_crowded() {
    let target_text = "a specific target thought";

    let ignites_alone = {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, handles) = build_actor(test_config(0.5), clock);
        handles.input_tx.send(target_text.to_string()).await.unwrap();
        settle(&mut actor).await;
        handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.text == target_text)
    };
    assert!(ignites_alone, "the target turn should ignite on its own when nothing else is competing");

    let ignites_when_crowded = {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, handles) = build_actor(test_config(0.5), clock);
        for i in 0..3 {
            handles.input_tx.send(format!("a genuinely strong competing thought number {i}")).await.unwrap();
            settle(&mut actor).await;
        }
        handles.input_tx.send(target_text.to_string()).await.unwrap();
        settle(&mut actor).await;
        let snapshot = handles.snapshot_rx.borrow().clone();
        assert!(snapshot.working_memory.len() > 1, "the crowd itself should genuinely be resident and competing, not evicted - working memory has room for all of it at capacity 20");
        snapshot.working_memory.iter().any(|m| m.text == target_text)
    };
    assert!(!ignites_when_crowded, "with several other strong candidates simultaneously competing, the identical target turn should fail to ignite - suppressed by crowding, not by capacity (there was room for it)");
}

/// **Negative control - `crowding_strength = 0.0` removes the effect
/// entirely.** The identical crowded scenario above, but with crowding
/// disabled: the target now ignites despite the crowd, proving the
/// suppression above is really attributable to crowding and not some other
/// side effect of having several turns arrive in sequence.
#[tokio::test]
async fn disabling_crowding_lets_the_identical_crowded_scenario_ignite_normally() {
    let target_text = "a specific target thought";
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (mut actor, handles) = build_actor(test_config(0.0), clock);

    for i in 0..3 {
        handles.input_tx.send(format!("a genuinely strong competing thought number {i}")).await.unwrap();
        settle(&mut actor).await;
    }
    handles.input_tx.send(target_text.to_string()).await.unwrap();
    settle(&mut actor).await;

    let snapshot = handles.snapshot_rx.borrow().clone();
    assert!(snapshot.working_memory.iter().any(|m| m.text == target_text), "with crowding disabled, the identical crowded scenario should no longer suppress the target's admission");
}
