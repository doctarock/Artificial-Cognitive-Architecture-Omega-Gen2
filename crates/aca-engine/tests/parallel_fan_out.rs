//! Phase 3 of the GWT-parity roadmap (see `docs/cognitive-capability-audit.md`'s
//! second addendum): real global broadcast makes ignited content available
//! to multiple independent consumers at once, not one linear pipeline where
//! every consequence has to win a single competitive vote. Before
//! `steps::memory_formation::maybe_automatic_remember`, `Operator::Remember`
//! only ever ran if it out-competed Speak/Ask/ContinueReflecting for the
//! Executive's one winning slot - a genuinely surprising broadcast winner
//! could be *either* remembered *or* spoken about in a given tick, never
//! both. This file proves both consequences now actually happen in the same
//! tick, through the real, unmodified actor.

use std::sync::Arc;
use std::time::Duration;

use aca_engine::{CognitiveLoopActor, LoopConfig, ToolRegistry};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::{MentalObject, Tier};
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

fn build_actor(config: LoopConfig, clock: Arc<ManualClock>, initial_objects: Vec<MentalObject>) -> (CognitiveLoopActor, aca_engine::LoopHandles) {
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
        initial_objects,
    )
}

/// **The positive case, through the live actor.** With the automatic-
/// remember threshold relaxed to `0.0` (any real surprise qualifies - a
/// controlled, deterministic stand-in for "genuinely surprising," the same
/// role `-100.0` plays for `attention_threshold` in the other Phase tests),
/// a fresh conversational turn is confirmed to both (a) get memory-formed
/// (a real `automatic_remember` event) and (b) get spoken about (a real
/// Act-phase `speak` event) - and, critically, on the *same* real
/// `cycle_seq`. Before this mechanism existed, these two consequences could
/// only ever have come from different ticks (whichever operator won that
/// tick's single Executive vote), never both at once.
#[tokio::test]
async fn memory_formation_and_communicative_decision_both_land_on_the_same_real_tick() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let mut config = LoopConfig::default();
    config.automatic_memory_formation_surprise_threshold = 0.0;
    let (mut actor, mut handles) = build_actor(config, clock, Vec::new());

    handles.input_tx.send("a genuinely novel thing has happened".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;
    // A few settle ticks so the Speak that follows Broadcast (Executive
    // proposes/selects the tick after admission, same as every other test
    // in this codebase that drives real turn-taking) has a real chance to
    // land, without assuming an exact tick offset.
    let mut remember_cycle = None;
    let mut speak_cycle = None;
    for _ in 0..5 {
        actor.tick().await;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.payload.get("automatic_remember").and_then(|v| v.as_bool()) == Some(true) {
                remember_cycle = Some(event.cycle_seq);
            }
            if event.phase == aca_store::CyclePhase::Act && event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                speak_cycle = Some(event.cycle_seq);
            }
        }
    }

    let remember_cycle = remember_cycle.expect("a genuinely surprising turn should trigger a real automatic_remember event");
    let speak_cycle = speak_cycle.expect("the same turn should still be spoken about - Remember no longer has to win a competition against it");
    assert_eq!(remember_cycle, speak_cycle, "memory formation and the communicative decision should land on the identical real tick - two independent consequences of the same broadcast winner, not a winner-take-all choice between them");
}

/// **Negative control - with the automatic path disabled (threshold
/// unreachable), the same turn still gets spoken about, but the automatic
/// event never fires.** Confirms the positive result above is really about
/// the automatic mechanism, not an artifact of this scenario always
/// producing both events regardless.
#[tokio::test]
async fn disabling_automatic_remember_leaves_communicative_decisions_unaffected_but_stops_the_automatic_event() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let mut config = LoopConfig::default();
    config.automatic_memory_formation_surprise_threshold = f32::INFINITY;
    let (mut actor, mut handles) = build_actor(config, clock, Vec::new());

    handles.input_tx.send("a genuinely novel thing has happened".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;

    let mut saw_automatic_remember = false;
    let mut saw_speak = false;
    for _ in 0..5 {
        actor.tick().await;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.payload.get("automatic_remember").and_then(|v| v.as_bool()) == Some(true) {
                saw_automatic_remember = true;
            }
            if event.phase == aca_store::CyclePhase::Act && event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                saw_speak = true;
            }
        }
    }

    assert!(!saw_automatic_remember, "with the threshold unreachable, the automatic path must never fire");
    assert!(saw_speak, "ordinary communicative decisions must be entirely unaffected by disabling the automatic path");
}
