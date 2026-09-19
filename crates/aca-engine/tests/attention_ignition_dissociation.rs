//! Phase 2 of the GWT-parity roadmap (see `docs/cognitive-capability-audit.md`'s
//! second addendum): GNW's claim that attention (selection into the
//! competition) and ignition (actually becoming globally available) are
//! dissociable - something can be attended without ever becoming conscious.
//! `steps::broadcast::decide_admission_with_hysteresis` (Phase 1) already
//! gave this a real mechanism, by construction, the moment it needed two
//! separate thresholds for hysteresis to mean anything. What this file
//! verifies is that the dissociation is genuinely observable and real
//! through the live actor, not just two numbers that happen to differ -
//! using `EngineSnapshot::attended_not_ignited`, computed fresh every tick
//! from real Coalition scores.

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

/// **The dissociation itself, through the live actor.** `ignition_threshold`
/// set deliberately out of reach (real ACT-R scores never approach it -
/// this sidesteps landing a live decay computation inside a narrow band,
/// the same precision problem the Phase 1 addendum notes was judged not
/// worth chasing) while `attention_threshold` stays at its ordinary
/// default, so a real conversational turn keeps clearing Coalition's own
/// bar (attended) but can never clear Broadcast's stricter one (never
/// ignites). Both facts are checked from the real live actor's own
/// published state, not asserted independently of each other.
#[tokio::test]
async fn a_real_candidate_is_attended_but_never_ignites_when_the_ignition_bar_is_out_of_reach() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let mut config = LoopConfig::default();
    config.ignition_threshold = 1_000.0;
    let (mut actor, handles) = build_actor(config, clock, Vec::new());

    handles.input_tx.send("a real conversational turn".to_string()).await.unwrap();

    let mut ever_attended = false;
    for _ in 0..3 {
        actor.tick().await;
        let snapshot = handles.snapshot_rx.borrow().clone();
        assert!(snapshot.working_memory.is_empty(), "with the ignition bar out of reach, nothing should ever actually broadcast");
        if snapshot.attended_not_ignited.iter().any(|m| m.text == "a real conversational turn") {
            ever_attended = true;
        }
    }
    tokio::task::yield_now().await;
    actor.tick().await;
    let final_snapshot = handles.snapshot_rx.borrow().clone();
    if final_snapshot.attended_not_ignited.iter().any(|m| m.text == "a real conversational turn") {
        ever_attended = true;
    }
    assert!(final_snapshot.working_memory.is_empty(), "still never ignited after embedding resolution settles");

    assert!(ever_attended, "the real turn should show up as a genuine Coalition candidate (attended) on at least the tick it was evaluated, even though it never ignites");
}

/// **Negative control - the same turn, ordinary thresholds, actually
/// ignites.** Proves the positive case above is really about the
/// artificially-raised bar, not some other reason this content could never
/// win Broadcast at all.
#[tokio::test]
async fn the_identical_turn_ignites_normally_under_ordinary_thresholds() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (mut actor, handles) = build_actor(LoopConfig::default(), clock, Vec::new());

    handles.input_tx.send("a real conversational turn".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;

    let snapshot = handles.snapshot_rx.borrow().clone();
    assert!(snapshot.working_memory.iter().any(|m| m.text == "a real conversational turn"), "under ordinary thresholds the identical turn should actually ignite, not just get attended");
}

#[tokio::test]
async fn an_attended_loser_is_reconsidered_on_events_then_expires() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let mut config = LoopConfig::default();
    config.ignition_threshold = 1_000.0;
    config.preconscious_trace_ms = 500;
    config.preconscious_reevaluation_ms = 100;
    let (mut actor, handles) = build_actor(config, clock.clone(), Vec::new());

    handles.input_tx.send("a briefly subliminal turn".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;
    assert!(handles.snapshot_rx.borrow().attended_not_ignited.iter().any(|member| member.text == "a briefly subliminal turn"));

    clock.advance(100);
    actor.tick().await;
    assert!(
        handles.snapshot_rx.borrow().attended_not_ignited.iter().any(|member| member.text == "a briefly subliminal turn"),
        "the scheduler-backed trace should nominate the candidate again at its event deadline"
    );

    clock.advance(500);
    actor.tick().await;
    assert!(
        handles.snapshot_rx.borrow().attended_not_ignited.is_empty(),
        "the subliminal trace must disappear once its bounded lifetime expires"
    );
}
