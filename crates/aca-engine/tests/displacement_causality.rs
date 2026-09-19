//! Verifies the mechanism behind a specific kind of introspective claim
//! Omega could make: "X entered my awareness because it displaced Y." An
//! LLM asked "why did you stop thinking about Y" can construct a plausible-
//! sounding answer regardless of what actually happened inside the engine -
//! that's exactly the failure mode `docs/cognitive-capability-audit.md`
//! already named for direct "are you conscious" questions. What's tested
//! here instead is the underlying mechanism itself, driven only through the
//! real, unmodified `CognitiveLoopActor`'s public `input_tx`/`tick()`/
//! `snapshot_rx`/`events_rx` surface, exactly as `cognitive_capabilities_
//! end_to_end.rs` already establishes as this codebase's bar for "wired
//! together," not just "the math works in isolation":
//!
//! 1. A real capacity-limited eviction, forced by genuine competition
//!    between two real conversational turns, produces a
//!    `snapshot.last_displacement` claim naming the correct real entrant and
//!    evicted object - not a hand-built scenario, not a mocked ranking.
//! 2. That claim is falsifiable by the exact intervention the mission asked
//!    for: disable the mechanism responsible (here, the capacity constraint
//!    that forces competition at all) and confirm the claim - and the real
//!    eviction behind it - both disappear, with both turns now coexisting.
//! 3. A negative control proves the mechanism isn't just labeling every
//!    release as a "displacement": an object that leaves Working Memory for
//!    an unrelated, non-competitive reason (simple decay-driven non-
//!    renomination, not eviction by a rival) must never produce a
//!    displacement claim naming some arbitrary "cause."

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

/// Submits two real conversational turns through `input_tx`, letting each
/// resolve its embedding and compete for admission exactly as production
/// does - no hand-built graph, no hand-supplied score.
async fn submit_turn(handles: &aca_engine::LoopHandles, actor: &mut CognitiveLoopActor, text: &str) {
    handles.input_tx.send(text.to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;
}

/// **Positive case, through the live actor.** Working Memory capacity
/// pinned to 1 - the tightest possible real competition - so a second real
/// turn can only be admitted by evicting *something*. (What exactly gets
/// evicted at capacity 1 isn't only ever the first turn verbatim: Omega's
/// own reply to it is a real object too and can itself win, then lose, the
/// single slot before the second turn ever arrives - confirmed live. The
/// assertions below deliberately don't hardcode which specific object that
/// turns out to be; they check that whatever `last_displacement` claims is
/// actually, verifiably true of the live actor's real state, which is the
/// property under test.) Confirms `steps::displacement::explain_release`'s
/// verdict is actually reachable end to end, not just correct in isolation,
/// and that the same fact is independently visible on the real event log.
#[tokio::test]
async fn a_real_eviction_under_capacity_pressure_produces_a_verified_displacement_claim() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let mut config = LoopConfig::default();
    config.working_memory_capacity = 1;
    let (mut actor, mut handles) = build_actor(config, clock, Vec::new());

    submit_turn(&handles, &mut actor, "the kettle is boiling").await;
    submit_turn(&handles, &mut actor, "someone is at the front door").await;
    let snapshot = handles.snapshot_rx.borrow().clone();

    assert_eq!(snapshot.working_memory.len(), 1, "capacity 1 should hold exactly one resident member");
    let resident_text = snapshot.working_memory[0].text.clone();

    let claim = snapshot.last_displacement.clone().expect("a real capacity-forced eviction should produce a verified displacement claim");
    assert_eq!(claim.entrant_text, resident_text, "the claim's entrant must be whichever real object is actually occupying the slot right now");
    assert_ne!(claim.evicted_text, resident_text, "the claim must name a genuinely different object as evicted, not the winner itself");
    assert!(!snapshot.working_memory.iter().any(|m| m.id == claim.evicted_id), "the object the claim names as evicted must not still be resident");

    let mut saw_matching_event = false;
    while let Ok(event) = handles.events_rx.try_recv() {
        if event.payload.get("displacement").and_then(|v| v.as_bool()) == Some(true)
            && event.payload.get("entrant").and_then(|v| v.as_str()) == Some(&claim.entrant_id.to_string())
            && event.payload.get("evicted").and_then(|v| v.as_str()) == Some(&claim.evicted_id.to_string())
        {
            saw_matching_event = true;
        }
    }
    assert!(saw_matching_event, "the same displacement fact should also be independently visible on the real event log, not only on the snapshot");
}

/// **The actual mission's intervention.** The identical scenario as the
/// positive case above, but with the mechanism responsible for forced
/// eviction - the capacity constraint itself - disabled by giving Working
/// Memory room to spare (4, versus the positive case's razor-thin 1). Note
/// what this does *not* claim: objects still naturally leave Working Memory
/// over a few ticks even at generous capacity (an Observation that's been
/// spoken to hands its role to whatever comes next - a real,
/// non-competitive turnover this codebase already treats as structurally
/// different from an eviction, see `ReleaseReason::NotRenominated`'s doc
/// comment). What must disappear specifically is the *competitive* claim:
/// with four free slots, nothing genuinely has to fight another live
/// candidate for a seat, so `explain_release` should never once report a
/// confirmed `Displaced` verdict across a real run that, at capacity 1,
/// reliably produces one. If the displacement claim in the positive test
/// were a narrative gloss rather than a real causal readout, disabling the
/// mechanism wouldn't change anything; it does.
#[tokio::test]
async fn disabling_the_capacity_constraint_makes_the_competitive_displacement_claim_disappear() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let mut config = LoopConfig::default();
    config.working_memory_capacity = 4; // room to spare - no forced competition
    let (mut actor, mut handles) = build_actor(config, clock, Vec::new());

    submit_turn(&handles, &mut actor, "the kettle is boiling").await;
    submit_turn(&handles, &mut actor, "someone is at the front door").await;
    // A few settle ticks so any delayed Reflection/Speak turnover (the
    // non-competitive kind) has room to happen too, giving the mechanism
    // every opportunity to (wrongly) report a displacement if it were going
    // to.
    for _ in 0..3 {
        actor.tick().await;
    }
    let snapshot = handles.snapshot_rx.borrow().clone();

    assert!(
        snapshot.last_displacement.is_none(),
        "with no forced competition under relaxed capacity, the most recent tick should carry no displacement claim - got: {:?}",
        snapshot.last_displacement.as_ref().map(|d| d.claim_text())
    );

    let mut saw_displacement_event = false;
    while let Ok(event) = handles.events_rx.try_recv() {
        if event.payload.get("displacement").and_then(|v| v.as_bool()) == Some(true) {
            saw_displacement_event = true;
        }
    }
    assert!(!saw_displacement_event, "the event log must not report a competitive displacement across a run where capacity never actually forced one");
}

/// **Negative control - non-competitive release must not be mislabeled.**
/// `steps::displacement::explain_release`'s `NotRenominated` case: an object
/// that simply stops being nominated (rather than losing a real fight for a
/// slot) must never be reported as having been displaced by whatever else
/// happens to be present. Exercised directly against the pure function
/// (mirrors `cognitive_capabilities.rs`'s mechanism-level tier) since
/// reproducing the specific "superseded by its own Reflection" non-
/// renomination path end to end would require standing up a full Reflection
/// cycle to test a claim this module already settles at the algorithm
/// level - the live-actor tests above already prove the positive path is
/// really wired in.
#[test]
fn an_object_that_was_never_a_real_candidate_this_tick_is_not_reported_as_displaced() {
    use aca_engine::{explain_release, ReleaseReason};
    use aca_types::MentalObjectId;

    let quietly_dropped = MentalObjectId::new();
    let unrelated_entrant = MentalObjectId::new();
    // `quietly_dropped` is absent from `raw_candidates` entirely - it never
    // competed this tick, so nothing "displaced" it.
    let raw_candidates = vec![(unrelated_entrant, 5.0, None)];

    let reason = explain_release(&raw_candidates, &[unrelated_entrant], 4, quietly_dropped);
    assert_eq!(reason, ReleaseReason::NotRenominated, "an object absent from this tick's real candidates must never be reported as displaced by another real candidate");
}
