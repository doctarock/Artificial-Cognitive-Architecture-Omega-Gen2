//! `cognitive_capabilities.rs` proves the underlying *math* is correct in
//! isolation - real embeddings, real ACT-R formulas - but every input to
//! those tests was hand-picked to demonstrate the mechanism, which means
//! none of them can actually fail against a system where the mechanisms
//! exist but were never wired together correctly. That is not proof the
//! *architecture* (the real, unmodified `CognitiveLoopActor`, driven only
//! through its public input/tick/snapshot surface, exactly as `omega-acad`
//! drives it in production) exhibits the capability at all.
//!
//! This file closes that gap with one adversarial pair per capability: a
//! positive case run against the live actor, and a negative control - the
//! identical scenario with the relevant cognitive mechanism switched off via
//! `LoopConfig::ablation_config` (a real, production-shipped kill-switch,
//! not a test-only shim) or via a structural variant that should defeat it.
//! If the negative control did NOT fail, the positive case would prove
//! nothing - so every test here is only trusted once both arms are seen to
//! diverge. No test in this file hand-supplies a surprise number, an
//! activation total, or a coalition score - every number is produced by the
//! real tick() pipeline (Predict/Observe/Compare/Broadcast/Learn) end to end.

use std::sync::Arc;
use std::time::Duration;

use aca_engine::steps::recall::RecallConfig;
use aca_engine::{CognitiveLoopActor, LoopConfig, ToolRegistry};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::{AssociativeEdge, EdgeKind, GoalStackId, GoalStackMembership, GoalStatus, MentalObject, MentalObjectKind, Tier};
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

/// Builds `anchor -> intermediate -> dormant`, a real two-hop associative
/// chain, plus `anchor`'s and `dormant`'s ids. `anchor` is a persistent,
/// Active `Intention` - the one Mental Object kind `steps::agenda::
/// surface_active_intentions` nominates for attention every tick regardless
/// of prior Working Memory membership, which is what lets it become a real
/// Working Memory member through the actor's own public tick() machinery
/// rather than a test reaching into private actor state to fake that
/// membership. `dormant`'s reference log is frozen to one stale reference
/// (matches the graph state `store.load_all()` would hand a freshly-booted
/// actor for a genuinely old memory - not a fabricated shortcut).
fn build_two_hop_chain_with_a_live_anchor() -> (Vec<MentalObject>, aca_types::MentalObjectId, aca_types::MentalObjectId) {
    let mut dormant = MentalObject::new_observation("a long-forgotten detail", EpochMillis(0), 0.5);
    dormant.activation.reference_log = aca_util::RingBuffer::new(64);
    dormant.activation.reference_log.push(EpochMillis(0));
    let dormant_id = dormant.id;

    let mut intermediate = MentalObject::new_observation("a bridging thought", EpochMillis(0), 0.5);
    intermediate.edges.push(AssociativeEdge { target_id: dormant_id, kind: EdgeKind::Associative, strength: 1.0, last_coactivated_at: EpochMillis(0) });
    let intermediate_id = intermediate.id;

    let mut anchor = MentalObject::new_observation("the current topic", EpochMillis(0), 0.5);
    anchor.kind = MentalObjectKind::Intention;
    anchor.goal = Some(GoalStackMembership { stack_id: GoalStackId::new(), parent_goal_id: None, status: GoalStatus::Active, priority: 0.9 });
    anchor.edges.push(AssociativeEdge { target_id: intermediate_id, kind: EdgeKind::Associative, strength: 1.0, last_coactivated_at: EpochMillis(0) });
    let anchor_id = anchor.id;

    (vec![dormant, intermediate, anchor], anchor_id, dormant_id)
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

/// **Adversarial pair 1 - Associative memory recall, through the live
/// actor.** A memory two real associative hops from anything currently
/// active, decayed 100 real seconds past the point its own activation alone
/// could win admission, still resurfaces into Working Memory once something
/// genuinely active (`anchor`) is within reach of it - using nothing but the
/// actor's public `tick()`. The negative control disables exactly the
/// mechanism this depends on (`ablation_config.disable_recall`, a real
/// production kill-switch) and asserts the *same* dormant memory now stays
/// gone - proving the positive result isn't a coincidence of timing or
/// decay math that would have happened regardless.
#[tokio::test]
async fn dormant_memory_resurfaces_via_the_live_actor_and_stays_gone_when_recall_is_disabled() {
    let recalled = {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.recall_config = RecallConfig { max_hops: 2 };
        let (objects, _anchor_id, dormant_id) = build_two_hop_chain_with_a_live_anchor();
        let (mut actor, handles) = build_actor(config, clock.clone(), objects);

        clock.advance(100_000);
        // Tick 1: `surface_active_intentions` (unconditional, every tick)
        // nominates the Active `anchor` intention for attention regardless
        // of it never having been in Working Memory before; Broadcast
        // admits it on its own merits (a fresh reference this instant).
        actor.tick().await;
        // Tick 2: Step 4.5 Recall now spreads from `anchor` (genuinely
        // resident in Working Memory as of tick 1's Broadcast) through
        // `intermediate` to `dormant`, two real hops, and folds it into
        // this tick's Coalition if it clears the attention threshold.
        actor.tick().await;

        handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.id == dormant_id)
    };
    assert!(recalled, "a two-hop dormant memory should resurface into Working Memory through the live actor's own tick(), not just the standalone recall() function");

    let recalled_with_ablation = {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.recall_config = RecallConfig { max_hops: 2 };
        config.ablation_config.disable_recall = true;
        let (objects, _anchor_id, dormant_id) = build_two_hop_chain_with_a_live_anchor();
        let (mut actor, handles) = build_actor(config, clock.clone(), objects);

        clock.advance(100_000);
        actor.tick().await;
        actor.tick().await;

        handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.id == dormant_id)
    };
    assert!(
        !recalled_with_ablation,
        "with recall disabled, the identical dormant memory must NOT resurface - if it did, the positive result above would be meaningless"
    );
}

/// **Adversarial pair 2 - Hebbian co-activation, through the live actor.**
/// Two genuinely unrelated conversational turns that actually overlap in
/// Working Memory (both real, freshly submitted via `input_tx`) end up with
/// a real, non-zero associative edge between them, visible on the public
/// snapshot - not asserted against a hand-built graph. The negative control
/// keeps the exact same two turns but separates them by enough idle time
/// that the first has already left Working Memory before the second ever
/// arrives, and asserts no edge forms - proving the edge in the positive
/// case reflects genuine, real-time co-occurrence rather than "any two
/// objects that ever existed get linked eventually."
#[tokio::test]
async fn concurrently_active_thoughts_get_a_real_edge_but_sequential_ones_that_never_overlap_do_not() {
    async fn run_two_turns_and_report_wm_edge_count(gap_ms: i64) -> usize {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, handles) = build_actor(LoopConfig::default(), clock.clone(), Vec::new());

        handles.input_tx.send("the weather today is unusually warm".to_string()).await.unwrap();
        actor.tick().await;
        tokio::task::yield_now().await;
        actor.tick().await;

        if gap_ms > 0 {
            clock.advance(gap_ms);
            // Enough idle ticks at the advanced clock for the first turn to
            // fully decay out of Working Memory before the second arrives.
            for _ in 0..5 {
                actor.tick().await;
            }
        }

        handles.input_tx.send("remind me to water the plants".to_string()).await.unwrap();
        actor.tick().await;
        tokio::task::yield_now().await;
        actor.tick().await;
        // A few settle ticks so same-tick admission/Learn timing can't
        // starve either arm of a fair chance to reinforce.
        for _ in 0..3 {
            actor.tick().await;
        }

        handles.snapshot_rx.borrow().associative_edges.len()
    }

    let overlapping_edges = run_two_turns_and_report_wm_edge_count(0).await;
    assert!(overlapping_edges > 0, "two real turns that overlap in Working Memory should leave a real, observable associative edge behind");

    let separated_edges = run_two_turns_and_report_wm_edge_count(200_000).await;
    assert_eq!(
        separated_edges, 0,
        "two turns separated by enough silence that the first fully decayed out before the second ever arrived must NOT show an associative edge - they never actually co-occurred"
    );
}

/// **Adversarial pair 3 - Coherent long-term behavior under real,
/// continuous ticking.** A single conversational turn, submitted once, must
/// resolve to speaking exactly once across a long run of continuous
/// ticking - not zero (dead), not many (a stuck loop re-executing the same
/// decision every tick). The negative control is structural, not an
/// ablation flag: an executive that ignores what it already did (the
/// documented historical failure mode - see `no_runaway_repetition.rs`)
/// would fail the upper bound here, and a broadcast/executive path that
/// never fires at all would fail the lower bound - this single assertion
/// range is bulletproof against both known failure directions at once.
#[tokio::test]
async fn a_single_turn_is_spoken_exactly_once_across_two_hundred_continuous_ticks() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (mut actor, mut handles) = build_actor(LoopConfig::default(), clock, Vec::new());

    handles.input_tx.send("is anyone there?".to_string()).await.unwrap();

    let mut speak_count = 0;
    for _ in 0..200 {
        actor.tick().await;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                speak_count += 1;
            }
        }
    }

    assert_eq!(speak_count, 1, "one real turn should produce exactly one Speak across 200 continuous ticks - not silence, not a runaway repeat loop");
}
