//! Phase 4's first real "specialist cloud" (see `docs/cognitive-capability-
//! audit.md`'s displacement addendum and the design discussion that followed
//! it): `steps::interlocutor::social_cloud_anchors` gives `steps::recall`
//! a second, independently-ablatable anchor set - one specific person's own
//! past utterances - instead of only ever spreading from whatever's
//! currently in Working Memory. This is heterogeneity in what a specialist
//! can even *see*, not just how it judges: the social pass and the generic
//! pass share the same scoring/Coalition/Broadcast machinery entirely: only
//! the anchor set differs.
//!
//! Same adversarial-pair discipline as `cognitive_capabilities_end_to_end.rs`:
//! a memory two hops from anything currently active, and completely
//! unreachable by the generic Working-Memory-anchored recall pass (it never
//! decayed back into anyone's current context), resurfaces specifically
//! because the *same* person who was once linked to it speaks again - and
//! stays gone when `disable_social_recall` is set, proving the positive
//! result isn't coincidental timing.

use std::sync::Arc;
use std::time::Duration;

use aca_engine::steps::recall::RecallConfig;
use aca_engine::{CognitiveLoopActor, LoopConfig, RoomInput, ToolRegistry};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::{AssociativeEdge, EdgeKind, MemoryRole, MentalObject, MentalObjectId, MentalObjectKind, Tier};
use aca_util::{EpochMillis, ManualClock, RingBuffer};
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

/// Builds a real `derek`-shaped interlocutor node (the exact shape
/// `steps::interlocutor::find_or_create` mints - `find_or_create` will find
/// and reuse this one rather than minting a duplicate, since it matches on
/// `data.interlocutor_hint`), plus a stale, deeply-decayed utterance from
/// him already `DerivedFrom`-linked to it, plus a second stale object
/// two-hops-from-Working-Memory-irrelevant that the utterance has a direct
/// `Associative` edge to (standing in for "whatever was actually co-active
/// with that utterance the first time it was ever broadcast" - a real edge,
/// hand-built rather than organically earned, exactly like
/// `cognitive_capabilities_end_to_end.rs`'s own two-hop chain helper, since
/// what's under test here is recall's *use* of the edge, not its formation).
/// Both stale objects carry a single ancient reference, so neither can clear
/// the attention threshold on its own activation alone - only real spreading
/// activation through the real edges can surface `dormant_content`.
fn build_derek_with_a_dormant_associated_memory() -> (Vec<MentalObject>, MentalObjectId, MentalObjectId) {
    let mut derek = MentalObject::new_observation("An interlocutor I've heard from, currently associated with the label 'derek'.", EpochMillis(0), 0.5);
    derek.kind = MentalObjectKind::Belief;
    derek.memory_roles = vec![MemoryRole::Semantic];
    derek.data = serde_json::json!({"interlocutor_hint": "derek"});
    let derek_id = derek.id;

    let mut dormant_content = MentalObject::new_observation("a detail derek once mentioned in passing", EpochMillis(0), 0.5);
    dormant_content.activation.reference_log = RingBuffer::new(64);
    dormant_content.activation.reference_log.push(EpochMillis(0));
    let dormant_id = dormant_content.id;

    let mut old_utterance = MentalObject::new_observation("derek's old message", EpochMillis(0), 0.5);
    old_utterance.activation.reference_log = RingBuffer::new(64);
    old_utterance.activation.reference_log.push(EpochMillis(0));
    old_utterance.edges.push(AssociativeEdge { target_id: derek_id, kind: EdgeKind::DerivedFrom, strength: 1.0, last_coactivated_at: EpochMillis(0) });
    old_utterance.edges.push(AssociativeEdge { target_id: dormant_id, kind: EdgeKind::Associative, strength: 1.0, last_coactivated_at: EpochMillis(0) });

    (vec![derek, dormant_content, old_utterance], derek_id, dormant_id)
}

async fn send_room_input_and_settle(handles: &aca_engine::LoopHandles, actor: &mut CognitiveLoopActor, text: &str, speaker_label: &str) {
    handles
        .room_input_tx
        .send(RoomInput { text: text.to_string(), speaker_label: Some(speaker_label.to_string()), stream_id: format!("track-{speaker_label}") })
        .await
        .unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;
}

/// **Positive case, through the live actor.** A fresh, unrelated utterance
/// from the *same* interlocutor a two-hop-dormant memory is associated with
/// resurfaces it into Working Memory, even though nothing in current
/// Working Memory is anywhere near it. The negative control disables
/// exactly the mechanism this depends on (`ablation_config.disable_social_recall`,
/// a real production kill-switch, independent of the generic
/// `disable_recall` flag) and asserts the identical dormant memory now stays
/// gone.
/// Every test in this file uses the same permissive `attention_threshold`
/// and `ignition_threshold` (`-100.0` each, not `LoopConfig::default()`'s
/// real `-2.0`/`-1.8`) - matching `steps::recall`'s own unit tests, not
/// `cognitive_capabilities_end_to_end.rs`'s dormant-memory test. That test's
/// anchor is a single, freshly-referenced, single-fan-out Working Memory
/// member, so real ACT-R decay numerics survive two hops undiluted. Here the
/// anchor set legitimately fans out to two things at once (the returning
/// interlocutor's node *and* the dormant content - both real
/// `reinforce_link`-created edges), which halves the spreading energy
/// reaching `dormant_content` relative to that single-fan-out case. What's
/// under test in this file is the anchor-selection and gating logic (whose
/// cloud, which trigger, real ablation), already proven distinct from raw
/// ACT-R threshold-clearing and ignition-hysteresis math by `steps::recall`'s
/// and `steps::broadcast`'s own unit suites - so, like those suites, these
/// permissive thresholds isolate exactly that, rather than re-relitigating
/// decay/ignition numerics a different test file already owns.
fn permissive_config() -> LoopConfig {
    let mut config = LoopConfig::default();
    config.recall_config = RecallConfig { max_hops: 2 };
    config.attention_threshold = -100.0;
    config.ignition_threshold = -100.0;
    config
}

#[tokio::test]
async fn a_memory_tied_to_an_interlocutor_resurfaces_when_they_speak_again_and_stays_gone_when_social_recall_is_disabled() {
    let resurfaced = {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (objects, _derek_id, dormant_id) = build_derek_with_a_dormant_associated_memory();
        let (mut actor, handles) = build_actor(permissive_config(), clock.clone(), objects);

        clock.advance(100_000);
        send_room_input_and_settle(&handles, &mut actor, "what's the weather like", "derek").await;

        handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.id == dormant_id)
    };
    assert!(resurfaced, "a memory two hops from a returning interlocutor's own old utterance should resurface into Working Memory through the live actor's own tick()");

    let resurfaced_with_ablation = {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = permissive_config();
        config.ablation_config.disable_social_recall = true;
        let (objects, _derek_id, dormant_id) = build_derek_with_a_dormant_associated_memory();
        let (mut actor, handles) = build_actor(config, clock.clone(), objects);

        clock.advance(100_000);
        send_room_input_and_settle(&handles, &mut actor, "what's the weather like", "derek").await;

        handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.id == dormant_id)
    };
    assert!(
        !resurfaced_with_ablation,
        "with social recall disabled, the identical dormant memory must NOT resurface - if it did, the positive result above would be meaningless"
    );
}

/// **Negative control - a different interlocutor's cloud must not leak.**
/// The exact same dormant memory, tied to derek, must not resurface when a
/// *different* recognized interlocutor speaks - proving this is real,
/// per-person scoping, not "anyone talking recalls everything."
#[tokio::test]
async fn an_unrelated_interlocutors_utterance_does_not_recall_a_different_persons_dormant_memory() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (objects, _derek_id, dormant_id) = build_derek_with_a_dormant_associated_memory();
    let (mut actor, handles) = build_actor(permissive_config(), clock.clone(), objects);

    clock.advance(100_000);
    send_room_input_and_settle(&handles, &mut actor, "what's the weather like", "alice").await;

    let resurfaced = handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.id == dormant_id);
    assert!(!resurfaced, "alice speaking must not recall a memory that's only ever been associated with derek");
}

/// **Negative control - no interlocutor this tick, no social recall at
/// all.** An ordinary typed turn (`input_tx`, no `speaker_label`) never
/// resolves an interlocutor, so `current_outcome_context`'s interlocutor is
/// `None` and the social pass simply never runs - it must not fall back to
/// "whichever interlocutor was last seen."
#[tokio::test]
async fn a_turn_with_no_resolved_interlocutor_never_triggers_social_recall() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (objects, _derek_id, dormant_id) = build_derek_with_a_dormant_associated_memory();
    let (mut actor, handles) = build_actor(permissive_config(), clock.clone(), objects);

    clock.advance(100_000);
    handles.input_tx.send("what's the weather like".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;

    let resurfaced = handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.id == dormant_id);
    assert!(!resurfaced, "a turn with no recognized speaker must not trigger any interlocutor's social cloud");
}
