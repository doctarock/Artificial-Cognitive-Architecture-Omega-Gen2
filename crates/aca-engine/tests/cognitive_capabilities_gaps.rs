//! Tests for the six gaps identified after reviewing
//! `cognitive_capabilities.rs` and `cognitive_capabilities_end_to_end.rs`:
//! those files proved specific mechanisms are correctly wired into the live
//! actor, but left six things completely unaddressed. This file addresses
//! each directly, and is honest where the answer is "this isn't
//! implemented yet" rather than forcing a pass.
//!
//! 1. Real semantic content (real models, not `FakeEmbeddingClient`/fixed
//!    `ChatClient` stubs) — `#[ignore]`d, needs the LAN inference hosts.
//! 2. Organic emergence — does a capability that requires seeded graph
//!    state (recall) have a real boundary when nothing is seeded?
//! 3. Content coherence — does contradiction get detected/resolved at all?
//! 4. A non-degenerate vs. degenerate baseline comparison, same scenario.
//! 5. Model independence — does swapping which model answers change
//!    anything it structurally shouldn't?
//! 6. Real wall-clock time and real concurrency, via the actual `run()`
//!    entry point production uses, not manually-sequenced `tick()` calls.

use std::sync::Arc;
use std::time::Duration;

use aca_engine::steps::recall::RecallConfig;
use aca_engine::{CognitiveLoopActor, LoopConfig, LoopHandles, ToolRegistry};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, EmbeddingClient, GenerateRequest, OllamaClient, TierError, TierPool, TierResponse};
use aca_types::{AssociativeEdge, EdgeKind, GoalStackId, GoalStackMembership, GoalStatus, MentalObject, MentalObjectId, MentalObjectKind, Tier};
use aca_util::{Clock, EpochMillis, ManualClock, SystemClock};
use async_trait::async_trait;

struct FixedChatClient(&'static str);

#[async_trait]
impl ChatClient for FixedChatClient {
    async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
        Ok(TierResponse { raw_text: self.0.to_string(), confidence: 0.9, tier: Tier::T3 })
    }
}

fn empty_pool(tier: Tier) -> DivergentPool {
    DivergentPool::new(tier, Duration::from_secs(5), Vec::new())
}

fn build_actor(
    config: LoopConfig,
    clock: Arc<dyn Clock>,
    embedding_client: Arc<dyn EmbeddingClient>,
    tier3_client: Arc<dyn ChatClient>,
    initial_objects: Vec<MentalObject>,
) -> (CognitiveLoopActor, LoopHandles) {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    CognitiveLoopActor::new(
        config,
        embedding_client,
        tier3_client.clone(),
        empty_pool(Tier::T1),
        empty_pool(Tier::T2),
        TierPool::new(Tier::T3, 1, Duration::from_secs(300)),
        TierPool::new(Tier::T4, 1, Duration::from_secs(300)),
        tier3_client,
        store.clone(),
        store,
        ToolRegistry::empty(),
        clock,
        initial_objects,
    )
}

// ---------------------------------------------------------------------
// Gap 1: real semantic content, via real network-backed models.
// ---------------------------------------------------------------------

/// `#[ignore]`d because it needs the actual LAN inference hosts from
/// `.env` up and reachable - this is not a claim any deterministic,
/// network-free test could make honestly. Run explicitly with
/// `cargo test -p aca-engine --test cognitive_capabilities_gaps -- --ignored`.
/// Everywhere else in this codebase's tests, "reflection" is a scripted
/// string returned with zero regard for the input - this is the one test
/// where a real model actually has to produce it.
#[tokio::test]
#[ignore = "requires the live LAN inference hosts (embedding + Tier 3) from .env to be reachable"]
async fn a_real_model_produces_genuine_non_placeholder_reflective_content() {
    // Endpoints come from the environment, never hardcoded here - this
    // repo's LAN inference hosts are private infrastructure that must not
    // appear in source under version control.
    let embedding_base_url = std::env::var("OMEGA_EMBEDDING_BASE_URL").expect("set OMEGA_EMBEDDING_BASE_URL to run this ignored test");
    let tier3_base_url = std::env::var("OMEGA_TIER3_BASE_URL").expect("set OMEGA_TIER3_BASE_URL to run this ignored test");
    let tier3_model = std::env::var("OMEGA_TIER3_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string());

    let http = reqwest::Client::new();
    let embedding_client: Arc<dyn EmbeddingClient> = Arc::new(OllamaClient::new(http.clone(), &embedding_base_url, "nomic-embed-text", Tier::T0));
    let tier3_client: Arc<dyn ChatClient> = Arc::new(OllamaClient::new(http, &tier3_base_url, &tier3_model, Tier::T3));

    // Prove this is a genuine network round trip, not `FakeEmbeddingClient`
    // (which always produces an 8-dimensional hash-based vector regardless
    // of model) - `nomic-embed-text` produces 768 dimensions.
    let real_embedding = embedding_client.embed("a distinctive test phrase").await.expect("the real embedding host should be reachable");
    assert_eq!(real_embedding.len(), 768, "expected a real nomic-embed-text embedding, not a stub");

    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (mut actor, mut handles) = build_actor(LoopConfig::default(), clock, embedding_client, tier3_client, Vec::new());

    let distinctive_input = "Without repeating my words back, name one thing that comes to mind about a lighthouse standing in a desert.";
    handles.input_tx.send(distinctive_input.to_string()).await.unwrap();

    let mut spoken = Vec::new();
    for i in 0..60 {
        actor.tick().await;
        while let Ok(event) = handles.events_rx.try_recv() {
            eprintln!("[tick {i}] phase={:?} kind={:?} tier_used={:?} payload={}", event.phase, event.event_type, event.tier_used, event.payload);
            if event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                spoken.push(event.payload.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string());
            }
        }
        if !spoken.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    assert!(!spoken.is_empty(), "a real Tier 3 model should have produced a spoken reply");
    let reply = spoken[0].trim();
    assert!(reply.len() > 15, "a real model's reply should be substantive, got: {reply:?}");
    assert_ne!(reply, distinctive_input, "the reply must be genuine synthesis, not an echo of the input");
    assert!(
        !aca_engine::SELF_MEMORY_SEED_TEXTS.contains(&reply),
        "the reply must be genuine content, not a verbatim recitation of a Self Memory seed belief"
    );
}

// ---------------------------------------------------------------------
// Gap 2: organic emergence - what actually happens with no seeded state.
// ---------------------------------------------------------------------

/// The sibling end-to-end suite proves recall *does* resurrect a decayed
/// memory - but only once a live `Intention` anchor gives it something
/// currently active to spread from. This test checks the honest opposite
/// case, entirely organically (no `initial_objects` seeding anywhere): two
/// ordinary conversational turns that really did co-occur (so a real edge
/// forms - see the Hebbian test in the sibling suite) but, once both have
/// fully decayed out through genuine silence, do NOT spontaneously resurface
/// just because *some* new, unrelated topic arrives. This is a real
/// architectural boundary, not a bug: nothing about specs.md's Memory Recall
/// section promises unprompted resurrection of forgotten, unrelated content,
/// and a system that did that on every new topic would be closer to noise
/// than memory.
#[tokio::test]
async fn a_decayed_pair_of_ordinary_memories_does_not_spontaneously_resurface_without_a_live_anchor() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (mut actor, handles) =
        build_actor(LoopConfig::default(), clock.clone(), Arc::new(FakeEmbeddingClient::default()), Arc::new(FixedChatClient("a reply")), Vec::new());

    handles.input_tx.send("the meeting is on Tuesday".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;
    handles.input_tx.send("remind me to call the dentist".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;

    let ids_before: Vec<MentalObjectId> = handles.snapshot_rx.borrow().working_memory.iter().map(|m| m.id).collect();
    assert_eq!(ids_before.len(), 2, "both real, organic turns should be co-resident in Working Memory - this is what lets a real edge form between them");

    // A long, genuine silence - long enough that both fully decay out.
    clock.advance(500_000);
    for _ in 0..5 {
        actor.tick().await;
    }
    assert!(
        handles.snapshot_rx.borrow().working_memory.is_empty(),
        "both organically-formed memories should have fully decayed out of Working Memory after 500 real seconds of silence"
    );

    // A brand-new, unrelated topic arrives. It never co-occurred with
    // either decayed memory, so it has no edge to either of them.
    handles.input_tx.send("what's the capital of France?".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;

    let ids_after: Vec<MentalObjectId> = handles.snapshot_rx.borrow().working_memory.iter().map(|m| m.id).collect();
    assert!(
        ids_before.iter().all(|id| !ids_after.contains(id)),
        "an unrelated new topic must not spontaneously resurrect unrelated, fully-decayed memories just because something new is happening"
    );
}

// ---------------------------------------------------------------------
// Gap 3: content coherence - is contradiction actually detected/resolved?
// ---------------------------------------------------------------------

/// Honest documentation of a real, current gap, not a demonstrated
/// capability. `aca_types::EdgeKind::Contradicts` exists as a type, but a
/// full source grep across `aca-engine` turns up zero call sites that ever
/// construct one - there is no belief-consistency or contradiction-detection
/// mechanism anywhere in the live pipeline today. This test exists so that
/// claim stays checked over time: it is written to FAIL the moment someone
/// adds real contradiction handling, which is the correct signal to come
/// back and replace it with a real capability test for whatever was built.
#[tokio::test]
async fn contradictory_statements_are_stored_side_by_side_with_no_contradiction_ever_flagged() {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (mut actor, handles) =
        build_actor(LoopConfig::default(), clock, Arc::new(FakeEmbeddingClient::default()), Arc::new(FixedChatClient("a reply")), Vec::new());

    handles.input_tx.send("the sky is blue".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;
    handles.input_tx.send("the sky is not blue, it is green".to_string()).await.unwrap();
    actor.tick().await;
    tokio::task::yield_now().await;
    actor.tick().await;

    let snapshot = handles.snapshot_rx.borrow().clone();
    assert_eq!(snapshot.working_memory.len(), 2, "both directly contradictory statements are currently stored as separate, equally-standing objects");
    assert!(
        !snapshot.associative_edges.iter().any(|e| e.kind == EdgeKind::Contradicts),
        "no Contradicts edge is ever produced today - if this assertion now fails, contradiction detection has been implemented and this test should be replaced with a real assertion about how it resolves the conflict"
    );
}

// ---------------------------------------------------------------------
// Gap 4: a degenerate-vs-default baseline comparison, identical scenario.
// ---------------------------------------------------------------------

fn build_two_hop_chain_with_a_live_anchor() -> (Vec<MentalObject>, MentalObjectId) {
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

    (vec![dormant, intermediate, anchor], dormant_id)
}

async fn dormant_memory_resurfaces_under(config: LoopConfig) -> bool {
    let clock = Arc::new(ManualClock::new(EpochMillis(0)));
    let (objects, dormant_id) = build_two_hop_chain_with_a_live_anchor();
    let (mut actor, handles) = build_actor(config, clock.clone(), Arc::new(FakeEmbeddingClient::default()), Arc::new(FixedChatClient("a reply")), objects);

    clock.advance(100_000);
    actor.tick().await;
    actor.tick().await;

    handles.snapshot_rx.borrow().working_memory.iter().any(|m| m.id == dormant_id)
}

/// The same exact scenario, three configurations of the same architecture:
/// fully wired, recall specifically disabled, and agenda specifically
/// disabled. This is the closest honest proxy for "compare against a
/// non-cognitive baseline" available without building and justifying an
/// entirely separate system: each ablation independently defeats the
/// capability for a genuinely different structural reason (no live source to
/// recall onto, vs. recall itself switched off), which is real evidence the
/// default configuration's success isn't a coincidence of the scenario.
#[tokio::test]
async fn a_fully_wired_configuration_passes_where_two_independently_degraded_ones_fail_on_the_identical_scenario() {
    let mut default_config = LoopConfig::default();
    default_config.recall_config = RecallConfig { max_hops: 2 };
    assert!(dormant_memory_resurfaces_under(default_config).await, "the default, fully-wired configuration should recall the dormant memory");

    let mut recall_disabled = LoopConfig::default();
    recall_disabled.recall_config = RecallConfig { max_hops: 2 };
    recall_disabled.ablation_config.disable_recall = true;
    assert!(!dormant_memory_resurfaces_under(recall_disabled).await, "disabling recall specifically should defeat this capability");

    let mut agenda_disabled = LoopConfig::default();
    agenda_disabled.recall_config = RecallConfig { max_hops: 2 };
    agenda_disabled.ablation_config.disable_agenda = true;
    assert!(
        !dormant_memory_resurfaces_under(agenda_disabled).await,
        "disabling agenda should ALSO defeat this capability, for a completely independent reason: the anchor Intention is never surfaced into Working Memory at all without it, so recall never gets a live source to spread from"
    );
}

// ---------------------------------------------------------------------
// Gap 5: model independence.
// ---------------------------------------------------------------------

/// Operationalizes specs.md's Core Principle directly: "identity, memory,
/// attention, and executive function exist independently of inference."
/// Two runs of the identical scenario, differing ONLY in which model
/// (represented here by two `ChatClient`s returning completely different
/// text at the same confidence/tier) answers Tier 3. Tier-0 bookkeeping -
/// whether/how many times Speak fires, Working Memory size, every memory
/// role count - must come out identical; only the actual spoken words may
/// differ. If swapping the model changed any of the Tier-0 numbers, "which
/// model answers is a free architectural choice" would be false.
#[tokio::test]
async fn identity_and_memory_bookkeeping_are_invariant_to_which_model_answers() {
    async fn run_with(raw_text: &'static str) -> (usize, aca_engine::MemoryRoleCounts, Vec<String>) {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, mut handles) =
            build_actor(LoopConfig::default(), clock, Arc::new(FakeEmbeddingClient::default()), Arc::new(FixedChatClient(raw_text)), Vec::new());
        handles.input_tx.send("is anyone there?".to_string()).await.unwrap();

        let mut spoken = Vec::new();
        for _ in 0..20 {
            actor.tick().await;
            while let Ok(event) = handles.events_rx.try_recv() {
                if event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                    spoken.push(event.payload.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string());
                }
            }
        }
        let snapshot = handles.snapshot_rx.borrow().clone();
        (snapshot.working_memory.len(), snapshot.memory_counts, spoken)
    }

    let (wm_len_a, counts_a, spoken_a) = run_with("Reply from Model Alpha, phrased however that model happens to phrase things.").await;
    let (wm_len_b, counts_b, spoken_b) = run_with("A completely different response from Model Beta - nothing alike in wording.").await;

    assert_eq!(wm_len_a, wm_len_b, "Working Memory admission is Tier-0 bookkeeping and must not depend on which model answered");
    assert_eq!(counts_a.working, counts_b.working);
    assert_eq!(counts_a.episodic, counts_b.episodic);
    assert_eq!(counts_a.semantic, counts_b.semantic);
    assert_eq!(counts_a.self_memory, counts_b.self_memory);
    assert_eq!(spoken_a.len(), spoken_b.len(), "the decision of whether/how many times to speak must not depend on which model produced the content");
    assert_ne!(spoken_a, spoken_b, "the spoken CONTENT should differ - proving the model was genuinely swapped, not that both runs coincidentally matched");
}

// ---------------------------------------------------------------------
// Gap 6: real wall-clock time and real concurrency.
// ---------------------------------------------------------------------

/// Every other test in both suites drives `tick()` synchronously, one call
/// at a time, on a `ManualClock`. Production never does this - `omega-acad`
/// calls the actual `run()` entry point (`loop { tick().await; yield_now()
/// .await }`) as a spawned background task under `SystemClock`, fed by
/// concurrent senders (the HTTP API, MCP, voice, sensors) all writing to the
/// same `mpsc` channels at once. This test is the one place that regime is
/// actually exercised: two independent tasks push real input concurrently
/// while the actor runs continuously under real time, and the assertion is
/// simply that nothing errors and real activity is observable afterward.
#[tokio::test]
async fn the_actor_runs_continuously_under_real_wall_clock_time_and_concurrent_input_without_erroring() {
    let (actor, mut handles) =
        build_actor(LoopConfig::default(), Arc::new(SystemClock), Arc::new(FakeEmbeddingClient::default()), Arc::new(FixedChatClient("a reply")), Vec::new());

    let input_tx_a = handles.input_tx.clone();
    let input_tx_b = handles.input_tx.clone();
    let run_handle = tokio::spawn(actor.run());

    let sender_a = tokio::spawn(async move {
        for i in 0..5 {
            input_tx_a.send(format!("concurrent thought A #{i}")).await.unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    let sender_b = tokio::spawn(async move {
        for i in 0..5 {
            input_tx_b.send(format!("concurrent thought B #{i}")).await.unwrap();
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    });
    sender_a.await.unwrap();
    sender_b.await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    let mut saw_error = false;
    let mut saw_any_event = false;
    loop {
        match handles.events_rx.try_recv() {
            Ok(event) => {
                saw_any_event = true;
                if event.event_type == aca_engine::CycleEventKind::Error {
                    saw_error = true;
                }
            }
            // A continuously-ticking actor with no real I/O delay easily
            // emits more events than the 256-slot broadcast buffer holds in
            // this span - `Lagged` is itself proof real activity happened
            // (just more of it than we care to drain here), not a failure;
            // only `Empty`/`Closed` should actually end the loop.
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                saw_any_event = true;
            }
            Err(_) => break,
        }
    }
    run_handle.abort();

    assert!(saw_any_event, "the actor should have produced real cycle events while running continuously under concurrent real-time input");
    assert!(!saw_error, "no tick should error under real concurrent input and real wall-clock scheduling");
    let snapshot = handles.snapshot_rx.borrow().clone();
    assert!(
        snapshot.memory_counts.episodic > 0 || !snapshot.working_memory.is_empty(),
        "real, concurrent conversational input should have produced some observable cognitive activity"
    );
}
