//! A runnable acceptance suite for Omega's cognitive capabilities, one test
//! per mechanism specs.md names as a theoretical foundation or design
//! objective. Unlike the unit tests scattered through each `steps::*`
//! module (which check one function in isolation with synthetic numbers),
//! these compose real public building blocks - real embeddings, real ACT-R
//! math, real GWT arbitration - the same way `full_cycle.rs` does, so a
//! failure here means an actual emergent capability regressed, not just an
//! internal implementation detail.
//!
//! Run with: `cargo test -p aca-engine --test cognitive_capabilities`

use std::collections::HashMap;
use std::collections::HashSet;

use aca_engine::steps::learn::reinforce_chunk_utility;
use aca_engine::{
    apply_learned_bias, broadcast, chunk_resolution, compare, decide_admission_deterministic, form_coalition, impasse_response, resolve_embedding,
    seed_self_memory_objects, select_operator, spawn_subgoal, CoalitionCandidate, ExecutiveConfig, ExecutiveDecision, ImpasseKind, ImpasseResponse, LoopConfig,
    Operator, OperatorProposal, PrecisionTracker, SourceChannel, DEFAULT_WORKING_MEMORY_CAPACITY,
};
use aca_graph::{compute_base_level, recompute_activation, spread_activation_multi_hop, Graph};
use aca_tiers::testing::FakeEmbeddingClient;
use aca_types::{AssociativeEdge, EdgeKind, MentalObject, MentalObjectId, MentalObjectKind};
use aca_util::{Clock, EpochMillis, ManualClock, RingBuffer};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// **Predictive Processing / Attention** (specs.md "Attention"): a candidate
/// whose content genuinely surprises the generative model (nothing like it
/// was predicted) clears the attention threshold; a candidate that exactly
/// matches what was already expected does not, even at identical underlying
/// ACT-R activation. This is the mechanism specs.md credits with making
/// "continuous" affordable: most content never reaches the expensive
/// Cognitive Core.
#[tokio::test]
async fn surprising_input_passes_the_attention_gate_while_a_correctly_predicted_repeat_is_filtered() {
    let embedding_client = FakeEmbeddingClient::default();
    let mut tracker = PrecisionTracker::default();

    let novel_embedding = resolve_embedding(&embedding_client, "a message never seen before").await.unwrap();
    let novel_comparison = compare(None, &novel_embedding, SourceChannel::ConversationInput, None, &mut tracker);

    let predicted_embedding = resolve_embedding(&embedding_client, "the usual greeting").await.unwrap();
    let mundane_comparison = compare(Some(&predicted_embedding), &predicted_embedding, SourceChannel::ConversationInput, None, &mut tracker);

    assert!(
        novel_comparison.error_magnitude > mundane_comparison.error_magnitude,
        "an unpredicted observation must be more surprising than one that exactly matched the prediction"
    );

    // Equal underlying activation for both, so only the surprise term can
    // decide which one clears the gate.
    let novel_id = MentalObjectId::new();
    let mundane_id = MentalObjectId::new();
    let raw_candidates = vec![
        (novel_id, 0.5, Some(novel_comparison.precision_weighted_surprise)),
        (mundane_id, 0.5, Some(mundane_comparison.precision_weighted_surprise)),
    ];
    let coalition = form_coalition(&raw_candidates, 1.0);

    assert!(coalition.iter().any(|c| c.id == novel_id), "genuinely surprising content should clear the attention threshold");
    assert!(
        !coalition.iter().any(|c| c.id == mundane_id),
        "content that exactly matched prediction should not clear the attention threshold on activation alone"
    );
}

/// **ACT-R Memory Decay and Reinforcement**: an unreinforced memory's
/// activation falls below the retrieval threshold over time (power-law
/// forgetting), and a single fresh reference (a recall or reinforcement)
/// restores it - "reinforcement restores activation," never storage itself.
/// Numbers here are grounded in `LoopConfig::default()`'s own documented
/// reasoning (a lone reference decays out of Working Memory around the
/// ~55-second mark at `decay_d = 0.5`, `attention_threshold = -2.0`).
#[test]
fn dormant_memory_decays_out_and_reinforcement_restores_it() {
    let config = LoopConfig::default();
    let mut log = RingBuffer::new(64);
    log.push(EpochMillis(0));

    let still_fresh = compute_base_level(&log, config.decay_d, EpochMillis(10_000));
    assert!(
        clears_retrieval_threshold_helper(still_fresh, config.attention_threshold),
        "a memory referenced 10s ago should still clear the retrieval threshold, got {still_fresh}"
    );

    let now_dormant = compute_base_level(&log, config.decay_d, EpochMillis(120_000));
    assert!(
        !clears_retrieval_threshold_helper(now_dormant, config.attention_threshold),
        "an unreinforced memory should have decayed below the retrieval threshold after 120s, got {now_dormant}"
    );

    // Reinforcement: a fresh reference (recall/co-activation) is recorded at
    // the dormant instant.
    log.push(EpochMillis(120_000));
    let reinforced = compute_base_level(&log, config.decay_d, EpochMillis(120_001));
    assert!(
        clears_retrieval_threshold_helper(reinforced, config.attention_threshold),
        "reinforcement should restore activation above the retrieval threshold, got {reinforced}"
    );
    assert!(reinforced > now_dormant, "reinforced activation ({reinforced}) should exceed the dormant reading ({now_dormant})");
}

fn clears_retrieval_threshold_helper(total: f32, threshold: f32) -> bool {
    aca_graph::clears_retrieval_threshold(total, threshold)
}

/// **Associative Memory (ACT-R spreading activation)**: a memory two
/// associative hops from anything currently in Working Memory has no direct
/// edge to spread from - `compute_spreading_activation`'s own single-hop
/// mechanism structurally cannot reach it - yet it is still recoverable
/// through the graph-wide multi-hop mechanism specs.md calls out ("dormant
/// memories remain recoverable through strong associative activation from a
/// sufficiently related context"), and that recovered activation is real
/// enough to rescue an otherwise too-decayed memory back into Coalition
/// contention.
#[test]
fn a_two_hop_dormant_memory_is_rescued_into_coalition_contention_by_spreading_activation() {
    let mut graph = Graph::new();
    let now = EpochMillis(0);

    let dormant = MentalObject::new_observation("a half-forgotten related idea", now, 0.5);
    let dormant_id = dormant.id;
    let mut intermediate = MentalObject::new_observation("a bridging thought", now, 0.5);
    intermediate.edges.push(AssociativeEdge { target_id: dormant_id, kind: EdgeKind::Associative, strength: 0.6, last_coactivated_at: now });
    let intermediate_id = intermediate.id;
    let mut source = MentalObject::new_observation("what's currently in Working Memory", now, 0.5);
    source.edges.push(AssociativeEdge { target_id: intermediate_id, kind: EdgeKind::Associative, strength: 0.7, last_coactivated_at: now });
    let source_id = source.id;

    graph.insert(dormant);
    graph.insert(intermediate);
    graph.insert(source);

    let one_hop = spread_activation_multi_hop(&graph, &[source_id], 1, 0.0, now);
    assert!(!one_hop.contains_key(&dormant_id), "a two-hop-away memory must not be reachable in a single hop");

    let two_hop = spread_activation_multi_hop(&graph, &[source_id], 2, 0.0, now);
    let recovered_spreading = *two_hop.get(&dormant_id).expect("a two-hop dormant memory should be discovered by multi-hop spreading activation");
    assert!(recovered_spreading > 0.0);

    // Simulate a long-decayed base-level activation, too weak to compete on
    // its own, but not so weak that spreading activation can't rescue it.
    let decayed_base_level = -2.3;
    let attention_threshold = -2.0;

    let isolated = form_coalition(&[(dormant_id, decayed_base_level, None)], attention_threshold);
    assert!(isolated.is_empty(), "on its own, a sufficiently decayed memory should not clear the attention threshold");

    let rescued = form_coalition(&[(dormant_id, decayed_base_level + recovered_spreading, None)], attention_threshold);
    assert_eq!(
        rescued.len(),
        1,
        "spreading activation from currently-active context should rescue an otherwise-too-decayed dormant memory into coalition contention"
    );
}

/// **Global Workspace Theory / Working Memory capacity**: broadcast is the
/// single, capacity-limited arbitration point. Even when many candidates are
/// all independently worth attending to, only the strongest
/// `DEFAULT_WORKING_MEMORY_CAPACITY` survive into Working Memory - "nothing
/// else can independently process the same input this cycle."
#[test]
fn broadcast_admits_only_the_strongest_candidates_up_to_capacity() {
    let mut graph = Graph::new();
    let now = EpochMillis(0);

    let mut candidates = Vec::new();
    for i in 0..10 {
        let object = MentalObject::new_observation(format!("candidate thought #{i}"), now, 0.5);
        let id = object.id;
        graph.insert(object);
        // Distinct, unambiguous scores: candidate i's score is i.0.
        candidates.push(CoalitionCandidate { id, score: i as f32 });
    }

    let admitted = decide_admission_deterministic(&candidates, DEFAULT_WORKING_MEMORY_CAPACITY);
    let result = broadcast(&mut graph, &admitted, &HashSet::new(), now);

    assert_eq!(
        result.working_memory.len(),
        DEFAULT_WORKING_MEMORY_CAPACITY,
        "Working Memory must never exceed its configured capacity, no matter how many candidates were eligible"
    );

    // The winners must be exactly the top-scoring candidates (ids 9,8,7,6).
    let scores_by_id: HashMap<MentalObjectId, f32> = candidates.iter().map(|c| (c.id, c.score)).collect();
    let winning_scores: Vec<f32> = result.working_memory.iter().map(|id| scores_by_id[id]).collect();
    let mut expected: Vec<f32> = (6..10).map(|i| i as f32).collect();
    expected.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(winning_scores, expected, "the admitted set should be exactly the top-{DEFAULT_WORKING_MEMORY_CAPACITY} candidates by score, strongest first");
}

/// **SOAR / Autonomous thought**: when nothing broadcast this cycle can be
/// decisively acted on, the Executive does not stall - it spawns a subgoal
/// Mental Object that is itself a real, activation-bearing candidate ready
/// to compete in the *next* cycle's Coalition. This is specs.md's
/// "unresolved impasses are a renewable source of cognition, not a special
/// idle-reflection bolt-on."
#[test]
fn a_missing_information_impasse_spawns_a_subgoal_that_re_enters_coalition_next_cycle() {
    let decision = select_operator(&[], &ExecutiveConfig::default());
    let ExecutiveDecision::Impasse { kind, .. } = decision else {
        panic!("no proposals at all should be classified as an impasse, got {decision:?}");
    };
    assert_eq!(kind, ImpasseKind::MissingInformation);
    assert_eq!(impasse_response(kind), ImpasseResponse::SpawnSubgoal);

    let mut graph = Graph::new();
    let clock = ManualClock::new(EpochMillis(0));
    let mut rng = StdRng::seed_from_u64(1);

    let subgoal_id = spawn_subgoal(&mut graph, "nothing broadcast this cycle was clearly worth acting on", clock.now(), 0.5);
    let subgoal = graph.get(&subgoal_id).expect("the subgoal must actually be inserted into the graph");
    assert_eq!(subgoal.kind, MentalObjectKind::Question, "a spawned subgoal is a Question Mental Object, not a special-cased type");

    // It must be a real candidate next cycle, not inert bookkeeping: recompute
    // its activation and confirm it clears the same coalition gate any other
    // fresh candidate would.
    {
        let object = graph.get_mut(&subgoal_id).unwrap();
        recompute_activation(&mut object.activation, subgoal_id, &[], 0.0, 0.0, &mut rng, &clock);
    }
    let activation_total = graph.get(&subgoal_id).unwrap().activation.total;
    let coalition = form_coalition(&[(subgoal_id, activation_total, None)], 0.1);
    assert_eq!(coalition.len(), 1, "a freshly-spawned subgoal should be a genuine Coalition candidate on the very next cycle");
}

/// **SOAR chunking + ACT-R utility learning / Continuous learning**: the
/// *same* ambiguous situation (a genuine tie between two operators)
/// recurring three times should get faster and more decisive each time, as
/// the chunk's utility drifts toward the reward its recommendation keeps
/// earning - "the same ambiguity doesn't have to be re-litigated by the
/// executive next time," and this is real reinforcement learning, not a
/// one-shot cache (specs.md's SOAR + ACT-R sections).
#[test]
fn a_recurring_impasse_is_resolved_more_decisively_each_time_it_recurs() {
    let mut graph = Graph::new();

    fn tied_situation() -> Vec<OperatorProposal> {
        vec![
            OperatorProposal { operator: Operator::Speak, target_id: MentalObjectId::new(), preference: 0.7, confidence: 0.9 },
            OperatorProposal { operator: Operator::Ask, target_id: MentalObjectId::new(), preference: 0.7, confidence: 0.9 },
        ]
    }

    // Occurrence 1: a genuine, unresolved tie - this IS the impasse.
    let first = tied_situation();
    let first_decision = select_operator(&first, &ExecutiveConfig::default());
    assert!(matches!(first_decision, ExecutiveDecision::Impasse { kind: ImpasseKind::Confidence, .. }));

    // The impasse resolves (e.g. via Tier 3/4 escalation) in favor of Speak,
    // and SOAR compiles that resolution into a chunk.
    let chunk_id = chunk_resolution(&mut graph, &first, Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

    let mut margins = Vec::new();
    for occurrence in 0..3 {
        let mut situation = tied_situation();
        apply_learned_bias(&graph, &mut situation);
        let decision = select_operator(&situation, &ExecutiveConfig::default());
        let winner = match decision {
            ExecutiveDecision::Selected(winner) => winner,
            other => panic!("occurrence {occurrence}: expected the learned chunk to resolve this decisively, got {other:?}"),
        };
        assert_eq!(winner.operator, Operator::Speak, "the chunk should keep recommending the operator it learned");

        let runner_up_preference = situation.iter().find(|p| p.operator != Operator::Speak).unwrap().preference;
        margins.push(winner.preference - runner_up_preference);

        // The recommendation paid off again - reward it, same as a real
        // consequence reaching `metacognition::reward_from_comparison`.
        reinforce_chunk_utility(&mut graph, chunk_id, 1.0);
    }

    assert!(
        margins.windows(2).all(|w| w[1] >= w[0]),
        "the winning margin should never shrink across repeated, consistently-rewarded occurrences of the same impasse: {margins:?}"
    );
    assert!(margins[2] > margins[0], "three rounds of positive reinforcement should make the decision measurably more decisive: {margins:?}");
}

/// **Self Memory / Persistent identity**: specs.md's Self Memory objective
/// is "rarely lose the competition for relevance even when dormant" - not
/// "never decays." Using the architecture's own shipped defaults
/// (`LoopConfig::self_memory_activation_bonus`, `attention_threshold`), a
/// Self Memory belief should still clear the attention threshold after 10
/// days of total silence, while an ordinary memory with the same one-time
/// reference has long since decayed out - and even Self Memory's own
/// elevated baseline eventually fades given long enough neglect, confirming
/// this is a durability bonus, not identity floating free of the same ACT-R
/// decay math everything else obeys.
#[test]
fn self_memory_outlasts_ordinary_memory_but_is_not_exempt_from_decay() {
    let config = LoopConfig::default();
    let now = EpochMillis(0);

    let self_belief = seed_self_memory_objects(now, config.decay_d).into_iter().next().unwrap();
    let ordinary_memory = MentalObject::new_observation("I went for a walk yesterday", now, config.decay_d);

    let ten_days_ms = 10 * 24 * 60 * 60 * 1000;
    let base_level_after_ten_days = compute_base_level(&self_belief.activation.reference_log, config.decay_d, EpochMillis(ten_days_ms));
    // Both objects share an identical single-reference activation history at
    // this point - the only difference the rest of the assertion isolates is
    // the Self Memory baseline bonus itself.
    assert_eq!(compute_base_level(&ordinary_memory.activation.reference_log, config.decay_d, EpochMillis(ten_days_ms)), base_level_after_ten_days);

    let ordinary_total = base_level_after_ten_days;
    let self_total = base_level_after_ten_days + config.self_memory_activation_bonus;

    assert!(
        !clears_retrieval_threshold_helper(ordinary_total, config.attention_threshold),
        "an ordinary, unreinforced memory should have decayed out of relevance after 10 days"
    );
    assert!(
        clears_retrieval_threshold_helper(self_total, config.attention_threshold),
        "Self Memory's baseline bonus should keep identity content competitive after the same 10 days of silence"
    );

    // But the bonus delays decay, it does not suspend it: far enough out,
    // even Self Memory fades.
    let twenty_days_ms = 20 * 24 * 60 * 60 * 1000;
    let base_level_after_twenty_days = compute_base_level(&self_belief.activation.reference_log, config.decay_d, EpochMillis(twenty_days_ms));
    let self_total_at_twenty_days = base_level_after_twenty_days + config.self_memory_activation_bonus;
    assert!(
        !clears_retrieval_threshold_helper(self_total_at_twenty_days, config.attention_threshold),
        "even Self Memory should eventually fade given long enough total neglect - the bonus is not an exemption from ACT-R decay"
    );
}
