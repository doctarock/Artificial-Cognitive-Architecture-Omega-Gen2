use aca_graph::{reinforce_edge, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH};
use aca_types::{EdgeKind, MemoryRole, MentalObject, MentalObjectId, MentalObjectKind};
use aca_util::EpochMillis;
use serde::{Deserialize, Serialize};

use super::executive::{Operator, OperatorProposal};

use aca_graph::Graph;

/// A freshly-created chunk's starting utility — chosen to match the old flat
/// `LEARNED_PREFERENCE_BONUS` this replaces, so a chunk's first application
/// (before any reward feedback has ever reached it - see
/// `reinforce_chunk_utility`) decisively breaks a tie exactly as before.
/// Everything after that first use is earned, not assumed.
const INITIAL_CHUNK_UTILITY: f32 = 0.5;

/// ACT-R's own utility-learning rate (its conventional default, per specs.md
/// line 173-175's "production rules with learned utility" - this constant is
/// what actually builds that half of the theory, chunking alone only ever
/// implemented the SOAR half). Applied as a Rescorla-Wagner-style update:
/// `U(n) = U(n-1) + α · (R(n) − U(n-1))` — utility drifts toward whatever
/// reward a chunk's recommendation has actually been earning lately, rather
/// than staying fixed at its birth value forever.
const UTILITY_LEARNING_RATE: f32 = 0.2;

#[derive(Debug, Serialize, Deserialize)]
struct ChunkPayload {
    situation_operators: Vec<Operator>,
    resolved_operator: Operator,
    /// Learned utility (see `reinforce_chunk_utility`) — the actual
    /// preference bonus this chunk contributes in `apply_learned_bias`,
    /// starting at `INITIAL_CHUNK_UTILITY` and adjusted every time this
    /// chunk's recommendation is followed and its consequence observed.
    utility: f32,
}

fn situation_signature(operators: &[Operator]) -> Vec<Operator> {
    let mut sorted: Vec<Operator> = operators.to_vec();
    sorted.sort_by_key(|op| format!("{op:?}"));
    sorted
}

/// Step 9 - Learn (SOAR chunking half): compiles a resolved impasse into a
/// semantic-memory Mental Object — "in situations like this (this set of
/// tied/ambiguous operators), prefer this one" — plus a reinforced edge
/// from each of the impasse's original candidate targets toward the chunk.
/// Named simplification: this is a summarized-resolution record, not a
/// compiled executable production rule (see the plan). Returns the new
/// chunk's id.
pub fn chunk_resolution(
    graph: &mut Graph,
    impasse_candidates: &[OperatorProposal],
    resolved_operator: Operator,
    resolution_confidence: f32,
    now: EpochMillis,
    decay_d: f32,
) -> MentalObjectId {
    let situation_operators = situation_signature(&impasse_candidates.iter().map(|p| p.operator).collect::<Vec<_>>());
    let payload = ChunkPayload {
        situation_operators: situation_operators.clone(),
        resolved_operator,
        utility: INITIAL_CHUNK_UTILITY,
    };

    let mut chunk = MentalObject::new_observation(
        format!("learned: among {situation_operators:?}, prefer {resolved_operator:?}"),
        now,
        decay_d,
    );
    chunk.kind = MentalObjectKind::Memory;
    chunk.memory_roles.push(MemoryRole::Semantic);
    chunk.confidence = resolution_confidence;
    chunk.data = serde_json::to_value(&payload).expect("ChunkPayload always serializes");
    let chunk_id = chunk.id;

    for proposal in impasse_candidates {
        if let Some(source) = graph.get_mut(&proposal.target_id) {
            reinforce_edge(
                &mut source.edges,
                chunk_id,
                EdgeKind::DerivedFrom,
                now,
                DEFAULT_HEBBIAN_INCREMENT,
                DEFAULT_MAX_EDGE_STRENGTH,
            );
        }
    }

    graph.insert(chunk);
    chunk_id
}

/// Looks for a previously-learned chunk matching this exact set of tied
/// operators (order-independent) and returns its id, resolved operator, and
/// current learned utility, if any.
fn find_learned_chunk(graph: &Graph, situation_operators: &[Operator]) -> Option<(MentalObjectId, Operator, f32)> {
    let target_signature = situation_signature(situation_operators);
    graph
        .iter()
        .filter(|object| object.memory_roles.contains(&MemoryRole::Semantic))
        .find_map(|object| {
            let payload: ChunkPayload = serde_json::from_value(object.data.clone()).ok()?;
            if situation_signature(&payload.situation_operators) == target_signature {
                Some((object.id, payload.resolved_operator, payload.utility))
            } else {
                None
            }
        })
}

/// Applies any learned bias to a fresh set of proposals *before* they reach
/// `select_operator`: if a chunk previously resolved this exact situation
/// (same set of candidate operators), that operator's preference gets a
/// bonus equal to the chunk's current learned utility — turning what would
/// otherwise be a repeat `Confidence` impasse into a fast, direct `Selected`
/// outcome. This is the concrete mechanism behind "the same ambiguity
/// doesn't have to be re-litigated by the executive next time," now with the
/// size of the nudge itself earned rather than fixed (see
/// `reinforce_chunk_utility`).
///
/// Returns the matched chunk's id and the operator it biased, so the caller
/// can register a `metacognition::PendingOutcome` for it — but only once
/// `select_operator` confirms that operator actually won; a chunk that
/// merely existed this tick without deciding the outcome earns no credit or
/// blame for what happens next.
pub fn apply_learned_bias(graph: &Graph, proposals: &mut [OperatorProposal]) -> Option<(MentalObjectId, Operator)> {
    let situation: Vec<Operator> = proposals.iter().map(|p| p.operator).collect();
    let (chunk_id, learned_operator, utility) = find_learned_chunk(graph, &situation)?;
    for proposal in proposals.iter_mut() {
        if proposal.operator == learned_operator {
            proposal.preference += utility;
        }
    }
    Some((chunk_id, learned_operator))
}

/// Step 9 - Learn (ACT-R procedural-utility half): updates a chunk's learned
/// utility from the reward its most recent recommendation actually earned,
/// via ACT-R's own utility-learning equation (specs.md line 173-175:
/// "production rules with learned utility"):
///
/// `U(n) = U(n-1) + α · (R(n) − U(n-1))`
///
/// This is what makes chunking genuinely reinforcement learning rather than
/// a one-shot cache: a chunk whose recommendation keeps paying off drifts
/// toward a larger, more decisive bonus, and one that stops paying off
/// drifts back down — without ever needing `select_operator` or
/// `apply_learned_bias` themselves to change. `reward` is expected to come
/// from `metacognition::reward_from_comparison`, itself Tier-0 (no LLM
/// call): reusing predictive processing's own prediction-error currency as
/// the reinforcement signal is what keeps this composing with the rest of
/// the architecture instead of bolting on a separate reward mechanism.
/// A no-op if `chunk_id` doesn't resolve to a real chunk (already discarded,
/// or a stale id) or its payload doesn't parse as a `ChunkPayload`.
pub fn reinforce_chunk_utility(graph: &mut Graph, chunk_id: MentalObjectId, reward: f32) {
    let Some(chunk) = graph.get_mut(&chunk_id) else { return };
    let Ok(mut payload) = serde_json::from_value::<ChunkPayload>(chunk.data.clone()) else { return };
    payload.utility += UTILITY_LEARNING_RATE * (reward - payload.utility);
    chunk.data = serde_json::to_value(&payload).expect("ChunkPayload always serializes");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steps::executive::{select_operator, ExecutiveConfig, ExecutiveDecision, ImpasseKind};

    fn tied_proposals() -> Vec<OperatorProposal> {
        vec![
            OperatorProposal { operator: Operator::Speak, target_id: MentalObjectId::new(), preference: 0.7, confidence: 0.9 },
            OperatorProposal { operator: Operator::Ask, target_id: MentalObjectId::new(), preference: 0.7, confidence: 0.9 },
        ]
    }

    #[test]
    fn chunk_resolution_writes_a_semantic_memory_object() {
        let mut graph = Graph::new();
        let candidates = tied_proposals();
        let chunk_id = chunk_resolution(&mut graph, &candidates, Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

        let chunk = graph.get(&chunk_id).expect("chunk should be inserted");
        assert!(chunk.memory_roles.contains(&MemoryRole::Semantic));
        assert_eq!(chunk.kind, MentalObjectKind::Memory);
    }

    #[test]
    fn chunk_resolution_reinforces_edges_from_original_candidates() {
        let mut graph = Graph::new();
        let candidates = tied_proposals();
        let target_ids: Vec<_> = candidates.iter().map(|p| p.target_id).collect();
        // Insert placeholder objects for the candidate targets so we can
        // observe their edges after chunking.
        for &id in &target_ids {
            let mut obj = MentalObject::new_observation("wm member", EpochMillis(0), 0.5);
            obj.id = id;
            graph.insert(obj);
        }

        let chunk_id = chunk_resolution(&mut graph, &candidates, Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

        for &id in &target_ids {
            let obj = graph.get(&id).unwrap();
            assert_eq!(obj.edges.len(), 1);
            assert_eq!(obj.edges[0].target_id, chunk_id);
        }
    }

    #[test]
    fn the_same_impasse_resolves_differently_the_second_time() {
        let mut graph = Graph::new();

        // First occurrence: a genuine tie -> Confidence impasse.
        let first_attempt = tied_proposals();
        let first_decision = select_operator(&first_attempt, &ExecutiveConfig::default());
        assert!(matches!(
            first_decision,
            ExecutiveDecision::Impasse { kind: ImpasseKind::Confidence, .. }
        ));

        // The impasse is resolved (e.g. via tier escalation) in favor of
        // Speak, and the resolution is chunked.
        chunk_resolution(&mut graph, &first_attempt, Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

        // Second occurrence: the identical situation recurs. Without
        // learning, this would be the same Confidence impasse again.
        let mut second_attempt = tied_proposals();
        apply_learned_bias(&graph, &mut second_attempt);
        let second_decision = select_operator(&second_attempt, &ExecutiveConfig::default());

        match second_decision {
            ExecutiveDecision::Selected(winner) => assert_eq!(winner.operator, Operator::Speak),
            other => panic!("expected the learned chunk to resolve this decisively, got {other:?}"),
        }
    }

    #[test]
    fn unrelated_situations_are_not_affected_by_an_unrelated_chunk() {
        let mut graph = Graph::new();
        chunk_resolution(&mut graph, &tied_proposals(), Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

        let mut unrelated = vec![
            OperatorProposal { operator: Operator::Remember, target_id: MentalObjectId::new(), preference: 0.6, confidence: 0.9 },
            OperatorProposal { operator: Operator::Ignore, target_id: MentalObjectId::new(), preference: 0.6, confidence: 0.9 },
        ];
        let preferences_before: Vec<f32> = unrelated.iter().map(|p| p.preference).collect();
        apply_learned_bias(&graph, &mut unrelated);
        let preferences_after: Vec<f32> = unrelated.iter().map(|p| p.preference).collect();

        assert_eq!(preferences_before, preferences_after, "an unrelated chunk must not bias an unrelated situation");
    }

    #[test]
    fn apply_learned_bias_reports_the_chunk_it_matched() {
        let mut graph = Graph::new();
        let candidates = tied_proposals();
        let chunk_id = chunk_resolution(&mut graph, &candidates, Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

        let mut second_attempt = tied_proposals();
        let applied = apply_learned_bias(&graph, &mut second_attempt);

        assert_eq!(applied, Some((chunk_id, Operator::Speak)));
    }

    #[test]
    fn apply_learned_bias_reports_nothing_when_no_chunk_matches() {
        let graph = Graph::new();
        let mut proposals = tied_proposals();
        assert_eq!(apply_learned_bias(&graph, &mut proposals), None);
    }

    #[test]
    fn reinforce_chunk_utility_moves_toward_the_reward_over_repeated_use() {
        let mut graph = Graph::new();
        let chunk_id = chunk_resolution(&mut graph, &tied_proposals(), Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

        // Repeatedly reward this chunk's recommendation with 1.0 - utility
        // should climb from its INITIAL_CHUNK_UTILITY seed toward 1.0.
        for _ in 0..20 {
            reinforce_chunk_utility(&mut graph, chunk_id, 1.0);
        }

        let mut proposals = tied_proposals();
        apply_learned_bias(&graph, &mut proposals);
        let speak_preference = proposals.iter().find(|p| p.operator == Operator::Speak).unwrap().preference;
        // Started at 0.7 + INITIAL_CHUNK_UTILITY (0.5) = 1.2; after enough
        // reward-1.0 updates the bonus should have grown well past that.
        assert!(speak_preference > 1.2, "utility should have grown toward the reward, got preference {speak_preference}");
    }

    #[test]
    fn reinforce_chunk_utility_shrinks_the_bonus_when_the_reward_is_poor() {
        let mut graph = Graph::new();
        let chunk_id = chunk_resolution(&mut graph, &tied_proposals(), Operator::Speak, 0.9, EpochMillis(1_000), 0.5);

        for _ in 0..20 {
            reinforce_chunk_utility(&mut graph, chunk_id, 0.0);
        }

        let mut proposals = tied_proposals();
        apply_learned_bias(&graph, &mut proposals);
        let speak_preference = proposals.iter().find(|p| p.operator == Operator::Speak).unwrap().preference;
        // Started at 0.7 + INITIAL_CHUNK_UTILITY (0.5) = 1.2; repeated
        // zero-reward feedback should have driven the bonus down toward 0,
        // shrinking the preference back toward the base 0.7.
        assert!(speak_preference < 0.8, "utility should have shrunk toward the reward, got preference {speak_preference}");
    }

    #[test]
    fn reinforce_chunk_utility_is_a_no_op_for_an_unknown_chunk_id() {
        let mut graph = Graph::new();
        // Should not panic even though this id was never inserted.
        reinforce_chunk_utility(&mut graph, MentalObjectId::new(), 1.0);
    }
}
