use std::collections::{HashMap, HashSet};

use aca_graph::{admit_top_n, rank_candidates, reinforce_edge, Graph, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH};
use aca_tiers::{AttentionDecision, AttentionOp};
use aca_types::{EdgeKind, MemoryRole, MentalObjectId};
use aca_util::EpochMillis;

use super::coalition::CoalitionCandidate;

/// Default Working Memory capacity — intentionally small (specs.md: "a
/// small number of concurrent Mental Objects, psychologically-plausible
/// values are in the 3-5 range"), a tuning parameter rather than a
/// theoretical commitment.
pub const DEFAULT_WORKING_MEMORY_CAPACITY: usize = 4;

/// What changed as a result of one Step 6 broadcast: the full new Working
/// Memory set (ranked, strongest first), which ids are newly admitted this
/// cycle (weren't in Working Memory last cycle), and which are released
/// (were in Working Memory last cycle, lost the competition this cycle).
/// The caller (the future `CognitiveLoopActor`) keeps `working_memory` as
/// its persistent state and passes it back in as `previous_working_memory`
/// next tick.
#[derive(Debug, Clone, Default)]
pub struct BroadcastResult {
    pub working_memory: Vec<MentalObjectId>,
    pub newly_admitted: Vec<MentalObjectId>,
    pub released: Vec<MentalObjectId>,
}

/// Step 6's deterministic admission rule (GWT's competition step, unchanged
/// from the original single-function `broadcast`): rank every Coalition
/// candidate by score, admit the top `capacity`. This is the mandatory
/// fallback path whenever the attention model is unconfigured, times out,
/// errors, or answers below `attention_min_confidence` — the cognitive
/// loop's liveness never depends on an unreachable local model.
pub fn decide_admission_deterministic(candidates: &[CoalitionCandidate], capacity: usize) -> Vec<(MentalObjectId, f32)> {
    let scores: HashMap<MentalObjectId, f32> = candidates.iter().map(|c| (c.id, c.score)).collect();
    let ranked = rank_candidates(&scores);
    admit_top_n(&ranked, capacity)
        .into_iter()
        .map(|id| (id, scores[&id]))
        .collect()
}

/// Ignition hysteresis (`LoopConfig::ignition_threshold`'s own doc comment
/// has the full ACT-R-grounded reasoning behind the default gap): the same
/// rank-and-admit-top-N rule `decide_admission_deterministic` already
/// applies, over a narrower pool. `candidates` has already been filtered by
/// Coalition at the lower `attention_threshold` - the "still a real
/// candidate at all" floor - so a member already in
/// `previous_working_memory` stays eligible at that same floor (sustaining
/// an already-ignited percept is the easy case), while anything not
/// previously admitted must additionally clear the stricter
/// `ignition_threshold` to be eligible to newly enter (crossing into
/// consciousness for the first time is the hard case). This is Dehaene's
/// own asymmetry, made literal and closed-form rather than a trained
/// judgment call - the same "no model where a deterministic rule already
/// suffices" reasoning `docs/coherence-arbitration-policy.md` applies
/// elsewhere in this engine.
///
/// Deliberately reuses `decide_admission_deterministic` for the actual
/// ranking rather than duplicating it - hysteresis only changes who's
/// *eligible* to compete this tick, never how the competition itself is
/// judged. A previously-admitted member that stays eligible can still lose
/// its seat to capacity pressure from a stronger competitor, exactly as
/// before (see `steps::displacement`) - hysteresis makes leaving harder to
/// trigger, not impossible once triggered.
pub fn decide_admission_with_hysteresis(candidates: &[CoalitionCandidate], previous_working_memory: &HashSet<MentalObjectId>, capacity: usize, ignition_threshold: f32) -> Vec<(MentalObjectId, f32)> {
    let eligible: Vec<CoalitionCandidate> = candidates.iter().copied().filter(|c| previous_working_memory.contains(&c.id) || c.score >= ignition_threshold).collect();
    decide_admission_deterministic(&eligible, capacity)
}

/// A parsed UUID still may not identify an eligible candidate this tick.
/// Treat that as an unusable actionable model vote so the actor falls back
/// to full deterministic admission, not a confident no-op.
pub fn attention_decision_target_is_eligible(decision: &AttentionDecision, coalition: &[CoalitionCandidate]) -> bool {
    match decision.operation {
        AttentionOp::Attend | AttentionOp::Switch | AttentionOp::Suppress => decision.target.is_some_and(|target| coalition.iter().any(|candidate| candidate.id == target)),
        AttentionOp::Maintain | AttentionOp::Ignore => true,
    }
}

/// Applies the Omega Attention model's single per-tick state-transition
/// decision directly to the persistent Working Memory set — this is what
/// "the model replaces Broadcast" means in practice: the model never emits
/// a ranked admission list (it wasn't trained to), so each tick's one
/// ATTEND/MAINTAIN/SWITCH/SUPPRESS/IGNORE call incrementally evolves
/// `previous_working_memory` instead of re-ranking everything from scratch.
/// See `training/attention-v0/SCHEMA_V0_2.md`.
///
/// - `Attend`/`Switch`: admits `target` (a no-op if it isn't a real
///   candidate this tick, or is already admitted). If this pushes Working
///   Memory over `capacity`, evicts the current weakest member by real
///   score — never the just-named target, since the model explicitly chose
///   it.
/// - `Maintain`: no voluntary membership change. A member absent from this
///   tick's real candidate set, or below Step 5's attention floor, is always
///   released, regardless of operation.
/// - `Suppress`: evicts `target` if present; otherwise a no-op.
/// - `Ignore`, or an unresolvable `target`: no-op — abstention, identical
///   in effect to a timed-out/unconfigured call.
///
/// Each returned member's score is its real `activation_total +
/// surprise.unwrap_or(0.0)` from `raw_candidates` when it's still a
/// candidate this tick, so `workspace.attention_score`'s meaning stays
/// identical regardless of which path produced the admission.
pub fn decide_admission_from_attention_model(
    previous_working_memory: &HashSet<MentalObjectId>,
    raw_candidates: &[(MentalObjectId, f32, Option<f32>)],
    decision: &AttentionDecision,
    attention_threshold: f32,
    capacity: usize,
) -> Vec<(MentalObjectId, f32)> {
    let real_score = |id: &MentalObjectId| -> f32 {
        raw_candidates
            .iter()
            .find(|(candidate_id, _, _)| candidate_id == id)
            .map(|(_, activation_total, surprise)| activation_total + surprise.unwrap_or(0.0))
            .unwrap_or(0.0)
    };

    // The specialist arbitrates *within* Step 5 Coalition; it cannot admit
    // an object that failed the architecture's attention floor, nor keep an
    // old winner that is no longer offered as an eligible candidate.
    let mut current: Vec<(MentalObjectId, f32)> = previous_working_memory
        .iter()
        .filter(|id| raw_candidates.iter().any(|(candidate_id, _, _)| candidate_id == *id && real_score(id) >= attention_threshold))
        .map(|id| (*id, real_score(id)))
        .collect();

    match decision.operation {
        AttentionOp::Attend | AttentionOp::Switch => {
            let Some(target) = decision.target else { return current };
            if !raw_candidates.iter().any(|(id, _, _)| *id == target) || !(real_score(&target) >= attention_threshold) {
                return current; // not a real candidate this tick -- no-op
            }
            let already_admitted = current.iter().any(|(id, _)| *id == target);
            if !already_admitted && capacity > 0 {
                if current.len() >= capacity
                    && let Some((weakest_idx, _)) =
                        current.iter().enumerate().min_by(|(_, (_, a)), (_, (_, b))| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                {
                    current.remove(weakest_idx);
                }
                current.push((target, real_score(&target)));
            }
        }
        AttentionOp::Suppress => {
            if let Some(target) = decision.target {
                current.retain(|(id, _)| *id != target);
            }
        }
        AttentionOp::Maintain | AttentionOp::Ignore => {
            // No membership change -- the returned set (with refreshed
            // real scores) still drives the unconditional per-tick
            // bookkeeping/Hebbian reinforcement below.
        }
    }
    current
}

/// Step 6 - Broadcast: GWT's single serial arbitration point. Given an
/// already-decided `admitted` list (from either
/// `decide_admission_deterministic` or `decide_admission_from_attention_model`
/// — this function has no opinion on how admission was decided), applies
/// the resulting workspace/memory-role bookkeeping and Hebbian coalescence
/// reinforcement. Only touches objects whose membership actually changed,
/// not every object in the graph.
pub fn broadcast(graph: &mut Graph, admitted: &[(MentalObjectId, f32)], previous_working_memory: &HashSet<MentalObjectId>, now: EpochMillis) -> BroadcastResult {
    let admitted_ids: Vec<MentalObjectId> = admitted.iter().map(|(id, _)| *id).collect();
    let admitted_set: HashSet<MentalObjectId> = admitted_ids.iter().copied().collect();

    let mut newly_admitted = Vec::new();
    for &(id, score) in admitted {
        if !previous_working_memory.contains(&id) {
            newly_admitted.push(id);
        }
        if let Some(object) = graph.get_mut(&id) {
            if !object.workspace.in_working_memory {
                object.workspace.broadcast_count += 1;
            }
            object.workspace.in_working_memory = true;
            object.workspace.last_broadcast_at = Some(now);
            object.workspace.attention_score = Some(score);
            if !object.memory_roles.contains(&MemoryRole::Working) {
                object.memory_roles.push(MemoryRole::Working);
            }
        }
    }

    let mut released = Vec::new();
    for &id in previous_working_memory {
        if !admitted_set.contains(&id) {
            released.push(id);
            if let Some(object) = graph.get_mut(&id) {
                object.workspace.in_working_memory = false;
                object.memory_roles.retain(|role| *role != MemoryRole::Working);
            }
        }
    }

    reinforce_coalescence(graph, &admitted_ids, now);

    BroadcastResult {
        working_memory: admitted_ids,
        newly_admitted,
        released,
    }
}

/// Dehaene's stronger reading of "what does information become when it
/// becomes globally available": global availability doesn't just make
/// broadcast content *eligible* for association-building done elsewhere
/// (memory formation, chunk resolution) — co-occurring in the workspace is
/// itself how new associations form. Made literal: every pair of objects
/// admitted to Working Memory together this tick gets a Hebbian nudge on
/// the associative edge between them, symmetric in both directions (unlike
/// `steps::interlocutor::reinforce_link`'s one-directional utterance-to-
/// speaker edge — two Working Memory peers co-occurring is not a
/// source/target relationship, so both ends should learn about the other).
///
/// Runs on the *whole* admitted set, not just `newly_admitted` — two
/// objects that have sat in Working Memory together for several ticks in a
/// row are still co-broadcast every one of those ticks, and should keep
/// accumulating association credit for it, not just on the tick either one
/// first arrived. `DEFAULT_MAX_EDGE_STRENGTH` is what keeps this from
/// growing without bound for a long-lived pair — the edge saturates instead
/// of dominating spreading activation forever.
///
/// Pure Tier-0 graph mutation, same cost class as everything else in this
/// function — `EdgeKind::Associative` already exists for exactly this
/// relationship (co-activation, not derivation/causation/subgoal structure),
/// so no new edge kind is needed.
fn reinforce_coalescence(graph: &mut Graph, admitted: &[MentalObjectId], now: EpochMillis) {
    for i in 0..admitted.len() {
        for j in (i + 1)..admitted.len() {
            let (a, b) = (admitted[i], admitted[j]);
            if let Some(object) = graph.get_mut(&a) {
                reinforce_edge(&mut object.edges, b, EdgeKind::Associative, now, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH);
            }
            if let Some(object) = graph.get_mut(&b) {
                reinforce_edge(&mut object.edges, a, EdgeKind::Associative, now, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::MentalObject;
    use aca_util::EpochMillis;

    fn insert_candidate(graph: &mut Graph, score: f32) -> MentalObjectId {
        let object = MentalObject::new_observation("candidate", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);
        let _ = score; // score is supplied separately to `broadcast`, not stored on insert
        id
    }

    fn admit(candidates: &[CoalitionCandidate], capacity: usize) -> Vec<(MentalObjectId, f32)> {
        decide_admission_deterministic(candidates, capacity)
    }

    #[test]
    fn admits_exactly_top_n_by_score() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let b = insert_candidate(&mut graph, 0.0);
        let c = insert_candidate(&mut graph, 0.0);
        let candidates = vec![
            CoalitionCandidate { id: a, score: 0.9 },
            CoalitionCandidate { id: b, score: 0.5 },
            CoalitionCandidate { id: c, score: 0.1 },
        ];
        let admitted = admit(&candidates, 2);
        let result = broadcast(&mut graph, &admitted, &HashSet::new(), EpochMillis(1000));
        assert_eq!(result.working_memory, vec![a, b]);
        assert_eq!(result.newly_admitted.len(), 2);
        assert!(result.released.is_empty());
    }

    #[test]
    fn admitted_objects_get_working_role_and_workspace_state() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let candidates = vec![CoalitionCandidate { id: a, score: 1.0 }];
        let admitted = admit(&candidates, 4);
        broadcast(&mut graph, &admitted, &HashSet::new(), EpochMillis(1000));

        let object = graph.get(&a).unwrap();
        assert!(object.workspace.in_working_memory);
        assert_eq!(object.workspace.broadcast_count, 1);
        assert_eq!(object.workspace.last_broadcast_at, Some(EpochMillis(1000)));
        assert!(object.memory_roles.contains(&MemoryRole::Working));
    }

    #[test]
    fn objects_dropped_from_working_memory_lose_the_role_and_flag() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let b = insert_candidate(&mut graph, 0.0);

        // Cycle 1: only `a` is admitted.
        let admitted1 = admit(&[CoalitionCandidate { id: a, score: 1.0 }], 1);
        let result1 = broadcast(&mut graph, &admitted1, &HashSet::new(), EpochMillis(1000));
        let previous: HashSet<_> = result1.working_memory.iter().copied().collect();

        // Cycle 2: `b` outranks `a`, capacity is still 1 -> `a` is released.
        let admitted2 = admit(&[CoalitionCandidate { id: b, score: 2.0 }], 1);
        let result2 = broadcast(&mut graph, &admitted2, &previous, EpochMillis(2000));

        assert_eq!(result2.working_memory, vec![b]);
        assert_eq!(result2.released, vec![a]);
        let a_object = graph.get(&a).unwrap();
        assert!(!a_object.workspace.in_working_memory);
        assert!(!a_object.memory_roles.contains(&MemoryRole::Working));
    }

    #[test]
    fn re_admission_in_the_same_slot_does_not_double_count_broadcast() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let candidates = vec![CoalitionCandidate { id: a, score: 1.0 }];

        let admitted1 = admit(&candidates, 4);
        let result1 = broadcast(&mut graph, &admitted1, &HashSet::new(), EpochMillis(1000));
        let previous: HashSet<_> = result1.working_memory.iter().copied().collect();
        let admitted2 = admit(&candidates, 4);
        broadcast(&mut graph, &admitted2, &previous, EpochMillis(2000));

        let object = graph.get(&a).unwrap();
        assert_eq!(object.workspace.broadcast_count, 1, "staying in Working Memory across ticks should not re-increment broadcast_count");
    }

    #[test]
    fn capacity_of_zero_admits_nothing() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let candidates = vec![CoalitionCandidate { id: a, score: 5.0 }];
        let admitted = admit(&candidates, 0);
        let result = broadcast(&mut graph, &admitted, &HashSet::new(), EpochMillis(1000));
        assert!(result.working_memory.is_empty());
    }

    #[test]
    fn co_broadcast_objects_get_a_symmetric_associative_edge() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let b = insert_candidate(&mut graph, 0.0);
        let candidates = vec![CoalitionCandidate { id: a, score: 0.9 }, CoalitionCandidate { id: b, score: 0.5 }];
        let admitted = admit(&candidates, 4);
        broadcast(&mut graph, &admitted, &HashSet::new(), EpochMillis(1000));

        let a_object = graph.get(&a).unwrap();
        assert_eq!(a_object.edges.len(), 1);
        assert_eq!(a_object.edges[0].target_id, b);
        assert_eq!(a_object.edges[0].kind, aca_types::EdgeKind::Associative);
        assert!(a_object.edges[0].strength > 0.0);

        let b_object = graph.get(&b).unwrap();
        assert_eq!(b_object.edges.len(), 1);
        assert_eq!(b_object.edges[0].target_id, a);
        assert_eq!(b_object.edges[0].kind, aca_types::EdgeKind::Associative);
    }

    #[test]
    fn a_lone_admitted_object_gets_no_coalescence_edge() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let candidates = vec![CoalitionCandidate { id: a, score: 0.9 }];
        let admitted = admit(&candidates, 4);
        broadcast(&mut graph, &admitted, &HashSet::new(), EpochMillis(1000));
        assert!(graph.get(&a).unwrap().edges.is_empty());
    }

    #[test]
    fn staying_co_broadcast_across_ticks_keeps_reinforcing_not_just_the_first_tick() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let b = insert_candidate(&mut graph, 0.0);
        let candidates = vec![CoalitionCandidate { id: a, score: 0.9 }, CoalitionCandidate { id: b, score: 0.5 }];

        let admitted1 = admit(&candidates, 4);
        let result1 = broadcast(&mut graph, &admitted1, &HashSet::new(), EpochMillis(1000));
        let previous: HashSet<_> = result1.working_memory.iter().copied().collect();
        let strength_after_first = graph.get(&a).unwrap().edges[0].strength;

        let admitted2 = admit(&candidates, 4);
        broadcast(&mut graph, &admitted2, &previous, EpochMillis(2000));
        let strength_after_second = graph.get(&a).unwrap().edges[0].strength;

        assert!(strength_after_second > strength_after_first, "a pair still co-broadcast on a later tick should keep accumulating association, not freeze after the first tick");
    }

    #[test]
    fn three_way_co_broadcast_links_every_pair_not_just_adjacent_ones() {
        let mut graph = Graph::new();
        let a = insert_candidate(&mut graph, 0.0);
        let b = insert_candidate(&mut graph, 0.0);
        let c = insert_candidate(&mut graph, 0.0);
        let candidates = vec![
            CoalitionCandidate { id: a, score: 0.9 },
            CoalitionCandidate { id: b, score: 0.6 },
            CoalitionCandidate { id: c, score: 0.3 },
        ];
        let admitted = admit(&candidates, 4);
        broadcast(&mut graph, &admitted, &HashSet::new(), EpochMillis(1000));

        for id in [a, b, c] {
            let object = graph.get(&id).unwrap();
            assert_eq!(object.edges.len(), 2, "each of 3 co-broadcast objects should link to the other 2");
        }
    }

    fn attention_decision(operation: AttentionOp, target: Option<MentalObjectId>) -> AttentionDecision {
        AttentionDecision { operation, target, confidence: 0.9, reason_code: "test".to_string() }
    }

    #[test]
    fn attend_admits_a_named_target_not_previously_in_working_memory() {
        let a = MentalObjectId::new();
        let raw_candidates = vec![(a, 1.5, None)];
        let decision = attention_decision(AttentionOp::Attend, Some(a));
        let admitted = decide_admission_from_attention_model(&HashSet::new(), &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(a, 1.5)]);
    }

    #[test]
    fn maintain_makes_no_membership_change() {
        let a = MentalObjectId::new();
        let previous: HashSet<_> = [a].into_iter().collect();
        let raw_candidates = vec![(a, 2.0, None)];
        let decision = attention_decision(AttentionOp::Maintain, Some(a));
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(a, 2.0)]);
    }

    #[test]
    fn maintain_releases_a_previous_member_not_present_this_tick() {
        let stale = MentalObjectId::new();
        let present = MentalObjectId::new();
        let previous: HashSet<_> = [stale, present].into_iter().collect();
        let raw_candidates = vec![(present, 2.0, None)];
        let decision = attention_decision(AttentionOp::Maintain, Some(stale));
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(present, 2.0)]);
    }

    #[test]
    fn model_cannot_override_the_coalition_attention_floor() {
        let stale_below_floor = MentalObjectId::new();
        let new_below_floor = MentalObjectId::new();
        let eligible = MentalObjectId::new();
        let previous: HashSet<_> = [stale_below_floor, eligible].into_iter().collect();
        let raw_candidates = vec![(stale_below_floor, -3.0, None), (new_below_floor, -4.0, None), (eligible, 1.0, None)];
        let decision = attention_decision(AttentionOp::Attend, Some(new_below_floor));
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(eligible, 1.0)]);
    }

    #[test]
    fn named_model_vote_must_resolve_to_this_ticks_coalition() {
        let eligible = MentalObjectId::new();
        let raw_only = MentalObjectId::new();
        let coalition = vec![CoalitionCandidate { id: eligible, score: 1.0 }];
        assert!(attention_decision_target_is_eligible(&attention_decision(AttentionOp::Attend, Some(eligible)), &coalition));
        assert!(!attention_decision_target_is_eligible(&attention_decision(AttentionOp::Switch, Some(raw_only)), &coalition));
        assert!(!attention_decision_target_is_eligible(&attention_decision(AttentionOp::Suppress, None), &coalition));
        assert!(attention_decision_target_is_eligible(&attention_decision(AttentionOp::Maintain, None), &coalition));
    }

    #[test]
    fn nonfinite_candidate_score_is_not_admitted() {
        let invalid = MentalObjectId::new();
        let raw_candidates = vec![(invalid, 1.0, Some(f32::NAN))];
        let decision = attention_decision(AttentionOp::Attend, Some(invalid));
        assert!(decide_admission_from_attention_model(&HashSet::new(), &raw_candidates, &decision, -2.0, 4).is_empty());
    }

    #[test]
    fn switch_evicts_the_weakest_member_when_over_capacity_never_the_new_target() {
        let weak = MentalObjectId::new();
        let strong = MentalObjectId::new();
        let target = MentalObjectId::new();
        let previous: HashSet<_> = [weak, strong].into_iter().collect();
        let raw_candidates = vec![(weak, -1.0, None), (strong, 5.0, None), (target, 0.1, None)];
        let decision = attention_decision(AttentionOp::Switch, Some(target));
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 2);
        let admitted_ids: HashSet<_> = admitted.iter().map(|(id, _)| *id).collect();
        assert!(admitted_ids.contains(&target), "the model's named target must never be the one evicted");
        assert!(admitted_ids.contains(&strong));
        assert!(!admitted_ids.contains(&weak), "the weakest existing member should be evicted to make room");
        assert_eq!(admitted.len(), 2);
    }

    #[test]
    fn suppress_evicts_the_named_target_if_present() {
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        let previous: HashSet<_> = [a, b].into_iter().collect();
        let raw_candidates = vec![(a, 1.0, None), (b, 2.0, None)];
        let decision = attention_decision(AttentionOp::Suppress, Some(a));
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(b, 2.0)]);
    }

    #[test]
    fn suppress_of_an_absent_target_is_a_no_op() {
        let a = MentalObjectId::new();
        let previous: HashSet<_> = [a].into_iter().collect();
        let raw_candidates = vec![(a, 1.0, None)];
        let decision = attention_decision(AttentionOp::Suppress, Some(MentalObjectId::new()));
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(a, 1.0)]);
    }

    #[test]
    fn ignore_is_a_no_op() {
        let a = MentalObjectId::new();
        let previous: HashSet<_> = [a].into_iter().collect();
        let raw_candidates = vec![(a, 1.0, None)];
        let decision = attention_decision(AttentionOp::Ignore, None);
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(a, 1.0)]);
    }

    #[test]
    fn attend_on_a_target_that_is_not_a_real_candidate_this_tick_is_a_no_op() {
        let a = MentalObjectId::new();
        let previous: HashSet<_> = [a].into_iter().collect();
        let raw_candidates = vec![(a, 1.0, None)]; // target below isn't in raw_candidates
        let decision = attention_decision(AttentionOp::Attend, Some(MentalObjectId::new()));
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(a, 1.0)]);
    }

    #[test]
    fn attend_with_no_target_is_a_no_op() {
        let a = MentalObjectId::new();
        let previous: HashSet<_> = [a].into_iter().collect();
        let raw_candidates = vec![(a, 1.0, None)];
        let decision = attention_decision(AttentionOp::Attend, None);
        let admitted = decide_admission_from_attention_model(&previous, &raw_candidates, &decision, -2.0, 4);
        assert_eq!(admitted, vec![(a, 1.0)]);
    }

    #[test]
    fn hysteresis_admits_a_brand_new_candidate_that_clears_the_stricter_ignition_bar() {
        let a = MentalObjectId::new();
        let candidates = vec![CoalitionCandidate { id: a, score: 1.0 }];
        let admitted = decide_admission_with_hysteresis(&candidates, &HashSet::new(), 4, 0.5);
        assert_eq!(admitted, vec![(a, 1.0)]);
    }

    #[test]
    fn hysteresis_refuses_a_brand_new_candidate_that_only_clears_the_looser_sustain_floor() {
        // Score 0.3 already cleared Coalition's own (lower) attention
        // threshold to be a candidate at all, but it's below the stricter
        // ignition_threshold of 0.5, and this id was never previously
        // admitted - the exact "subliminal, never quite crosses into
        // consciousness" case.
        let a = MentalObjectId::new();
        let candidates = vec![CoalitionCandidate { id: a, score: 0.3 }];
        let admitted = decide_admission_with_hysteresis(&candidates, &HashSet::new(), 4, 0.5);
        assert!(admitted.is_empty());
    }

    #[test]
    fn hysteresis_sustains_a_previously_admitted_member_at_a_score_that_would_reject_a_newcomer() {
        // The actual hysteresis signature: the identical score (0.3) is
        // refused for a fresh id above, but accepted here purely because
        // `a` was already in `previous_working_memory` - sustaining an
        // already-ignited percept is the easy case, entering fresh is the
        // hard one.
        let a = MentalObjectId::new();
        let previous: HashSet<_> = [a].into_iter().collect();
        let candidates = vec![CoalitionCandidate { id: a, score: 0.3 }];
        let admitted = decide_admission_with_hysteresis(&candidates, &previous, 4, 0.5);
        assert_eq!(admitted, vec![(a, 0.3)]);
    }

    #[test]
    fn hysteresis_still_lets_a_sustained_member_lose_its_seat_to_real_capacity_pressure() {
        // Hysteresis relaxes *eligibility*, not the ranking itself - a
        // previously-admitted member that stays eligible can still be
        // outranked and evicted by a stronger competitor at capacity 1,
        // exactly like ordinary competitive displacement.
        let sustained = MentalObjectId::new();
        let stronger_newcomer = MentalObjectId::new();
        let previous: HashSet<_> = [sustained].into_iter().collect();
        let candidates = vec![CoalitionCandidate { id: sustained, score: 0.3 }, CoalitionCandidate { id: stronger_newcomer, score: 5.0 }];
        let admitted = decide_admission_with_hysteresis(&candidates, &previous, 1, 0.5);
        assert_eq!(admitted, vec![(stronger_newcomer, 5.0)]);
    }

    #[test]
    fn an_equal_ignition_threshold_and_candidate_floor_reduces_to_plain_deterministic_admission() {
        // The production-safe default shape: with ignition_threshold set to
        // whatever Coalition's own floor already was, every candidate that
        // reached this function at all already clears it, so hysteresis is
        // a no-op and this must match `decide_admission_deterministic`
        // exactly - the mechanism this file's `#[allow(...)]`-free
        // `decide_admission_deterministic` already proves reusable, not a
        // second, silently-diverging implementation.
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        let candidates = vec![CoalitionCandidate { id: a, score: 0.9 }, CoalitionCandidate { id: b, score: 0.1 }];
        let via_hysteresis = decide_admission_with_hysteresis(&candidates, &HashSet::new(), 4, f32::NEG_INFINITY);
        let via_plain = decide_admission_deterministic(&candidates, 4);
        assert_eq!(via_hysteresis, via_plain);
    }
}
