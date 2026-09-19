use std::collections::HashSet;

use aca_graph::{clears_retrieval_threshold, compute_base_level, sample_noise, spread_activation_multi_hop, Graph};
use aca_types::MentalObjectId;
use aca_util::Clock;
use rand::Rng;

/// Tuning knobs for `recall` - a threshold/cadence tuning parameter, not a
/// theoretical commitment (see `LoopConfig`'s own doc comment on this
/// framing).
#[derive(Debug, Clone, Copy)]
pub struct RecallConfig {
    /// How many associative hops `spread_activation_multi_hop` propagates
    /// out from current Working Memory before stopping. Further than Step
    /// 4's zero-hop reach, shallow enough to stay tractable at the
    /// household-scale, fully-memory-resident graph the rest of this
    /// codebase already assumes (see `aca-store`'s `load_all_objects` doc
    /// comment) - tune against real usage once it exists.
    pub max_hops: usize,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self { max_hops: 2 }
    }
}

/// Step 4.5 - Recall: SYNAPSE-style graph-wide multi-hop spreading
/// activation from a set of anchor nodes, discovering dormant memories
/// Step 4's single-hop spread structurally cannot reach (specs.md's Memory
/// Recall section: "the graph nodes that clear the retrieval threshold
/// become recall candidates" - unqualified, i.e. any node in the graph, not
/// just ones already in Working Memory). Only objects NOT already in
/// `anchors` or `excluded` (everything Step 5 already nominated this tick
/// from the newly-observed object/`pending_admission`) are considered.
///
/// `anchors` is ordinary current Working Memory for the generic recall pass
/// `loop_actor` always runs, but the parameter is deliberately not named or
/// typed as such: a specialist recall pass (e.g. `steps::interlocutor::
/// social_cloud_anchors`, spreading from one person's own utterances rather
/// than from whatever's currently broadcast) reuses this exact function with
/// a different anchor set - the spreading/threshold/bookkeeping logic below
/// has no dependency on the anchors actually being Working Memory, only on
/// them being real, currently-active graph nodes to spread from.
///
/// Writes the recomputed `base_level`/`spreading`/`noise`/`total` back onto
/// each winner's `ActivationState` - the same fields Step 4 already writes
/// - and returns ids clearing `attention_threshold`, for the caller to fold
/// into Step 5's `raw_candidates` with no surprise term, exactly like an
/// ordinary recalled Working Memory member.
#[allow(clippy::too_many_arguments)]
pub fn recall(
    graph: &mut Graph,
    anchors: &HashSet<MentalObjectId>,
    excluded: &HashSet<MentalObjectId>,
    config: &RecallConfig,
    edge_decay_rate_per_ms: f64,
    attention_threshold: f32,
    self_memory_activation_bonus: f32,
    max_noise: f32,
    rng: &mut impl Rng,
    clock: &dyn Clock,
) -> Vec<MentalObjectId> {
    let now = clock.now();
    let source_ids: Vec<MentalObjectId> = anchors.iter().copied().collect();
    let spreading_map = spread_activation_multi_hop(graph, &source_ids, config.max_hops, edge_decay_rate_per_ms, now);

    let mut winners = Vec::new();
    for (id, spreading) in spreading_map {
        if anchors.contains(&id) || excluded.contains(&id) {
            continue;
        }
        let Some(object) = graph.get_mut(&id) else {
            continue;
        };
        let base_level = compute_base_level(&object.activation.reference_log, object.activation.decay_d, now);
        let noise = sample_noise(rng, max_noise);
        object.activation.base_level = base_level;
        object.activation.spreading = spreading;
        object.activation.noise = noise;
        object.activation.total = base_level + spreading + noise;
        object.activation.last_computed_at = now;
        crate::loop_actor::apply_self_memory_activation_bonus(object, self_memory_activation_bonus);
        if clears_retrieval_threshold(object.activation.total, attention_threshold) {
            winners.push(id);
        }
    }
    winners
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::{AssociativeEdge, EdgeKind, MentalObject};
    use aca_util::{EpochMillis, ManualClock};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn edge_to(target: MentalObjectId, strength: f32, at: EpochMillis) -> AssociativeEdge {
        AssociativeEdge { target_id: target, kind: EdgeKind::Associative, strength, last_coactivated_at: at }
    }

    /// Builds `source(in WM) -> intermediate -> dormant`, none of them
    /// sharing an edge directly with `source` except the first hop, so a
    /// non-trivial `max_hops` is required to reach `dormant` at all.
    fn two_hop_graph() -> (Graph, MentalObjectId, MentalObjectId, MentalObjectId) {
        let mut graph = Graph::new();
        let mut dormant = MentalObject::new_observation("dormant", EpochMillis(0), 0.5);
        // A stale, single, ancient reference: base_level alone stays deeply negative.
        dormant.activation.reference_log = aca_util::RingBuffer::new(64);
        dormant.activation.reference_log.push(EpochMillis(0));
        let dormant_id = dormant.id;
        graph.insert(dormant);

        let mut intermediate = MentalObject::new_observation("intermediate", EpochMillis(0), 0.5);
        intermediate.edges.push(edge_to(dormant_id, 1.0, EpochMillis(0)));
        let intermediate_id = intermediate.id;
        graph.insert(intermediate);

        let mut source = MentalObject::new_observation("source", EpochMillis(0), 0.5);
        source.edges.push(edge_to(intermediate_id, 1.0, EpochMillis(0)));
        let source_id = source.id;
        graph.insert(source);

        (graph, source_id, intermediate_id, dormant_id)
    }

    #[test]
    fn discovers_a_multi_hop_dormant_object_and_writes_back_activation() {
        let (mut graph, source_id, _intermediate_id, dormant_id) = two_hop_graph();
        let working_memory: HashSet<MentalObjectId> = [source_id].into_iter().collect();
        let config = RecallConfig { max_hops: 2 };
        let clock = ManualClock::new(EpochMillis(1_000));
        let mut rng = StdRng::seed_from_u64(7);

        let winners = recall(&mut graph, &working_memory, &HashSet::new(), &config, 0.0, -100.0, 0.0, 0.0, &mut rng, &clock);

        assert!(winners.contains(&dormant_id), "a two-hop-reachable dormant object should be recalled");
        let object = graph.get(&dormant_id).unwrap();
        assert!(object.activation.spreading > 0.0, "spreading activation should have been written back");
        assert_eq!(object.activation.last_computed_at, EpochMillis(1_000));
    }

    #[test]
    fn excludes_ids_already_in_working_memory_or_excluded_set() {
        let (mut graph, source_id, _intermediate_id, dormant_id) = two_hop_graph();
        let working_memory: HashSet<MentalObjectId> = [source_id].into_iter().collect();
        let excluded: HashSet<MentalObjectId> = [dormant_id].into_iter().collect();
        let config = RecallConfig { max_hops: 2 };
        let clock = ManualClock::new(EpochMillis(1_000));
        let mut rng = StdRng::seed_from_u64(7);

        let winners = recall(&mut graph, &working_memory, &excluded, &config, 0.0, -100.0, 0.0, 0.0, &mut rng, &clock);
        assert!(!winners.contains(&dormant_id), "an already-excluded id must never be returned even if it clears threshold");
    }

    #[test]
    fn respects_max_hops_one_does_not_discover_two_hop_targets() {
        let (mut graph, source_id, _intermediate_id, dormant_id) = two_hop_graph();
        let working_memory: HashSet<MentalObjectId> = [source_id].into_iter().collect();
        let config = RecallConfig { max_hops: 1 };
        let clock = ManualClock::new(EpochMillis(1_000));
        let mut rng = StdRng::seed_from_u64(7);

        let winners = recall(&mut graph, &working_memory, &HashSet::new(), &config, 0.0, -100.0, 0.0, 0.0, &mut rng, &clock);
        assert!(!winners.contains(&dormant_id), "max_hops=1 must not reach a two-hop-away object");
    }

    #[test]
    fn returns_nothing_below_threshold() {
        let (mut graph, source_id, _intermediate_id, dormant_id) = two_hop_graph();
        let working_memory: HashSet<MentalObjectId> = [source_id].into_iter().collect();
        let config = RecallConfig { max_hops: 2 };
        let clock = ManualClock::new(EpochMillis(1_000));
        let mut rng = StdRng::seed_from_u64(7);

        // An unreasonably high threshold nothing can clear.
        let winners = recall(&mut graph, &working_memory, &HashSet::new(), &config, 0.0, 1_000.0, 0.0, 0.0, &mut rng, &clock);
        assert!(winners.is_empty());
        assert!(!winners.contains(&dormant_id));
    }
}
