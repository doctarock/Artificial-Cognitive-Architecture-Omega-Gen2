use std::collections::{HashMap, HashSet};

use aca_types::{EdgeKind, MentalObjectId, ObjectStatus};
use aca_util::EpochMillis;

use crate::edges::effective_strength;
use crate::graph::Graph;

/// Iterative multi-hop spreading activation across the *entire* graph,
/// starting from `source_ids` (current Working Memory). Unlike
/// `activation::compute_spreading_activation` (single-hop, caller supplies
/// the target), this *discovers* every node reachable within `max_hops`
/// hops and returns its accumulated spreading activation in one pass — this
/// is what lets a dormant memory several associative hops from Working
/// Memory re-enter the competition at all (specs.md: "Dormant memories
/// remain recoverable through strong associative activation from a
/// sufficiently related context").
///
/// Uses the same v1 simplification `compute_spreading_activation` already
/// documents (fixed uniform per-source weight `1/|source_ids|`, not a
/// formally-derived attentional allocation) as the seed weight at hop 0.
///
/// Fan effect: at each hop, a node's outgoing energy is divided by its
/// out-degree (`edges.len()`) before being distributed to its neighbors —
/// keeps a highly-connected hub node from radiating undiminished activation
/// to everything it touches.
///
/// Edge weights are time-decayed via `effective_strength`, never raw
/// `strength`.
///
/// This is a bounded frontier relaxation (BFS by hop), not a full-graph
/// scan: cost is proportional to edges actually reachable within
/// `max_hops`, not `graph.len()`. Source ids are excluded from both the
/// returned map and from re-entering the frontier (prevents a cycle routing
/// back through a source from re-amplifying through it). Non-`Active`
/// objects (discarded) neither propagate further nor appear in the result.
pub fn spread_activation_multi_hop(
    graph: &Graph,
    source_ids: &[MentalObjectId],
    max_hops: usize,
    decay_rate_per_ms: f64,
    now: EpochMillis,
) -> HashMap<MentalObjectId, f32> {
    let mut accumulated: HashMap<MentalObjectId, f32> = HashMap::new();
    if source_ids.is_empty() || max_hops == 0 {
        return accumulated;
    }
    let source_set: HashSet<MentalObjectId> = source_ids.iter().copied().collect();
    let uniform_weight = 1.0 / source_ids.len() as f32;
    let mut frontier: HashMap<MentalObjectId, f32> = source_ids.iter().map(|&id| (id, uniform_weight)).collect();

    for _ in 0..max_hops {
        let mut next_frontier: HashMap<MentalObjectId, f32> = HashMap::new();
        for (&node_id, &energy) in &frontier {
            let Some(node) = graph.get(&node_id) else {
                continue;
            };
            if node.status != ObjectStatus::Active || node.edges.is_empty() {
                continue;
            }
            let fan_out = node.edges.len() as f32;
            for edge in &node.edges {
                if source_set.contains(&edge.target_id) {
                    continue; // never re-enter/re-amplify through a source
                }
                // A discarded (or otherwise missing) target neither
                // receives activation nor propagates further — matches
                // `Graph::active_ids`' "the only ones eligible as
                // recall/coalition candidates" rule.
                let target_active = graph.get(&edge.target_id).is_some_and(|t| t.status == ObjectStatus::Active);
                if !target_active {
                    continue;
                }
                let contribution = energy * effective_strength(edge, now, decay_rate_per_ms) / fan_out;
                if contribution <= 0.0 {
                    continue;
                }
                if matches!(edge.kind, EdgeKind::Inhibitory | EdgeKind::Contradicts) {
                    // Signed suppression reaches this target but does not
                    // spread as negative energy through unrelated neighbors.
                    *accumulated.entry(edge.target_id).or_insert(0.0) -= contribution;
                } else {
                    *accumulated.entry(edge.target_id).or_insert(0.0) += contribution;
                    *next_frontier.entry(edge.target_id).or_insert(0.0) += contribution;
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    accumulated
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::{AssociativeEdge, EdgeKind, MentalObject};

    fn edge_to(target: MentalObjectId, strength: f32, at: EpochMillis) -> AssociativeEdge {
        AssociativeEdge {
            target_id: target,
            kind: EdgeKind::Associative,
            strength,
            last_coactivated_at: at,
        }
    }

    #[test]
    fn inhibitory_and_contradictory_targets_are_suppressed_without_forwarding_negative_energy() {
        for kind in [EdgeKind::Inhibitory, EdgeKind::Contradicts] {
            let mut graph = Graph::new();
            let downstream = insert_node(&mut graph, vec![]);
            let inhibited = insert_node(&mut graph, vec![edge_to(downstream, 1.0, EpochMillis(0))]);
            let source = insert_node(&mut graph, vec![AssociativeEdge {
                target_id: inhibited, kind, strength: 0.8,
                last_coactivated_at: EpochMillis(0),
            }]);
            let result = spread_activation_multi_hop(&graph, &[source], 2, 0.0, EpochMillis(0));
            assert!(result.get(&inhibited).is_some_and(|value| *value < 0.0), "{kind:?}");
            assert!(!result.contains_key(&downstream), "{kind:?}");
        }
    }

    fn insert_node(graph: &mut Graph, edges: Vec<AssociativeEdge>) -> MentalObjectId {
        let mut obj = MentalObject::new_observation("n", EpochMillis(0), 0.5);
        obj.edges = edges;
        let id = obj.id;
        graph.insert(obj);
        id
    }

    #[test]
    fn no_sources_returns_empty() {
        let graph = Graph::new();
        let result = spread_activation_multi_hop(&graph, &[], 2, 0.0, EpochMillis(0));
        assert!(result.is_empty());
    }

    #[test]
    fn zero_hops_returns_empty() {
        let mut graph = Graph::new();
        let target = insert_node(&mut graph, Vec::new());
        let source = insert_node(&mut graph, vec![edge_to(target, 0.8, EpochMillis(0))]);
        let result = spread_activation_multi_hop(&graph, &[source], 0, 0.0, EpochMillis(0));
        assert!(result.is_empty());
    }

    #[test]
    fn single_hop_matches_uniform_weight_times_effective_strength() {
        let mut graph = Graph::new();
        let target = insert_node(&mut graph, Vec::new());
        let source = insert_node(&mut graph, vec![edge_to(target, 0.8, EpochMillis(0))]);
        let result = spread_activation_multi_hop(&graph, &[source], 1, 0.0, EpochMillis(0));
        // one source (uniform weight 1.0), one outgoing edge (fan_out 1.0): 1.0 * 0.8 / 1.0 = 0.8
        assert!((result[&target] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn two_hop_reaches_a_node_with_no_direct_edge_from_sources() {
        let mut graph = Graph::new();
        let dormant = insert_node(&mut graph, Vec::new());
        let intermediate = insert_node(&mut graph, vec![edge_to(dormant, 0.5, EpochMillis(0))]);
        let source = insert_node(&mut graph, vec![edge_to(intermediate, 0.6, EpochMillis(0))]);

        let one_hop = spread_activation_multi_hop(&graph, &[source], 1, 0.0, EpochMillis(0));
        assert!(!one_hop.contains_key(&dormant), "one hop must not reach a two-hop-away node");
        assert!((one_hop[&intermediate] - 0.6).abs() < 1e-6);

        let two_hop = spread_activation_multi_hop(&graph, &[source], 2, 0.0, EpochMillis(0));
        // hop 1: source -> intermediate, energy 1.0 * 0.6 / 1.0 = 0.6
        // hop 2: intermediate -> dormant, energy 0.6 * 0.5 / 1.0 = 0.3
        assert!((two_hop[&dormant] - 0.3).abs() < 1e-6, "got {:?}", two_hop.get(&dormant));
    }

    #[test]
    fn fan_effect_normalizes_by_out_degree() {
        let mut graph = Graph::new();
        let narrow_target = insert_node(&mut graph, Vec::new());
        let wide_target = insert_node(&mut graph, Vec::new());
        let decoy_a = insert_node(&mut graph, Vec::new());
        let decoy_b = insert_node(&mut graph, Vec::new());

        // narrow_source has one outgoing edge (fan_out 1) to narrow_target.
        let narrow_source = insert_node(&mut graph, vec![edge_to(narrow_target, 0.9, EpochMillis(0))]);
        // wide_source has three equal-strength outgoing edges (fan_out 3), one of them to wide_target.
        let wide_source = insert_node(
            &mut graph,
            vec![
                edge_to(wide_target, 0.9, EpochMillis(0)),
                edge_to(decoy_a, 0.9, EpochMillis(0)),
                edge_to(decoy_b, 0.9, EpochMillis(0)),
            ],
        );

        let narrow_result = spread_activation_multi_hop(&graph, &[narrow_source], 1, 0.0, EpochMillis(0));
        let wide_result = spread_activation_multi_hop(&graph, &[wide_source], 1, 0.0, EpochMillis(0));

        assert!(
            wide_result[&wide_target] < narrow_result[&narrow_target],
            "a higher out-degree intermediate should contribute proportionally less to each neighbor"
        );
        assert!((wide_result[&wide_target] - narrow_result[&narrow_target] / 3.0).abs() < 1e-6);
    }

    #[test]
    fn decayed_edge_contributes_less_than_a_fresh_one() {
        let mut graph = Graph::new();
        let target = insert_node(&mut graph, Vec::new());
        let source = insert_node(&mut graph, vec![edge_to(target, 1.0, EpochMillis(0))]);

        let no_decay = spread_activation_multi_hop(&graph, &[source], 1, 0.0, EpochMillis(2_000));
        let with_decay = spread_activation_multi_hop(&graph, &[source], 1, 0.001, EpochMillis(2_000));

        assert!((no_decay[&target] - 1.0).abs() < 1e-6, "decay_rate=0.0 must be a true no-op");
        assert!(with_decay[&target] < no_decay[&target], "a decayed edge should contribute less than a fresh one");
        assert!(with_decay[&target] > 0.0);
    }

    #[test]
    fn sources_never_appear_in_the_returned_map_even_via_a_cycle() {
        let mut graph = Graph::new();
        let source_id = MentalObjectId::new();
        let other_id = MentalObjectId::new();

        let mut source = MentalObject::new_observation("source", EpochMillis(0), 0.5);
        source.id = source_id;
        source.edges = vec![edge_to(other_id, 0.7, EpochMillis(0))];
        graph.insert(source);

        let mut other = MentalObject::new_observation("other", EpochMillis(0), 0.5);
        other.id = other_id;
        other.edges = vec![edge_to(source_id, 0.7, EpochMillis(0))];
        graph.insert(other);

        let result = spread_activation_multi_hop(&graph, &[source_id], 5, 0.0, EpochMillis(0));
        assert!(!result.contains_key(&source_id), "a source must never appear in its own result, even via a cycle");
        assert!(result.contains_key(&other_id));
    }

    #[test]
    fn discarded_intermediate_node_does_not_propagate_further() {
        let mut graph = Graph::new();
        let dormant = insert_node(&mut graph, Vec::new());
        let intermediate_id = MentalObjectId::new();
        let mut intermediate = MentalObject::new_observation("intermediate", EpochMillis(0), 0.5);
        intermediate.id = intermediate_id;
        intermediate.edges = vec![edge_to(dormant, 0.5, EpochMillis(0))];
        intermediate.status = ObjectStatus::Discarded;
        graph.insert(intermediate);
        let source = insert_node(&mut graph, vec![edge_to(intermediate_id, 0.6, EpochMillis(0))]);

        let result = spread_activation_multi_hop(&graph, &[source], 2, 0.0, EpochMillis(0));
        assert!(!result.contains_key(&intermediate_id), "a discarded node must not appear in the result");
        assert!(!result.contains_key(&dormant), "propagation must stop at a discarded node");
    }
}
