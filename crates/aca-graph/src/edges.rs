use aca_types::{AssociativeEdge, EdgeKind, MentalObjectId};
use aca_util::EpochMillis;

/// Fixed Hebbian increment applied to an edge's strength each time its two
/// endpoints co-activate (e.g. both broadcast together). Named
/// simplification: this is a fixed-increment counter, not ACT-R's formally
/// derived fan equation.
pub const DEFAULT_HEBBIAN_INCREMENT: f32 = 0.15;

/// Ceiling on edge strength — prevents unbounded reinforcement from making
/// one association dominate spreading activation forever.
pub const DEFAULT_MAX_EDGE_STRENGTH: f32 = 1.0;

/// Reinforces (or creates) the edge from an object's outgoing edge list to
/// `target_id`: increments strength by `increment` (clamped to
/// `max_strength`) and stamps `last_coactivated_at`. This is the "reference
/// count" for associative strength, mirroring how `record_reference` works
/// for base-level activation.
pub fn reinforce_edge(
    edges: &mut Vec<AssociativeEdge>,
    target_id: MentalObjectId,
    kind: EdgeKind,
    at: EpochMillis,
    increment: f32,
    max_strength: f32,
) {
    if let Some(edge) = edges
        .iter_mut()
        .find(|edge| edge.target_id == target_id && edge.kind == kind)
    {
        edge.strength = (edge.strength + increment).min(max_strength);
        edge.last_coactivated_at = at;
    } else {
        edges.push(AssociativeEdge {
            target_id,
            kind,
            strength: increment.min(max_strength),
            last_coactivated_at: at,
        });
    }
}

/// The *effective* (decayed) strength of an edge at time `now`, computed
/// lazily rather than by mutating the stored value on every tick —
/// exponential decay from the last co-activation: `strength *
/// exp(-decay_rate_per_ms * elapsed_ms)`. Keeps `aca-graph` from needing a
/// full-graph sweep every cycle just to age edges nobody is currently
/// looking at.
pub fn effective_strength(edge: &AssociativeEdge, now: EpochMillis, decay_rate_per_ms: f64) -> f32 {
    let elapsed_ms = edge.last_coactivated_at.elapsed_ms_until(now).max(0) as f64;
    let decayed = edge.strength as f64 * (-decay_rate_per_ms * elapsed_ms).exp();
    decayed as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reinforce_edge_creates_when_absent() {
        let mut edges = Vec::new();
        let target = MentalObjectId::new();
        reinforce_edge(&mut edges, target, EdgeKind::Associative, EpochMillis(0), 0.2, 1.0);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].strength, 0.2);
    }

    #[test]
    fn reinforce_edge_increments_when_present() {
        let mut edges = Vec::new();
        let target = MentalObjectId::new();
        reinforce_edge(&mut edges, target, EdgeKind::Associative, EpochMillis(0), 0.2, 1.0);
        reinforce_edge(&mut edges, target, EdgeKind::Associative, EpochMillis(100), 0.3, 1.0);
        assert_eq!(edges.len(), 1, "same target+kind reinforces in place");
        assert!((edges[0].strength - 0.5).abs() < 1e-6);
        assert_eq!(edges[0].last_coactivated_at, EpochMillis(100));
    }

    #[test]
    fn reinforce_edge_clamps_to_max_strength() {
        let mut edges = Vec::new();
        let target = MentalObjectId::new();
        for _ in 0..10 {
            reinforce_edge(&mut edges, target, EdgeKind::Associative, EpochMillis(0), 0.5, 1.0);
        }
        assert_eq!(edges[0].strength, 1.0);
    }

    #[test]
    fn distinct_edge_kinds_to_the_same_target_are_independent() {
        let mut edges = Vec::new();
        let target = MentalObjectId::new();
        reinforce_edge(&mut edges, target, EdgeKind::Associative, EpochMillis(0), 0.2, 1.0);
        reinforce_edge(&mut edges, target, EdgeKind::Causal, EpochMillis(0), 0.3, 1.0);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn effective_strength_decays_with_elapsed_time() {
        let edge = AssociativeEdge {
            target_id: MentalObjectId::new(),
            kind: EdgeKind::Associative,
            strength: 1.0,
            last_coactivated_at: EpochMillis(0),
        };
        let immediate = effective_strength(&edge, EpochMillis(0), 0.001);
        let later = effective_strength(&edge, EpochMillis(1_000), 0.001);
        assert!((immediate - 1.0).abs() < 1e-6);
        assert!(later < immediate, "strength should decay over elapsed time");
        assert!(later > 0.0);
    }
}
