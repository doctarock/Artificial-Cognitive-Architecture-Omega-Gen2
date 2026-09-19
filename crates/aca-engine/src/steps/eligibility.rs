use std::collections::HashMap;

use aca_graph::{DEFAULT_MAX_EDGE_STRENGTH, Graph};
use aca_types::{EdgeKind, MentalObjectId};
use aca_util::EpochMillis;

#[derive(Debug, Clone, Copy)]
pub struct EligibilityConfig {
    pub tau_ms: f32,
    pub learning_rate: f32,
    pub min_trace: f32,
}

impl Default for EligibilityConfig {
    fn default() -> Self {
        Self {
            tau_ms: 10_000.0,
            learning_rate: 0.05,
            min_trace: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeKey {
    source: MentalObjectId,
    target: MentalObjectId,
    kind: EdgeKind,
}

#[derive(Debug, Clone, Copy)]
struct Trace {
    value: f32,
    updated_at: EpochMillis,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeCredit {
    pub source: MentalObjectId,
    pub target: MentalObjectId,
    pub trace: f32,
    pub delta: f32,
}

/// Transient synaptic eligibility: co-active edges become credit candidates,
/// then a later global reward modulates only those still carrying a trace.
/// The registry is actor-local on purpose; eligibility is short-lived process
/// state and should not survive a daemon restart.
#[derive(Debug, Default)]
pub struct EligibilityTraceRegistry {
    traces: HashMap<EdgeKey, Trace>,
}

impl EligibilityTraceRegistry {
    pub fn mark(
        &mut self,
        source: MentalObjectId,
        target: MentalObjectId,
        kind: EdgeKind,
        now: EpochMillis,
    ) {
        self.traces.insert(
            EdgeKey {
                source,
                target,
                kind,
            },
            Trace {
                value: 1.0,
                updated_at: now,
            },
        );
    }

    pub fn apply_reward(
        &mut self,
        graph: &mut Graph,
        reward: f32,
        config: &EligibilityConfig,
        now: EpochMillis,
    ) -> Vec<EdgeCredit> {
        let signed_reward = (reward.clamp(0.0, 1.0) - 0.5) * 2.0;
        let mut credits = Vec::new();
        self.traces.retain(|key, trace| {
            let elapsed_ms = trace.updated_at.elapsed_ms_until(now).max(0) as f32;
            trace.value *= (-elapsed_ms / config.tau_ms.max(1.0)).exp();
            trace.updated_at = now;
            if trace.value < config.min_trace {
                return false;
            }
            let delta = config.learning_rate * signed_reward * trace.value;
            if delta != 0.0
                && let Some(edge) = graph.get_mut(&key.source).and_then(|source| {
                    source
                        .edges
                        .iter_mut()
                        .find(|edge| edge.target_id == key.target && edge.kind == key.kind)
                })
            {
                edge.strength = (edge.strength + delta).clamp(0.0, DEFAULT_MAX_EDGE_STRENGTH);
                credits.push(EdgeCredit {
                    source: key.source,
                    target: key.target,
                    trace: trace.value,
                    delta,
                });
            }
            true
        });
        credits
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.traces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_graph::reinforce_edge;

    fn linked_graph(now: EpochMillis) -> (Graph, MentalObjectId, MentalObjectId) {
        let mut graph = Graph::new();
        let mut source = aca_types::MentalObject::new_observation("source", now, 0.5);
        let target = aca_types::MentalObject::new_observation("target", now, 0.5);
        let (source_id, target_id) = (source.id, target.id);
        reinforce_edge(
            &mut source.edges,
            target_id,
            EdgeKind::Associative,
            now,
            0.5,
            1.0,
        );
        graph.insert(source);
        graph.insert(target);
        (graph, source_id, target_id)
    }

    #[test]
    fn delayed_reward_strengthens_a_recently_eligible_edge() {
        let (mut graph, source, target) = linked_graph(EpochMillis(0));
        let mut traces = EligibilityTraceRegistry::default();
        traces.mark(source, target, EdgeKind::Associative, EpochMillis(0));
        let credits = traces.apply_reward(
            &mut graph,
            1.0,
            &EligibilityConfig::default(),
            EpochMillis(1_000),
        );
        assert_eq!(credits.len(), 1);
        assert!(graph.get(&source).unwrap().edges[0].strength > 0.5);
    }

    #[test]
    fn poor_outcome_weakens_the_same_recent_edge() {
        let (mut graph, source, target) = linked_graph(EpochMillis(0));
        let mut traces = EligibilityTraceRegistry::default();
        traces.mark(source, target, EdgeKind::Associative, EpochMillis(0));
        traces.apply_reward(
            &mut graph,
            0.0,
            &EligibilityConfig::default(),
            EpochMillis(1_000),
        );
        assert!(graph.get(&source).unwrap().edges[0].strength < 0.5);
    }

    #[test]
    fn expired_trace_receives_no_credit_and_is_removed() {
        let (mut graph, source, target) = linked_graph(EpochMillis(0));
        let mut traces = EligibilityTraceRegistry::default();
        traces.mark(source, target, EdgeKind::Associative, EpochMillis(0));
        let credits = traces.apply_reward(
            &mut graph,
            1.0,
            &EligibilityConfig::default(),
            EpochMillis(1_000_000),
        );
        assert!(credits.is_empty());
        assert_eq!(traces.len(), 0);
        assert_eq!(graph.get(&source).unwrap().edges[0].strength, 0.5);
    }
}
