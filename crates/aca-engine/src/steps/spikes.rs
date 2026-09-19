use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use aca_graph::{Graph, effective_strength};
use aca_types::{EdgeKind, MentalObjectId, ObjectStatus};
use aca_util::EpochMillis;

#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    pub due_at: EpochMillis,
    sequence: u64,
    pub source: MentalObjectId,
    pub target: MentalObjectId,
    pub pulse: f32,
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        (self.due_at, self.sequence) == (other.due_at, other.sequence)
    }
}
impl Eq for SpikeEvent {}
impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.due_at, self.sequence).cmp(&(other.due_at, other.sequence))
    }
}

/// Actor-local sparse propagation queue. An object fires only on actual
/// input; its bounded outgoing neighborhood receives delayed signed pulses.
/// There is no dense sweep and no per-object decay timer.
#[derive(Debug, Default)]
pub struct SpikeEventQueue {
    pending: BinaryHeap<Reverse<SpikeEvent>>,
    next_sequence: u64,
}

impl SpikeEventQueue {
    pub fn next_deadline(&self) -> Option<EpochMillis> {
        self.pending.peek().map(|event| event.0.due_at)
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn clear(&mut self) {
        self.pending.clear();
    }

    pub fn schedule_from(
        &mut self,
        graph: &Graph,
        source_id: MentalObjectId,
        now: EpochMillis,
        delay_ms: i64,
        edge_decay_rate_per_ms: f64,
        max_fanout: usize,
        max_pending: usize,
    ) -> usize {
        if max_fanout == 0 || self.pending.len() >= max_pending {
            return 0;
        }
        let Some(source) = graph.get(&source_id) else {
            return 0;
        };
        if source.status != ObjectStatus::Active {
            return 0;
        }
        let mut outgoing: Vec<_> = source
            .edges
            .iter()
            .filter_map(|edge| {
                let sign = match edge.kind {
                    EdgeKind::Associative | EdgeKind::Causal | EdgeKind::Supports => 1.0,
                    EdgeKind::Inhibitory | EdgeKind::Contradicts => -1.0,
                    _ => return None,
                };
                let target = graph.get(&edge.target_id)?;
                let strength = effective_strength(edge, now, edge_decay_rate_per_ms);
                (target.status == ObjectStatus::Active && target.id != source_id && strength > 0.0)
                    .then(|| (edge.target_id, sign * strength))
            })
            .collect();
        outgoing.sort_by(|(id_a, pulse_a), (id_b, pulse_b)| {
            pulse_b
                .abs()
                .total_cmp(&pulse_a.abs())
                .then_with(|| id_a.cmp(id_b))
        });
        outgoing.truncate(max_fanout.min(max_pending.saturating_sub(self.pending.len())));
        let fan = (outgoing.len() as f32).sqrt().max(1.0);
        for (target, pulse) in &outgoing {
            let event = SpikeEvent {
                due_at: EpochMillis(now.0.saturating_add(delay_ms.max(0))),
                sequence: self.next_sequence,
                source: source_id,
                target: *target,
                pulse: *pulse / fan,
            };
            self.next_sequence = self.next_sequence.wrapping_add(1);
            self.pending.push(Reverse(event));
        }
        outgoing.len()
    }

    pub fn drain_due(&mut self, now: EpochMillis, limit: usize) -> Vec<SpikeEvent> {
        let mut due = Vec::new();
        while due.len() < limit
            && self
                .pending
                .peek()
                .is_some_and(|event| event.0.due_at <= now)
        {
            due.push(
                self.pending
                    .pop()
                    .expect("peek just confirmed a queued event")
                    .0,
            );
        }
        due
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::MentalObject;

    #[test]
    fn only_a_sparse_signed_neighborhood_is_scheduled_and_drained_when_due() {
        let mut graph = Graph::new();
        let mut source = MentalObject::new_observation("source", EpochMillis(0), 0.5);
        let positive = MentalObject::new_observation("positive", EpochMillis(0), 0.5);
        let negative = MentalObject::new_observation("negative", EpochMillis(0), 0.5);
        let contradictory = MentalObject::new_observation("contradictory", EpochMillis(0), 0.5);
        for (target, kind, strength) in [
            (positive.id, EdgeKind::Associative, 1.0),
            (negative.id, EdgeKind::Inhibitory, 0.5),
            (contradictory.id, EdgeKind::Contradicts, 0.4),
        ] {
            aca_graph::reinforce_edge(
                &mut source.edges,
                target,
                kind,
                EpochMillis(0),
                strength,
                1.0,
            );
        }
        let source_id = source.id;
        for object in [source, positive, negative, contradictory] {
            graph.insert(object);
        }
        let mut queue = SpikeEventQueue::default();
        assert_eq!(
            queue.schedule_from(&graph, source_id, EpochMillis(0), 10, 0.0, 3, 16),
            3
        );
        assert_eq!(queue.next_deadline(), Some(EpochMillis(10)));
        assert!(queue.drain_due(EpochMillis(9), 16).is_empty());
        let due = queue.drain_due(EpochMillis(10), 16);
        assert_eq!(due.len(), 3);
        assert!(due.iter().any(|event| event.pulse > 0.0));
        assert!(due.iter().any(|event| event.pulse < 0.0));
        assert_eq!(due.iter().filter(|event| event.pulse < 0.0).count(), 2);
        assert_eq!(queue.len(), 0);
    }
}
