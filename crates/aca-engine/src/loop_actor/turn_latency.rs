use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use aca_graph::Graph;
use aca_types::MentalObjectId;

const MAX_PENDING_TURNS: usize = 256;
const MAX_TURN_AGE: Duration = Duration::from_secs(30 * 60);

/// Monotonic actor-dequeue-to-terminal-action timing, linked by object provenance.
/// This is transient measurement state, never written into the memory graph.
#[derive(Default)]
pub(super) struct ForegroundTurnTracker {
    started: HashMap<MentalObjectId, Instant>,
}

impl ForegroundTurnTracker {
    pub(super) fn discard(&mut self, observation_id: MentalObjectId) {
        self.started.remove(&observation_id);
    }

    pub(super) fn start(&mut self, observation_id: MentalObjectId, at: Instant) {
        self.started
            .retain(|_, started| at.saturating_duration_since(*started) <= MAX_TURN_AGE);
        if self.started.len() >= MAX_PENDING_TURNS {
            if let Some(oldest) = self
                .started
                .iter()
                .min_by_key(|(_, at)| *at)
                .map(|(id, _)| *id)
            {
                self.started.remove(&oldest);
            }
        }
        self.started.insert(observation_id, at);
    }

    pub(super) fn finish_action(
        &mut self,
        graph: &Graph,
        target_id: MentalObjectId,
        at: Instant,
    ) -> Option<(MentalObjectId, u64)> {
        if let Some(started) = self.started.remove(&target_id) {
            return Some((target_id, elapsed_us(started, at)));
        }
        let mut frontier = vec![target_id];
        let mut seen = HashSet::new();
        let mut matched = Vec::new();
        let mut truncated = false;
        while let Some(id) = frontier.pop() {
            if seen.len() >= 16 {
                truncated = true;
                break;
            }
            if !seen.insert(id) {
                continue;
            }
            if self.started.contains_key(&id) {
                matched.push(id);
            }
            if let Some(object) = graph.get(&id) {
                frontier.extend(object.source_object_ids.iter().copied());
            }
        }
        // A response grounded in two live inputs cannot honestly be charged
        // to just one; leave both pending rather than fabricating attribution.
        if matched.len() != 1 || truncated {
            return None;
        }
        let id = matched[0];
        self.started
            .remove(&id)
            .map(|started| (id, elapsed_us(started, at)))
    }
}

fn elapsed_us(started: Instant, at: Instant) -> u64 {
    at.saturating_duration_since(started)
        .as_micros()
        .min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::MentalObject;
    use aca_util::EpochMillis;

    #[test]
    fn tracks_a_spoken_reflection_back_to_its_observation_once() {
        let mut graph = Graph::new();
        let observation = MentalObject::new_observation("question", EpochMillis(0), 0.5);
        let id = observation.id;
        let mut reflection = MentalObject::new_observation("answer", EpochMillis(0), 0.5);
        reflection.source_object_ids.push(id);
        let reflection_id = reflection.id;
        graph.insert(observation);
        graph.insert(reflection);
        let started = Instant::now();
        let mut tracker = ForegroundTurnTracker::default();
        tracker.start(id, started);
        assert_eq!(
            tracker.finish_action(&graph, reflection_id, started + Duration::from_millis(7)),
            Some((id, 7_000))
        );
        assert_eq!(
            tracker.finish_action(&graph, reflection_id, started + Duration::from_millis(8)),
            None
        );
    }

    #[test]
    fn abstains_when_a_spoken_reflection_uses_two_pending_foreground_inputs() {
        let mut graph = Graph::new();
        let first = MentalObject::new_observation("first", EpochMillis(0), 0.5);
        let second = MentalObject::new_observation("second", EpochMillis(0), 0.5);
        let mut reflection = MentalObject::new_observation("combined", EpochMillis(0), 0.5);
        reflection.source_object_ids.extend([first.id, second.id]);
        let target_id = reflection.id;
        let first_id = first.id;
        graph.insert(first);
        graph.insert(second);
        graph.insert(reflection);
        let started = Instant::now();
        let mut tracker = ForegroundTurnTracker::default();
        tracker.start(first_id, started);
        tracker.start(graph.get(&target_id).unwrap().source_object_ids[1], started);
        assert_eq!(
            tracker.finish_action(&graph, target_id, started + Duration::from_millis(7)),
            None
        );
        assert_eq!(tracker.started.len(), 2);
    }
}
