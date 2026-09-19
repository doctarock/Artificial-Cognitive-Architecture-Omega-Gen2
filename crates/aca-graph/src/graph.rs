use std::collections::{HashMap, HashSet};

use aca_types::{MentalObject, MentalObjectId, ObjectStatus};

/// The in-memory activation graph: every Mental Object, keyed by id.
/// Associative edges live directly on each object (`MentalObject::edges`,
/// outgoing), so this is a plain id-keyed map rather than a separate
/// adjacency structure — ACT-R's math needs no generic graph traversal, so a
/// `petgraph`-style structure isn't justified for v1 (see the build plan).
///
/// This type has no `tokio` dependency and performs no I/O — it is owned
/// exclusively by `aca-engine`'s `CognitiveLoopActor` and mutated only from
/// within a single tick.
#[derive(Debug, Default)]
pub struct Graph {
    objects: HashMap<MentalObjectId, MentalObject>,
    goal_ids: HashSet<MentalObjectId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, object: MentalObject) {
        if object.goal.is_some() { self.goal_ids.insert(object.id); }
        else { self.goal_ids.remove(&object.id); }
        self.objects.insert(object.id, object);
    }

    pub fn get(&self, id: &MentalObjectId) -> Option<&MentalObject> {
        self.objects.get(id)
    }

    pub fn get_mut(&mut self, id: &MentalObjectId) -> Option<&mut MentalObject> {
        self.objects.get_mut(id)
    }

    /// Marks an object discarded rather than removing it — memories remain
    /// available indefinitely unless explicitly forgotten; decay affects
    /// activation, never storage.
    pub fn discard(&mut self, id: &MentalObjectId, discarded_at: aca_util::EpochMillis) {
        if let Some(object) = self.objects.get_mut(id) {
            object.status = ObjectStatus::Discarded;
            object.discarded_at = Some(discarded_at);
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MentalObject> {
        self.objects.values()
    }

    /// Sparse goal subset; status may change in place, so callers still
    /// filter for Active when that is the property they require.
    pub fn goal_objects(&self) -> impl Iterator<Item = &MentalObject> {
        self.goal_ids.iter().filter_map(|id| self.objects.get(id))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut MentalObject> {
        self.objects.values_mut()
    }

    /// Active (non-archived, non-discarded) objects — the only ones
    /// eligible as recall/coalition candidates.
    pub fn active_ids(&self) -> impl Iterator<Item = MentalObjectId> + '_ {
        self.objects
            .values()
            .filter(|o| o.status == ObjectStatus::Active)
            .map(|o| o.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_util::EpochMillis;

    #[test]
    fn insert_and_get_round_trip() {
        let mut graph = Graph::new();
        let obj = MentalObject::new_observation("hello", EpochMillis(0), 0.5);
        let id = obj.id;
        graph.insert(obj);
        assert_eq!(graph.get(&id).unwrap().text, "hello");
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn discard_marks_status_but_keeps_the_object() {
        let mut graph = Graph::new();
        let obj = MentalObject::new_observation("hello", EpochMillis(0), 0.5);
        let id = obj.id;
        graph.insert(obj);
        graph.discard(&id, EpochMillis(100));
        assert_eq!(graph.len(), 1, "discard must not remove the object");
        let discarded = graph.get(&id).unwrap();
        assert_eq!(discarded.status, ObjectStatus::Discarded);
        assert_eq!(discarded.discarded_at, Some(EpochMillis(100)));
        assert_eq!(graph.active_ids().count(), 0);
    }

    #[test]
    fn goal_index_tracks_insertions_and_replacements() {
        let mut graph = Graph::new();
        let mut object = MentalObject::new_observation("goal", EpochMillis(0), 0.5);
        object.goal = Some(aca_types::GoalStackMembership {
            stack_id: aca_types::GoalStackId::new(), parent_goal_id: None,
            status: aca_types::GoalStatus::Active, priority: 1.0,
        });
        assert_eq!(graph.goal_objects().count(), 0);
        graph.insert(object.clone());
        assert_eq!(graph.goal_objects().count(), 1);
        object.goal = None;
        graph.insert(object);
        assert_eq!(graph.goal_objects().count(), 0);
    }
}
