use std::collections::HashSet;

use aca_graph::Graph;
use aca_types::{EdgeKind, MemoryRole, MentalObject, MentalObjectId, MentalObjectKind, ObjectStatus};
use aca_util::EpochMillis;

/// Recognition of *other* people, kept structurally separate from
/// `self_memory` (which is exclusively Omega's own identity) even though the
/// shape mirrors it closely: a persistent graph node an utterance can be
/// linked to and reinforced against, grown the same way every other memory
/// in this graph grows.
///
/// Deliberately keyed only by a *named/enrolled* `speaker_label` - never by
/// the voice service's own ephemeral "Unknown speaker N" diarization slot.
/// Confirmed live against a real session transcript: an enrolled label like
/// "derek" maps to exactly one underlying `speaker_track_id` for the whole
/// session, while every "Unknown speaker N" label maps to several different
/// `speaker_track_id`s over the same window - it's a rotating bucket the
/// voice service assigns to not-yet-enrolled voices, not a stable identity.
/// Callers (`loop_actor`'s room-input handling) only ever pass a
/// `speaker_label` here once it's already been filtered down to enrolled
/// names - see that filtering for where the line is actually drawn.
/// Finds the interlocutor node previously recognized under `speaker_label`,
/// or mints one on first recognition - a linear scan, deliberately: the
/// number of distinct enrolled interlocutors is small, so no index is
/// justified yet (same reasoning as `self_memory::is_self_memory_seeded`'s
/// own scan-by-tag lookup).
///
/// The new node is never itself a Coalition candidate (nothing nominates
/// it for Working Memory) - it exists purely as a memory anchor other
/// objects associate with and reinforce, not something Omega "thinks about"
/// directly, so it deliberately carries no embedding (see `is_embedding_resolved`'s
/// gate elsewhere, which this node is never subject to).
pub fn find_or_create(graph: &mut Graph, speaker_label: &str, now: EpochMillis, decay_d: f32) -> MentalObjectId {
    if let Some(existing) = graph.iter().find(|object| object.data.get("interlocutor_hint").and_then(|v| v.as_str()) == Some(speaker_label)) {
        return existing.id;
    }
    let mut object = MentalObject::new_observation(format!("An interlocutor I've heard from, currently associated with the label '{speaker_label}'."), now, decay_d);
    object.kind = MentalObjectKind::Belief;
    object.memory_roles = vec![MemoryRole::Semantic];
    object.data = serde_json::json!({"interlocutor_hint": speaker_label});
    let id = object.id;
    graph.insert(object);
    id
}

/// Reinforces the link from a fresh utterance to the interlocutor it came
/// from - the exact same Hebbian machinery `steps::memory_formation::reinforce_coactivation`
/// already uses for Working-Memory co-occurrence, applied here to
/// utterance-to-speaker association instead. `record_reference` on the
/// interlocutor's own activation is what makes repetition (not a bonus
/// dial, unlike `self_memory_activation_bonus`) the entire mechanism behind
/// familiarity: a frequently-heard-from voice's node accumulates references
/// and stays salient through ordinary ACT-R base-level decay, the same way
/// any other frequently-referenced memory does.
pub fn reinforce_link(graph: &mut Graph, utterance_id: MentalObjectId, interlocutor_id: MentalObjectId, now: EpochMillis) {
    if let Some(utterance) = graph.get_mut(&utterance_id) {
        aca_graph::reinforce_edge(&mut utterance.edges, interlocutor_id, EdgeKind::DerivedFrom, now, aca_graph::DEFAULT_HEBBIAN_INCREMENT, aca_graph::DEFAULT_MAX_EDGE_STRENGTH);
    }
    if let Some(interlocutor) = graph.get_mut(&interlocutor_id) {
        aca_graph::record_reference(&mut interlocutor.activation, now);
    }
}

/// The "social cloud" - `steps::recall::recall`'s anchor set for a
/// specialist recall pass scoped to one specific person, rather than to
/// whatever's currently in Working Memory. Every `DerivedFrom` edge into
/// `interlocutor_id` was put there by `reinforce_link` above, one per real
/// utterance actually heard from them - so this is exactly "everything this
/// person has actually said," not a topical or embedding-similarity guess.
///
/// Deliberately a plain reverse-edge scan, not `spread_activation_multi_hop`
/// starting *from* the interlocutor node itself: that node is built with no
/// outgoing edges at all (see `find_or_create`'s own doc comment - it's a
/// pure sink other objects point at, never a spreading source), so a
/// same-direction spread from it would terminate immediately with zero
/// results. This returns each such utterance directly as an anchor instead;
/// `recall`'s own multi-hop spread then continues outward from *their*
/// edges exactly as it would from any other anchor set.
///
/// `ObjectStatus::Active`-only, matching `aca_graph::recall`'s own "only
/// active nodes are eligible recall candidates" convention - a discarded
/// utterance shouldn't anchor a fresh recall pass any more than it would
/// naturally spread activation elsewhere in the graph.
pub fn social_cloud_anchors(graph: &Graph, interlocutor_id: MentalObjectId) -> HashSet<MentalObjectId> {
    graph
        .iter()
        .filter(|object| object.status == ObjectStatus::Active)
        .filter(|object| object.edges.iter().any(|edge| edge.kind == EdgeKind::DerivedFrom && edge.target_id == interlocutor_id))
        .map(|object| object.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_or_create_mints_a_semantic_belief_node_tagged_with_the_hint() {
        let mut graph = Graph::new();
        let id = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);
        let object = graph.get(&id).expect("interlocutor node should be in the graph");
        assert_eq!(object.kind, MentalObjectKind::Belief);
        assert_eq!(object.memory_roles, vec![MemoryRole::Semantic]);
        assert_eq!(object.data.get("interlocutor_hint").and_then(|v| v.as_str()), Some("derek"));
        assert!(object.embedding.is_none(), "an interlocutor anchor is never itself a Coalition candidate");
    }

    #[test]
    fn find_or_create_is_idempotent_per_label() {
        let mut graph = Graph::new();
        let first = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);
        let second = find_or_create(&mut graph, "derek", EpochMillis(1_000), 0.5);
        assert_eq!(first, second, "the same label must resolve to the same persistent node, not a fresh one each time");
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn find_or_create_gives_different_labels_different_nodes() {
        let mut graph = Graph::new();
        let derek = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);
        let someone_else = find_or_create(&mut graph, "someone_else", EpochMillis(0), 0.5);
        assert_ne!(derek, someone_else);
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn reinforce_link_strengthens_the_edge_and_refreshes_the_interlocutors_activation() {
        let mut graph = Graph::new();
        let interlocutor_id = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);
        let utterance = MentalObject::new_observation("hello Omega", EpochMillis(1_000), 0.5);
        let utterance_id = utterance.id;
        graph.insert(utterance);

        reinforce_link(&mut graph, utterance_id, interlocutor_id, EpochMillis(1_000));

        let utterance = graph.get(&utterance_id).unwrap();
        assert_eq!(utterance.edges.len(), 1);
        assert_eq!(utterance.edges[0].target_id, interlocutor_id);
        assert_eq!(utterance.edges[0].kind, EdgeKind::DerivedFrom);
        assert!(utterance.edges[0].strength > 0.0);

        let interlocutor = graph.get(&interlocutor_id).unwrap();
        assert_eq!(interlocutor.activation.reference_log.len(), 2, "creation reference plus the reinforcement reference");
    }

    #[test]
    fn repeated_reinforcement_from_the_same_interlocutor_accumulates_edge_strength() {
        let mut graph = Graph::new();
        let interlocutor_id = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);

        let first = MentalObject::new_observation("hello", EpochMillis(1_000), 0.5);
        let first_id = first.id;
        graph.insert(first);
        reinforce_link(&mut graph, first_id, interlocutor_id, EpochMillis(1_000));

        let second = MentalObject::new_observation("how are you", EpochMillis(2_000), 0.5);
        let second_id = second.id;
        graph.insert(second);
        reinforce_link(&mut graph, second_id, interlocutor_id, EpochMillis(2_000));

        let interlocutor = graph.get(&interlocutor_id).unwrap();
        assert_eq!(interlocutor.activation.reference_log.len(), 3, "creation plus two reinforcements");
    }

    #[test]
    fn social_cloud_anchors_returns_exactly_this_interlocutors_own_utterances() {
        let mut graph = Graph::new();
        let derek_id = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);
        let alice_id = find_or_create(&mut graph, "alice", EpochMillis(0), 0.5);

        let derek_utterance = MentalObject::new_observation("derek's message", EpochMillis(1_000), 0.5);
        let derek_utterance_id = derek_utterance.id;
        graph.insert(derek_utterance);
        reinforce_link(&mut graph, derek_utterance_id, derek_id, EpochMillis(1_000));

        let alice_utterance = MentalObject::new_observation("alice's message", EpochMillis(1_000), 0.5);
        let alice_utterance_id = alice_utterance.id;
        graph.insert(alice_utterance);
        reinforce_link(&mut graph, alice_utterance_id, alice_id, EpochMillis(1_000));

        let unrelated = MentalObject::new_observation("never linked to anyone", EpochMillis(1_000), 0.5);
        graph.insert(unrelated);

        let anchors = social_cloud_anchors(&graph, derek_id);
        assert_eq!(anchors, [derek_utterance_id].into_iter().collect(), "should contain exactly derek's own utterance, not alice's and not the unrelated object");
    }

    #[test]
    fn social_cloud_anchors_excludes_a_discarded_utterance() {
        let mut graph = Graph::new();
        let derek_id = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);

        let utterance = MentalObject::new_observation("an old message", EpochMillis(1_000), 0.5);
        let utterance_id = utterance.id;
        graph.insert(utterance);
        reinforce_link(&mut graph, utterance_id, derek_id, EpochMillis(1_000));
        graph.discard(&utterance_id, EpochMillis(2_000));

        let anchors = social_cloud_anchors(&graph, derek_id);
        assert!(anchors.is_empty(), "a discarded utterance must not anchor a fresh recall pass");
    }

    #[test]
    fn social_cloud_anchors_is_empty_for_an_interlocutor_never_linked_to_anything() {
        let mut graph = Graph::new();
        let derek_id = find_or_create(&mut graph, "derek", EpochMillis(0), 0.5);
        assert!(social_cloud_anchors(&graph, derek_id).is_empty());
    }
}
