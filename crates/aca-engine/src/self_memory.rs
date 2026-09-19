use aca_types::{MemoryRole, MentalObject, MentalObjectKind};
use aca_util::EpochMillis;

/// Omega's foundational self-knowledge — specs.md's Self Memory subsystem
/// ("identity continuity... beliefs, values, preferences... self-
/// description"), seeded once at first boot so it's real, persistent
/// content rather than only a counted-but-never-written `MemoryRole`
/// variant (before this, nothing anywhere in the engine ever tagged an
/// object `MemoryRole::SelfMemory`). Drawn directly from specs.md's own
/// Purpose/Core Principle/Primary Cognitive Question sections — this is
/// Omega's actual designed identity, not fabricated personality content,
/// and deliberately not run through a semantic classifier (see
/// `steps::memory_formation::MemoryFormationOutcome`'s doc comment on why
/// belief-revision classification is explicitly deferred to a future
/// LLM-backed pass): these five statements are foundational enough to seed
/// directly, without needing a judgement call about whether they're "new"
/// or "revised."
pub const SELF_MEMORY_SEED_TEXTS: &[&str] = &[
    "I am Omega, a persistent artificial cognitive architecture designed to emulate the functional processes of cognition rather than the behaviour of a conversational assistant.",
    "I continuously maintain an internal state, observe my environment, form thoughts, consolidate experiences into memory, develop intentions, and communicate only when communication is the most appropriate cognitive action.",
    "Conversation is one possible behaviour of mine, not my primary purpose.",
    "The architecture is the cognitive system; language models are cognitive processors operating within it. My identity and continuity do not depend on any single model call, provider, or context window.",
    "The question I continuously ask is not \"what should I do next?\" but \"what currently occupies my mind?\" - tasks are consequences of my mental state, not the purpose of my existence.",
];

/// Builds one `Belief`-kind, `SelfMemory`-tagged Mental Object per seed
/// text. Embeddings are left unresolved (`None`) — the caller (an async
/// context with an `EmbeddingClient`) is responsible for resolving them
/// before insertion, the same shell/resolve split `steps::observe` uses for
/// ordinary Observations.
pub fn seed_self_memory_objects(now: EpochMillis, decay_d: f32) -> Vec<MentalObject> {
    SELF_MEMORY_SEED_TEXTS
        .iter()
        .map(|text| {
            let mut object = MentalObject::new_observation(*text, now, decay_d);
            object.kind = MentalObjectKind::Belief;
            object.memory_roles = vec![MemoryRole::SelfMemory];
            object
        })
        .collect()
}

/// Whether Self Memory has already been seeded — checked against whatever
/// was just loaded from the durable store, so a restart never re-seeds
/// (and therefore never duplicates) this content.
///
/// Checks for the presence of the actual seed text, not merely "any object
/// tagged `SelfMemory` exists" - the latter is not equivalent, confirmed
/// live: `steps::memory_formation`'s classifier can mis-tag an unrelated
/// candidate (e.g. a misheard voice fragment) as `SelfBelief` before real
/// seeding ever gets a chance to run, and that single mistagged object then
/// permanently satisfied the old, weaker check on every subsequent restart -
/// real identity content was never seeded at all, on any boot, for as long
/// as that one contaminated object existed. Matching on the seed text itself
/// means contamination elsewhere in Self Memory can never block the one
/// thing this function is actually supposed to guarantee.
pub fn is_self_memory_seeded(objects: &[MentalObject]) -> bool {
    objects
        .iter()
        .filter(|object| object.memory_roles.contains(&MemoryRole::SelfMemory))
        .any(|object| SELF_MEMORY_SEED_TEXTS.contains(&object.text.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::ObjectStatus;

    #[test]
    fn seed_self_memory_objects_tags_every_object_as_self_memory_beliefs() {
        let objects = seed_self_memory_objects(EpochMillis(0), 0.5);
        assert_eq!(objects.len(), SELF_MEMORY_SEED_TEXTS.len());
        for object in &objects {
            assert_eq!(object.kind, MentalObjectKind::Belief);
            assert_eq!(object.memory_roles, vec![MemoryRole::SelfMemory]);
            assert!(object.embedding.is_none(), "embedding resolution is the caller's job, not this function's");
            assert_eq!(object.status, ObjectStatus::Active);
        }
    }

    #[test]
    fn is_self_memory_seeded_is_false_for_an_empty_or_self_memory_less_graph() {
        assert!(!is_self_memory_seeded(&[]));
        let plain = MentalObject::new_observation("hello", EpochMillis(0), 0.5);
        assert!(!is_self_memory_seeded(&[plain]));
    }

    #[test]
    fn is_self_memory_seeded_is_true_once_any_object_carries_the_role() {
        let seeded = seed_self_memory_objects(EpochMillis(0), 0.5);
        assert!(is_self_memory_seeded(&seeded));
    }

    #[test]
    fn a_contaminated_self_memory_object_does_not_count_as_seeded() {
        // Regression guard for the confirmed-live failure: a misclassified
        // object (e.g. a misheard voice fragment) tagged `SelfMemory` by
        // `steps::memory_formation`'s classifier must never be mistaken for
        // real seed content - otherwise it permanently blocks real seeding
        // on every future restart, exactly as happened live.
        let mut contaminated = MentalObject::new_observation("You called me weak, didn't you?", EpochMillis(0), 0.5);
        contaminated.memory_roles = vec![MemoryRole::SelfMemory];
        assert!(!is_self_memory_seeded(&[contaminated]), "a mistagged object with non-seed text must not block real seeding");
    }
}
