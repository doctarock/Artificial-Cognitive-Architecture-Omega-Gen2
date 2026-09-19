use serde::{Deserialize, Serialize};

/// The content-type axis of a Mental Object — what kind of cognitive
/// artifact this is. Orthogonal to `MemoryRole` (which memory subsystem
/// view(s) currently include the object).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MentalObjectKind {
    Observation,
    Thought,
    Reflection,
    Question,
    Idea,
    Goal,
    Belief,
    Intention,
    Decision,
    Memory,
    Hypothesis,
}

/// Which memory subsystem view(s) a Mental Object currently belongs to.
/// Not a separate storage system — every object lives in the one activation
/// graph; this is just a tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryRole {
    Working,
    Episodic,
    Semantic,
    #[serde(rename = "self")]
    SelfMemory,
}

/// The relationship an associative edge represents between two Mental
/// Objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EdgeKind {
    Associative,
    Causal,
    SubgoalOf,
    DerivedFrom,
    Contradicts,
    Supports,
    /// Suppresses rather than excites the target during spreading activation.
    Inhibitory,
}

/// SOAR-style goal-stack membership status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GoalStatus {
    Active,
    Suspended,
    Satisfied,
    Abandoned,
    Impassed,
}

/// Lifecycle status of a Mental Object in the graph. Discarding is a status
/// change, never a delete — memories remain available indefinitely unless
/// explicitly forgotten (decay affects activation, not storage).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObjectStatus {
    Active,
    Archived,
    Discarded,
}

/// The model-tier ladder: 0 = pure architecture (no model), 1 = many
/// concurrent small models, 2 = a handful of mid models, 3 = one
/// single-flight capable model, 4 = one large emergency-escalation-only
/// model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    T0,
    T1,
    T2,
    T3,
    T4,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_kind_serializes_kebab_case() {
        assert_eq!(
            serde_json::to_string(&EdgeKind::SubgoalOf).unwrap(),
            "\"subgoal-of\""
        );
        assert_eq!(
            serde_json::to_string(&EdgeKind::DerivedFrom).unwrap(),
            "\"derived-from\""
        );
    }

    #[test]
    fn memory_role_self_memory_serializes_as_self() {
        assert_eq!(
            serde_json::to_string(&MemoryRole::SelfMemory).unwrap(),
            "\"self\""
        );
    }

    #[test]
    fn tier_ordering_reflects_the_ladder() {
        assert!(Tier::T0 < Tier::T1);
        assert!(Tier::T1 < Tier::T2);
        assert!(Tier::T2 < Tier::T3);
        assert!(Tier::T3 < Tier::T4);
    }

    #[test]
    fn round_trips_through_json() {
        let kind = MentalObjectKind::Hypothesis;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"hypothesis\"");
        let back: MentalObjectKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
    }
}
