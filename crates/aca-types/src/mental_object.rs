use aca_util::{EpochMillis, RingBuffer};
use serde::{Deserialize, Serialize};

use crate::enums::{EdgeKind, GoalStatus, MemoryRole, MentalObjectKind, ObjectStatus, Tier};
use crate::ids::{GoalStackId, MentalObjectId};

/// How many past reference timestamps ACT-R's base-level activation sum
/// keeps per object before the oldest is evicted. A bound, not a theoretical
/// commitment — recent references dominate the power-law sum in practice,
/// so an unbounded log would cost memory for no real activation-math
/// benefit.
pub const DEFAULT_REFERENCE_LOG_CAPACITY: usize = 64;

/// A learned association from one Mental Object to another. `strength` is
/// `S_ki` in ACT-R's spreading-activation term — Hebbian-incremented on
/// co-activation, with its own decay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssociativeEdge {
    pub target_id: MentalObjectId,
    pub kind: EdgeKind,
    pub strength: f32,
    pub last_coactivated_at: EpochMillis,
}

/// The ACT-R activation state of a Mental Object: `total = base_level +
/// spreading + noise`. `base_level` and `spreading` are cached, recomputed
/// each tick from `reference_log` and the current Working Memory context
/// respectively — see `aca-graph::activation`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationState {
    pub base_level: f32,
    pub reference_log: RingBuffer<EpochMillis>,
    pub decay_d: f32,
    pub spreading: f32,
    pub noise: f32,
    pub total: f32,
    pub last_computed_at: EpochMillis,
}

impl ActivationState {
    /// A freshly-created object: one reference (its own creation), zero
    /// spreading/noise/total until the first activation pass computes them.
    pub fn new_at(created_at: EpochMillis, decay_d: f32) -> Self {
        let mut reference_log = RingBuffer::new(DEFAULT_REFERENCE_LOG_CAPACITY);
        reference_log.push(created_at);
        Self {
            base_level: 0.0,
            reference_log,
            decay_d,
            spreading: 0.0,
            noise: 0.0,
            total: 0.0,
            last_computed_at: created_at,
        }
    }
}

/// Predictive-processing state: the expectation this object was compared
/// against, the resulting prediction error, and its precision weight.
/// `None` fields mean "not yet computed" (e.g. an object still awaiting its
/// embedding — see the embedding-is-I/O wrinkle in the build plan).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionState {
    pub expected_embedding: Option<Vec<f32>>,
    pub error_magnitude: Option<f32>,
    pub precision: Option<f32>,
}

/// Short-lived, event-driven state layered beneath ACT-R activation. ACT-R
/// answers whether an object is retrievable over seconds to days; this state
/// answers whether recent local input is strong enough for the object to emit
/// an activation event right now. It is deliberately transient when loaded
/// from durable memory: membrane potential and adaptation are process state,
/// not autobiographical content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalObjectDynamics {
    pub potential: f32,
    pub threshold: f32,
    pub adaptation: f32,
    pub last_updated_at: EpochMillis,
    pub refractory_until: Option<EpochMillis>,
    pub last_fired_at: Option<EpochMillis>,
}

impl MentalObjectDynamics {
    pub fn new_at(created_at: EpochMillis) -> Self {
        Self {
            potential: 0.0,
            threshold: 1.0,
            adaptation: 0.0,
            last_updated_at: created_at,
            refractory_until: None,
            last_fired_at: None,
        }
    }
}

impl Default for MentalObjectDynamics {
    fn default() -> Self {
        Self::new_at(EpochMillis(0))
    }
}

/// SOAR-style goal-stack membership. `stack_id == "self"` (by convention, a
/// fixed well-known `GoalStackId`) marks the root Self Memory stack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalStackMembership {
    pub stack_id: GoalStackId,
    pub parent_goal_id: Option<MentalObjectId>,
    pub status: GoalStatus,
    pub priority: f32,
}

/// GWT broadcast bookkeeping.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceState {
    pub attention_score: Option<f32>,
    pub in_working_memory: bool,
    pub broadcast_count: u32,
    pub last_broadcast_at: Option<EpochMillis>,
}

/// Whether an internally-classified memory has been independently confirmed
/// yet. `Candidate` means only the system's own classifier has ever vouched
/// for this content — it can still be recalled and reinforced like any other
/// object, but readers that treat content as durable self-knowledge
/// (`loop_actor::memory_coordinator::build_self_summary`, the Self Memory
/// activation bonus) must not trust it until something *outside* the
/// classifier that produced it touches it again on a later tick
/// (`steps::memory_formation::form_memory`'s dedup-reinforcement scan,
/// `steps::synthesize`'s own dedup path). This is deliberately not
/// `ObjectStatus`: a staged candidate stays `Active` throughout, since it
/// must remain eligible for exactly the recall/reinforcement that would
/// confirm it — this is an orthogonal trust dimension, not a lifecycle
/// state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromotionStatus {
    Candidate,
    Confirmed,
}

/// See `PromotionStatus`'s own doc comment for what the distinction means.
/// Every object not produced by an internal classifier (raw Observations,
/// hand-seeded Self Memory, Episodic memories - see `form_memory`'s doc
/// comment on why Episodic formation is a computable event, never
/// judgement-gated) is `Confirmed` from the moment it's created; this is
/// the safe default `new_observation` uses, so nothing outside
/// `form_memory`'s own Semantic/SelfBelief arm has to opt in explicitly.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PromotionState {
    pub status: PromotionStatus,
    pub staged_at: EpochMillis,
    pub confirmed_at: Option<EpochMillis>,
    /// How many times an independent later touch has reconfirmed this
    /// content since it was first staged - `0` for a `Candidate` that
    /// hasn't been confirmed yet, and for anything `Confirmed` at creation
    /// (nothing to count).
    pub confirming_references: u32,
}

impl PromotionState {
    pub fn confirmed(at: EpochMillis) -> Self {
        Self { status: PromotionStatus::Confirmed, staged_at: at, confirmed_at: Some(at), confirming_references: 0 }
    }

    pub fn candidate(at: EpochMillis) -> Self {
        Self { status: PromotionStatus::Candidate, staged_at: at, confirmed_at: None, confirming_references: 0 }
    }

    /// Promotes a `Candidate` to `Confirmed` - a no-op (doesn't bump
    /// `confirmed_at`/`confirming_references` again) if already confirmed,
    /// so repeated reinforcement of an already-trusted memory doesn't need
    /// its own separate guard at every call site.
    pub fn confirm(&mut self, at: EpochMillis) {
        if self.status == PromotionStatus::Confirmed {
            return;
        }
        self.status = PromotionStatus::Confirmed;
        self.confirmed_at = Some(at);
        self.confirming_references += 1;
    }

    /// The mirror of `confirm()`: walks a `Confirmed` object back to
    /// `Candidate` - the promotion "cache" invalidating instead of staying
    /// stale once the live `confidence` it was trusted on has eroded too
    /// far (see `steps::confidence_revision::apply_contradiction_penalty_and_maybe_demote`,
    /// the only caller). A no-op if already `Candidate`, symmetric to
    /// `confirm()`'s own no-op. `confirming_references` is deliberately
    /// left untouched - a historical count of how many times this content
    /// has ever earned trust, not a current streak that demotion should
    /// erase.
    pub fn demote(&mut self, at: EpochMillis) {
        if self.status == PromotionStatus::Candidate {
            return;
        }
        self.status = PromotionStatus::Candidate;
        self.staged_at = at;
        self.confirmed_at = None;
    }
}

impl Default for PromotionState {
    /// `#[serde(default)]`'s fallback for a row written before this field
    /// existed - same "confirmed, not demoted" reasoning as this struct's
    /// own doc comment. `EpochMillis(0)` is a placeholder timestamp, same
    /// convention `MentalObjectDynamics::default()` already uses.
    fn default() -> Self {
        Self::confirmed(EpochMillis(0))
    }
}

/// The one universal cognitive data structure. Every candidate, memory,
/// goal, prediction, coalition, and operator output is a `MentalObject` —
/// `kind` says what it represents; `memory_roles` says which memory
/// subsystem view(s) currently include it; the rest carries whichever of
/// the four theories' bookkeeping is relevant to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalObject {
    pub id: MentalObjectId,
    pub kind: MentalObjectKind,
    pub created_at: EpochMillis,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    /// Kind-specific structured payload. Deliberately loose (`Value`) for
    /// v1 — tighten to a `#[serde(tag = "kind")]` enum per `MentalObjectKind`
    /// once real usage settles the shapes (see the plan's named
    /// simplifications).
    pub data: serde_json::Value,

    pub activation: ActivationState,
    pub edges: Vec<AssociativeEdge>,
    pub prediction: PredictionState,
    #[serde(default)]
    pub dynamics: MentalObjectDynamics,
    pub goal: Option<GoalStackMembership>,
    pub workspace: WorkspaceState,
    pub memory_roles: Vec<MemoryRole>,

    pub confidence: f32,
    pub tier_used: Option<Tier>,
    pub produced_by_operator: Option<String>,
    pub source_object_ids: Vec<MentalObjectId>,
    /// See `PromotionState`'s own doc comment. `#[serde(default)]` so rows
    /// written before this field existed deserialize as `Confirmed`
    /// (`PromotionState`'s own `Default` impl below), never retroactively
    /// demoted.
    #[serde(default = "PromotionState::default")]
    pub promotion: PromotionState,

    pub status: ObjectStatus,
    pub discarded_at: Option<EpochMillis>,
}

impl MentalObject {
    /// Construct a freshly-observed object: active, not yet in Working
    /// Memory, no edges/goal/tier/provenance yet.
    pub fn new_observation(
        text: impl Into<String>,
        created_at: EpochMillis,
        decay_d: f32,
    ) -> Self {
        Self {
            id: MentalObjectId::new(),
            kind: MentalObjectKind::Observation,
            created_at,
            text: text.into(),
            embedding: None,
            data: serde_json::Value::Null,
            activation: ActivationState::new_at(created_at, decay_d),
            edges: Vec::new(),
            prediction: PredictionState::default(),
            dynamics: MentalObjectDynamics::new_at(created_at),
            goal: None,
            workspace: WorkspaceState::default(),
            memory_roles: Vec::new(),
            confidence: 0.5,
            tier_used: None,
            produced_by_operator: None,
            source_object_ids: Vec::new(),
            promotion: PromotionState::confirmed(created_at),
            status: ObjectStatus::Active,
            discarded_at: None,
        }
    }

    /// An object is a valid recall/coalition candidate only while active and
    /// past the pending-embedding stage (see the Observe/Compare wrinkle) —
    /// this is the one gate that keeps Compare/Coalition entirely free of
    /// I/O concerns.
    pub fn is_embedding_resolved(&self) -> bool {
        self.embedding.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_observation_seeds_one_reference() {
        let now = EpochMillis(1_000);
        let obj = MentalObject::new_observation("hello", now, 0.5);
        assert_eq!(obj.activation.reference_log.len(), 1);
        assert_eq!(obj.dynamics.last_updated_at, now);
        assert_eq!(obj.dynamics.potential, 0.0);
        assert_eq!(obj.status, ObjectStatus::Active);
        assert!(!obj.is_embedding_resolved());
    }

    #[test]
    fn embedding_resolution_gate() {
        let now = EpochMillis(1_000);
        let mut obj = MentalObject::new_observation("hello", now, 0.5);
        assert!(!obj.is_embedding_resolved());
        obj.embedding = Some(vec![0.1, 0.2, 0.3]);
        assert!(obj.is_embedding_resolved());
    }

    #[test]
    fn new_observation_is_confirmed_by_default() {
        let now = EpochMillis(1_000);
        let obj = MentalObject::new_observation("hello", now, 0.5);
        assert_eq!(obj.promotion.status, PromotionStatus::Confirmed);
        assert_eq!(obj.promotion.confirmed_at, Some(now));
        assert_eq!(obj.promotion.confirming_references, 0);
    }

    #[test]
    fn promotion_state_default_is_confirmed_for_pre_migration_rows() {
        assert_eq!(PromotionState::default().status, PromotionStatus::Confirmed);
    }

    #[test]
    fn confirm_promotes_a_candidate_and_counts_the_reference() {
        let mut promotion = PromotionState::candidate(EpochMillis(0));
        assert_eq!(promotion.status, PromotionStatus::Candidate);
        promotion.confirm(EpochMillis(5_000));
        assert_eq!(promotion.status, PromotionStatus::Confirmed);
        assert_eq!(promotion.confirmed_at, Some(EpochMillis(5_000)));
        assert_eq!(promotion.confirming_references, 1);
    }

    #[test]
    fn confirm_is_a_no_op_on_an_already_confirmed_state() {
        let mut promotion = PromotionState::confirmed(EpochMillis(0));
        promotion.confirm(EpochMillis(5_000));
        assert_eq!(promotion.confirmed_at, Some(EpochMillis(0)), "an already-confirmed state must not be re-stamped");
        assert_eq!(promotion.confirming_references, 0, "an already-confirmed state must not double-count");
    }

    #[test]
    fn demote_walks_a_confirmed_state_back_to_candidate() {
        let mut promotion = PromotionState::confirmed(EpochMillis(0));
        promotion.confirming_references = 2;
        promotion.demote(EpochMillis(5_000));
        assert_eq!(promotion.status, PromotionStatus::Candidate);
        assert_eq!(promotion.staged_at, EpochMillis(5_000));
        assert_eq!(promotion.confirmed_at, None);
        assert_eq!(promotion.confirming_references, 2, "confirming_references is historical and must survive a demotion");
    }

    #[test]
    fn demote_is_a_no_op_on_an_already_candidate_state() {
        let mut promotion = PromotionState::candidate(EpochMillis(0));
        promotion.demote(EpochMillis(5_000));
        assert_eq!(promotion.staged_at, EpochMillis(0), "an already-candidate state must not be re-staged");
    }
}
