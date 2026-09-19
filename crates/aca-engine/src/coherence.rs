//! Deterministic conflict-resolution rules for the four single-owner
//! surfaces named in `scale-strategy.md`'s "Keep Single-Owner" list, as
//! specified in `docs/coherence-arbitration-policy.md`. Nothing in this
//! module is wired into `CognitiveLoopActor::tick` or any other live path —
//! `CognitiveLoopActor` remains the sole mutator of `Graph` and Working
//! Memory. These functions exist so the policy can be sanity-checked
//! (see the tests below) before any concurrency is introduced.
//!
//! This is a permanent hard path, not a placeholder for a future trained
//! model — see the policy doc's "Why no model" section. Every rule here is
//! closed-form arithmetic over fields the engine already computes, with no
//! outcome signal to ever train against.

use std::cmp::Ordering;

use aca_types::MentalObjectId;
use aca_util::EpochMillis;

use crate::steps::executive::{ExecutiveConfig, ExecutiveDecision, OperatorProposal};

// ---------------------------------------------------------------------
// 1. Working Memory admission (Broadcast, Step 6)
// ---------------------------------------------------------------------

/// One thread's proposed admission for a given tick. `score` is the same
/// `activation_total + surprise.unwrap_or(0.0)` value
/// `decide_admission_from_attention_model` already computes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmissionProposal {
    pub candidate_id: MentalObjectId,
    pub score: f32,
    pub created_at: EpochMillis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionOutcome {
    AdmitA,
    AdmitB,
}

/// Resolves two competing admission proposals for the same tick. Higher
/// `score` wins. Tie-break: older `created_at` wins (more corroborated
/// evidence). Second tie-break: lower `MentalObjectId` — a stable,
/// arbitrary total order (UUIDv7 is time-orderable, so this is itself a
/// weak recency signal, never a coin flip). Total: every input pair
/// resolves to exactly one outcome, never a panic or an unresolved tie.
pub fn resolve_admission_conflict(a: AdmissionProposal, b: AdmissionProposal) -> AdmissionOutcome {
    match a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal) {
        Ordering::Greater => AdmissionOutcome::AdmitA,
        Ordering::Less => AdmissionOutcome::AdmitB,
        Ordering::Equal => match a.created_at.cmp(&b.created_at) {
            Ordering::Less => AdmissionOutcome::AdmitA,
            Ordering::Greater => AdmissionOutcome::AdmitB,
            Ordering::Equal => {
                if a.candidate_id <= b.candidate_id {
                    AdmissionOutcome::AdmitA
                } else {
                    AdmissionOutcome::AdmitB
                }
            }
        },
    }
}

// ---------------------------------------------------------------------
// 2. Activation graph mutation
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationField {
    Activation,
    Status,
    Edges,
    Workspace,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MutationProposal {
    pub object_id: MentalObjectId,
    pub field: MutationField,
    pub confidence: f32,
    pub tick: EpochMillis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationOutcome {
    ApplyA,
    ApplyB,
    ApplyBoth,
}

/// Resolves two proposed writes. Different `object_id`, or same object but
/// different `field`: both apply (fields are independent columns).
/// Same object and field: higher `confidence` wins. Equal confidence:
/// `Status` prefers the *earlier* tick (a `Discarded` transition is
/// one-way per `Graph::discard`'s contract — a slower concurrent writer
/// that hasn't seen the discard yet must not undo it); every other field
/// prefers the *later* tick (freshest evidence wins for decaying/refreshing
/// state).
pub fn resolve_mutation_conflict(a: MutationProposal, b: MutationProposal) -> MutationOutcome {
    if a.object_id != b.object_id || a.field != b.field {
        return MutationOutcome::ApplyBoth;
    }
    match a.confidence.partial_cmp(&b.confidence).unwrap_or(Ordering::Equal) {
        Ordering::Greater => MutationOutcome::ApplyA,
        Ordering::Less => MutationOutcome::ApplyB,
        Ordering::Equal => {
            let a_wins = match a.field {
                MutationField::Status => a.tick <= b.tick,
                MutationField::Activation | MutationField::Edges | MutationField::Workspace => a.tick >= b.tick,
            };
            if a_wins {
                MutationOutcome::ApplyA
            } else {
                MutationOutcome::ApplyB
            }
        }
    }
}

// ---------------------------------------------------------------------
// 3. Executive operator selection
// ---------------------------------------------------------------------

/// Resolves cross-thread competition for the same tick's operator
/// selection by concatenating every thread's proposals and running the
/// engine's existing single-owner selection logic
/// (`steps::executive::select_operator`) unchanged. Verified in this
/// module's tests to be identical to calling `select_operator` directly on
/// the merged list: preference ordering plus the confidence-threshold
/// impasse gate is already order-independent across proposal sources, so
/// this surface needs no new merge rule — a low-confidence or genuinely
/// tied cross-thread result degrades to the same `ImpasseKind::Confidence`
/// escalation path a single thread would hit today.
pub fn resolve_executive_conflict(thread_a: &[OperatorProposal], thread_b: &[OperatorProposal], config: &ExecutiveConfig) -> ExecutiveDecision {
    let mut combined = Vec::with_capacity(thread_a.len() + thread_b.len());
    combined.extend_from_slice(thread_a);
    combined.extend_from_slice(thread_b);
    crate::steps::executive::select_operator(&combined, config)
}

// ---------------------------------------------------------------------
// 4. Tier 3/Tier 4 escalation seats
// ---------------------------------------------------------------------

/// A `Confidence`-impasse escalation request contending for the single
/// Tier 3/4 permit. `MissingInformation` impasses never reach this
/// function — they take the subgoal-respawn path instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EscalationRequest {
    pub request_id: u64,
    pub tick: EpochMillis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationOutcome {
    Grant,
    DeferToHeuristic,
}

/// Orders two escalation requests arriving close enough together to both
/// plausibly `try_acquire` the one permit before the tick's existing
/// `DeferredToHeuristic` fallback (`resolve_confidence_impasse`) would
/// trigger on its own. Earlier `tick` wins (FIFO); ties broken by lower
/// `request_id`. Exactly one of the two is granted — the other gets
/// precisely what it would already get today (the heuristic fallback),
/// never a wait queue, since the single-flight permit itself never grows.
pub fn resolve_escalation_conflict(a: EscalationRequest, b: EscalationRequest) -> (EscalationOutcome, EscalationOutcome) {
    let a_first = match a.tick.cmp(&b.tick) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.request_id <= b.request_id,
    };
    if a_first {
        (EscalationOutcome::Grant, EscalationOutcome::DeferToHeuristic)
    } else {
        (EscalationOutcome::DeferToHeuristic, EscalationOutcome::Grant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steps::executive::Operator;

    fn admission(score: f32, created_at: i64) -> AdmissionProposal {
        AdmissionProposal { candidate_id: MentalObjectId::new(), score, created_at: EpochMillis(created_at) }
    }

    #[test]
    fn admission_higher_score_wins() {
        let a = admission(0.9, 100);
        let b = admission(0.4, 100);
        assert_eq!(resolve_admission_conflict(a, b), AdmissionOutcome::AdmitA);
        assert_eq!(resolve_admission_conflict(b, a), AdmissionOutcome::AdmitB);
    }

    #[test]
    fn admission_tie_breaks_on_older_created_at() {
        let older = admission(0.5, 50);
        let newer = admission(0.5, 100);
        assert_eq!(resolve_admission_conflict(older, newer), AdmissionOutcome::AdmitA);
        assert_eq!(resolve_admission_conflict(newer, older), AdmissionOutcome::AdmitB);
    }

    #[test]
    fn admission_full_tie_breaks_on_id_and_is_symmetric() {
        let a = admission(0.5, 100);
        let mut b = a;
        b.candidate_id = MentalObjectId::new();
        let (lower, higher) = if a.candidate_id <= b.candidate_id { (a, b) } else { (b, a) };
        assert_eq!(resolve_admission_conflict(lower, higher), AdmissionOutcome::AdmitA);
        assert_eq!(resolve_admission_conflict(higher, lower), AdmissionOutcome::AdmitB);
    }

    fn mutation(object_id: MentalObjectId, field: MutationField, confidence: f32, tick: i64) -> MutationProposal {
        MutationProposal { object_id, field, confidence, tick: EpochMillis(tick) }
    }

    #[test]
    fn mutation_different_objects_apply_both() {
        let a = mutation(MentalObjectId::new(), MutationField::Activation, 0.9, 1);
        let b = mutation(MentalObjectId::new(), MutationField::Activation, 0.1, 1);
        assert_eq!(resolve_mutation_conflict(a, b), MutationOutcome::ApplyBoth);
    }

    #[test]
    fn mutation_same_object_different_field_apply_both() {
        let id = MentalObjectId::new();
        let a = mutation(id, MutationField::Activation, 0.9, 1);
        let b = mutation(id, MutationField::Status, 0.1, 1);
        assert_eq!(resolve_mutation_conflict(a, b), MutationOutcome::ApplyBoth);
    }

    #[test]
    fn mutation_same_field_higher_confidence_wins() {
        let id = MentalObjectId::new();
        let a = mutation(id, MutationField::Activation, 0.9, 1);
        let b = mutation(id, MutationField::Activation, 0.2, 1);
        assert_eq!(resolve_mutation_conflict(a, b), MutationOutcome::ApplyA);
        assert_eq!(resolve_mutation_conflict(b, a), MutationOutcome::ApplyB);
    }

    #[test]
    fn mutation_status_tie_prefers_earlier_tick_never_undoing_a_discard() {
        let id = MentalObjectId::new();
        let earlier_discard = mutation(id, MutationField::Status, 0.5, 10);
        let later_write = mutation(id, MutationField::Status, 0.5, 20);
        assert_eq!(resolve_mutation_conflict(earlier_discard, later_write), MutationOutcome::ApplyA);
        assert_eq!(resolve_mutation_conflict(later_write, earlier_discard), MutationOutcome::ApplyB);
    }

    #[test]
    fn mutation_activation_tie_prefers_later_tick() {
        let id = MentalObjectId::new();
        let earlier = mutation(id, MutationField::Activation, 0.5, 10);
        let later = mutation(id, MutationField::Activation, 0.5, 20);
        assert_eq!(resolve_mutation_conflict(earlier, later), MutationOutcome::ApplyB);
        assert_eq!(resolve_mutation_conflict(later, earlier), MutationOutcome::ApplyA);
    }

    fn proposal(op: Operator, preference: f32, confidence: f32) -> OperatorProposal {
        OperatorProposal { operator: op, target_id: MentalObjectId::new(), preference, confidence }
    }

    #[test]
    fn executive_conflict_picks_highest_preference_across_threads() {
        let thread_a = vec![proposal(Operator::Speak, 0.3, 0.9)];
        let thread_b = vec![proposal(Operator::Ask, 0.9, 0.9)];
        let config = ExecutiveConfig::default();
        let decision = resolve_executive_conflict(&thread_a, &thread_b, &config);
        match decision {
            ExecutiveDecision::Selected(winner) => assert_eq!(winner.operator, Operator::Ask),
            other => panic!("expected a decisive winner, got {other:?}"),
        }
    }

    #[test]
    fn executive_conflict_matches_calling_select_operator_on_the_merged_list() {
        let thread_a = vec![proposal(Operator::Speak, 0.5, 0.9), proposal(Operator::Remember, 0.2, 0.9)];
        let thread_b = vec![proposal(Operator::Ask, 0.5, 0.9)];
        let config = ExecutiveConfig::default();

        let via_conflict_fn = resolve_executive_conflict(&thread_a, &thread_b, &config);

        let mut merged = thread_a.clone();
        merged.extend(thread_b.clone());
        let via_direct_call = crate::steps::executive::select_operator(&merged, &config);

        assert_eq!(via_conflict_fn, via_direct_call);
    }

    #[test]
    fn executive_conflict_still_escalates_on_low_confidence() {
        let thread_a = vec![proposal(Operator::Speak, 0.9, 0.1)];
        let thread_b: Vec<OperatorProposal> = vec![];
        let config = ExecutiveConfig::default();
        let decision = resolve_executive_conflict(&thread_a, &thread_b, &config);
        assert!(matches!(decision, ExecutiveDecision::Impasse { .. }));
    }

    fn escalation(request_id: u64, tick: i64) -> EscalationRequest {
        EscalationRequest { request_id, tick: EpochMillis(tick) }
    }

    #[test]
    fn escalation_earlier_tick_wins() {
        let earlier = escalation(1, 10);
        let later = escalation(2, 20);
        assert_eq!(resolve_escalation_conflict(earlier, later), (EscalationOutcome::Grant, EscalationOutcome::DeferToHeuristic));
        assert_eq!(resolve_escalation_conflict(later, earlier), (EscalationOutcome::DeferToHeuristic, EscalationOutcome::Grant));
    }

    #[test]
    fn escalation_tie_breaks_on_lower_request_id() {
        let a = escalation(1, 10);
        let b = escalation(2, 10);
        assert_eq!(resolve_escalation_conflict(a, b), (EscalationOutcome::Grant, EscalationOutcome::DeferToHeuristic));
        assert_eq!(resolve_escalation_conflict(b, a), (EscalationOutcome::DeferToHeuristic, EscalationOutcome::Grant));
    }

    #[test]
    fn escalation_exactly_one_granted_never_both_never_neither() {
        let a = escalation(1, 10);
        let b = escalation(2, 10);
        let (outcome_a, outcome_b) = resolve_escalation_conflict(a, b);
        let grants = [outcome_a, outcome_b].iter().filter(|o| **o == EscalationOutcome::Grant).count();
        assert_eq!(grants, 1);
    }
}
