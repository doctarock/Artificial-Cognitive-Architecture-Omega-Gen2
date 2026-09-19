use std::collections::HashMap;

use aca_graph::{admit_top_n, rank_candidates};
use aca_types::MentalObjectId;

/// A real, checkable causal claim about one tick's Broadcast competition:
/// `entrant` is the specific candidate whose presence in this tick's
/// coalition is the reason `evicted` lost its seat in Working Memory. Every
/// value this module hands out has already been confirmed by
/// `verify_displacement` below - a real counterfactual re-run of the actual
/// admission algorithm with `entrant` removed - so "X entered my awareness
/// because it displaced Y" is never asserted on the strength of ranking
/// order alone; it's the mechanism's own output under an actual
/// intervention, not a narrative label attached after the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Displacement {
    pub entrant: MentalObjectId,
    pub evicted: MentalObjectId,
}

/// Why a previously-admitted object is no longer in this tick's Working
/// Memory. These are structurally different release paths and must never be
/// conflated: `loop_actor`'s Step 5 already has a case (a raw Observation
/// superseded by its own Reflection) where an object simply stops competing,
/// never beaten by anything, so a "displaced by X" claim about it would be
/// false. `Outranked` is the second honest non-claim: the object really did
/// compete and really did lose, but lost to the *combined* weight of the
/// admitted set rather than to any single identifiable entrant (see
/// `explain_release`'s doc comment for why this case exists and can't be
/// papered over).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseReason {
    /// Wasn't among this tick's candidates at all - dropped out of
    /// competition rather than losing it.
    NotRenominated,
    /// Competed this tick, lost, and no single admitted candidate's absence
    /// would have changed the outcome - too far below the capacity cutoff
    /// for a one-cause story to be true.
    Outranked,
    /// Competed this tick and lost specifically because `entrant` was
    /// admitted - confirmed, not inferred (see `Displacement`'s doc
    /// comment).
    Displaced(Displacement),
}

/// Re-runs Step 6's real admission algorithm (`rank_candidates` +
/// `admit_top_n`, the same pair `steps::broadcast::
/// decide_admission_deterministic` calls) with `entrant` removed from the
/// candidate pool, and checks whether `evicted` - which was NOT admitted in
/// the real tick - would now be admitted. `true` is a live, mechanical
/// confirmation that `entrant`'s presence is what kept `evicted` out this
/// tick, not a description of the ranking after the fact: this is the same
/// intervention an experimenter would run by hand - remove the candidate,
/// rerun the mechanism, see whether the outcome changes. This is the
/// function to reach for when the actual mission is "disable the mechanism
/// responsible and see whether the behaviour disappears," not just
/// `explain_release`'s already-computed verdict.
pub fn verify_displacement(raw_candidates: &[(MentalObjectId, f32, Option<f32>)], capacity: usize, entrant: MentalObjectId, evicted: MentalObjectId) -> bool {
    let scores: HashMap<MentalObjectId, f32> = raw_candidates
        .iter()
        .filter(|(id, _, _)| *id != entrant)
        .map(|(id, activation_total, surprise)| (*id, activation_total + surprise.unwrap_or(0.0)))
        .collect();
    if !scores.contains_key(&evicted) {
        return false;
    }
    let ranked = rank_candidates(&scores);
    admit_top_n(&ranked, capacity).contains(&evicted)
}

/// Classifies why `released_id` - already known to have left Working Memory
/// this tick (`steps::broadcast::BroadcastResult::released`) - actually left
/// it, and if it was genuinely displaced, names the real entrant
/// responsible.
///
/// Candidate entrants are `newly_admitted` members that outrank
/// `released_id` on real score, tried weakest-first: the weakest such
/// entrant is the one whose removal alone is most likely to flip the
/// outcome (a stronger entrant's slot would very likely just be refilled by
/// the next-strongest loser, not `released_id`). Every candidate is checked
/// against `verify_displacement` - the real algorithm, re-run - before it's
/// ever reported; nothing here is asserted on ranking order alone.
///
/// `Outranked` (not a guess at a "closest" cause) is returned when
/// `released_id` competed and lost but sits far enough below the capacity
/// cutoff that no single entrant's absence would have saved it - removing
/// any one admitted candidate still leaves capacity binding against the
/// rest. That is a real, different shape of loss, and reporting it as
/// "displaced by X" for some arbitrarily-chosen X would be exactly the kind
/// of unfalsifiable claim this module exists to rule out.
pub fn explain_release(raw_candidates: &[(MentalObjectId, f32, Option<f32>)], newly_admitted: &[MentalObjectId], capacity: usize, released_id: MentalObjectId) -> ReleaseReason {
    let Some(&(_, released_activation, released_surprise)) = raw_candidates.iter().find(|(id, _, _)| *id == released_id) else {
        return ReleaseReason::NotRenominated;
    };
    let released_score = released_activation + released_surprise.unwrap_or(0.0);

    let mut candidate_entrants: Vec<(MentalObjectId, f32)> = newly_admitted
        .iter()
        .filter_map(|id| raw_candidates.iter().find(|(candidate_id, _, _)| candidate_id == id).map(|(_, activation_total, surprise)| (*id, activation_total + surprise.unwrap_or(0.0))))
        .filter(|(_, score)| *score > released_score)
        .collect();
    candidate_entrants.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for (entrant, _) in candidate_entrants {
        if verify_displacement(raw_candidates, capacity, entrant, released_id) {
            return ReleaseReason::Displaced(Displacement { entrant, evicted: released_id });
        }
    }
    ReleaseReason::Outranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_displacement_confirms_a_real_single_swap() {
        let entrant = MentalObjectId::new();
        let evicted = MentalObjectId::new();
        let stays = MentalObjectId::new();
        // capacity 2: {stays=5.0, evicted=3.0} admitted before entrant
        // arrives at 4.0 and outranks evicted.
        let raw = vec![(stays, 5.0, None), (evicted, 3.0, None), (entrant, 4.0, None)];
        assert!(verify_displacement(&raw, 2, entrant, evicted), "removing the entrant should let the evicted candidate back into the top 2");
    }

    #[test]
    fn verify_displacement_is_false_when_evicted_still_would_not_fit() {
        let entrant = MentalObjectId::new();
        let evicted = MentalObjectId::new();
        let stronger_a = MentalObjectId::new();
        let stronger_b = MentalObjectId::new();
        // capacity 2, but two other candidates already outrank `evicted`
        // regardless of `entrant` - removing entrant alone can't save it.
        let raw = vec![(stronger_a, 9.0, None), (stronger_b, 8.0, None), (entrant, 4.0, None), (evicted, 1.0, None)];
        assert!(!verify_displacement(&raw, 2, entrant, evicted));
    }

    #[test]
    fn verify_displacement_is_false_when_evicted_was_never_a_real_candidate() {
        let entrant = MentalObjectId::new();
        let phantom = MentalObjectId::new();
        let raw = vec![(entrant, 4.0, None)];
        assert!(!verify_displacement(&raw, 2, entrant, phantom));
    }

    #[test]
    fn explain_release_reports_a_confirmed_single_swap() {
        let entrant = MentalObjectId::new();
        let evicted = MentalObjectId::new();
        let stays = MentalObjectId::new();
        let raw = vec![(stays, 5.0, None), (evicted, 3.0, None), (entrant, 4.0, None)];
        let reason = explain_release(&raw, &[entrant], 2, evicted);
        assert_eq!(reason, ReleaseReason::Displaced(Displacement { entrant, evicted }));
    }

    #[test]
    fn explain_release_reports_not_renominated_for_an_object_absent_from_this_ticks_candidates() {
        let released = MentalObjectId::new();
        let raw = vec![(MentalObjectId::new(), 1.0, None)];
        let reason = explain_release(&raw, &[], 2, released);
        assert_eq!(reason, ReleaseReason::NotRenominated);
    }

    #[test]
    fn explain_release_reports_outranked_when_no_single_entrant_explains_the_loss() {
        let entrant = MentalObjectId::new();
        let evicted = MentalObjectId::new();
        let stronger_a = MentalObjectId::new();
        let stronger_b = MentalObjectId::new();
        let raw = vec![(stronger_a, 9.0, None), (stronger_b, 8.0, None), (entrant, 4.0, None), (evicted, 1.0, None)];
        let reason = explain_release(&raw, &[entrant], 2, evicted);
        assert_eq!(reason, ReleaseReason::Outranked, "a loss that survives removing any single entrant must not be attributed to that entrant");
    }

    #[test]
    fn explain_release_picks_the_weakest_entrant_that_actually_explains_it() {
        // Two newly-admitted entrants outrank `evicted`; only removing the
        // weaker of the two (closest to the cutoff) actually frees a slot
        // for it at capacity 3, since the stronger one's slot would just be
        // refilled by the other entrant.
        let weak_entrant = MentalObjectId::new();
        let strong_entrant = MentalObjectId::new();
        let evicted = MentalObjectId::new();
        let stays = MentalObjectId::new();
        let raw = vec![(stays, 9.0, None), (strong_entrant, 7.0, None), (weak_entrant, 3.0, None), (evicted, 2.0, None)];
        let reason = explain_release(&raw, &[weak_entrant, strong_entrant], 3, evicted);
        assert_eq!(reason, ReleaseReason::Displaced(Displacement { entrant: weak_entrant, evicted }));
    }
}
