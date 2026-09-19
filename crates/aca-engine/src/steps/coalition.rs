use aca_types::{EdgeKind, MentalObjectId};
use aca_util::EpochMillis;

/// A candidate's combined ranking score: recency/frequency-driven ACT-R
/// activation, plus (for freshly-compared observations) this cycle's
/// precision-weighted surprise. This is the single scale that lets
/// surprising new observations, high-activation recalled memories, due
/// goals, and pending subgoals all compete fairly in one ranking — the
/// alternative would be several separate ad hoc priority rules, which
/// specs.md explicitly rejects ("Importance, recency... are not five
/// separate ad hoc factors").
pub fn attention_score(activation_total: f32, precision_weighted_surprise: Option<f32>) -> f32 {
    activation_total + precision_weighted_surprise.unwrap_or(0.0)
}

/// One candidate entering the Step 5 coalition, already scored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoalitionCandidate {
    pub id: MentalObjectId,
    pub score: f32,
}

/// Step 5 - Form coalitions. Filters candidates by the attention threshold
/// (reusing `aca_graph::clears_retrieval_threshold` — the same "is this
/// above the line" gate, applied here to the combined attention score
/// rather than raw ACT-R activation) and scores the survivors. Pure
/// computation: ranking itself (Step 6) is where GWT's serial arbitration
/// happens, not here — this step may admit many candidates; Broadcast
/// narrows them to the capacity-limited winners.
pub fn form_coalition(
    raw_candidates: &[(MentalObjectId, f32, Option<f32>)],
    attention_threshold: f32,
) -> Vec<CoalitionCandidate> {
    raw_candidates
        .iter()
        .map(|(id, activation_total, surprise)| CoalitionCandidate {
            id: *id,
            score: attention_score(*activation_total, *surprise),
        })
        .filter(|candidate| aca_graph::clears_retrieval_threshold(candidate.score, attention_threshold))
        .collect()
}

/// Divisive normalization / crowding (Reynolds & Heeger's normalization
/// model of attention, made literal - see `docs/cognitive-capability-audit.md`'s
/// "Phase 5, revisited" section for the design discussion this closes).
/// Distinct from both `form_coalition` (which only asks "is this a real
/// candidate at all") and Broadcast's hysteresis (which is about *time* -
/// how long something stays eligible, not how many other things are
/// competing right now): this is the one place how *crowded* this tick's
/// competing field is becomes a real, measurable factor in what can win,
/// independent of any single candidate's own ACT-R activation.
///
/// Each candidate's effective score is divided by `1 + crowding_strength *
/// (sum of every OTHER candidate's positive score this tick)` - a quiet
/// tick (this candidate alone, or alongside only weak/negative-scoring
/// competitors) leaves the score essentially untouched; a busy tick
/// (several other strong candidates active at once) suppresses it, even
/// though nothing about the candidate's own activation changed. Only
/// *positive* scores contribute to the suppressive pool - a weak or
/// negative-scoring bystander should never make a genuinely strong
/// candidate easier to admit, and a candidate never contributes to its own
/// denominator. `crowding_strength = 0.0` is an exact identity - the real,
/// checkable pre-normalization baseline, not a decorative default.
pub fn apply_crowding_normalization(candidates: &[CoalitionCandidate], crowding_strength: f32) -> Vec<CoalitionCandidate> {
    // A negative strength has no defined meaning here (this models a
    // suppressive divisor, never an amplifying one) and would otherwise let
    // `denominator` reach exactly 0 or go negative on a crowded tick,
    // sending a score to +/-infinity or silently flipping its sign.
    let crowding_strength = crowding_strength.max(0.0);
    if crowding_strength == 0.0 {
        return candidates.to_vec();
    }
    let total_positive: f32 = candidates.iter().map(|c| c.score.max(0.0)).sum();
    candidates
        .iter()
        .map(|candidate| {
            let others_positive = total_positive - candidate.score.max(0.0);
            let denominator = 1.0 + crowding_strength * others_positive;
            CoalitionCandidate { id: candidate.id, score: candidate.score / denominator }
        })
        .collect()
}

/// A freshly ignited winner suppresses the strongest losing competitors via
/// sparse, learned signed edges. Existing winners are not reinforced merely
/// because a maintenance deadline caused another arbitration pass.
pub fn reinforce_lateral_inhibition(
    graph: &mut aca_graph::Graph,
    fresh_winners: &[MentalObjectId],
    coalition: &[CoalitionCandidate],
    admitted: &std::collections::HashSet<MentalObjectId>,
    now: EpochMillis,
    increment: f32,
    max_losers_per_winner: usize,
) -> Vec<MentalObjectId> {
    if increment <= 0.0 || max_losers_per_winner == 0 {
        return Vec::new();
    }
    let mut losers: Vec<_> = coalition.iter().filter(|candidate| !admitted.contains(&candidate.id)).copied().collect();
    losers.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    losers.truncate(max_losers_per_winner);
    let mut changed = Vec::new();
    for winner in fresh_winners {
        if let Some(source) = graph.get_mut(winner) {
            for loser in &losers {
                aca_graph::reinforce_edge(&mut source.edges, loser.id, EdgeKind::Inhibitory, now, increment, 1.0);
            }
            if !losers.is_empty() {
                changed.push(*winner);
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::MentalObjectId;

    #[test]
    fn only_fresh_winners_gain_bounded_inhibitory_edges_to_losers() {
        let mut graph = aca_graph::Graph::new();
        let winner = aca_types::MentalObject::new_observation("winner", EpochMillis(0), 0.5);
        let loser_a = aca_types::MentalObject::new_observation("a", EpochMillis(0), 0.5);
        let loser_b = aca_types::MentalObject::new_observation("b", EpochMillis(0), 0.5);
        let ids = (winner.id, loser_a.id, loser_b.id);
        for object in [winner, loser_a, loser_b] { graph.insert(object); }
        let coalition = [CoalitionCandidate { id: ids.0, score: 3.0 }, CoalitionCandidate { id: ids.1, score: 2.0 }, CoalitionCandidate { id: ids.2, score: 1.0 }];
        let admitted = std::collections::HashSet::from([ids.0]);
        let changed = reinforce_lateral_inhibition(&mut graph, &[ids.0], &coalition, &admitted, EpochMillis(0), 0.05, 1);
        assert_eq!(changed, vec![ids.0]);
        let edges = &graph.get(&ids.0).unwrap().edges;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target_id, ids.1);
        assert_eq!(edges[0].kind, EdgeKind::Inhibitory);
        assert!(reinforce_lateral_inhibition(&mut graph, &[], &coalition, &admitted, EpochMillis(1), 0.05, 1).is_empty());
        assert_eq!(graph.get(&ids.0).unwrap().edges[0].strength, 0.05);
    }

    #[test]
    fn attention_score_adds_surprise_to_activation() {
        assert!((attention_score(1.0, Some(0.5)) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn attention_score_with_no_surprise_is_just_activation() {
        assert!((attention_score(1.0, None) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn form_coalition_filters_below_threshold_candidates() {
        let below = MentalObjectId::new();
        let above = MentalObjectId::new();
        let raw = vec![
            (below, 0.1, None),
            (above, 5.0, Some(0.5)),
        ];
        let coalition = form_coalition(&raw, 1.0);
        assert_eq!(coalition.len(), 1);
        assert_eq!(coalition[0].id, above);
    }

    #[test]
    fn form_coalition_includes_a_surprising_candidate_with_low_activation() {
        // A brand-new, low-activation object that was highly surprising
        // should still be able to clear the threshold via its surprise term.
        let id = MentalObjectId::new();
        let raw = vec![(id, 0.2, Some(2.0))];
        let coalition = form_coalition(&raw, 1.0);
        assert_eq!(coalition.len(), 1);
    }

    #[test]
    fn crowding_leaves_a_lone_candidate_untouched() {
        let id = MentalObjectId::new();
        let candidates = vec![CoalitionCandidate { id, score: 5.0 }];
        let normalized = apply_crowding_normalization(&candidates, 0.1);
        assert!((normalized[0].score - 5.0).abs() < 1e-6, "with nothing else competing, crowding should have no effect at all");
    }

    #[test]
    fn zero_crowding_strength_is_an_exact_identity() {
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        let candidates = vec![CoalitionCandidate { id: a, score: 5.0 }, CoalitionCandidate { id: b, score: 4.0 }];
        let normalized = apply_crowding_normalization(&candidates, 0.0);
        assert_eq!(normalized, candidates);
    }

    #[test]
    fn two_strong_candidates_suppress_each_other() {
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        let candidates = vec![CoalitionCandidate { id: a, score: 5.0 }, CoalitionCandidate { id: b, score: 4.0 }];
        let normalized = apply_crowding_normalization(&candidates, 0.1);
        // a's denominator is 1 + 0.1*4 = 1.4; b's is 1 + 0.1*5 = 1.5.
        assert!((normalized[0].score - 5.0 / 1.4).abs() < 1e-4);
        assert!((normalized[1].score - 4.0 / 1.5).abs() < 1e-4);
        assert!(normalized[0].score < 5.0 && normalized[1].score < 4.0, "both candidates should be suppressed relative to their raw scores");
    }

    #[test]
    fn a_negative_scoring_bystander_does_not_suppress_a_strong_candidate() {
        let strong = MentalObjectId::new();
        let weak = MentalObjectId::new();
        let candidates = vec![CoalitionCandidate { id: strong, score: 5.0 }, CoalitionCandidate { id: weak, score: -3.0 }];
        let normalized = apply_crowding_normalization(&candidates, 0.5);
        let strong_after = normalized.iter().find(|c| c.id == strong).unwrap().score;
        assert!((strong_after - 5.0).abs() < 1e-6, "a below-baseline bystander must never contribute to the suppressive pool, got {strong_after}");
    }

    #[test]
    fn higher_crowding_strength_suppresses_more() {
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        let low = apply_crowding_normalization(&[CoalitionCandidate { id: a, score: 5.0 }, CoalitionCandidate { id: b, score: 5.0 }], 0.05);
        let high = apply_crowding_normalization(&[CoalitionCandidate { id: a, score: 5.0 }, CoalitionCandidate { id: b, score: 5.0 }], 0.5);
        let low_a = low.iter().find(|c| c.id == a).unwrap().score;
        let high_a = high.iter().find(|c| c.id == a).unwrap().score;
        assert!(high_a < low_a, "a stronger crowding_strength should suppress the identical field more, not less");
    }
}
