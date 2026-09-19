use std::collections::HashMap;

use aca_types::MentalObjectId;

/// Ranks candidates by score descending, breaking ties deterministically by
/// id (ascending) so ranking/admission is reproducible in tests and across
/// runs given identical input scores. This is GWT's competition step: pure
/// ranking math, no model call, the one serial arbitration point content
/// must pass through before it can be acted on.
pub fn rank_candidates(scores: &HashMap<MentalObjectId, f32>) -> Vec<MentalObjectId> {
    let mut ranked: Vec<MentalObjectId> = scores.keys().copied().collect();
    ranked.sort_by(|a, b| {
        let score_a = scores[a];
        let score_b = scores[b];
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    ranked
}

/// Admits the top `capacity` ranked candidates into Working Memory — GWT's
/// capacity-limited broadcast. Returns fewer than `capacity` if there
/// weren't enough candidates; never more.
pub fn admit_top_n(ranked: &[MentalObjectId], capacity: usize) -> Vec<MentalObjectId> {
    ranked.iter().take(capacity).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_by_score_descending() {
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        let c = MentalObjectId::new();
        let mut scores = HashMap::new();
        scores.insert(a, 0.2);
        scores.insert(b, 0.9);
        scores.insert(c, 0.5);

        let ranked = rank_candidates(&scores);
        assert_eq!(ranked, vec![b, c, a]);
    }

    #[test]
    fn ties_break_deterministically_by_id() {
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        let mut scores = HashMap::new();
        scores.insert(a, 0.5);
        scores.insert(b, 0.5);

        let ranked_once = rank_candidates(&scores);
        let ranked_again = rank_candidates(&scores);
        assert_eq!(ranked_once, ranked_again, "tie-break must be stable across calls");

        let expected_order = if a < b { vec![a, b] } else { vec![b, a] };
        assert_eq!(ranked_once, expected_order);
    }

    #[test]
    fn admit_top_n_never_exceeds_capacity() {
        let ids: Vec<MentalObjectId> = (0..5).map(|_| MentalObjectId::new()).collect();
        let admitted = admit_top_n(&ids, 3);
        assert_eq!(admitted.len(), 3);
        assert_eq!(admitted, ids[..3]);
    }

    #[test]
    fn admit_top_n_handles_fewer_candidates_than_capacity() {
        let ids: Vec<MentalObjectId> = (0..2).map(|_| MentalObjectId::new()).collect();
        let admitted = admit_top_n(&ids, 5);
        assert_eq!(admitted.len(), 2);
    }
}
