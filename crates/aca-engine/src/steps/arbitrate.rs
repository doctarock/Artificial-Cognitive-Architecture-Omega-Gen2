use aca_tiers::TierResponse;
use aca_util::cosine_similarity;

/// The result of arbitrating between multiple tier candidates: which one to
/// use, and how much independent corroboration it actually has - a Tier-0
/// (pure, no model call) replacement for trusting a candidate's own
/// self-reported confidence. `agreement` is real evidence (other
/// independently-generated answers landed near the same place in embedding
/// space), not a claim any one candidate made about itself.
#[derive(Debug, Clone)]
pub struct ArbitrationOutcome {
    pub chosen: TierResponse,
    /// Mean cosine similarity of `chosen` to every other candidate, clamped
    /// to `[0.0, 1.0]`. `0.0` when there was only one usable candidate -
    /// explicitly "unverifiable," never silently treated as trustworthy.
    pub agreement: f32,
    /// How many candidates actually had a usable embedding and took part in
    /// arbitration - distinct from how many were originally sampled, since
    /// some may have failed to embed.
    pub sample_count: usize,
}

/// Step: Arbitrate. Picks the candidate with the strongest mutual agreement
/// with the rest of the field, and reports that agreement as the real
/// confidence signal - never a candidate's own self-reported number, which
/// live testing showed carries no reliable signal at small model sizes (see
/// `cognitive_core::reflect`'s doc comment). `embeddings[i]` must correspond
/// to `candidates[i]`; a mismatched length is treated as no usable
/// candidates.
///
/// `None` when there's nothing to arbitrate over (no candidates, or lengths
/// don't line up). A single candidate is still returned - as
/// `sample_count: 1`, `agreement: 0.0` - so callers can see "only one
/// answer, no corroboration" and decide for themselves whether that's
/// trustworthy (it generally isn't, for a cheap tier - see
/// `cognitive_core::TIER1_AGREEMENT_THRESHOLD`/`TIER2_AGREEMENT_THRESHOLD`).
pub fn arbitrate_by_agreement(candidates: Vec<TierResponse>, embeddings: &[Vec<f32>]) -> Option<ArbitrationOutcome> {
    if candidates.len() != embeddings.len() || candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        let mut candidates = candidates;
        return Some(ArbitrationOutcome { chosen: candidates.remove(0), agreement: 0.0, sample_count: 1 });
    }

    let sample_count = candidates.len();
    let mut best_index = 0;
    let mut best_agreement = f32::MIN;
    for i in 0..sample_count {
        let mut total = 0.0f32;
        for j in 0..sample_count {
            if i == j {
                continue;
            }
            total += cosine_similarity(&embeddings[i], &embeddings[j]);
        }
        let mean_similarity = total / (sample_count - 1) as f32;
        if mean_similarity > best_agreement {
            best_agreement = mean_similarity;
            best_index = i;
        }
    }

    let mut candidates = candidates;
    let chosen = candidates.remove(best_index);
    Some(ArbitrationOutcome {
        chosen,
        agreement: best_agreement.clamp(0.0, 1.0),
        sample_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_tiers::testing::FakeEmbeddingClient;
    use aca_tiers::EmbeddingClient;
    use aca_types::Tier;

    fn response(text: &str) -> TierResponse {
        TierResponse { raw_text: text.to_string(), confidence: 1.0, tier: Tier::T1 }
    }

    async fn embed_all(texts: &[&str]) -> Vec<Vec<f32>> {
        let client = FakeEmbeddingClient::default();
        let mut out = Vec::new();
        for t in texts {
            out.push(client.embed(t).await.unwrap());
        }
        out
    }

    #[test]
    fn no_candidates_yields_none() {
        assert!(arbitrate_by_agreement(vec![], &[]).is_none());
    }

    #[test]
    fn mismatched_lengths_yields_none() {
        let candidates = vec![response("a")];
        assert!(arbitrate_by_agreement(candidates, &[]).is_none());
    }

    #[tokio::test]
    async fn a_single_candidate_is_returned_unverified() {
        let candidates = vec![response("solo answer")];
        let embeddings = embed_all(&["solo answer"]).await;
        let outcome = arbitrate_by_agreement(candidates, &embeddings).unwrap();
        assert_eq!(outcome.sample_count, 1);
        assert_eq!(outcome.agreement, 0.0);
        assert_eq!(outcome.chosen.raw_text, "solo answer");
    }

    #[tokio::test]
    async fn identical_answers_yield_maximal_agreement() {
        let texts = ["hi, I can hear you", "hi, I can hear you", "hi, I can hear you"];
        let candidates = texts.iter().map(|t| response(t)).collect();
        let embeddings = embed_all(&texts).await;
        let outcome = arbitrate_by_agreement(candidates, &embeddings).unwrap();
        assert_eq!(outcome.sample_count, 3);
        assert!((outcome.agreement - 1.0).abs() < 1e-5);
        assert_eq!(outcome.chosen.raw_text, "hi, I can hear you");
    }

    #[tokio::test]
    async fn divergent_answers_yield_low_agreement() {
        // Reproduces the observed failure shape: independent samples that
        // don't actually agree with each other at all.
        let texts = ["I am a Super Mario character", "the weather is nice today", "purple elephants dance slowly"];
        let candidates = texts.iter().map(|t| response(t)).collect();
        let embeddings = embed_all(&texts).await;
        let outcome = arbitrate_by_agreement(candidates, &embeddings).unwrap();
        assert_eq!(outcome.sample_count, 3);
        assert!(outcome.agreement < 0.5, "uncorrelated fake embeddings should not look like agreement: {}", outcome.agreement);
    }

    #[tokio::test]
    async fn majority_agreement_wins_over_a_lone_outlier() {
        let texts = ["the sky is blue", "the sky is blue", "bananas are purple today"];
        let candidates = texts.iter().map(|t| response(t)).collect();
        let embeddings = embed_all(&texts).await;
        let outcome = arbitrate_by_agreement(candidates, &embeddings).unwrap();
        assert_eq!(outcome.chosen.raw_text, "the sky is blue");
    }
}
