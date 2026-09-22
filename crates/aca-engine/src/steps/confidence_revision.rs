use aca_types::{MentalObject, PromotionStatus};
use aca_util::EpochMillis;

/// Tuning knobs for confidence revision - thresholds, not theoretical
/// commitments (see `LoopConfig`'s own doc comment on this framing).
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceRevisionConfig {
    /// How much `apply_contradiction_penalty` subtracts from `confidence`
    /// per newly-observed contradiction.
    pub contradiction_penalty: f32,
    /// `confidence` never drops below this - a contradicted memory is
    /// evidence to weigh less heavily, not a claim it's certainly false; a
    /// floor above `0.0` keeps it a real, if weak, competitor rather than
    /// pinning it to "definitely wrong."
    pub min_confidence_floor: f32,
    /// How much `apply_corroboration_bonus` adds to `confidence` per
    /// newly-observed piece of consistent evidence - the mirror of
    /// `contradiction_penalty`, same magnitude, so one act of evidence
    /// moves confidence by the same amount in either direction.
    pub corroboration_bonus: f32,
    /// Below this, a `Confirmed` object is demoted back to `Candidate` by
    /// `apply_contradiction_penalty_and_maybe_demote` - the promotion
    /// "cache" invalidating once the live confidence it was trusted on has
    /// eroded too far. Set just above `min_confidence_floor` so demotion
    /// reflects *sustained* erosion (several independent contradictions),
    /// not one hit alone - the same reasoning as `contradiction_penalty`'s
    /// own doc comment.
    pub demotion_confidence_threshold: f32,
    /// Whether `apply_contradiction_penalty`/`apply_corroboration_bonus`
    /// are ever called at all - `false` reproduces the old "confidence is
    /// set once at creation, never revised" behavior exactly, for A/B
    /// comparison.
    pub enabled: bool,
}

impl Default for ConfidenceRevisionConfig {
    fn default() -> Self {
        Self {
            // A single contradiction is real evidence, not proof - several
            // independent contradictions should compound toward the floor
            // rather than one alone driving confidence there outright.
            contradiction_penalty: 0.2,
            min_confidence_floor: 0.05,
            corroboration_bonus: 0.2,
            demotion_confidence_threshold: 0.15,
            enabled: true,
        }
    }
}

/// Lowers `object.confidence` by `penalty`, floored at `floor` - the
/// consumer side of `EdgeKind::Contradicts`: this architecture's `confidence`
/// field was set once at creation/calibration and never revised by later
/// evidence, distinct from ACT-R `activation` (already decays with disuse,
/// but that's a *relevance* signal, not a *correctness* one - disuse-based
/// fading alone was found too slow to catch fast pollution).
///
/// Callers own the "only on a genuinely new edge" discipline (a lookup
/// against `source.edges` before calling `reinforce_edge`, not something
/// this function can enforce on its own) - applying this unconditionally on
/// every tick a standing `Contradicts` edge merely gets re-coactivated would
/// crater confidence without bound instead of penalizing the one real event
/// (the contradiction being *observed*) exactly once.
pub fn apply_contradiction_penalty(object: &mut MentalObject, config: &ConfidenceRevisionConfig) {
    if !config.enabled {
        return;
    }
    object.confidence = (object.confidence - config.contradiction_penalty).max(config.min_confidence_floor);
}

/// The mirror of `apply_contradiction_penalty`: raises `object.confidence`
/// by `corroboration_bonus`, ceilinged at `1.0` - the consumer side of a
/// `detect_contradiction` vote that comes back "consistent" rather than
/// "contradicts", or a verified positive outcome (see
/// `apply_corroboration_bonus_and_maybe_confirm`'s own doc comment for why
/// only the latter also re-confirms). Same "callers own the only-on-a-
/// genuinely-new-edge discipline" contract as the penalty side.
pub fn apply_corroboration_bonus(object: &mut MentalObject, config: &ConfidenceRevisionConfig) {
    if !config.enabled {
        return;
    }
    object.confidence = (object.confidence + config.corroboration_bonus).min(1.0);
}

/// `apply_contradiction_penalty`, plus the missing invalidation edge:
/// once confidence has eroded to `demotion_confidence_threshold` or below,
/// an already-`Confirmed` object no longer deserves the trust that status
/// grants - demote it back to `Candidate` so every filter that reads
/// `promotion.status` (not `confidence`) stops treating it as settled.
/// Kept separate from `apply_contradiction_penalty` itself (rather than
/// folded in) so the pure confidence math stays independently testable and
/// every existing call site/test of that function is untouched.
pub fn apply_contradiction_penalty_and_maybe_demote(object: &mut MentalObject, config: &ConfidenceRevisionConfig, at: EpochMillis) {
    apply_contradiction_penalty(object, config);
    if !config.enabled {
        return;
    }
    if object.confidence <= config.demotion_confidence_threshold && object.promotion.status == PromotionStatus::Confirmed {
        object.promotion.demote(at);
    }
}

/// `apply_corroboration_bonus`, plus re-confirmation of a still-`Candidate`
/// object - reserved for evidence at least as strong as the existing
/// role-matched confirmation scan's own bar (a verified real-world
/// outcome, e.g. the goal-outcome learner's `Supports` case), which is why
/// `steps::memory_formation`'s "consistent" vote - genuinely weaker
/// evidence, "related but not the same claim" - calls the bare
/// `apply_corroboration_bonus` instead of this one.
pub fn apply_corroboration_bonus_and_maybe_confirm(object: &mut MentalObject, config: &ConfidenceRevisionConfig, at: EpochMillis) {
    apply_corroboration_bonus(object, config);
    if !config.enabled {
        return;
    }
    if object.promotion.status == PromotionStatus::Candidate {
        object.promotion.confirm(at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penalty_lowers_confidence_by_the_configured_amount() {
        let mut object = MentalObject::new_observation("a contradicted belief", EpochMillis(0), 0.5);
        object.confidence = 0.8;
        apply_contradiction_penalty(&mut object, &ConfidenceRevisionConfig::default());
        assert!((object.confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn penalty_never_drops_confidence_below_the_floor() {
        let mut object = MentalObject::new_observation("a repeatedly contradicted belief", EpochMillis(0), 0.5);
        object.confidence = 0.1;
        let config = ConfidenceRevisionConfig::default();
        apply_contradiction_penalty(&mut object, &config);
        apply_contradiction_penalty(&mut object, &config);
        apply_contradiction_penalty(&mut object, &config);
        assert!((object.confidence - config.min_confidence_floor).abs() < 1e-6);
    }

    #[test]
    fn disabled_config_leaves_confidence_untouched() {
        let mut object = MentalObject::new_observation("a contradicted belief", EpochMillis(0), 0.5);
        object.confidence = 0.8;
        let config = ConfidenceRevisionConfig { enabled: false, ..ConfidenceRevisionConfig::default() };
        apply_contradiction_penalty(&mut object, &config);
        assert_eq!(object.confidence, 0.8, "ablated revision must reproduce the old never-revised behavior exactly");
    }

    #[test]
    fn corroboration_raises_confidence_by_the_configured_amount() {
        let mut object = MentalObject::new_observation("a corroborated belief", EpochMillis(0), 0.5);
        object.confidence = 0.4;
        apply_corroboration_bonus(&mut object, &ConfidenceRevisionConfig::default());
        assert!((object.confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn corroboration_never_exceeds_full_confidence() {
        let mut object = MentalObject::new_observation("a repeatedly corroborated belief", EpochMillis(0), 0.5);
        object.confidence = 0.95;
        apply_corroboration_bonus(&mut object, &ConfidenceRevisionConfig::default());
        assert_eq!(object.confidence, 1.0);
    }

    #[test]
    fn disabled_config_leaves_corroboration_untouched() {
        let mut object = MentalObject::new_observation("a corroborated belief", EpochMillis(0), 0.5);
        object.confidence = 0.4;
        let config = ConfidenceRevisionConfig { enabled: false, ..ConfidenceRevisionConfig::default() };
        apply_corroboration_bonus(&mut object, &config);
        assert_eq!(object.confidence, 0.4);
    }

    #[test]
    fn sustained_contradiction_demotes_a_confirmed_object() {
        let mut object = MentalObject::new_observation("a once-trusted belief", EpochMillis(0), 0.5);
        object.confidence = 0.5;
        let config = ConfidenceRevisionConfig::default();
        // 0.5 -> 0.3 -> 0.1 (floored at min_confidence_floor 0.05) - the
        // second penalty already crosses demotion_confidence_threshold's
        // 0.15, so this exercises the real "sustained, not single-hit"
        // path the threshold is meant to reflect.
        apply_contradiction_penalty_and_maybe_demote(&mut object, &config, EpochMillis(1_000));
        assert_eq!(object.promotion.status, PromotionStatus::Confirmed, "a single contradiction must not demote outright");
        apply_contradiction_penalty_and_maybe_demote(&mut object, &config, EpochMillis(2_000));
        assert_eq!(object.promotion.status, PromotionStatus::Candidate, "sustained contradiction should invalidate the promotion cache");
        assert_eq!(object.promotion.staged_at, EpochMillis(2_000));
    }

    #[test]
    fn demotion_never_fires_while_already_disabled() {
        let mut object = MentalObject::new_observation("a belief", EpochMillis(0), 0.5);
        object.confidence = 0.1;
        let config = ConfidenceRevisionConfig { enabled: false, ..ConfidenceRevisionConfig::default() };
        apply_contradiction_penalty_and_maybe_demote(&mut object, &config, EpochMillis(1_000));
        assert_eq!(object.promotion.status, PromotionStatus::Confirmed, "an ablated config must not demote either");
    }

    #[test]
    fn corroboration_confirms_a_still_candidate_object() {
        let mut object = MentalObject::new_observation("a staged belief", EpochMillis(0), 0.5);
        object.promotion = aca_types::PromotionState::candidate(EpochMillis(0));
        object.confidence = 0.5;
        apply_corroboration_bonus_and_maybe_confirm(&mut object, &ConfidenceRevisionConfig::default(), EpochMillis(5_000));
        assert_eq!(object.promotion.status, PromotionStatus::Confirmed);
        assert_eq!(object.promotion.confirmed_at, Some(EpochMillis(5_000)));
    }

    #[test]
    fn corroboration_confirm_is_a_no_op_on_an_already_confirmed_object() {
        let mut object = MentalObject::new_observation("a belief", EpochMillis(0), 0.5);
        object.confidence = 0.5;
        apply_corroboration_bonus_and_maybe_confirm(&mut object, &ConfidenceRevisionConfig::default(), EpochMillis(5_000));
        assert_eq!(object.promotion.confirmed_at, Some(EpochMillis(0)), "an already-confirmed object must not be re-stamped");
    }
}
