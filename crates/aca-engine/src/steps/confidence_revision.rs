use aca_types::MentalObject;

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
    /// Whether `apply_contradiction_penalty` is ever called at all - `false`
    /// reproduces the old "confidence is set once at creation, never
    /// revised" behavior exactly, for A/B comparison.
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

#[cfg(test)]
mod tests {
    use super::*;
    use aca_util::EpochMillis;

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
}
