use aca_types::MentalObjectDynamics;
use aca_util::EpochMillis;

use super::compare::{ComparisonResult, SourceChannel};

/// Pure-numeric orienting-layer weights and dynamics. This layer decides how
/// strongly a resolved observation should perturb local object state; it does
/// not generate language and performs no I/O.
#[derive(Debug, Clone, Copy)]
pub struct OrientingConfig {
    pub novelty_weight: f32,
    pub prediction_error_weight: f32,
    pub goal_relevance_weight: f32,
    pub affective_salience_weight: f32,
    pub social_relevance_weight: f32,
    pub threat_weight: f32,
    pub urgency_weight: f32,
    pub firing_threshold: f32,
    pub attention_pulse: f32,
    pub potential_tau_ms: f32,
    pub adaptation_tau_ms: f32,
    pub refractory_ms: i64,
    pub adaptation_increment: f32,
}

impl Default for OrientingConfig {
    fn default() -> Self {
        Self {
            novelty_weight: 0.30,
            prediction_error_weight: 0.25,
            goal_relevance_weight: 0.20,
            affective_salience_weight: 0.15,
            social_relevance_weight: 0.10,
            threat_weight: 0.25,
            urgency_weight: 0.25,
            firing_threshold: 0.5,
            attention_pulse: 0.25,
            potential_tau_ms: 250.0,
            adaptation_tau_ms: 2_000.0,
            refractory_ms: 400,
            adaptation_increment: 0.15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientingResult {
    pub novelty: f32,
    pub prediction_error: f32,
    pub goal_relevance: f32,
    pub affective_salience: f32,
    pub social_relevance: f32,
    pub threat: f32,
    pub urgency: f32,
    pub score: f32,
    pub fired: bool,
}

/// Conservative sensory habituation: only a familiar, well-predicted
/// environment reading without identity or calibrated urgency may stay
/// below Coalition. Human-directed input is never muted by this gate.
pub fn should_habituate_environment(
    channel: SourceChannel,
    has_expectation: bool,
    error_magnitude: f32,
    orienting: Option<OrientingResult>,
    has_identity: bool,
    threat: f32,
    urgency: f32,
    threshold: f32,
) -> bool {
    channel == SourceChannel::Environment
        && has_expectation
        && !has_identity
        && threat <= 0.0
        && urgency <= 0.0
        && error_magnitude < 0.2
        && orienting.is_some_and(|result| !result.fired && result.score < threshold)
}

/// Evaluates a cheap orienting response and applies it as a local pulse to
/// the observation's membrane-like state. Novelty and prediction error are
/// kept as separate weighted terms intentionally: the former asks whether
/// this departs from expectation at all, while the latter retains the
/// precision-weighted magnitude already used by Coalition.
pub fn orient(
    dynamics: &mut MentalObjectDynamics,
    comparison: &ComparisonResult,
    source_surprise: f32,
    goal_relevance: f32,
    affect_valence: f32,
    channel: SourceChannel,
    threat: f32,
    urgency: f32,
    config: &OrientingConfig,
    now: EpochMillis,
) -> OrientingResult {
    let novelty = comparison
        .error_magnitude
        .max(source_surprise)
        .clamp(0.0, 1.0);
    let prediction_error = comparison.precision_weighted_surprise.clamp(0.0, 1.0);
    let goal_relevance = goal_relevance.clamp(0.0, 1.0);
    let affective_salience = (-affect_valence).clamp(0.0, 1.0);
    let social_relevance = matches!(channel, SourceChannel::ConversationInput) as u8 as f32;
    let threat = threat.clamp(0.0, 1.0);
    let urgency = urgency.clamp(0.0, 1.0);
    let score = (config.novelty_weight * novelty
        + config.prediction_error_weight * prediction_error
        + config.goal_relevance_weight * goal_relevance
        + config.affective_salience_weight * affective_salience
        + config.social_relevance_weight * social_relevance
        + config.threat_weight * threat
        + config.urgency_weight * urgency)
        .clamp(0.0, 1.0);

    dynamics.threshold = config.firing_threshold;
    aca_graph::stimulate_dynamics(
        dynamics,
        score,
        now,
        config.potential_tau_ms,
        config.adaptation_tau_ms,
    );
    let fired = aca_graph::try_fire_dynamics(
        dynamics,
        now,
        config.refractory_ms,
        config.adaptation_increment,
    );

    OrientingResult {
        novelty,
        prediction_error,
        goal_relevance,
        affective_salience,
        social_relevance,
        threat,
        urgency,
        score,
        fired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comparison(error: f32, surprise: f32) -> ComparisonResult {
        ComparisonResult {
            error_magnitude: error,
            precision: 1.0,
            precision_weighted_surprise: surprise,
            epistemic_value: 1.0,
        }
    }

    #[test]
    fn habituation_never_mutes_unexpected_urgent_identified_or_human_input() {
        let mut dynamics = MentalObjectDynamics::new_at(EpochMillis(0));
        let quiet = orient(&mut dynamics, &comparison(0.0, 0.0), 0.0, 0.0, 0.0,
            SourceChannel::Environment, 0.0, 0.0, &OrientingConfig::default(), EpochMillis(0));
        let gate = |channel, has_expectation, error, has_identity, threat, urgency|
            should_habituate_environment(channel, has_expectation, error, Some(quiet),
                has_identity, threat, urgency, 0.2);
        assert!(gate(SourceChannel::Environment, true, 0.0, false, 0.0, 0.0));
        assert!(!gate(SourceChannel::ConversationInput, true, 0.0, false, 0.0, 0.0));
        assert!(!gate(SourceChannel::Environment, false, 0.0, false, 0.0, 0.0));
        assert!(!gate(SourceChannel::Environment, true, 0.8, false, 0.0, 0.0));
        assert!(!gate(SourceChannel::Environment, true, 0.0, true, 0.0, 0.0));
        assert!(!gate(SourceChannel::Environment, true, 0.0, false, 0.1, 0.0));
        assert!(!gate(SourceChannel::Environment, true, 0.0, false, 0.0, 0.1));
    }

    #[test]
    fn expected_background_input_stays_subthreshold() {
        let mut dynamics = MentalObjectDynamics::new_at(EpochMillis(0));
        let result = orient(
            &mut dynamics,
            &comparison(0.0, 0.0),
            0.0,
            0.0,
            0.0,
            SourceChannel::Environment,
            0.0,
            0.0,
            &OrientingConfig::default(),
            EpochMillis(0),
        );
        assert_eq!(result.score, 0.0);
        assert!(!result.fired);
    }

    #[test]
    fn novel_social_input_fires_without_a_model() {
        let mut dynamics = MentalObjectDynamics::new_at(EpochMillis(0));
        let result = orient(
            &mut dynamics,
            &comparison(1.0, 1.0),
            1.0,
            0.0,
            0.0,
            SourceChannel::ConversationInput,
            0.0,
            0.0,
            &OrientingConfig::default(),
            EpochMillis(0),
        );
        assert!((result.score - 0.65).abs() < 1e-6);
        assert!(result.fired);
    }

    #[test]
    fn negative_affect_and_goal_relevance_can_accumulate_into_a_later_firing() {
        let mut dynamics = MentalObjectDynamics::new_at(EpochMillis(0));
        let config = OrientingConfig::default();
        let first = orient(
            &mut dynamics,
            &comparison(0.0, 0.0),
            0.0,
            1.0,
            -1.0,
            SourceChannel::Environment,
            0.0,
            0.0,
            &config,
            EpochMillis(0),
        );
        assert!(!first.fired);
        let second = orient(
            &mut dynamics,
            &comparison(0.0, 0.0),
            0.0,
            1.0,
            -1.0,
            SourceChannel::Environment,
            0.0,
            0.0,
            &config,
            EpochMillis(10),
        );
        assert!(second.fired);
    }

    #[test]
    fn calibrated_sensor_threat_and_urgency_fire_without_semantic_surprise() {
        let mut dynamics = MentalObjectDynamics::new_at(EpochMillis(0));
        let result = orient(
            &mut dynamics,
            &comparison(0.0, 0.0),
            0.0,
            0.0,
            0.0,
            SourceChannel::Environment,
            1.0,
            1.0,
            &OrientingConfig::default(),
            EpochMillis(0),
        );
        assert_eq!(result.score, 0.5);
        assert!(result.fired);
    }
}
