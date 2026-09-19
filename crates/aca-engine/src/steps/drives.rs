use aca_graph::Graph;
use aca_types::MentalObjectKind;
use aca_util::EpochMillis;

use crate::snapshot::TierStatus;

/// EMA smoothing factor for every drive - same value and same reasoning as
/// `steps::affect::AffectTracker`'s own `smoothing` default: a fresh
/// reading moves a drive by 10% per nominal 250ms maintenance interval,
/// slow enough that one noisy reading cannot swing a drive on its own.
const DRIVE_SMOOTHING: f32 = 0.1;

/// How many unresolved discrepancies (open `Question`s or low-confidence
/// Reflections, neither yet consulted-on) saturate `curiosity` at `1.0` - a
/// handful, not hundreds: this engine runs at the "tens-low-hundreds of
/// nodes" scale already assumed elsewhere (see
/// `executive::has_reflection_for`'s doc comment), so even a few genuinely
/// unresolved threads should already read as noticeably curious. A tuning
/// choice, not a theoretical commitment.
const CURIOSITY_SATURATION_COUNT: f32 = 4.0;

/// A mean `|reported_confidence - reward|` this large already saturates
/// `competence` pressure at `1.0` - `CalibrationTracker::bias_for`'s value
/// is mathematically bounded to `[-1.0, 1.0]` (both operands are `[0,1]`),
/// but a bias anywhere near that extreme would mean a tier's self-reported
/// confidence is essentially uncorrelated with outcomes; 0.5 is already a
/// severe, saturating miscalibration in practice.
const COMPETENCE_BIAS_SATURATION: f32 = 0.5;

/// Wall-clock absence from real conversational input, in milliseconds,
/// that saturates `social_connection` pressure at `1.0`. A tuning knob
/// (same framing as `BoredomConfig::idle_threshold_ms`), not a claim about
/// how quickly isolation actually matters.
const SOCIAL_CONNECTION_SATURATION_MS: i64 = 30 * 60 * 1000;

/// Shared with `steps::affect::AffectTracker::update`, which folds valence
/// with the identical formula.
pub(super) fn ema(smoothing: f32, sample: f32, previous: f32) -> f32 {
    smoothing * sample + (1.0 - smoothing) * previous
}

/// A small set of continuously-varying, actor-local, non-persisted
/// cognitive pressures - the same category as `steps::affect::
/// AffectTracker::valence`, generalized from one scalar to several. Each
/// field is grounded in a signal this engine already computes somewhere
/// else in `tick()` (Compare's prediction error, unresolved graph state,
/// `CalibrationTracker`'s existing bias tracking, `ExecutionTracker`'s
/// real tool-call failure rate, conversational recency, tier-pool
/// saturation) - never a new model call, never a fabricated number.
/// Deliberately *not* the full 11-drive list a will/desire
/// architecture could in principle track (novelty, agency, exploration,
/// coherence, self-maintenance are all missing) - there is no real signal
/// grounding any of those in this engine yet, and inventing one would be
/// exactly the "manufactured pressure" this module exists to avoid.
///
/// Every field is a pressure, not a virtue: higher always means "this
/// matters more right now," including `competence` (rises when recent
/// tier calibration has been *poor*, not when it's been good) - consistent
/// polarity across every field is what lets `steps::agenda` compare them
/// on one shared scale when deciding what pressure is currently strongest.
#[derive(Debug, Clone, Copy)]
pub struct DriveState {
    /// How uncertain Omega's own predictive model has recently been -
    /// tracks `steps::compare::ComparisonResult::epistemic_value` (already
    /// `1.0 / precision`, active inference's own uncertainty term).
    pub uncertainty: f32,
    /// How much currently sits genuinely unresolved - tracks
    /// `curiosity_pressure`'s reading of open Questions and low-confidence
    /// Reflections.
    pub curiosity: f32,
    /// Pressure to become more competent - rises when either
    /// `CalibrationTracker` shows Tier 3's self-reported confidence has
    /// recently been miscalibrated (over- or under-confident), or
    /// `ExecutionTracker` shows recent tool invocations failing.
    pub competence: f32,
    /// Pressure toward reconnecting - rises the longer it's been since a
    /// real conversational turn (`SourceChannel::ConversationInput`).
    pub social_connection: f32,
    /// How saturated the constrained, single-flight model tiers (T3/T4)
    /// currently are.
    pub resource_pressure: f32,
    smoothing: f32,
    last_updated_at: Option<EpochMillis>,
}

impl DriveState {
    pub fn new(smoothing: f32) -> Self {
        Self { uncertainty: 0.0, curiosity: 0.0, competence: 0.0, social_connection: 0.0, resource_pressure: 0.0, smoothing, last_updated_at: None }
    }

    /// Folds one fresh tick's readings into each drive's running estimate.
    /// `epistemic_value`, `calibration_bias_magnitude`, and
    /// `execution_failure_rate` are `Option` -
    /// `None` when this tick has no fresh reading to offer (no Compare ran
    /// this tick; fewer than two calibration samples exist yet
    /// respectively) - and in that case the corresponding drive is simply
    /// left unchanged, the same "neutral until real evidence" convention
    /// `CalibrationTracker`/`PrecisionTracker` already use. The other three
    /// readings are always computable (a graph scan, a tier-pool read, and
    /// elapsed wall-clock time all exist on every tick), so they always
    /// fold in.
    pub fn update(&mut self, epistemic_value: Option<f32>, curiosity_reading: f32, calibration_bias_magnitude: Option<f32>, execution_failure_rate: Option<f32>, social_connection_reading: f32, resource_reading: f32) {
        self.update_with_smoothing(self.smoothing, epistemic_value, curiosity_reading,
            calibration_bias_magnitude, execution_failure_rate, social_connection_reading, resource_reading);
    }

    /// Actor path: retain the same 250ms nominal EMA while making its
    /// effective smoothing independent of scheduler wake frequency.
    pub fn update_at(&mut self, now: EpochMillis, epistemic_value: Option<f32>, curiosity_reading: f32, calibration_bias_magnitude: Option<f32>, execution_failure_rate: Option<f32>, social_connection_reading: f32, resource_reading: f32) {
        let elapsed_ms = self.last_updated_at.map_or(250,
            |last| now.0.saturating_sub(last.0).max(0));
        self.last_updated_at = Some(now);
        let smoothing = 1.0 - (1.0 - self.smoothing.clamp(0.0, 1.0)).powf(elapsed_ms as f32 / 250.0);
        self.update_with_smoothing(smoothing, epistemic_value, curiosity_reading,
            calibration_bias_magnitude, execution_failure_rate, social_connection_reading, resource_reading);
    }

    fn update_with_smoothing(&mut self, smoothing: f32, epistemic_value: Option<f32>, curiosity_reading: f32, calibration_bias_magnitude: Option<f32>, execution_failure_rate: Option<f32>, social_connection_reading: f32, resource_reading: f32) {
        if let Some(epistemic_value) = epistemic_value {
            self.uncertainty = ema(smoothing, epistemic_value.clamp(0.0, 1.0), self.uncertainty);
        }
        self.curiosity = ema(smoothing, curiosity_reading.clamp(0.0, 1.0), self.curiosity);
        let calibration_pressure = calibration_bias_magnitude.map(|bias_magnitude| (bias_magnitude.abs() / COMPETENCE_BIAS_SATURATION).clamp(0.0, 1.0));
        let execution_pressure = execution_failure_rate.map(|failure_rate| failure_rate.clamp(0.0, 1.0));
        let competence_pressure = match (calibration_pressure, execution_pressure) {
            (Some(calibration), Some(execution)) => Some(calibration.max(execution)),
            (Some(calibration), None) => Some(calibration),
            (None, Some(execution)) => Some(execution),
            (None, None) => None,
        };
        if let Some(competence_pressure) = competence_pressure {
            self.competence = ema(smoothing, competence_pressure, self.competence);
        }
        self.social_connection = ema(smoothing, social_connection_reading.clamp(0.0, 1.0), self.social_connection);
        self.resource_pressure = ema(smoothing, resource_reading.clamp(0.0, 1.0), self.resource_pressure);
    }

    /// The strongest of the three drives that reflect how well Omega's own
    /// internal machinery is currently functioning, as opposed to the state
    /// of the world or the conversation - Seth/Craig's interoceptive-
    /// inference ("beast machine") framing of self-monitoring: attention
    /// turns inward when homeostasis is actually disrupted, not on a fixed
    /// schedule. `uncertainty` (the predictive model doing poorly),
    /// `competence` (miscalibrated self-reports or failing tool calls), and
    /// `resource_pressure` (the constrained tiers running hot) are all
    /// literally about Omega's own functioning; `curiosity` (what's
    /// genuinely unresolved out there) and `social_connection` (how long
    /// since real conversational contact) are about the world and the
    /// relationship, not the machinery, so they're deliberately excluded
    /// here even though `strongest` above still considers them. Used by
    /// `steps::boredom::generate` to let a self-status check fire on real
    /// internal disruption rather than only a wall-clock interval.
    pub fn self_monitoring_pressure(&self) -> f32 {
        self.uncertainty.max(self.competence).max(self.resource_pressure)
    }

    /// The strongest currently-tracked pressure and its own value -
    /// `steps::agenda`'s spawn trigger reads this to decide both *whether*
    /// a new intention is worth forming and *which* drive it should be
    /// tagged as originating from (`data.source_drive`). Named by field,
    /// not by an enum, so a caller gets the same string this struct's own
    /// field is called - one source of truth for the name, not two.
    pub fn strongest(&self) -> (&'static str, f32) {
        [
            ("uncertainty", self.uncertainty),
            ("curiosity", self.curiosity),
            ("competence", self.competence),
            ("social_connection", self.social_connection),
            ("resource_pressure", self.resource_pressure),
        ]
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("the array literal above is never empty")
    }
}

    impl Default for DriveState {
    fn default() -> Self {
        Self::new(DRIVE_SMOOTHING)
    }
}

/// Counts what's actually still unresolved in `graph` right now: an open
/// `Question` (spawned by `executive::spawn_subgoal` on a
/// `MissingInformation` impasse, or by `steps::agenda`) or a Reflection
/// whose confidence is below `low_confidence_threshold` - excluding either
/// once already consulted on. These are exactly the two conditions
/// `executive::propose_operators`'s own `is_unconsulted_question`/
/// `is_unresolved_low_confidence_reflection` triggers already check;
/// this reuses the same definition of "unresolved" as a *count* rather than
/// a per-object gate, so `curiosity` tracks precisely what the Executive
/// itself would still treat as open questions, not a separate notion of
/// its own.
pub fn curiosity_pressure(graph: &Graph, low_confidence_threshold: f32) -> f32 {
    let count = graph
        .iter()
        .filter(|object| {
            let already_consulted = object.produced_by_operator.as_deref() == Some("ConsultKnowledgeLibrary");
            if already_consulted {
                return false;
            }
            match object.kind {
                MentalObjectKind::Question => true,
                MentalObjectKind::Reflection => object.confidence < low_confidence_threshold,
                _ => false,
            }
        })
        .count();
    (count as f32 / CURIOSITY_SATURATION_COUNT).clamp(0.0, 1.0)
}

/// How saturated the constrained, single-flight tiers (T3/T4) currently
/// are: `0.0` with full capacity free, `1.0` when every permit across both
/// is checked out. T1/T2 are deliberately excluded - they're many-
/// concurrent-model pools by design (`DivergentPool`), so their own
/// busy-ness was never meant to signal scarcity the way a single-flight
/// tier filling up does. `0.0` (no pressure) if `tier_status` is empty
/// (nothing to be saturated), never a division by zero.
pub fn resource_pressure(tier_status: &[TierStatus]) -> f32 {
    let (used, capacity) = tier_status.iter().fold((0usize, 0usize), |(used, capacity), status| {
        (used + status.capacity.saturating_sub(status.available_permits), capacity + status.capacity)
    });
    if capacity == 0 {
        return 0.0;
    }
    (used as f32 / capacity as f32).clamp(0.0, 1.0)
}

/// Pressure toward reconnecting: `0.0` (neutral) if there is no recorded
/// prior conversational turn at all yet - the same "no evidence, no
/// pressure" convention `CalibrationTracker`/`PrecisionTracker` already
/// use, not an assumption that a fresh actor is already lonely - otherwise
/// the elapsed time since `last_conversation_input_at`, normalized against
/// `SOCIAL_CONNECTION_SATURATION_MS`.
pub fn social_connection_pressure(last_conversation_input_at: Option<EpochMillis>, now: EpochMillis) -> f32 {
    match last_conversation_input_at {
        None => 0.0,
        Some(last) => ((now.0 - last.0) as f32 / SOCIAL_CONNECTION_SATURATION_MS as f32).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::{MemoryRole, MentalObject};

    #[test]
    fn starts_neutral_with_no_history() {
        let drives = DriveState::default();
        assert_eq!(drives.uncertainty, 0.0);
        assert_eq!(drives.curiosity, 0.0);
        assert_eq!(drives.competence, 0.0);
        assert_eq!(drives.social_connection, 0.0);
        assert_eq!(drives.resource_pressure, 0.0);
    }

    #[test]
    fn drive_smoothing_depends_on_elapsed_time_not_scheduler_wake_count() {
        let mut sparse = DriveState::default();
        let mut frequent = DriveState::default();
        for drives in [&mut sparse, &mut frequent] {
            drives.update_at(EpochMillis(0), None, 1.0, None, None, 0.0, 0.0);
        }
        sparse.update_at(EpochMillis(1_000), None, 1.0, None, None, 0.0, 0.0);
        for at in [250, 500, 750, 1_000] {
            frequent.update_at(EpochMillis(at), None, 1.0, None, None, 0.0, 0.0);
        }
        assert!((sparse.curiosity - frequent.curiosity).abs() < 1e-6);
    }

    #[test]
    fn a_run_of_high_epistemic_value_readings_raises_uncertainty() {
        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(Some(1.0), 0.0, None, None, 0.0, 0.0);
        }
        assert!(drives.uncertainty > 0.9, "got {}", drives.uncertainty);
    }

    #[test]
    fn a_missing_epistemic_value_reading_leaves_uncertainty_unchanged() {
        let mut drives = DriveState::default();
        drives.update(Some(1.0), 0.0, None, None, 0.0, 0.0);
        let after_first = drives.uncertainty;
        drives.update(None, 0.0, None, None, 0.0, 0.0);
        assert_eq!(drives.uncertainty, after_first, "None must skip the update entirely, not fold in as zero");
    }

    #[test]
    fn a_missing_calibration_reading_leaves_competence_unchanged() {
        let mut drives = DriveState::default();
        drives.update(None, 0.0, Some(0.5), None, 0.0, 0.0);
        let after_first = drives.competence;
        drives.update(None, 0.0, None, None, 0.0, 0.0);
        assert_eq!(drives.competence, after_first);
    }

    #[test]
    fn execution_failures_raise_competence_pressure() {
        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(None, 0.0, None, Some(1.0), 0.0, 0.0);
        }
        assert!(drives.competence > 0.9, "got {}", drives.competence);
    }

    #[test]
    fn competence_uses_the_stronger_of_calibration_and_execution_pressure() {
        let mut drives = DriveState::default();
        drives.update(None, 0.0, Some(0.1), Some(0.8), 0.0, 0.0);
        assert!((drives.competence - 0.08).abs() < 1e-6, "got {}", drives.competence);
    }

    #[test]
    fn self_monitoring_pressure_ignores_curiosity_and_social_connection() {
        let mut drives = DriveState::default();
        // Curiosity and social_connection both saturate; the three
        // self-monitoring drives stay untouched (no epistemic_value,
        // calibration, or execution readings folded in).
        for _ in 0..50 {
            drives.update(None, 1.0, None, None, 1.0, 0.0);
        }
        assert_eq!(drives.self_monitoring_pressure(), 0.0, "curiosity/social_connection must not leak into self-monitoring pressure");
    }

    #[test]
    fn self_monitoring_pressure_tracks_the_strongest_internal_drive() {
        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(None, 0.0, None, Some(1.0), 0.0, 0.3);
        }
        assert!((drives.self_monitoring_pressure() - drives.competence).abs() < 1e-6);
        assert!(drives.self_monitoring_pressure() > drives.resource_pressure);
    }

    #[test]
    fn strongest_names_the_field_with_the_highest_value() {
        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(None, 1.0, None, None, 0.0, 0.0);
        }
        let (name, value) = drives.strongest();
        assert_eq!(name, "curiosity");
        assert!(value > 0.9);
    }

    fn question(now: EpochMillis) -> MentalObject {
        let mut object = MentalObject::new_observation("what is that", now, 0.5);
        object.kind = MentalObjectKind::Question;
        object
    }

    #[test]
    fn curiosity_pressure_counts_unconsulted_questions_and_low_confidence_reflections() {
        let mut graph = Graph::new();
        graph.insert(question(EpochMillis(0)));

        let mut low_confidence_reflection = MentalObject::new_observation("probably something", EpochMillis(0), 0.5);
        low_confidence_reflection.kind = MentalObjectKind::Reflection;
        low_confidence_reflection.confidence = 0.2;
        graph.insert(low_confidence_reflection);

        let mut high_confidence_reflection = MentalObject::new_observation("definitely something", EpochMillis(0), 0.5);
        high_confidence_reflection.kind = MentalObjectKind::Reflection;
        high_confidence_reflection.confidence = 0.9;
        graph.insert(high_confidence_reflection);

        let mut already_consulted_question = question(EpochMillis(0));
        already_consulted_question.produced_by_operator = Some("ConsultKnowledgeLibrary".to_string());
        graph.insert(already_consulted_question);

        // 2 counted (the open question, the low-confidence reflection);
        // saturation is 4, so this should read as exactly 0.5.
        assert_eq!(curiosity_pressure(&graph, 0.5), 0.5);
    }

    #[test]
    fn curiosity_pressure_ignores_unrelated_object_kinds() {
        let mut graph = Graph::new();
        graph.insert(MentalObject::new_observation("just an observation", EpochMillis(0), 0.5));
        let mut memory = MentalObject::new_observation("a semantic memory", EpochMillis(0), 0.5);
        memory.kind = MentalObjectKind::Memory;
        memory.memory_roles.push(MemoryRole::Semantic);
        graph.insert(memory);
        assert_eq!(curiosity_pressure(&graph, 0.5), 0.0);
    }

    fn tier_status(capacity: usize, available_permits: usize) -> TierStatus {
        TierStatus { tier: aca_types::Tier::T3, available_permits, capacity }
    }

    #[test]
    fn resource_pressure_is_zero_with_full_capacity_free() {
        assert_eq!(resource_pressure(&[tier_status(3, 3), tier_status(1, 1)]), 0.0);
    }

    #[test]
    fn resource_pressure_is_one_when_fully_saturated() {
        assert_eq!(resource_pressure(&[tier_status(3, 0), tier_status(1, 0)]), 1.0);
    }

    #[test]
    fn resource_pressure_reflects_partial_saturation() {
        // 2 of 4 total permits in use.
        assert_eq!(resource_pressure(&[tier_status(3, 2), tier_status(1, 0)]), 0.5);
    }

    #[test]
    fn resource_pressure_is_zero_with_no_tiers_at_all() {
        assert_eq!(resource_pressure(&[]), 0.0);
    }

    #[test]
    fn social_connection_pressure_is_neutral_with_no_prior_conversation() {
        assert_eq!(social_connection_pressure(None, EpochMillis(1_000_000)), 0.0);
    }

    #[test]
    fn social_connection_pressure_rises_with_elapsed_absence() {
        let last = EpochMillis(0);
        let halfway = EpochMillis(SOCIAL_CONNECTION_SATURATION_MS / 2);
        let saturated = EpochMillis(SOCIAL_CONNECTION_SATURATION_MS * 2);
        assert!((social_connection_pressure(Some(last), halfway) - 0.5).abs() < 1e-6);
        assert_eq!(social_connection_pressure(Some(last), saturated), 1.0);
    }
}
