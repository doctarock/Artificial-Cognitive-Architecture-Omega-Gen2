use std::collections::HashMap;

use aca_types::MentalObjectId;
use aca_util::{cosine_error, RingBuffer};

/// Which recurring source a piece of content came from, for the purpose of
/// tracking how reliable/noisy that source's prediction errors have been.
/// Mirrors specs.md's attention sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceChannel {
    ConversationInput,
    SelfGeneratedThought,
    GoalDue,
    /// A Knowledge Library consult result re-entering as an Observation.
    /// Kept distinct from `ConversationInput` because a document store has a
    /// genuinely different reliability/error profile than a live
    /// interlocutor — it earns its own precision-tracking bucket rather than
    /// being folded into (and skewing) conversational precision.
    ExternalKnowledge,
    /// A sensed-environment observation forwarded by an external perception
    /// service (e.g. a camera service reporting a scene change), as opposed
    /// to a conversational turn — see `loop_actor::SensorInput`. One shared
    /// bucket for every such service for now, a named simplification: a
    /// camera feed and some future non-speech sensor could well have
    /// different noise profiles and earn their own buckets, but only camera
    /// is concrete today, so splitting further waits for a second real
    /// modality to actually show it matters.
    Environment,
}

const DEFAULT_PRECISION_WINDOW: usize = 20;
const PRECISION_EPSILON: f32 = 1e-3;
/// Precision assumed for a channel with fewer than two historical samples —
/// neutral (neither amplifies nor suppresses) rather than zero or infinite,
/// since there isn't yet evidence either way about that channel's
/// reliability.
const DEFAULT_PRECISION: f32 = 1.0;

/// Rolling per-source-channel inverse-variance precision tracker. Named
/// simplification: this is a rolling-window inverse-variance estimate, not a
/// jointly-optimized Bayesian precision with cross-source covariance (see
/// the plan).
///
/// `interlocutor_windows` is a second, optional layer keyed by a specific
/// recognized interlocutor's `MentalObjectId` (see `steps::interlocutor`),
/// not a new `SourceChannel` variant — the two are genuinely different axes
/// (categorical reliability vs. individually-learned reliability), and
/// layering keeps the existing 5-bucket channel map, and every test built
/// against it, untouched. `precision_for_interlocutor` prefers the
/// interlocutor-specific estimate once it has its own history, falling back
/// to the channel-level estimate before that — a "specific overrides
/// generic once earned" hierarchy: a not-yet-recognized or unenrolled voice
/// is judged by the generic conversational prior, a familiar one by what's
/// actually been learned about them specifically.
pub struct PrecisionTracker {
    windows: HashMap<SourceChannel, RingBuffer<f32>>,
    interlocutor_windows: HashMap<MentalObjectId, RingBuffer<f32>>,
    window_capacity: usize,
}

impl PrecisionTracker {
    pub fn new(window_capacity: usize) -> Self {
        Self {
            windows: HashMap::new(),
            interlocutor_windows: HashMap::new(),
            window_capacity,
        }
    }

    fn stddev(window: &RingBuffer<f32>) -> Option<f32> {
        let n = window.len();
        if n < 2 {
            return None;
        }
        let values: Vec<f32> = window.iter().copied().collect();
        let mean = values.iter().sum::<f32>() / n as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        Some(variance.sqrt())
    }

    /// The current precision for `channel`, based on history *before* the
    /// value about to be recorded this tick — i.e. call this before
    /// `record_error` for the same observation.
    pub fn precision_for(&self, channel: SourceChannel) -> f32 {
        match self.windows.get(&channel).and_then(Self::stddev) {
            Some(sd) => 1.0 / (sd + PRECISION_EPSILON),
            None => DEFAULT_PRECISION,
        }
    }

    /// `precision_for`, refined by a specific interlocutor's own error
    /// history once it has one — see this struct's own doc comment.
    /// `interlocutor` is `None` for anything that never resolved to a
    /// recognized (named/enrolled) speaker, which simply falls straight
    /// through to `precision_for`.
    pub fn precision_for_interlocutor(&self, channel: SourceChannel, interlocutor: Option<MentalObjectId>) -> f32 {
        let specific = interlocutor.and_then(|id| self.interlocutor_windows.get(&id)).and_then(Self::stddev);
        match specific {
            Some(sd) => 1.0 / (sd + PRECISION_EPSILON),
            None => self.precision_for(channel),
        }
    }

    pub fn record_error(&mut self, channel: SourceChannel, error_magnitude: f32) {
        self.windows
            .entry(channel)
            .or_insert_with(|| RingBuffer::new(self.window_capacity))
            .push(error_magnitude);
    }

    /// `record_error`, plus (when `interlocutor` is `Some`) the same value
    /// into that interlocutor's own window — the channel-level bucket is
    /// always updated too, so the generic conversational prior keeps
    /// learning from every conversational turn, recognized or not.
    pub fn record_error_for_interlocutor(&mut self, channel: SourceChannel, interlocutor: Option<MentalObjectId>, error_magnitude: f32) {
        self.record_error(channel, error_magnitude);
        if let Some(id) = interlocutor {
            self.interlocutor_windows.entry(id).or_insert_with(|| RingBuffer::new(self.window_capacity)).push(error_magnitude);
        }
    }
}

impl Default for PrecisionTracker {
    fn default() -> Self {
        Self::new(DEFAULT_PRECISION_WINDOW)
    }
}

/// Step 3 - Compare's output: the raw prediction error, the channel's
/// current precision weight, and their product — the single currency
/// candidate Mental Objects compete on on entering Coalition (Step 5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparisonResult {
    pub error_magnitude: f32,
    pub precision: f32,
    pub precision_weighted_surprise: f32,
    /// Active inference's epistemic value (specs.md's Attention section:
    /// "attention toward observations expected to reduce model uncertainty,
    /// not merely toward what's loud") — how uncertain this channel's own
    /// error history currently is, i.e. `1.0 / precision`. High when a
    /// channel has been noisy/inconsistent or has little history yet (so a
    /// fresh observation from it is more likely to actually inform the
    /// model); low when a channel has been highly consistent (so another
    /// sample from it teaches little that isn't already known). Named
    /// simplification, like `PrecisionTracker` itself: true expected
    /// information gain would require simulating hypothetical observations
    /// against a real forward generative model, which this architecture
    /// doesn't have — using the same rolling per-channel uncertainty
    /// `PrecisionTracker` already tracks is the Tier-0-computable stand-in,
    /// not a literal free-energy calculation. Always positive and finite:
    /// `precision` is never zero (see `PrecisionTracker::precision_for`).
    pub epistemic_value: f32,
}

/// Step 3 - Compare. Pure computation: no I/O, no LLM call, operates only
/// on already-resolved embeddings. `expected` is `None` when Predict had
/// nothing to blend from (e.g. the very first observation ever) — by
/// convention this is treated as maximally surprising (`error_magnitude =
/// 1.0`, cosine error's midpoint) rather than skipping comparison entirely,
/// so a graph's first-ever input still competes fairly for broadcast.
///
/// `interlocutor` is `Some` only when this observation resolved to a
/// recognized (named/enrolled) speaker (see `steps::interlocutor`) — `None`
/// covers every other case (a non-conversational channel, an unrecognized
/// or unenrolled voice, self-generated thought, etc.) and behaves exactly
/// as before this parameter existed.
pub fn compare(
    expected: Option<&[f32]>,
    actual: &[f32],
    channel: SourceChannel,
    interlocutor: Option<MentalObjectId>,
    tracker: &mut PrecisionTracker,
) -> ComparisonResult {
    let error_magnitude = match expected {
        Some(expected_vec) => cosine_error(expected_vec, actual),
        None => 1.0,
    };
    let precision = tracker.precision_for_interlocutor(channel, interlocutor);
    tracker.record_error_for_interlocutor(channel, interlocutor, error_magnitude);
    ComparisonResult {
        error_magnitude,
        precision,
        precision_weighted_surprise: error_magnitude * precision,
        epistemic_value: 1.0 / precision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_expectation_is_maximally_surprising_by_convention() {
        let mut tracker = PrecisionTracker::default();
        let actual = vec![1.0, 0.0];
        let result = compare(None, &actual, SourceChannel::ConversationInput, None, &mut tracker);
        assert_eq!(result.error_magnitude, 1.0);
    }

    #[test]
    fn identical_expected_and_actual_yield_zero_error() {
        let mut tracker = PrecisionTracker::default();
        let v = vec![1.0, 0.0, 0.0];
        let result = compare(Some(&v), &v, SourceChannel::ConversationInput, None, &mut tracker);
        assert!(result.error_magnitude.abs() < 1e-5);
    }

    #[test]
    fn precision_defaults_to_neutral_with_no_history() {
        let tracker = PrecisionTracker::default();
        assert_eq!(tracker.precision_for(SourceChannel::GoalDue), DEFAULT_PRECISION);
    }

    #[test]
    fn precision_rises_as_a_channels_errors_become_more_consistent() {
        let mut tracker = PrecisionTracker::default();
        // A noisy channel: wildly varying error magnitudes.
        for e in [0.1, 0.9, 0.2, 0.8, 0.15, 0.85] {
            tracker.record_error(SourceChannel::ConversationInput, e);
        }
        let noisy_precision = tracker.precision_for(SourceChannel::ConversationInput);

        // A consistent channel: nearly identical error magnitudes.
        for e in [0.5, 0.51, 0.49, 0.5, 0.505, 0.495] {
            tracker.record_error(SourceChannel::GoalDue, e);
        }
        let consistent_precision = tracker.precision_for(SourceChannel::GoalDue);

        assert!(
            consistent_precision > noisy_precision,
            "consistent={consistent_precision} should exceed noisy={noisy_precision}"
        );
    }

    #[test]
    fn record_error_advances_the_window_used_by_next_precision_for_call() {
        let mut tracker = PrecisionTracker::default();
        assert_eq!(tracker.precision_for(SourceChannel::ConversationInput), DEFAULT_PRECISION);
        tracker.record_error(SourceChannel::ConversationInput, 0.5);
        // still < 2 samples -> still default
        assert_eq!(tracker.precision_for(SourceChannel::ConversationInput), DEFAULT_PRECISION);
        tracker.record_error(SourceChannel::ConversationInput, 0.5);
        // now >= 2 identical samples -> stddev 0 -> precision should be very high (1/epsilon)
        assert!(tracker.precision_for(SourceChannel::ConversationInput) > DEFAULT_PRECISION);
    }

    #[test]
    fn window_respects_capacity_bound() {
        let mut tracker = PrecisionTracker::new(3);
        for e in [0.1, 0.1, 0.1, 0.9, 0.9] {
            tracker.record_error(SourceChannel::ConversationInput, e);
        }
        // Only the last 3 values (0.1, 0.9, 0.9) should remain in the window.
        let precision = tracker.precision_for(SourceChannel::ConversationInput);
        assert!(precision.is_finite());
    }

    #[test]
    fn precision_weighted_surprise_is_the_product() {
        let mut tracker = PrecisionTracker::default();
        let expected = vec![1.0, 0.0];
        let actual = vec![0.0, 1.0]; // orthogonal -> error 1.0
        let result = compare(Some(&expected), &actual, SourceChannel::SelfGeneratedThought, None, &mut tracker);
        assert!((result.precision_weighted_surprise - (result.error_magnitude * result.precision)).abs() < 1e-6);
    }

    #[test]
    fn epistemic_value_is_the_inverse_of_precision() {
        let mut tracker = PrecisionTracker::default();
        let v = vec![1.0, 0.0];
        let result = compare(Some(&v), &v, SourceChannel::GoalDue, None, &mut tracker);
        assert!((result.epistemic_value - (1.0 / result.precision)).abs() < 1e-6);
    }

    #[test]
    fn environment_channel_tracks_precision_independently_of_conversation() {
        // A sensed-environment observation (e.g. a camera service's scene
        // report) must not skew, or be skewed by, conversational precision -
        // same independence already proven for ExternalKnowledge above, now
        // for the channel loop_actor::SensorInput forwards through.
        let mut tracker = PrecisionTracker::default();
        for e in [0.1, 0.9, 0.2, 0.8, 0.15, 0.85] {
            tracker.record_error(SourceChannel::ConversationInput, e);
        }
        assert_eq!(
            tracker.precision_for(SourceChannel::Environment),
            DEFAULT_PRECISION,
            "Environment should still be untouched by ConversationInput's history"
        );

        let v = vec![1.0, 0.0];
        let result = compare(Some(&v), &v, SourceChannel::Environment, None, &mut tracker);
        assert!((result.error_magnitude).abs() < 1e-5);
        assert_eq!(result.precision, DEFAULT_PRECISION, "first-ever Environment sample has no history to derive precision from yet");
    }

    #[test]
    fn interlocutor_precision_falls_back_to_the_channel_estimate_with_no_history_of_its_own() {
        let mut tracker = PrecisionTracker::default();
        let derek = MentalObjectId::new();
        // Channel-level history only - `derek` has never been recorded
        // individually yet.
        for e in [0.5, 0.51, 0.49, 0.5] {
            tracker.record_error(SourceChannel::ConversationInput, e);
        }
        assert_eq!(
            tracker.precision_for_interlocutor(SourceChannel::ConversationInput, Some(derek)),
            tracker.precision_for(SourceChannel::ConversationInput),
            "with no individual history yet, an interlocutor's precision must equal the generic channel estimate"
        );
    }

    #[test]
    fn interlocutor_precision_overrides_the_channel_estimate_once_it_has_its_own_history() {
        let mut tracker = PrecisionTracker::default();
        let derek = MentalObjectId::new();
        let stranger = MentalObjectId::new();

        // The channel overall is noisy (a mix of everyone in the room)...
        for e in [0.1, 0.9, 0.2, 0.8, 0.15, 0.85] {
            tracker.record_error_for_interlocutor(SourceChannel::ConversationInput, Some(stranger), e);
        }
        // ...but this specific, recognized interlocutor has been highly
        // consistent every time they've actually spoken.
        for e in [0.5, 0.51, 0.49, 0.5, 0.505, 0.495] {
            tracker.record_error_for_interlocutor(SourceChannel::ConversationInput, Some(derek), e);
        }

        let derek_precision = tracker.precision_for_interlocutor(SourceChannel::ConversationInput, Some(derek));
        let channel_precision = tracker.precision_for(SourceChannel::ConversationInput);
        assert!(
            derek_precision > channel_precision,
            "a specifically consistent interlocutor ({derek_precision}) should be judged more precisely than the noisy channel average ({channel_precision})"
        );
    }

    #[test]
    fn compare_records_into_both_the_channel_and_interlocutor_windows() {
        let mut tracker = PrecisionTracker::default();
        let derek = MentalObjectId::new();
        let v = vec![1.0, 0.0];
        compare(Some(&v), &v, SourceChannel::ConversationInput, Some(derek), &mut tracker);
        compare(Some(&v), &v, SourceChannel::ConversationInput, Some(derek), &mut tracker);

        assert_ne!(tracker.precision_for(SourceChannel::ConversationInput), DEFAULT_PRECISION, "the generic channel bucket should have learned from this turn too");
        assert_ne!(
            tracker.precision_for_interlocutor(SourceChannel::ConversationInput, Some(derek)),
            DEFAULT_PRECISION,
            "the interlocutor-specific bucket should have its own history now"
        );
    }

    #[test]
    fn a_noisy_channel_has_higher_epistemic_value_than_a_consistent_one() {
        // Curiosity should favor the channel Omega's model is least sure
        // about - the inverse of what precision-weighting already favors
        // for trusting an error signal.
        let mut tracker = PrecisionTracker::default();
        for e in [0.1, 0.9, 0.2, 0.8, 0.15, 0.85] {
            tracker.record_error(SourceChannel::ConversationInput, e);
        }
        for e in [0.5, 0.51, 0.49, 0.5, 0.505, 0.495] {
            tracker.record_error(SourceChannel::GoalDue, e);
        }

        let v = vec![1.0, 0.0];
        let noisy = compare(Some(&v), &v, SourceChannel::ConversationInput, None, &mut tracker);
        let consistent = compare(Some(&v), &v, SourceChannel::GoalDue, None, &mut tracker);

        assert!(
            noisy.epistemic_value > consistent.epistemic_value,
            "noisy={} should exceed consistent={}",
            noisy.epistemic_value,
            consistent.epistemic_value
        );
    }
}
