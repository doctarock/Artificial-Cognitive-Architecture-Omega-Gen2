use std::collections::HashMap;

use aca_types::{GoalStatus, MentalObjectId, Tier};
use aca_util::RingBuffer;

use super::compare::{ComparisonResult, SourceChannel};

/// Converts a `ComparisonResult`'s prediction error into a `[0, 1]` reward:
/// `1.0` means the next moment on that channel was well-predicted (whatever
/// happened after the decision this reward is paying off was benign/
/// expected); `0.0` means it was maximally surprising. This is deliberately
/// *not* a new signal - predictive processing's own currency
/// (`error_magnitude`) already answers "how well did reality match what was
/// expected," which is exactly what "did that decision's consequence turn
/// out okay" reduces to when there's no explicit external reward channel.
/// Reusing it is what keeps both consumers of this module (chunk-utility
/// learning and confidence calibration, below) Tier-0: no LLM call, no new
/// I/O, just a different read of a number the architecture already computes
/// every tick.
pub fn reward_from_comparison(comparison: &ComparisonResult) -> f32 {
    (1.0 - comparison.error_magnitude).clamp(0.0, 1.0)
}

/// A second, independent reward channel from `reward_from_comparison`
/// above: whether a goal's status just moved toward or away from being
/// satisfied, rather than whether the world matched Omega's prediction.
/// Agency/normativity's own critique of predictive-processing valence:
/// `reward_from_comparison` alone conflates *surprising* with *bad* - a
/// well-predicted but goal-thwarting tick and a startling-but-welcome one
/// currently register identically under it, since it only ever measures
/// prediction error. This doesn't replace that signal; `steps::affect::
/// AffectTracker::update` blends the two.
///
/// `Some(1.0)` when `previous` was `Active` and `current` is `Satisfied`
/// (real progress); `Some(0.0)` when it moved to `Abandoned` instead (a real
/// setback, not just an unexpected observation); `None` for every other
/// transition, including "no transition at all" - the same "neutral until
/// there's real evidence" convention `CalibrationTracker::bias_for` and
/// `PrecisionTracker::precision_for` both already follow in this module and
/// its sibling.
pub fn goal_progress_reward(previous: GoalStatus, current: GoalStatus) -> Option<f32> {
    match (previous, current) {
        (GoalStatus::Active, GoalStatus::Satisfied) => Some(1.0),
        (GoalStatus::Active, GoalStatus::Abandoned) => Some(0.0),
        _ => None,
    }
}

/// Scans every goal-carrying object currently in `graph` for a status that
/// differs from what `previous` recorded for it last tick, returning the
/// reward for the first such transition (`goal_progress_reward`) alongside
/// a fresh snapshot of every current goal's status for the caller to keep as
/// next tick's `previous`. The graph's sparse goal index avoids scanning
/// unrelated autobiographical objects on every cognitive event.
///
/// At most one transition's reward is returned per tick even if several
/// goals transitioned in the same tick - a named simplification (which one
/// wins is graph-iteration-order, unspecified), not a claim that only one
/// goal can ever change per tick.
pub fn detect_goal_progress(graph: &aca_graph::Graph, previous: &HashMap<MentalObjectId, GoalStatus>) -> (Option<f32>, HashMap<MentalObjectId, GoalStatus>) {
    let mut current = HashMap::new();
    let mut reward = None;
    for object in graph.goal_objects() {
        let Some(membership) = &object.goal else { continue };
        current.insert(object.id, membership.status);
        if reward.is_none() {
            reward = previous.get(&object.id).and_then(|&previous_status| goal_progress_reward(previous_status, membership.status));
        }
    }
    (reward, current)
}

/// One decision awaiting its consequence: registered the tick a decision is
/// made (a chunk's bias won operator selection, or a Tier 3/4 self-reported
/// confidence was trusted), and resolved - or expired, uncredited - on a
/// later tick's `compare()` for the same channel/interlocutor. Exactly one
/// of `chunk_id`/`tier` is expected to be `Some` per outcome: which field is
/// set determines which learning process (`reinforce_chunk_utility` in
/// `steps::learn`, or `CalibrationTracker::record` below) the resolved
/// reward feeds.
#[derive(Debug, Clone, Copy)]
pub struct PendingOutcome {
    pub channel: SourceChannel,
    pub interlocutor: Option<MentalObjectId>,
    pub chunk_id: Option<MentalObjectId>,
    pub tier: Option<Tier>,
    pub reported_confidence: f32,
    pub created_at: aca_util::EpochMillis,
}

/// Holds decisions awaiting their consequence. A `Vec` rather than a map
/// keyed by channel: at most one or two decisions are made per tick (see
/// `loop_actor`'s Steps 7-9), so a linear scan on `resolve` is cheap at this
/// scale - same "not a hot path" reasoning already applied elsewhere in this
/// codebase (e.g. `executive::has_reflection_for`).
pub struct OutcomeRegistry {
    pending: Vec<PendingOutcome>,
    /// How long (wall-clock) a pending outcome stays eligible for credit
    /// before being dropped, uncredited. ACT-R-consistent temporal-proximity
    /// credit assignment: an outcome too far removed from the decision that
    /// produced it isn't good evidence about that decision anymore (a lot
    /// can have changed on the same channel in between).
    ///
    /// Wall-clock rather than a tick/cycle count deliberately: this used to
    /// be `expiry_cycles`, counted in ticks, which meant its real-world
    /// window shrank or grew with however fast ticks happened to be firing -
    /// a window meant to matter most while the system is actively engaged
    /// (rapid ticking) shrank to milliseconds in exactly that regime, and
    /// stretched out during any lull. `CognitiveScheduler`'s idle-sleep
    /// ceiling (`LoopConfig::max_idle_interval_ms`) made ticks-per-second
    /// variable by design, so a tick-counted expiry could no longer be
    /// treated as a stand-in for a real duration.
    expiry_ms: i64,
}

impl OutcomeRegistry {
    pub fn new(expiry_ms: i64) -> Self {
        Self { pending: Vec::new(), expiry_ms }
    }

    pub fn register(&mut self, outcome: PendingOutcome) {
        self.pending.push(outcome);
    }

    /// Drains every pending outcome matching `(channel, interlocutor)`,
    /// dropping (without returning) anything older than `expiry_ms`
    /// regardless of whether it matches - both kinds of removal happen in
    /// the same pass so expired entries never accumulate indefinitely just
    /// because their channel never recurs.
    pub fn resolve(&mut self, channel: SourceChannel, interlocutor: Option<MentalObjectId>, now: aca_util::EpochMillis) -> Vec<PendingOutcome> {
        let expiry_ms = self.expiry_ms;
        let mut resolved = Vec::new();
        self.pending.retain(|outcome| {
            let age_ms = now.0.saturating_sub(outcome.created_at.0);
            if age_ms > expiry_ms {
                return false;
            }
            if outcome.channel == channel && outcome.interlocutor == interlocutor {
                resolved.push(*outcome);
                return false;
            }
            true
        });
        resolved
    }
}

const DEFAULT_CALIBRATION_WINDOW: usize = 20;

/// Rolling per-tier calibration tracker: how far off has this tier's own
/// self-reported confidence been from how things actually turned out. Same
/// shape as `compare::PrecisionTracker` deliberately - same rolling-window
/// approach, same "neutral until proven otherwise" default, same
/// non-persistence (actor-local state, reset on restart).
///
/// Only ever meant to be consulted for Tier 3/4 - Tier 1/2 already earn a
/// trustworthy confidence structurally, via cross-sample agreement rather
/// than self-report (see `cognitive_core::try_tier_via_agreement`), and
/// applying calibration on top of an already-honest number would be
/// second-guessing a signal that doesn't need it.
pub struct CalibrationTracker {
    /// Per tier, a rolling window of `(reported_confidence - reward)`
    /// values - positive means this tier tends to be overconfident, negative
    /// means underconfident, centered on zero means well-calibrated.
    windows: HashMap<Tier, RingBuffer<f32>>,
    window_capacity: usize,
}

impl CalibrationTracker {
    pub fn new(window_capacity: usize) -> Self {
        Self { windows: HashMap::new(), window_capacity }
    }

    /// This tier's mean overconfidence bias so far. `0.0` (neutral) with
    /// fewer than two samples - not enough history yet to say anything about
    /// this tier's calibration, so it's trusted at face value. `pub` (not
    /// just used internally by `calibrate`) so `steps::drives`'s `competence`
    /// signal can read the same real, already-tracked miscalibration number
    /// rather than inventing a separate one - see that module's doc comment.
    pub fn bias_for(&self, tier: Tier) -> f32 {
        match self.windows.get(&tier) {
            Some(window) if window.len() >= 2 => {
                let values: Vec<f32> = window.iter().copied().collect();
                values.iter().sum::<f32>() / values.len() as f32
            }
            _ => 0.0,
        }
    }

    /// Records one resolved `(reported_confidence, reward)` pair for `tier`.
    pub fn record(&mut self, tier: Tier, reported_confidence: f32, reward: f32) {
        let bias = reported_confidence - reward;
        self.windows.entry(tier).or_insert_with(|| RingBuffer::new(self.window_capacity)).push(bias);
    }

    /// `raw_confidence`, discounted by `tier`'s learned overconfidence bias
    /// (or boosted, if the tier has been running underconfident), clamped to
    /// `[0, 1]`. With no history for this tier, returns `raw_confidence`
    /// unchanged - calibration only ever acts on evidence actually
    /// accumulated, never a default assumption of untrustworthiness.
    pub fn calibrate(&self, tier: Tier, raw_confidence: f32) -> f32 {
        (raw_confidence - self.bias_for(tier)).clamp(0.0, 1.0)
    }
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new(DEFAULT_CALIBRATION_WINDOW)
    }
}

const DEFAULT_EXECUTION_WINDOW: usize = 20;

/// Rolling tracker of how often Act's tool invocations actually succeed -
/// `CalibrationTracker`'s counterpart for the *executing* half of "how well
/// is this going," not the *predicting* half. Same shape deliberately: same
/// rolling-window approach, same "neutral until proven otherwise" default,
/// same non-persistence.
///
/// What it's fed is the real difference from `CalibrationTracker`: not a
/// gap between a self-reported confidence and a reward, but a raw
/// `Result::is_ok()` off `steps::act::ActOutcome::Acted` - did the tool
/// call this tick actually succeed, independent of whether anything was
/// predicted about it beforehand. Motor/proprioceptive signal, not
/// perceptual/predictive. See `steps::drives::DriveState::update`'s doc
/// comment for how the two feed the same `competence` pressure.
pub struct ExecutionTracker {
    window: RingBuffer<f32>,
}

impl ExecutionTracker {
    pub fn new(window_capacity: usize) -> Self {
        Self { window: RingBuffer::new(window_capacity) }
    }

    /// Records one resolved tool invocation: `true` if `ActOutcome::Acted`
    /// carried `Ok`, `false` if it carried `Err`.
    pub fn record(&mut self, succeeded: bool) {
        self.window.push(if succeeded { 0.0 } else { 1.0 });
    }

    /// The recent fraction of tool invocations that failed, in `[0, 1]`.
    /// `0.0` (neutral - assume competent) with fewer than two samples, the
    /// same "no evidence, no pressure" convention `CalibrationTracker::
    /// bias_for` already uses.
    pub fn failure_rate(&self) -> f32 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let values: Vec<f32> = self.window.iter().copied().collect();
        values.iter().sum::<f32>() / values.len() as f32
    }
}

impl Default for ExecutionTracker {
    fn default() -> Self {
        Self::new(DEFAULT_EXECUTION_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_util::EpochMillis;

    fn comparison_with_error(error_magnitude: f32) -> ComparisonResult {
        ComparisonResult { error_magnitude, precision: 1.0, precision_weighted_surprise: error_magnitude, epistemic_value: 1.0 }
    }

    #[test]
    fn reward_is_the_complement_of_error_magnitude() {
        assert!((reward_from_comparison(&comparison_with_error(0.3)) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn reward_is_clamped_to_zero_one() {
        assert_eq!(reward_from_comparison(&comparison_with_error(-1.0)), 1.0);
        assert_eq!(reward_from_comparison(&comparison_with_error(2.0)), 0.0);
    }

    fn outcome(channel: SourceChannel, interlocutor: Option<MentalObjectId>, created_at_ms: i64) -> PendingOutcome {
        PendingOutcome { channel, interlocutor, chunk_id: None, tier: Some(Tier::T3), reported_confidence: 0.8, created_at: EpochMillis(created_at_ms) }
    }

    #[test]
    fn resolve_returns_only_outcomes_matching_channel_and_interlocutor() {
        let mut registry = OutcomeRegistry::new(10);
        registry.register(outcome(SourceChannel::ConversationInput, None, 0));
        registry.register(outcome(SourceChannel::GoalDue, None, 0));

        let resolved = registry.resolve(SourceChannel::ConversationInput, None, EpochMillis(1));
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].channel, SourceChannel::ConversationInput);

        // The GoalDue outcome should still be pending, not consumed by the
        // ConversationInput resolve above.
        let resolved_goal = registry.resolve(SourceChannel::GoalDue, None, EpochMillis(1));
        assert_eq!(resolved_goal.len(), 1);
    }

    #[test]
    fn resolve_drops_expired_outcomes_without_returning_them() {
        let mut registry = OutcomeRegistry::new(2);
        registry.register(outcome(SourceChannel::ConversationInput, None, 0));

        // Too old (age 5ms > expiry 2ms) - dropped, not credited.
        let resolved = registry.resolve(SourceChannel::ConversationInput, None, EpochMillis(5));
        assert!(resolved.is_empty());

        // Confirm it's actually gone, not just skipped this once.
        registry.register(outcome(SourceChannel::ConversationInput, None, 5));
        let resolved_again = registry.resolve(SourceChannel::ConversationInput, None, EpochMillis(6));
        assert_eq!(resolved_again.len(), 1);
    }

    #[test]
    fn resolve_leaves_unmatched_outcomes_pending_for_a_later_tick() {
        let mut registry = OutcomeRegistry::new(10);
        registry.register(outcome(SourceChannel::ConversationInput, None, 0));

        let resolved_wrong_channel = registry.resolve(SourceChannel::Environment, None, EpochMillis(1));
        assert!(resolved_wrong_channel.is_empty());

        let resolved_right_channel = registry.resolve(SourceChannel::ConversationInput, None, EpochMillis(2));
        assert_eq!(resolved_right_channel.len(), 1);
    }

    #[test]
    fn calibration_defaults_to_neutral_with_no_history() {
        let tracker = CalibrationTracker::default();
        assert_eq!(tracker.calibrate(Tier::T3, 0.9), 0.9);
    }

    #[test]
    fn calibration_discounts_a_consistently_overconfident_tier() {
        let mut tracker = CalibrationTracker::default();
        // T3 keeps reporting 0.9 confidence but the outcomes only earn ~0.4
        // reward - a real, consistent overconfidence bias of ~0.5.
        for _ in 0..6 {
            tracker.record(Tier::T3, 0.9, 0.4);
        }
        let calibrated = tracker.calibrate(Tier::T3, 0.9);
        assert!(calibrated < 0.9, "an overconfident tier's reported confidence should be discounted, got {calibrated}");
        assert!((calibrated - 0.4).abs() < 1e-4, "calibrated confidence should track the learned bias closely, got {calibrated}");
    }

    #[test]
    fn calibration_boosts_a_consistently_underconfident_tier() {
        let mut tracker = CalibrationTracker::default();
        for _ in 0..6 {
            tracker.record(Tier::T3, 0.3, 0.8);
        }
        let calibrated = tracker.calibrate(Tier::T3, 0.3);
        assert!(calibrated > 0.3, "an underconfident tier's reported confidence should be boosted, got {calibrated}");
    }

    #[test]
    fn calibration_is_clamped_to_zero_one() {
        let mut tracker = CalibrationTracker::default();
        for _ in 0..6 {
            tracker.record(Tier::T3, 1.0, 0.0);
        }
        assert_eq!(tracker.calibrate(Tier::T3, 1.0), 0.0);
    }

    #[test]
    fn goal_progress_reward_is_full_when_an_active_goal_becomes_satisfied() {
        assert_eq!(goal_progress_reward(GoalStatus::Active, GoalStatus::Satisfied), Some(1.0));
    }

    #[test]
    fn goal_progress_reward_is_zero_when_an_active_goal_is_abandoned() {
        assert_eq!(goal_progress_reward(GoalStatus::Active, GoalStatus::Abandoned), Some(0.0));
    }

    #[test]
    fn goal_progress_reward_is_neutral_with_no_transition() {
        assert_eq!(goal_progress_reward(GoalStatus::Active, GoalStatus::Active), None);
    }

    #[test]
    fn goal_progress_reward_is_neutral_for_transitions_it_has_no_opinion_about() {
        assert_eq!(goal_progress_reward(GoalStatus::Active, GoalStatus::Suspended), None);
        assert_eq!(goal_progress_reward(GoalStatus::Suspended, GoalStatus::Satisfied), None);
    }

    fn goal_object(status: GoalStatus) -> aca_types::MentalObject {
        let mut object = aca_types::MentalObject::new_observation("a goal", aca_util::EpochMillis(0), 0.5);
        object.goal = Some(aca_types::GoalStackMembership {
            stack_id: aca_types::GoalStackId::new(),
            parent_goal_id: None,
            status,
            priority: 0.5,
        });
        object
    }

    #[test]
    fn detect_goal_progress_finds_no_reward_with_no_prior_history() {
        let mut graph = aca_graph::Graph::new();
        let goal = goal_object(GoalStatus::Active);
        let id = goal.id;
        graph.insert(goal);

        let (reward, current) = detect_goal_progress(&graph, &HashMap::new());
        assert!(reward.is_none(), "a goal seen for the first time has no 'previous' to diff against");
        assert_eq!(current.get(&id), Some(&GoalStatus::Active));
    }

    #[test]
    fn detect_goal_progress_finds_a_transition_from_the_previous_snapshot() {
        let mut graph = aca_graph::Graph::new();
        let goal = goal_object(GoalStatus::Satisfied);
        let id = goal.id;
        graph.insert(goal);

        let mut previous = HashMap::new();
        previous.insert(id, GoalStatus::Active);

        let (reward, current) = detect_goal_progress(&graph, &previous);
        assert_eq!(reward, Some(1.0));
        assert_eq!(current.get(&id), Some(&GoalStatus::Satisfied));
    }

    #[test]
    fn detect_goal_progress_ignores_non_goal_objects() {
        let mut graph = aca_graph::Graph::new();
        graph.insert(aca_types::MentalObject::new_observation("not a goal", aca_util::EpochMillis(0), 0.5));

        let (reward, current) = detect_goal_progress(&graph, &HashMap::new());
        assert!(reward.is_none());
        assert!(current.is_empty());
    }

    #[test]
    fn calibration_for_one_tier_does_not_affect_another() {
        let mut tracker = CalibrationTracker::default();
        for _ in 0..6 {
            tracker.record(Tier::T3, 0.9, 0.2);
        }
        // T4 has no history of its own - must stay neutral regardless of
        // how miscalibrated T3 has turned out to be.
        assert_eq!(tracker.calibrate(Tier::T4, 0.9), 0.9);
    }

    #[test]
    fn execution_tracker_defaults_to_neutral_with_no_history() {
        let tracker = ExecutionTracker::default();
        assert_eq!(tracker.failure_rate(), 0.0);
    }

    #[test]
    fn execution_tracker_waits_for_two_samples() {
        let mut tracker = ExecutionTracker::default();
        tracker.record(false);
        assert_eq!(tracker.failure_rate(), 0.0);
    }

    #[test]
    fn execution_tracker_reports_recent_failure_fraction() {
        let mut tracker = ExecutionTracker::default();
        tracker.record(false);
        tracker.record(true);
        tracker.record(false);
        tracker.record(true);
        assert!((tracker.failure_rate() - 0.5).abs() < 1e-6);
    }
}
