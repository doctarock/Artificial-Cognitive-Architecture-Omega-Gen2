/// How strongly `AffectTracker::precision_gain` amplifies/dampens
/// precision-weighted surprise per unit of valence - see that method's own
/// doc comment. Kept modest: at the extremes (valence = +-1.0) this is
/// still only a 30% swing, not a signal capable of drowning out or
/// inventing surprise on its own.
const AFFECT_GAIN_STRENGTH: f32 = 0.3;

/// How much weight `update` gives `goal_progress_reward` relative to
/// `epistemic_reward` when both are present this tick - see `update`'s own
/// doc comment for why affect needs both at all. Weighted toward goal
/// progress, deliberately not 50/50: at even weighting, a goal-thwarting
/// tick that was also perfectly predicted would cancel out to exactly
/// neutral rather than reading as bad, which is a smaller fix than the one
/// this channel exists for (`steps::metacognition::goal_progress_reward`'s
/// own doc comment) - agency should be able to outweigh prediction accuracy
/// in how a tick actually felt, not just cancel it to a wash. A tuning
/// choice, not a theoretical commitment.
const GOAL_PROGRESS_WEIGHT: f32 = 0.7;

/// A persistent, slowly-moving read of "how has Omega's own predictive
/// model been doing lately" - Barrett's constructed-emotion theory and
/// Damasio's somatic-marker hypothesis, both read through the predictive-
/// processing account of interoception and affect (Clark/Hohwy) that
/// specs.md's Attention section already draws on for `epistemic_value`.
/// Built from exactly the reward signal `steps::metacognition::
/// reward_from_comparison` already computes every tick - no new inference,
/// no new I/O, just a slower-moving (EMA-smoothed) read of a number the
/// architecture already produces, the same "not a new signal" discipline
/// that module's own doc comment states for reuse elsewhere in this
/// engine.
///
/// This is deliberately *not* a Mental Object or a Self Memory belief: it's
/// a continuous scalar, actor-local and non-persisted (same category as
/// `PrecisionTracker`/`CalibrationTracker`), that modulates *how* Omega
/// attends rather than being something Omega thinks *about*. A future pass
/// could surface it as a genuine interoceptive Mental Object competing for
/// Working Memory in its own right, closer to what a felt mood actually is;
/// this is the minimal, honest version, the mechanism rather than the
/// self-representation of it.
#[derive(Debug, Clone, Copy)]
pub struct AffectTracker {
    /// Range `[-1.0, 1.0]`. `0.0` (neutral) until enough history
    /// accumulates to move it - never assumes good or bad standing by
    /// default.
    valence: f32,
    /// EMA smoothing factor in `(0.0, 1.0]` - how much weight one fresh
    /// sample gets against the running estimate. Small by design: affect
    /// should track a *trend* across many ticks, not whipsaw on every
    /// single observation the way `PrecisionTracker`'s per-channel rolling
    /// window can.
    smoothing: f32,
}

impl AffectTracker {
    pub fn new(smoothing: f32) -> Self {
        Self { valence: 0.0, smoothing }
    }

    pub fn valence(&self) -> f32 {
        self.valence
    }

    /// Folds one fresh tick's reward into the running valence estimate.
    /// Takes *two* independent reward channels, not one: `epistemic_reward`
    /// (`steps::metacognition::reward_from_comparison`'s `[0, 1]` output -
    /// `1.0` well-predicted, `0.0` maximally surprising) and
    /// `goal_progress_reward` (`steps::metacognition::goal_progress_reward`'s
    /// `Option<[0, 1]>` - `Some` only on a tick where a goal's status
    /// actually transitioned).
    ///
    /// Epistemic reward alone conflates *surprising* with *bad*: a
    /// well-predicted but goal-thwarting tick and a startling-but-welcome
    /// one would register identically under it. When goal-progress evidence
    /// exists this tick, it's blended in (`GOAL_PROGRESS_WEIGHT`) rather
    /// than substituted - "is my model accurate" and "am I getting what I
    /// want" are both real inputs to how a tick actually went, not one
    /// standing in for the other. On a tick with no goal transition at all
    /// (`None`, the overwhelmingly common case - nothing in this engine
    /// transitions a goal's status on every tick), this degrades exactly to
    /// the single-channel behavior this method used to have.
    ///
    /// Whichever combined value results is remapped to `[-1, 1]` before
    /// being folded in: a reward of exactly `0.5` ("about as well as usual")
    /// is affectively neutral, not positive, so only a genuine run of
    /// better- or worse-than-typical outcomes should move valence away from
    /// zero.
    pub fn update(&mut self, epistemic_reward: f32, goal_progress_reward: Option<f32>) {
        let combined = match goal_progress_reward {
            Some(goal_reward) => GOAL_PROGRESS_WEIGHT * goal_reward.clamp(0.0, 1.0) + (1.0 - GOAL_PROGRESS_WEIGHT) * epistemic_reward.clamp(0.0, 1.0),
            None => epistemic_reward,
        };
        let signed = (combined.clamp(0.0, 1.0) - 0.5) * 2.0;
        self.valence = super::drives::ema(self.smoothing, signed, self.valence);
    }

    /// The predictive-processing account of anxiety/mood made literal
    /// (specs.md's own framing: attention is precision-weighted prediction
    /// error): a recent run of surprising, poorly-predicted experience
    /// (negative valence) raises the gain applied to fresh prediction
    /// error - the same amplification anxiety produces on threat-salient
    /// error in the human literature, i.e. heightened vigilance; a recent
    /// run of well-predicted experience (positive valence) loosens it
    /// slightly. `1.0` (no modulation) at neutral valence, so a
    /// fresh/short-lived actor behaves exactly as it would with no affect
    /// mechanism at all until real history accumulates. Floored well above
    /// zero - affect modulates attention, it never zeroes out surprise
    /// entirely.
    pub fn precision_gain(&self) -> f32 {
        (1.0 - AFFECT_GAIN_STRENGTH * self.valence).max(0.1)
    }
}

impl Default for AffectTracker {
    fn default() -> Self {
        // A fresh sample moves valence by 10% of the way toward it - slow
        // enough that one surprising or one calm tick can't swing mood on
        // its own, fast enough that a genuine sustained run (the whole
        // point of this mechanism) is visible within a few dozen ticks
        // rather than hundreds.
        Self::new(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_neutral_with_no_history() {
        let tracker = AffectTracker::default();
        assert_eq!(tracker.valence(), 0.0);
        assert_eq!(tracker.precision_gain(), 1.0);
    }

    #[test]
    fn a_run_of_well_predicted_rewards_moves_valence_positive() {
        let mut tracker = AffectTracker::default();
        for _ in 0..50 {
            tracker.update(1.0, None);
        }
        assert!(tracker.valence() > 0.5, "valence should have moved strongly positive, got {}", tracker.valence());
    }

    #[test]
    fn a_run_of_surprising_rewards_moves_valence_negative() {
        let mut tracker = AffectTracker::default();
        for _ in 0..50 {
            tracker.update(0.0, None);
        }
        assert!(tracker.valence() < -0.5, "valence should have moved strongly negative, got {}", tracker.valence());
    }

    #[test]
    fn neutral_reward_of_one_half_does_not_move_valence() {
        let mut tracker = AffectTracker::default();
        for _ in 0..20 {
            tracker.update(0.5, None);
        }
        assert!((tracker.valence()).abs() < 1e-6, "a run of exactly-typical rewards should leave valence at zero, got {}", tracker.valence());
    }

    #[test]
    fn negative_valence_raises_precision_gain_above_one() {
        let mut tracker = AffectTracker::default();
        for _ in 0..50 {
            tracker.update(0.0, None);
        }
        assert!(tracker.precision_gain() > 1.0, "negative (surprising) valence should raise gain (vigilance), got {}", tracker.precision_gain());
    }

    #[test]
    fn positive_valence_lowers_precision_gain_below_one() {
        let mut tracker = AffectTracker::default();
        for _ in 0..50 {
            tracker.update(1.0, None);
        }
        assert!(tracker.precision_gain() < 1.0, "positive (well-predicted) valence should lower gain, got {}", tracker.precision_gain());
    }

    #[test]
    fn precision_gain_never_reaches_zero_even_at_maximal_negative_valence() {
        let mut tracker = AffectTracker::new(1.0); // fully overwrite each sample
        tracker.update(1.0, None); // valence == 1.0 exactly, worst case for the floor check below is -1.0
        // Drive valence hard toward -1.0.
        for _ in 0..10 {
            tracker.update(0.0, None);
        }
        assert!(tracker.valence() <= -0.99);
        assert!(tracker.precision_gain() >= 0.1, "gain must stay floored well above zero, got {}", tracker.precision_gain());
    }

    #[test]
    fn a_goal_thwarting_tick_reads_as_bad_even_when_well_predicted() {
        // The exact conflation `goal_progress_reward` exists to fix: a
        // perfectly-predicted (epistemic_reward = 1.0) tick that also
        // abandoned an Active goal (goal_progress_reward = Some(0.0))
        // should not read as affectively good just because it was expected.
        let mut tracker = AffectTracker::new(1.0); // full overwrite, so one call is legible
        tracker.update(1.0, Some(0.0));
        assert!(tracker.valence() < 0.0, "a goal-thwarting outcome should pull valence negative even when fully well-predicted, got {}", tracker.valence());
    }

    #[test]
    fn a_surprising_but_welcome_tick_reads_as_good_even_when_poorly_predicted() {
        let mut tracker = AffectTracker::new(1.0);
        tracker.update(0.0, Some(1.0));
        assert!(tracker.valence() > 0.0, "goal progress should pull valence positive even when maximally surprising, got {}", tracker.valence());
    }

    #[test]
    fn with_no_goal_transition_this_tick_only_epistemic_reward_moves_valence() {
        let mut tracker = AffectTracker::new(1.0);
        tracker.update(1.0, None);
        assert_eq!(tracker.valence(), 1.0, "None should degrade to exactly the old single-channel behavior");
    }
}
