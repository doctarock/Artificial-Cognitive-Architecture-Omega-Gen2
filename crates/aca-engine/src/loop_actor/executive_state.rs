use crate::snapshot::{DisplacementSummary, OperatorProposalSummary, WorkingMemoryMember};
use crate::steps::metacognition::{CalibrationTracker, ExecutionTracker, OutcomeRegistry};

/// Executive's own decision/reward bookkeeping: the metacognitive trackers
/// that turn a resolved outcome into learned confidence calibration and
/// execution competence, plus the most recent tick's reportable
/// decision/displacement/attention state that `publish_snapshot` republishes
/// verbatim.
pub(crate) struct ExecutiveState {
    /// Decisions (a chunk's utility bias winning selection, a Tier 3/4
    /// self-reported confidence being trusted) awaiting the consequence that
    /// turns them into reward - see `steps::metacognition`.
    pub(crate) outcome_registry: OutcomeRegistry,
    /// Per-tier learned confidence-calibration bias, applied only to Tier
    /// 3/4 self-reports (Tier 1/2 already earn a trustworthy confidence
    /// structurally - see `CalibrationTracker`'s doc comment).
    pub(crate) calibration_tracker: CalibrationTracker,
    /// Rolling real tool-call success/failure rate for the executing half of
    /// competence.
    pub(crate) execution_tracker: ExecutionTracker,
    /// The most recently *selected* (not merely proposed) Executive
    /// operator - set only on `ExecutiveDecision::Selected`, so a
    /// self-assessment tool can report what Omega itself last actually
    /// decided to do, not just what's currently in Working Memory.
    /// Republished verbatim into `EngineSnapshot::last_operator_proposal` by
    /// `publish_snapshot`.
    pub(crate) last_operator_proposal: Option<OperatorProposalSummary>,
    /// The most recent tick's verified displacement claim - see
    /// `DisplacementSummary`'s own doc comment. Set from
    /// `steps::displacement::explain_release`'s `ReleaseReason::Displaced`
    /// case only, computed fresh every tick against that tick's real
    /// `raw_candidates`/`newly_admitted` - never carried forward on a tick
    /// with no competitive eviction, so a stale claim never lingers past the
    /// tick it was actually true of. Republished verbatim into
    /// `EngineSnapshot::last_displacement` by `publish_snapshot`.
    pub(crate) last_displacement: Option<DisplacementSummary>,
    /// This tick's real Coalition candidates that were attended (cleared
    /// `attention_threshold`, genuinely competed) but did not ignite (never
    /// made it into the fresh working memory) - see `EngineSnapshot::
    /// attended_not_ignited`'s own doc comment. Recomputed from scratch
    /// every tick immediately after Broadcast - never carried forward, same
    /// discipline as `last_displacement`.
    pub(crate) last_attended_not_ignited: Vec<WorkingMemoryMember>,
}

impl ExecutiveState {
    pub(crate) fn new(outcome_feedback_window_ms: i64) -> Self {
        Self {
            outcome_registry: OutcomeRegistry::new(outcome_feedback_window_ms),
            calibration_tracker: CalibrationTracker::default(),
            execution_tracker: ExecutionTracker::default(),
            last_operator_proposal: None,
            last_displacement: None,
            last_attended_not_ignited: Vec::new(),
        }
    }
}
