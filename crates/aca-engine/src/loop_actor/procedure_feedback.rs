use std::collections::HashMap;

use aca_types::MentalObjectId;
use aca_util::EpochMillis;
use tokio::sync::mpsc;

use super::{OutcomeFeedbackCommand, ProcedureFeedbackCommand};

const MAX_PENDING_ATTEMPTS: usize = 256;
const ATTEMPT_EXPIRY_MS: i64 = 30 * 60 * 1000;
const MAX_COMPARED_OBSERVATIONS: usize = 2048;

pub(crate) struct ProcedureAttempt {
    pub(crate) stimulus: String,
    pub(crate) response: String,
    pub(crate) stimulus_embedding: Vec<f32>,
    pub(crate) reflected_then_spoke: bool,
    pub(crate) compiled_spoke: bool,
    pub(crate) reflected_then_asked: bool,
    pub(crate) compiled_asked: bool,
    pub(crate) reflected_then_ignored: bool,
    pub(crate) compiled_ignored: bool,
    pub(crate) at: EpochMillis,
}

/// Actor-owned provenance for outcome-verified macro compilation. Ordinary
/// input text cannot create a success event; only the host feedback channel
/// can resolve an *actual* previously spoken observation id. Neither a model
/// response nor a repeated utterance grades itself.
pub(crate) struct ProcedureFeedbackState {
    pub(crate) feedback_rx: mpsc::Receiver<ProcedureFeedbackCommand>,
    pub(crate) feedback_primed: Option<ProcedureFeedbackCommand>,
    pub(crate) outcome_rx: mpsc::Receiver<OutcomeFeedbackCommand>,
    pub(crate) outcome_primed: Option<OutcomeFeedbackCommand>,
    reflection_sources: HashMap<MentalObjectId, MentalObjectId>,
    attempts: HashMap<MentalObjectId, ProcedureAttempt>,
    compared_observations: HashMap<MentalObjectId, (EpochMillis, bool)>,
}

impl ProcedureFeedbackState {
    pub(crate) fn new(feedback_rx: mpsc::Receiver<ProcedureFeedbackCommand>, outcome_rx: mpsc::Receiver<OutcomeFeedbackCommand>) -> Self {
        Self { feedback_rx, feedback_primed: None, outcome_rx, outcome_primed: None,
            reflection_sources: HashMap::new(), attempts: HashMap::new(), compared_observations: HashMap::new() }
    }

    pub(crate) fn take_feedback(&mut self) -> Option<ProcedureFeedbackCommand> {
        self.feedback_primed.take().or_else(|| self.feedback_rx.try_recv().ok())
    }

    pub(crate) fn take_outcome_feedback(&mut self) -> Option<OutcomeFeedbackCommand> {
        self.outcome_primed.take().or_else(|| self.outcome_rx.try_recv().ok())
    }

    pub(crate) fn record_compared_observation(&mut self, observation_id: MentalObjectId, now: EpochMillis) {
        self.compared_observations.retain(|_, (at, _)| now.0.saturating_sub(at.0) <= ATTEMPT_EXPIRY_MS);
        if self.compared_observations.len() >= MAX_COMPARED_OBSERVATIONS {
            if let Some(oldest) = self.compared_observations.iter().min_by_key(|(_, (at, _))| at.0).map(|(id, _)| *id) {
                self.compared_observations.remove(&oldest);
            }
        }
        self.compared_observations.insert(observation_id, (now, false));
    }

    /// A host verdict is accepted once for a recent observation that really
    /// passed Compare. It is not inferred from the model's own text or from
    /// deterministic Working Memory admission.
    pub(crate) fn label_compared_observation(&mut self, observation_id: MentalObjectId, now: EpochMillis) -> bool {
        let Some((at, labeled)) = self.compared_observations.get_mut(&observation_id) else { return false };
        if *labeled || now.0.saturating_sub(at.0) > ATTEMPT_EXPIRY_MS { return false }
        *labeled = true;
        true
    }

    pub(crate) fn record_reflection(&mut self, reflection_id: MentalObjectId, source_id: MentalObjectId) {
        if self.reflection_sources.len() >= MAX_PENDING_ATTEMPTS {
            self.reflection_sources.clear(); // stale unspoken reflection provenance
        }
        self.reflection_sources.insert(reflection_id, source_id);
    }

    pub(crate) fn take_reflection_source(&mut self, reflection_id: MentalObjectId) -> Option<MentalObjectId> {
        self.reflection_sources.remove(&reflection_id)
    }

    pub(crate) fn record_attempt(&mut self, observation_id: MentalObjectId, attempt: ProcedureAttempt, now: EpochMillis) {
        self.attempts.retain(|_, pending| now.0.saturating_sub(pending.at.0) <= ATTEMPT_EXPIRY_MS);
        if self.attempts.len() >= MAX_PENDING_ATTEMPTS {
            if let Some(oldest) = self.attempts.iter().min_by_key(|(_, pending)| pending.at.0).map(|(id, _)| *id) {
                self.attempts.remove(&oldest);
            }
        }
        self.attempts.insert(observation_id, attempt);
    }

    pub(crate) fn take_attempt(&mut self, observation_id: MentalObjectId, now: EpochMillis) -> Option<ProcedureAttempt> {
        let attempt = self.attempts.remove(&observation_id)?;
        (now.0.saturating_sub(attempt.at.0) <= ATTEMPT_EXPIRY_MS).then_some(attempt)
    }
}
