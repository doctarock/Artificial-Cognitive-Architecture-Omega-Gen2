use aca_types::{EdgeKind, GoalStatus, MentalObject, MentalObjectId, ObjectStatus};
use aca_util::{weighted_blend, EpochMillis};

use super::compare::{PrecisionTracker, SourceChannel};

/// The three sources Predict blends into an expectation, per specs.md's
/// "Expectation-Vector Prediction": the previous Observation, the current
/// Working Memory members, and any currently-due Goal. Any of them may be
/// absent (e.g. nothing is due, or this is the very first observation
/// ever) — absent sources simply drop out of the blend.
///
/// `previous_observation_embedding` is deliberately generic here even though
/// the caller (`loop_actor`) now keys it per `SourceChannel` rather than
/// keeping one global "last observation" slot — see that module's
/// `channel_embeddings` field. This function stays agnostic to *which*
/// channel supplied it; only the caller needs to know that.
#[derive(Debug, Default)]
pub struct PredictionInputs<'a> {
    pub previous_observation_embedding: Option<&'a [f32]>,
    pub working_memory_embeddings: Vec<&'a [f32]>,
    pub due_goal_embedding: Option<&'a [f32]>,
}

/// Working Memory's blend weight when derived via `PredictWeights::from_precision`
/// — a fixed baseline, not an adaptive estimate. Unlike `previous_observation`
/// and `due_goal`, Working Memory is a multi-object blend with no single
/// tracked channel of its own to derive a precision from; adapting it too
/// would mean inventing a second, different kind of estimate rather than
/// reusing `PrecisionTracker` as-is. Matches `PrecisionTracker::DEFAULT_PRECISION`'s
/// own "neutral, no evidence either way yet" value, so a cold-start blend
/// (no precision history anywhere) treats all three sources roughly evenly
/// rather than favoring one by construction.
pub const WORKING_MEMORY_BASELINE_WEIGHT: f32 = 1.0;

/// Blend weights — either the fixed `default()` (a tuning baseline, not a
/// theoretical commitment) or, in production, `from_precision` (see its own
/// doc comment): a source's weight tracks how reliable it's actually been,
/// not a hand-picked constant.
#[derive(Debug, Clone, Copy)]
pub struct PredictWeights {
    pub previous_observation: f32,
    pub working_memory: f32,
    pub due_goal: f32,
}

impl Default for PredictWeights {
    fn default() -> Self {
        Self {
            previous_observation: 0.4,
            working_memory: 0.4,
            due_goal: 0.2,
        }
    }
}

impl PredictWeights {
    /// Derives the blend's weights from each source's currently-tracked
    /// reliability (`compare::PrecisionTracker`) instead of a fixed split —
    /// predictive processing's own "higher-precision priors dominate the
    /// posterior" (specs.md's Predictive Processing section), applied to the
    /// generative model's blend itself rather than only to downstream
    /// attention. `weighted_blend` normalizes by total weight internally, so
    /// these raw precision values don't need to be pre-normalized to sum to
    /// anything in particular - only their values *relative to each other*
    /// matter.
    ///
    /// `incoming_channel` is this tick's actual observation source (known
    /// before Predict runs - see `loop_actor`'s Steps 1-3 ordering); its
    /// tracked precision governs `previous_observation`'s weight, since that
    /// slot is now keyed to the same channel (see `PredictionInputs`'s doc
    /// comment). `due_goal`'s weight comes from `SourceChannel::GoalDue`'s
    /// tracked precision, which stays at `PrecisionTracker`'s neutral
    /// default until a live pathway actually produces that channel (a
    /// separate, larger feature - not this one).
    pub fn from_precision(tracker: &PrecisionTracker, incoming_channel: SourceChannel) -> Self {
        Self {
            previous_observation: tracker.precision_for(incoming_channel),
            working_memory: WORKING_MEMORY_BASELINE_WEIGHT,
            due_goal: tracker.precision_for(SourceChannel::GoalDue),
        }
    }
}

/// Step 1 - Predict. Pure computation: no I/O, no LLM call. Returns `None`
/// when there is nothing at all to predict from (e.g. the very first
/// observation in a fresh graph) — Compare treats a missing expectation as
/// maximally surprising by convention, rather than this function inventing
/// a default vector.
pub fn predict_expected_embedding(
    inputs: &PredictionInputs,
    weights: &PredictWeights,
) -> Option<Vec<f32>> {
    let mut parts: Vec<(&[f32], f32)> = Vec::new();

    if let Some(prev) = inputs.previous_observation_embedding {
        parts.push((prev, weights.previous_observation));
    }
    if !inputs.working_memory_embeddings.is_empty() {
        let share = weights.working_memory / inputs.working_memory_embeddings.len() as f32;
        for embedding in &inputs.working_memory_embeddings {
            parts.push((embedding, share));
        }
    }
    if let Some(goal) = inputs.due_goal_embedding {
        parts.push((goal, weights.due_goal));
    }

    if parts.is_empty() {
        return None;
    }
    Some(weighted_blend(&parts))
}

/// specs.md's "currently-due Goal" input, made real: the currently `Active`
/// goal with the highest current ACT-R activation - the same competitive
/// currency Coalition itself uses to decide Working Memory admission,
/// applied here to the goal subset of the graph. `GoalStackMembership`
/// carries no deadline field at all (only `stack_id`/`parent_goal_id`/
/// `status`/`priority`), so "due" here means "most alive right now," not
/// "closest to a deadline" - inventing a deadline system would be new
/// mechanism this pass deliberately avoids, when activation already answers
/// the same question every other competition in this architecture asks.
/// Returns `None` when nothing in the graph is currently an `Active` goal.
pub fn most_salient_active_goal(graph: &aca_graph::Graph) -> Option<&MentalObject> {
    graph
        .goal_objects()
        .filter(|object| matches!(&object.goal, Some(membership) if membership.status == GoalStatus::Active))
        .max_by(|a, b| a.activation.total.partial_cmp(&b.activation.total).unwrap_or(std::cmp::Ordering::Equal))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalEventPrediction {
    pub expected_source: Option<SourceChannel>,
    /// Empirical P(actual source | previous source). This turns source
    /// surprise into a graded distributional error instead of a binary
    /// expected/unexpected flag once more than one successor has occurred.
    pub actual_source_probability: Option<f32>,
    pub expected_interval_ms: Option<f32>,
    pub expected_interval_stddev_ms: Option<f32>,
    pub expected_affect: Option<f32>,
    pub expected_affect_stddev: Option<f32>,
    pub source_surprise: f32,
    pub interval_error_ms: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningDistribution {
    mean: f32,
    variance_ema: f32,
    count: u64,
}

impl RunningDistribution {
    fn observe(&mut self, sample: f32, smoothing: f32) {
        if self.count == 0 {
            self.mean = sample;
            self.variance_ema = 0.0;
        } else {
            let delta = sample - self.mean;
            self.mean += smoothing * delta;
            self.variance_ema = (1.0 - smoothing) * (self.variance_ema + smoothing * delta * delta);
        }
        self.count += 1;
    }

    fn stddev(self) -> Option<f32> {
        (self.count >= 2).then(|| self.variance_ema.max(0.0).sqrt())
    }
}

/// Tiny online forward model for event shape rather than event prose. It
/// learns channel transitions, timing, and affect with counters/EMAs only;
/// no embedding or generative call is involved.
#[derive(Debug, Default)]
pub struct LocalEventPredictor {
    transitions: std::collections::HashMap<(SourceChannel, SourceChannel), u64>,
    interval_distribution: std::collections::HashMap<SourceChannel, RunningDistribution>,
    affect_distribution: std::collections::HashMap<SourceChannel, RunningDistribution>,
    last_event: Option<(SourceChannel, EpochMillis)>,
}

impl LocalEventPredictor {
    pub fn observe(&mut self, actual_source: SourceChannel, affect: f32, now: EpochMillis) -> LocalEventPrediction {
        let expected_source = self.last_event.and_then(|(previous, _)| self.most_likely_successor(previous));
        let actual_source_probability = self.last_event.and_then(|(previous, _)| self.successor_probability(previous, actual_source));
        let expected_interval = expected_source.and_then(|source| self.interval_distribution.get(&source).copied());
        let expected_affect_distribution = expected_source.and_then(|source| self.affect_distribution.get(&source).copied());
        let expected_interval_ms = expected_interval.map(|distribution| distribution.mean);
        let expected_interval_stddev_ms = expected_interval.and_then(RunningDistribution::stddev);
        let expected_affect = expected_affect_distribution.map(|distribution| distribution.mean);
        let expected_affect_stddev = expected_affect_distribution.and_then(RunningDistribution::stddev);
        let source_surprise = actual_source_probability.map_or(0.5, |probability| 1.0 - probability);
        let actual_interval = self.last_event.map(|(_, at)| at.elapsed_ms_until(now).max(0) as f32);
        let interval_error_ms = expected_interval_ms.zip(actual_interval).map(|(expected, actual)| (actual - expected).abs());

        if let Some((previous, _)) = self.last_event {
            *self.transitions.entry((previous, actual_source)).or_default() += 1;
        }
        if let Some(interval) = actual_interval {
            self.interval_distribution.entry(actual_source).or_default().observe(interval, 0.2);
        }
        self.affect_distribution.entry(actual_source).or_default().observe(affect.clamp(-1.0, 1.0), 0.1);
        self.last_event = Some((actual_source, now));

        LocalEventPrediction {
            expected_source,
            actual_source_probability,
            expected_interval_ms,
            expected_interval_stddev_ms,
            expected_affect,
            expected_affect_stddev,
            source_surprise,
            interval_error_ms,
        }
    }

    fn most_likely_successor(&self, previous: SourceChannel) -> Option<SourceChannel> {
        const CHANNELS: [SourceChannel; 5] = [
            SourceChannel::ConversationInput,
            SourceChannel::SelfGeneratedThought,
            SourceChannel::GoalDue,
            SourceChannel::ExternalKnowledge,
            SourceChannel::Environment,
        ];
        CHANNELS
            .into_iter()
            .enumerate()
            .filter_map(|(order, channel)| self.transitions.get(&(previous, channel)).copied().map(|count| (count, std::cmp::Reverse(order), channel)))
            .max_by_key(|(count, order, _)| (*count, *order))
            .map(|(_, _, channel)| channel)
    }

    fn successor_probability(&self, previous: SourceChannel, actual: SourceChannel) -> Option<f32> {
        let (actual_count, total) = self.transitions.iter()
            .filter(|((source, _), _)| *source == previous)
            .fold((0_u64, 0_u64), |(actual_count, total), ((_, target), count)| {
                (actual_count + if *target == actual { *count } else { 0 }, total + *count)
            });
        (total > 0).then(|| actual_count as f32 / total as f32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalObjectPrediction {
    pub expected_object: MentalObjectId,
    pub embedding_error: f32,
    pub edge_strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalGoalImpactPrediction {
    pub goal_id: MentalObjectId,
    /// Signed expected consequence: positive helps, negative thwarts.
    pub expected_impact: f32,
    pub evidence_strength: f32,
}

/// Below this magnitude, a goal's net signed edge strength is treated as no
/// real predicted consequence rather than a genuine (if tiny) one - guards
/// against near-perfectly-cancelling supporting/contradicting edges (e.g.
/// 0.7 vs -0.6999999) landing just shy of, rather than exactly at, zero and
/// still winning `max_by` over goals with no evidence at all.
const GOAL_IMPACT_NOISE_FLOOR: f32 = 1e-4;

/// Reads learned supportive/contradictory links from an object to active
/// goals. The strongest signed consequence is selected in O(out-degree)
/// time; this is a local forecast, not a causal proof.
pub fn predict_goal_impact(
    graph: &aca_graph::Graph,
    source_id: MentalObjectId,
    now: EpochMillis,
    edge_decay_rate_per_ms: f64,
) -> Option<LocalGoalImpactPrediction> {
    let source = graph.get(&source_id)?;
    let mut net = std::collections::HashMap::<MentalObjectId, f32>::new();
    for edge in &source.edges {
        let sign = match edge.kind {
            EdgeKind::Supports | EdgeKind::Causal => 1.0,
            EdgeKind::Contradicts | EdgeKind::Inhibitory => -1.0,
            _ => continue,
        };
        let Some(goal) = graph.get(&edge.target_id) else { continue };
        if goal.status != ObjectStatus::Active || !matches!(&goal.goal, Some(membership) if membership.status == GoalStatus::Active) { continue; }
        *net.entry(edge.target_id).or_default() += sign * aca_graph::effective_strength(edge, now, edge_decay_rate_per_ms);
    }
    let (goal_id, impact) = net.into_iter().filter(|(_, impact)| impact.abs() > GOAL_IMPACT_NOISE_FLOOR)
        .max_by(|(id_a, impact_a), (id_b, impact_b)| impact_a.abs().total_cmp(&impact_b.abs()).then_with(|| id_b.cmp(id_a)))?;
    Some(LocalGoalImpactPrediction { goal_id, expected_impact: impact.clamp(-1.0, 1.0), evidence_strength: impact.abs().clamp(0.0, 1.0) })
}

/// Predicts the next object from one active source's learned positive links.
/// This is an O(out-degree) Tier-0 operation, not a model invocation.
pub fn predict_successor_object(
    graph: &aca_graph::Graph,
    source_id: MentalObjectId,
    observed_embedding: &[f32],
    now: EpochMillis,
    edge_decay_rate_per_ms: f64,
) -> Option<LocalObjectPrediction> {
    let source = graph.get(&source_id)?;
    if observed_embedding.is_empty() { return None; }
    let mut net_strength = std::collections::HashMap::<MentalObjectId, f32>::new();
    for edge in &source.edges {
        let sign = match edge.kind {
            EdgeKind::Causal | EdgeKind::Associative | EdgeKind::Supports => 1.0,
            EdgeKind::Inhibitory | EdgeKind::Contradicts => -1.0,
            _ => continue,
        };
        let Some(target) = graph.get(&edge.target_id) else { continue };
        if target.status != ObjectStatus::Active
            || !target.embedding.as_ref().is_some_and(|embedding| embedding.len() == observed_embedding.len())
            || target.data.get("interlocutor_hint").is_some() {
            continue;
        }
        *net_strength.entry(edge.target_id).or_default() += sign * aca_graph::effective_strength(edge, now, edge_decay_rate_per_ms);
    }
    let (target, strength) = net_strength.into_iter()
        .filter(|(_, strength)| *strength > 0.0)
        .max_by(|(id_a, strength_a), (id_b, strength_b)| strength_a.total_cmp(strength_b).then_with(|| id_b.cmp(id_a)))?;
    let expected = graph.get(&target)?.embedding.as_deref()?;
    let embedding_error = (1.0 - aca_util::cosine_similarity(expected, observed_embedding)).clamp(0.0, 1.0);
    Some(LocalObjectPrediction { expected_object: target, embedding_error, edge_strength: strength })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sources_yields_no_prediction() {
        let inputs = PredictionInputs::default();
        assert!(predict_expected_embedding(&inputs, &PredictWeights::default()).is_none());
    }

    #[test]
    fn single_source_dominates_when_alone() {
        let prev = vec![1.0, 0.0];
        let inputs = PredictionInputs {
            previous_observation_embedding: Some(&prev),
            ..Default::default()
        };
        let predicted = predict_expected_embedding(&inputs, &PredictWeights::default()).unwrap();
        assert!((predicted[0] - 1.0).abs() < 1e-5);
        assert!((predicted[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn multiple_working_memory_members_share_their_weight_evenly() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let inputs = PredictionInputs {
            working_memory_embeddings: vec![&a, &b],
            ..Default::default()
        };
        let predicted = predict_expected_embedding(&inputs, &PredictWeights::default()).unwrap();
        // both share the full working_memory weight equally -> equal contribution
        assert!((predicted[0] - predicted[1]).abs() < 1e-5);
    }

    #[test]
    fn all_three_sources_blend_together() {
        let prev = vec![1.0, 0.0, 0.0];
        let wm = vec![0.0, 1.0, 0.0];
        let goal = vec![0.0, 0.0, 1.0];
        let inputs = PredictionInputs {
            previous_observation_embedding: Some(&prev),
            working_memory_embeddings: vec![&wm],
            due_goal_embedding: Some(&goal),
        };
        let weights = PredictWeights::default();
        let predicted = predict_expected_embedding(&inputs, &weights).unwrap();
        assert!((predicted[0] - weights.previous_observation).abs() < 1e-5);
        assert!((predicted[1] - weights.working_memory).abs() < 1e-5);
        assert!((predicted[2] - weights.due_goal).abs() < 1e-5);
    }

    #[test]
    fn from_precision_defaults_to_the_neutral_baseline_with_no_history() {
        let tracker = PrecisionTracker::default();
        let weights = PredictWeights::from_precision(&tracker, SourceChannel::ConversationInput);
        // No history anywhere yet - every source should land at the same
        // neutral weight, matching `PrecisionTracker::DEFAULT_PRECISION`.
        assert!((weights.previous_observation - WORKING_MEMORY_BASELINE_WEIGHT).abs() < 1e-5);
        assert!((weights.due_goal - WORKING_MEMORY_BASELINE_WEIGHT).abs() < 1e-5);
        assert_eq!(weights.working_memory, WORKING_MEMORY_BASELINE_WEIGHT);
    }

    #[test]
    fn from_precision_favors_a_more_reliable_incoming_channel() {
        let mut tracker = PrecisionTracker::default();
        // ConversationInput has been noisy; Environment has been highly
        // consistent - Environment should earn a larger blend weight when
        // it's the incoming channel.
        for e in [0.1, 0.9, 0.2, 0.8, 0.15, 0.85] {
            tracker.record_error(SourceChannel::ConversationInput, e);
        }
        for e in [0.5, 0.51, 0.49, 0.5, 0.505, 0.495] {
            tracker.record_error(SourceChannel::Environment, e);
        }

        let noisy_weights = PredictWeights::from_precision(&tracker, SourceChannel::ConversationInput);
        let consistent_weights = PredictWeights::from_precision(&tracker, SourceChannel::Environment);
        assert!(
            consistent_weights.previous_observation > noisy_weights.previous_observation,
            "a more reliable incoming channel should earn a larger blend weight"
        );
    }

    fn active_goal(activation_total: f32, embedding: Vec<f32>) -> MentalObject {
        let mut object = MentalObject::new_observation("a goal", aca_util::EpochMillis(0), 0.5);
        object.embedding = Some(embedding);
        object.activation.total = activation_total;
        object.goal = Some(aca_types::GoalStackMembership {
            stack_id: aca_types::GoalStackId::new(),
            parent_goal_id: None,
            status: GoalStatus::Active,
            priority: 0.5,
        });
        object
    }

    #[test]
    fn most_salient_active_goal_picks_the_highest_activation_among_active_goals() {
        let mut graph = aca_graph::Graph::new();
        graph.insert(active_goal(1.0, vec![1.0, 0.0]));
        let winner = active_goal(5.0, vec![0.0, 1.0]);
        let winner_id = winner.id;
        graph.insert(winner);

        let salient = most_salient_active_goal(&graph).expect("an active goal exists");
        assert_eq!(salient.id, winner_id);
    }

    #[test]
    fn most_salient_active_goal_ignores_non_active_goals_even_with_higher_activation() {
        let mut graph = aca_graph::Graph::new();
        let mut suspended = active_goal(100.0, vec![1.0, 0.0]);
        suspended.goal.as_mut().unwrap().status = GoalStatus::Suspended;
        graph.insert(suspended);
        let active = active_goal(1.0, vec![0.0, 1.0]);
        let active_id = active.id;
        graph.insert(active);

        let salient = most_salient_active_goal(&graph).expect("the active goal should still be found");
        assert_eq!(salient.id, active_id);
    }

    #[test]
    fn most_salient_active_goal_is_none_with_no_active_goals_in_the_graph() {
        let mut graph = aca_graph::Graph::new();
        graph.insert(MentalObject::new_observation("not a goal", aca_util::EpochMillis(0), 0.5));
        assert!(most_salient_active_goal(&graph).is_none());
    }

    #[test]
    fn local_event_model_learns_source_timing_and_affect_without_embeddings() {
        let mut model = LocalEventPredictor::default();
        model.observe(SourceChannel::ConversationInput, 0.2, EpochMillis(0));
        model.observe(SourceChannel::Environment, -0.5, EpochMillis(100));
        model.observe(SourceChannel::ConversationInput, 0.2, EpochMillis(200));
        let prediction = model.observe(SourceChannel::Environment, -0.5, EpochMillis(300));
        assert_eq!(prediction.expected_source, Some(SourceChannel::Environment));
        assert_eq!(prediction.source_surprise, 0.0);
        assert_eq!(prediction.actual_source_probability, Some(1.0));
        assert_eq!(prediction.expected_interval_ms, Some(100.0));
        assert_eq!(prediction.expected_interval_stddev_ms, None);
        assert_eq!(prediction.expected_affect, Some(-0.5));
        assert_eq!(prediction.expected_affect_stddev, None);
        assert_eq!(prediction.interval_error_ms, Some(0.0));
    }

    #[test]
    fn unexpected_source_is_maximally_surprising_after_a_pattern_is_learned() {
        let mut model = LocalEventPredictor::default();
        model.observe(SourceChannel::ConversationInput, 0.0, EpochMillis(0));
        model.observe(SourceChannel::Environment, 0.0, EpochMillis(10));
        model.observe(SourceChannel::ConversationInput, 0.0, EpochMillis(20));
        let prediction = model.observe(SourceChannel::GoalDue, 0.0, EpochMillis(30));
        assert_eq!(prediction.expected_source, Some(SourceChannel::Environment));
        assert_eq!(prediction.source_surprise, 1.0);
        assert_eq!(prediction.actual_source_probability, Some(0.0));
    }

    #[test]
    fn source_surprise_uses_the_learned_successor_distribution() {
        let mut model = LocalEventPredictor::default();
        for source in [
            SourceChannel::ConversationInput,
            SourceChannel::Environment,
            SourceChannel::ConversationInput,
            SourceChannel::Environment,
            SourceChannel::ConversationInput,
            SourceChannel::GoalDue,
            SourceChannel::ConversationInput,
        ] {
            model.observe(source, 0.0, EpochMillis(0));
        }
        let prediction = model.observe(SourceChannel::Environment, 0.0, EpochMillis(0));
        assert_eq!(prediction.expected_source, Some(SourceChannel::Environment));
        assert!((prediction.actual_source_probability.unwrap() - 2.0 / 3.0).abs() < 1e-6);
        assert!((prediction.source_surprise - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn successor_prediction_uses_learned_links_and_ignores_inhibition() {
        let mut graph = aca_graph::Graph::new();
        let mut source = MentalObject::new_observation("source", EpochMillis(0), 0.5);
        let mut expected = MentalObject::new_observation("expected", EpochMillis(0), 0.5);
        expected.embedding = Some(vec![1.0, 0.0]);
        let mut suppressed = MentalObject::new_observation("suppressed", EpochMillis(0), 0.5);
        suppressed.embedding = Some(vec![0.0, 1.0]);
        let (source_id, expected_id, suppressed_id) = (source.id, expected.id, suppressed.id);
        aca_graph::reinforce_edge(&mut source.edges, expected_id, EdgeKind::Associative, EpochMillis(0), 0.6, 1.0);
        aca_graph::reinforce_edge(&mut source.edges, suppressed_id, EdgeKind::Inhibitory, EpochMillis(0), 0.9, 1.0);
        aca_graph::reinforce_edge(&mut source.edges, suppressed_id, EdgeKind::Associative, EpochMillis(0), 0.3, 1.0);
        for object in [source, expected, suppressed] { graph.insert(object); }
        let result = predict_successor_object(&graph, source_id, &[1.0, 0.0], EpochMillis(0), 0.0).unwrap();
        assert_eq!(result.expected_object, expected_id);
        assert!(result.embedding_error < 1e-6);
        let unexpected = predict_successor_object(&graph, source_id, &[0.0, 1.0], EpochMillis(0), 0.0).unwrap();
        assert_eq!(unexpected.embedding_error, 1.0);
    }

    #[test]
    fn goal_impact_prediction_reads_signed_outcome_links_only_for_active_goals() {
        let mut graph = aca_graph::Graph::new();
        let mut source = MentalObject::new_observation("decision", EpochMillis(0), 0.5);
        let mut goal = MentalObject::new_observation("goal", EpochMillis(0), 0.5);
        goal.goal = Some(aca_types::GoalStackMembership {
            stack_id: aca_types::GoalStackId::new(), parent_goal_id: None,
            status: GoalStatus::Active, priority: 1.0,
        });
        let (source_id, goal_id) = (source.id, goal.id);
        aca_graph::reinforce_edge(&mut source.edges, goal_id, EdgeKind::Contradicts, EpochMillis(0), 0.7, 1.0);
        for object in [source, goal] { graph.insert(object); }
        let prediction = predict_goal_impact(&graph, source_id, EpochMillis(0), 0.0).unwrap();
        assert_eq!(prediction.goal_id, goal_id);
        assert!((prediction.expected_impact + 0.7).abs() < 1e-6);
        graph.get_mut(&goal_id).unwrap().goal.as_mut().unwrap().status = GoalStatus::Satisfied;
        assert!(predict_goal_impact(&graph, source_id, EpochMillis(1), 0.0).is_none());
    }
}
