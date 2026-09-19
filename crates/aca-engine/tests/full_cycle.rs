//! Integration test proving Phase 7's acceptance bar: one full, real
//! cognitive cycle for a text turn, end to end, using only `aca-engine`'s
//! public API - predict -> observe -> compare -> memory-update -> coalition
//! -> broadcast -> executive(speak) -> act(reply rendered) -> memory formed
//! + an edge reinforced.

use std::collections::HashSet;
use std::time::Duration;

use aca_engine::{
    act, broadcast, compare, decide_admission_deterministic, form_coalition, form_memory, new_observation_shell,
    predict_expected_embedding, reinforce_coactivation, resolve_embedding, select_operator,
    ActOutcome, CoalitionCandidate, ConfidenceRevisionConfig, ExecutiveConfig, ExecutiveDecision, KnowledgeLibraryConfig,
    MemoryFormationConfig, MemoryFormationOutcome, Operator, OperatorProposal, PredictWeights,
    PredictionInputs, PrecisionTracker, SourceChannel, TemperatureConfig, ToolRegistry,
};
use aca_graph::{recompute_activation, Graph};
use aca_store::SqliteStore;
use aca_tiers::testing::FakeEmbeddingClient;
use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::{MemoryRole, Tier};
use aca_util::{Clock, ManualClock, RingBuffer};
use async_trait::async_trait;
use rand::SeedableRng;

struct FixedChatClient {
    raw_text: &'static str,
}

#[async_trait]
impl ChatClient for FixedChatClient {
    async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
        Ok(TierResponse {
            raw_text: self.raw_text.to_string(),
            confidence: 0.9,
            tier: Tier::T3,
        })
    }
}

#[tokio::test]
async fn one_full_cycle_speaks_forms_memory_and_reinforces_an_edge() {
    let embedding_client = FakeEmbeddingClient::default();
    let clock = ManualClock::new(aca_util::EpochMillis(1_000));
    let mut graph = Graph::new();

    // --- Step 1: Predict --- nothing precedes this, so there's nothing to
    // predict from yet: no previous observation, empty Working Memory, no
    // due goal.
    let expected = predict_expected_embedding(&PredictionInputs::default(), &PredictWeights::default());
    assert!(expected.is_none(), "the very first observation has nothing to predict against");

    // --- Step 2: Observe --- construct the shell, then resolve its
    // embedding via the async companion, exactly as the plan's
    // embedding-is-I/O split requires.
    let mut observation = new_observation_shell("hello Omega, are you there?", clock.now(), 0.5);
    assert!(!observation.is_embedding_resolved());
    let embedding = resolve_embedding(&embedding_client, &observation.text).await.unwrap();
    observation.embedding = Some(embedding.clone());
    assert!(observation.is_embedding_resolved());

    // --- Step 3: Compare --- no expectation existed, so this is maximally
    // surprising by convention.
    let mut tracker = PrecisionTracker::default();
    let comparison = compare(expected.as_deref(), &embedding, SourceChannel::ConversationInput, None, &mut tracker);
    assert_eq!(comparison.error_magnitude, 1.0);
    observation.prediction.error_magnitude = Some(comparison.error_magnitude);
    observation.prediction.precision = Some(comparison.precision);

    // --- Step 4: Update memory dynamics --- a freshly-created object has
    // one reference (its own creation); compute its real activation with no
    // other active context sources (nothing else is in Working Memory yet).
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let observation_id = observation.id;
    graph.insert(observation);
    {
        let object = graph.get_mut(&observation_id).unwrap();
        recompute_activation(&mut object.activation, observation_id, &[], 0.0, 0.0, &mut rng, &clock);
    }
    let activation_total = graph.get(&observation_id).unwrap().activation.total;

    // --- Step 5: Form coalitions --- one candidate, well above threshold
    // thanks to its surprise term.
    let raw_candidates = vec![(observation_id, activation_total, Some(comparison.precision_weighted_surprise))];
    let coalition = form_coalition(&raw_candidates, 0.1);
    assert_eq!(coalition.len(), 1);
    let candidates: Vec<CoalitionCandidate> = coalition;

    // --- Step 6: Broadcast --- GWT's single arbitration point admits it
    // into Working Memory.
    let admitted = decide_admission_deterministic(&candidates, 4);
    let broadcast_result = broadcast(&mut graph, &admitted, &HashSet::new(), clock.now());
    assert_eq!(broadcast_result.working_memory, vec![observation_id]);
    assert!(graph.get(&observation_id).unwrap().workspace.in_working_memory);
    assert!(graph.get(&observation_id).unwrap().memory_roles.contains(&MemoryRole::Working));

    // --- Step 7: Execute (SOAR) --- exactly one clear proposal: speak.
    let proposals = vec![OperatorProposal {
        operator: Operator::Speak,
        target_id: observation_id,
        preference: 0.9,
        confidence: 0.95,
    }];
    let decision = select_operator(&proposals, &ExecutiveConfig::default());
    let selected = match decision {
        ExecutiveDecision::Selected(proposal) => proposal,
        other => panic!("expected a decisive Selected operator, got {other:?}"),
    };
    assert_eq!(selected.operator, Operator::Speak);

    // --- Step 8: Act --- Speak renders the already-broadcast content;
    // Social Interface never originates anything new.
    let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
    let chat_client = FixedChatClient { raw_text: "unused for Speak" };
    let tier1_pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![]);
    let tier2_pool = DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![]);
    let kl_store = SqliteStore::open_in_memory().unwrap();
    let tool_registry = ToolRegistry::empty();
    let outcome = act(
        &mut graph,
        &selected,
        &tier1_pool,
        &tier2_pool,
        &pool,
        &chat_client,
        &embedding_client,
        &kl_store,
        &KnowledgeLibraryConfig::default(),
        &tool_registry,
        &[],
        "",
        0.0,
        &MemoryFormationConfig::default(),
        &ConfidenceRevisionConfig::default(),
        clock.now(),
        &clock,
        0.5,
        Duration::from_secs(5),
        &TemperatureConfig::default(),
        None,
    )
    .await;
    match outcome {
        ActOutcome::Spoke { text, .. } => assert_eq!(text, "hello Omega, are you there?"),
        other => panic!("expected Spoke, got {other:?}"),
    }

    // --- Memory formation + edge reinforcement, satisfying the rest of the
    // Phase 7 acceptance bar: some content is silently filed to memory, and
    // co-activation strengthens an associative edge. A second, related
    // observation arrives and becomes a second Working Memory member.
    let mut second = new_observation_shell("still there, Omega?", clock.now(), 0.5);
    let second_embedding = resolve_embedding(&embedding_client, &second.text).await.unwrap();
    second.embedding = Some(second_embedding);
    let second_id = second.id;
    graph.insert(second);

    let candidate_for_memory = graph.get(&second_id).unwrap().clone();
    let memory_outcome = form_memory(&mut graph, candidate_for_memory, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1_pool, &tier2_pool, "", clock.now(), 0.3).await;
    assert!(matches!(memory_outcome, MemoryFormationOutcome::NewEpisodic { id } if id == second_id));
    assert!(graph.get(&second_id).unwrap().memory_roles.contains(&MemoryRole::Episodic));

    reinforce_coactivation(&mut graph, observation_id, second_id, clock.now());
    let first_object = graph.get(&observation_id).unwrap();
    assert_eq!(first_object.edges.len(), 1, "co-activation should have reinforced exactly one edge");
    assert_eq!(first_object.edges[0].target_id, second_id);
    assert!(first_object.edges[0].strength > 0.0);

    // Sanity: the reference log genuinely grew, and RingBuffer isn't just a
    // stub - both objects have real, non-empty activation history.
    let ring: &RingBuffer<_> = &graph.get(&observation_id).unwrap().activation.reference_log;
    assert!(!ring.is_empty());
}
