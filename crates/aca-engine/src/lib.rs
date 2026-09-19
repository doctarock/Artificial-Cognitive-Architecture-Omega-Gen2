//! The `CognitiveLoopActor` and the ten-step cognitive cycle: predict,
//! observe, compare, memory-dynamics update, coalition formation, GWT
//! broadcast, SOAR executive, act, learn, repeat. Built incrementally,
//! module by module, per the approved MVP plan.

mod attention_workspace;
pub mod coherence;
mod embedding_worker;
pub mod cognitive_core;
pub mod config;
pub mod loop_actor;
pub mod prompt_templates;
pub mod self_memory;
pub mod snapshot;
pub mod steps;

pub use cognitive_core::{reflect, CognitiveCoreOutput};
pub use config::{AblationConfig, LoopConfig, ResponsivenessConfig, TemperatureConfig};
pub use loop_actor::{CognitiveLoopActor, CuratedAnswerCommand, ExternalAgentInput, LoopHandles, OutcomeFeedbackCommand, ProcedureFeedbackCommand, RoomInput, SensorInput, SensorSignal};
pub use self_memory::{is_self_memory_seeded, seed_self_memory_objects, SELF_MEMORY_SEED_TEXTS};
pub use snapshot::{DisplacementSummary, EdgeSummary, EngineSnapshot, GoalSummary, LiveModelHandles, MemoryRoleCounts, ModelStatus, TierStatus, WorkingMemoryMember};
pub use steps::act::{act, ActOutcome, SpeechRenderPath};
pub use steps::arbitrate::{arbitrate_by_agreement, ArbitrationOutcome};
pub use steps::boredom::{generate as generate_boredom_stimulus, BoredomConfig, BoredomStimulus};
pub use steps::broadcast::{broadcast, decide_admission_deterministic, decide_admission_from_attention_model, decide_admission_with_hysteresis, BroadcastResult, DEFAULT_WORKING_MEMORY_CAPACITY};
pub use steps::coalition::{apply_crowding_normalization, attention_score, form_coalition, CoalitionCandidate};
pub use steps::communicative_intent::{CommunicativeIntentModelError, CommunicativeIntentPrediction, CommunicativeIntentSpecialist};
pub use steps::compare::{compare, ComparisonResult, PrecisionTracker, SourceChannel};
pub use steps::confidence_revision::{apply_contradiction_penalty, ConfidenceRevisionConfig};
pub use steps::displacement::{explain_release, verify_displacement, Displacement, ReleaseReason};
pub use steps::eligibility::{EdgeCredit, EligibilityConfig, EligibilityTraceRegistry};
pub use steps::executive::{
    impasse_response, propose_operators, resolve_confidence_impasse, select_operator, select_operator_sampled,
    spawn_subgoal, ExecutiveConfig, ExecutiveDecision, ImpasseKind, ImpasseResolution,
    ImpasseResponse, Operator, OperatorProposal, SamplingConfig,
};
pub use steps::knowledge_library::{consult, KnowledgeLibraryConfig, KnowledgeLibraryResult};
pub use steps::learn::{apply_learned_bias, chunk_resolution};
pub use steps::memory_formation::{
    form_memory, maybe_automatic_remember, reinforce_coactivation, MemoryFormationConfig, MemoryFormationOutcome,
};
pub use steps::metacognition::{reward_from_comparison, CalibrationTracker, ExecutionTracker, OutcomeRegistry, PendingOutcome};
pub use steps::observe::{new_observation_shell, resolve_embedding};
pub use steps::orient::{orient, OrientingConfig, OrientingResult};
pub use steps::predict::{predict_expected_embedding, LocalEventPrediction, LocalEventPredictor, PredictWeights, PredictionInputs};
pub use steps::predict::{predict_successor_object, LocalObjectPrediction};
pub use steps::predict::{predict_goal_impact, LocalGoalImpactPrediction};
pub use steps::procedural::{compiled_procedure, compiled_response, is_routine_stimulus, record_response, record_response_with_embedding,
    CompiledCondition, CompiledExecution, CompiledOperatorStep, CompiledProcedure, ExpectedConsequence};
pub use steps::procedural::innate_social_response;
pub use steps::orient_outcome::{OrientOutcomeModelError, OrientOutcomeSpecialist};
pub use steps::spikes::{SpikeEvent, SpikeEventQueue};
pub use steps::social_interface::render_speech;
pub use steps::known_answers::{CuratedAnswer, curated_answer_index, record_curated_answer, revoke_curated_answer, static_question_key};
pub use steps::synthesize::{synthesize, SynthesisConfig};
pub use steps::tools::{match_tool_intent, CurrentTimeTool, SelfStatusTool, Tool, ToolRegistry, ToolRiskTier};
pub use steps::tool_intent::{ToolIntentModelError, ToolIntentPrediction, ToolIntentSpecialist};

// Re-exported so downstream crates (aca-api, omega-acad) don't need a
// direct aca-store dependency just to name `CycleEvent` et al.
pub use aca_store::{CycleEvent, CycleEventKind, CyclePhase, KnowledgeLibraryStore, MemoryStore};
