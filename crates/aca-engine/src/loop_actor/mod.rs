use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use aca_graph::{recompute_activation, reinforce_edge, Graph, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH};
use aca_store::{CycleEvent, CycleEventKind, CyclePhase, DirtyBatch, KnowledgeLibraryStore, MemoryStore};
use aca_tiers::{AttentionClient, ChatClient, DivergentPool, EmbeddingClient, GenerateRequest, TierPool};
use aca_types::{AssociativeEdge, EdgeKind, MemoryRole, MentalObject, MentalObjectId};
use aca_util::{Clock, EpochMillis};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;
use tokio::sync::{broadcast, mpsc, watch};

use crate::config::LoopConfig;
use crate::snapshot::{EdgeSummary, EngineSnapshot, GoalSummary, LiveModelHandles, MemoryRoleCounts, ProvisionalAbstractionSummary, WorkingMemoryMember};
use crate::steps::act::act;
use crate::steps::compare::{compare, PrecisionTracker, SourceChannel};
use crate::steps::executive::{
    has_reflection_for, impasse_response, operator_keyword, parse_operator_keyword, propose_operators, resolve_confidence_impasse, select_operator_sampled,
    spawn_subgoal, ExecutiveDecision, ImpasseResolution, ImpasseResponse, OperatorProposal,
};
use crate::steps::learn::{apply_learned_bias, chunk_resolution, reinforce_chunk_utility};
use crate::steps::memory_formation::{reinforce_coactivation, MemoryFormationOutcome};
use crate::steps::affect::AffectTracker;
use crate::steps::metacognition::{detect_goal_progress, goal_progress_reward as reward_for_goal_transition, reward_from_comparison, PendingOutcome};
use crate::embedding_worker::{run_embedding_worker, PendingObservation};
use crate::steps::observe::new_observation_shell;
use crate::steps::predict::{most_salient_active_goal, predict_expected_embedding, PredictWeights, PredictionInputs};
use crate::steps::tools::ToolRegistry;
use crate::steps::broadcast::{attention_decision_target_is_eligible, broadcast as broadcast_step, decide_admission_from_attention_model, decide_admission_with_hysteresis};
use crate::attention_workspace::{build_attention_workspace_json, workspace_in_distribution};
use crate::prompt_templates::attention_workspace_prompt;

mod model_coordinator;
use model_coordinator::ModelCoordinator;
mod drive_system;
use drive_system::DriveSystem;
mod telemetry_sink;
use telemetry_sink::TelemetrySink;
mod turn_latency;
use turn_latency::ForegroundTurnTracker;
mod procedure_feedback;
use procedure_feedback::{ProcedureAttempt, ProcedureFeedbackState};
#[cfg(test)]
use telemetry_sink::log_signature;
mod prediction_system;
use prediction_system::PredictionSystem;
mod executive_state;
use executive_state::ExecutiveState;
mod memory_coordinator;
use memory_coordinator::MemoryCoordinator;
mod perception_state;
use perception_state::PerceptionState;
mod scheduler;
use scheduler::CognitiveScheduler;

/// One turn of input from another agent on the household network (an MCP
/// `send_input` call), kept distinct from `input_tx`'s plain `String`
/// specifically so it can carry `agent_id` for provenance tagging without
/// touching the existing human/voice/API input path at all.
pub struct ExternalAgentInput {
    pub text: String,
    pub agent_id: String,
}

/// One observation forwarded by an external perception service (a camera
/// service reporting a scene change, and any future non-conversational
/// sensor) - as opposed to `input_tx`/`external_agent_input_tx`, which are
/// always conversational turns. `channel` is caller-chosen rather than
/// hardcoded to `SourceChannel::Environment` so a future service that
/// genuinely produces conversational content (unlikely, but the point of
/// `SourceChannel` is per-source reliability tracking, not a fixed
/// conversation/environment binary) isn't forced into the wrong bucket.
/// `source` is free-form provenance (e.g. "camera"), mirroring the
/// `data`/`data_tag` json pattern the other `Incoming` arms already use.
/// `entity_label` is `RoomInput::speaker_label`'s exact counterpart for a
/// non-conversational sensor - `Some` only when the sensor itself already
/// resolved this observation to a confidently *recognized*, named identity
/// (e.g. a face match against an enrollment gallery), never a raw/anonymous
/// track id. It is folded into the same `data_tag["speaker_label"]` key
/// `Incoming::Room` already sets, which is what lets it reach
/// `steps::interlocutor::find_or_create` - see that function's doc comment.
/// Deliberately the *same* key rather than a parallel `entity_label`-keyed
/// resolution path: a person recognized by voice and by camera under the
/// same enrolled name should land on the very same interlocutor node, not
/// two - see `docs/video-interaction-plan.md` section 4. Callers must apply
/// their own confidence bar before setting this (see
/// `omega-acad::video::VideoClient`'s `LABEL_TRUST_THRESHOLD`) - a bad
/// identity claim here contaminates a shared, persistent memory node, not
/// just one observation (see that plan's section 8.4).
pub struct SensorInput {
    pub text: String,
    pub channel: SourceChannel,
    pub source: &'static str,
    pub entity_label: Option<String>,
}

/// A sensor-owned numerical reflex signal. The producer, not a language
/// model, is responsible for calibrating threat and urgency in [0, 1].
/// An optional already-computed embedding lets a trusted local sensor avoid
/// the embedding worker on urgent events.
pub struct SensorSignal {
    pub text: String,
    pub channel: SourceChannel,
    pub source: &'static str,
    pub entity_label: Option<String>,
    pub threat: f32,
    pub urgency: f32,
    pub embedding: Option<Vec<f32>>,
}

/// Host-only curation command. Ordinary text/agent/sensor inputs cannot
/// enter this channel; the owning application must supply a vetted answer
/// and a finite question embedding explicitly.
pub enum CuratedAnswerCommand {
    Upsert { question: String, answer: String, question_embedding: Vec<f32> },
    Revoke { question: String },
}

/// Host-only evaluation of a previously observed terminal turn. The id is
/// published on its Speak, Ask, or Ignore Act event; text, agent, and sensor ingress
/// cannot fabricate feedback on this channel. A successful label credits only
/// an actual allowlisted two-step provenance chain, never an output alone.
pub struct ProcedureFeedbackCommand {
    pub observation_id: MentalObjectId,
    pub successful: bool,
}

/// Host-only judgment of a real observed consequence, including a useful
/// or harmful non-response. This does not itself authorize a procedural
/// macro; it supplies independent local labels for future specialist data.
pub struct OutcomeFeedbackCommand {
    pub observation_id: MentalObjectId,
    pub successful: bool,
}

/// One utterance heard from a shared room-audio feed (as opposed to
/// `input_tx`'s plain `String`, which is a single-party channel - stdin, the
/// local HTTP API - with no "who said this" to preserve). `stream_id` is the
/// voice service's own diarization track id, kept regardless of whether the
/// speaker is recognized, purely so same-tick backlog from *different*
/// speakers never gets coalesced into one blended Observation (see
/// `tick()`'s `pending_room_input` handling) - it is never treated as a
/// persistent identity.
///
/// `speaker_label` is `Some` only for a *named/enrolled* speaker - never for
/// the voice service's own ephemeral "Unknown speaker N" diarization slot
/// (that filtering happens at the producer, e.g. `omega-acad`'s voice
/// client, which is the one place that actually sees the raw label). See
/// `steps::interlocutor`'s doc comment for why that distinction matters:
/// confirmed live, an unenrolled slot's numbering gets reassigned to
/// different real voices over a session, so it is never safe to anchor a
/// persistent memory to.
pub struct RoomInput {
    pub text: String,
    pub speaker_label: Option<String>,
    pub stream_id: String,
}

/// Channels handed back to whoever constructs a `CognitiveLoopActor`: a
/// sender for submitting external text input, a sender for input from other
/// agents (tagged with their `agent_id`, otherwise treated identically - no
/// privilege), a sender for observations from external sensing services
/// (tagged by `SourceChannel`, never treated as a conversational turn), a
/// sender for room-audio utterances (tagged with stream/speaker provenance,
/// see `RoomInput`), a host-only curation sender, a receiver for the live `cycle_events` feed (one per
/// subscriber via `.resubscribe()`), and a receiver for the always-latest
/// Working Memory snapshot.
pub struct LoopHandles {
    pub input_tx: mpsc::Sender<String>,
    pub external_agent_input_tx: mpsc::Sender<ExternalAgentInput>,
    pub sensor_input_tx: mpsc::Sender<SensorInput>,
    pub sensor_signal_tx: mpsc::Sender<SensorSignal>,
    pub curated_answer_tx: mpsc::Sender<CuratedAnswerCommand>,
    pub procedure_feedback_tx: mpsc::Sender<ProcedureFeedbackCommand>,
    pub outcome_feedback_tx: mpsc::Sender<OutcomeFeedbackCommand>,
    pub room_input_tx: mpsc::Sender<RoomInput>,
    pub events_rx: broadcast::Receiver<CycleEvent>,
    pub snapshot_rx: watch::Receiver<EngineSnapshot>,
    /// Live model-tier state, readable independent of `snapshot_rx`'s own
    /// tick-cadence publishes - see `LiveModelHandles`'s doc comment for why
    /// the two are deliberately not the same channel.
    pub live_models: LiveModelHandles,
}

/// The continuous cognitive cycle driver. Owns the in-memory activation
/// graph, Working Memory membership, and the model-tier clients — no
/// `Arc<Mutex<_>>` sharing of the graph itself; external access is only
/// through the channels in `LoopHandles`, per the build plan's
/// single-owner-actor concurrency model.
pub struct CognitiveLoopActor {
    /// The shared cognitive substrate - see `MemoryCoordinator`'s own doc
    /// comment.
    memory: MemoryCoordinator,
    /// The last-seen embedding *per source channel* (specs.md's "the
    /// conversation, the environment, and itself" as genuinely separate
    /// generative streams, not one global slot every observation
    /// overwrites) - see `steps::predict::PredictWeights::from_precision`'s
    /// doc comment for why keying on channel, not just remembering "the
    /// last thing that happened," is what keeps a surprising environment
    /// reading from corrupting the model's expectation for the next
    /// conversational turn.
    /// Everything Predict/Observe/Compare need to form an expectation, embed
    /// fresh input, and track how reliable each generative stream has been -
    /// see `PredictionSystem`'s own doc comment.
    prediction: PredictionSystem,
    /// Executive's own decision/reward bookkeeping - see `ExecutiveState`'s
    /// own doc comment.
    executive: ExecutiveState,
    /// The continuous cognitive pressures driving Agenda's spawn decisions -
    /// see `DriveSystem`'s own doc comment. Actor-local, non-persisted, same
    /// category as `affect_tracker`.
    drives: DriveSystem,
    /// Every goal-carrying object's status as of the end of the previous
    /// tick - `steps::metacognition::detect_goal_progress`'s `previous`
    /// argument, diffed against the graph's current goal statuses each tick
    /// to notice a transition (e.g. `Active` -> `Satisfied`) the instant it
    /// happens, regardless of what caused it. Actor-local, non-persisted,
    /// same as `precision_tracker` - losing this across a restart just means
    /// the first tick after restart can't detect a transition that happened
    /// before it (nothing to diff against yet), not that any transition is
    /// silently misread.
    previous_goal_statuses: HashMap<MentalObjectId, aca_types::GoalStatus>,
    rng: StdRng,
    clock: Arc<dyn Clock>,
    config: LoopConfig,

    /// Every substrate model client/pool this actor coordinates, plus the
    /// Omega Attention model's optional endpoint and its reconciliation
    /// bookkeeping - see `ModelCoordinator`'s own doc comment.
    model: ModelCoordinator,

    tool_registry: ToolRegistry,
    /// This actor's own event bus and `tracing` mirror - see
    /// `TelemetrySink`'s own doc comment.
    telemetry: TelemetrySink,
    /// Every raw-input channel, the room-audio backlog, the Act/Knowledge-
    /// Library reentry loopbacks, and the boredom/idle timers - see
    /// `PerceptionState`'s own doc comment.
    perception: PerceptionState,
    turn_latency: ForegroundTurnTracker,
    procedure_feedback: ProcedureFeedbackState,
    orient_outcome_specialist: Option<crate::steps::orient_outcome::OrientOutcomeSpecialist>,
    communicative_intent_specialist: Option<crate::steps::communicative_intent::CommunicativeIntentSpecialist>,

    cycle_seq: u64,
    /// Wall-clock cadence for the periodic write-behind flush - see
    /// `LoopConfig::flush_interval_ms`'s doc comment for why this is
    /// elapsed time rather than a tick count. `None` means "never flushed
    /// yet," which is always due.
    last_flush_at: Option<EpochMillis>,
}

impl CognitiveLoopActor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: LoopConfig,
        embedding_client: Arc<dyn EmbeddingClient>,
        chat_client: Arc<dyn ChatClient>,
        tier1_pool: DivergentPool,
        tier2_pool: DivergentPool,
        tier3_pool: TierPool,
        tier4_pool: TierPool,
        tier4_client: Arc<dyn ChatClient>,
        store: Arc<dyn MemoryStore>,
        kl_store: Arc<dyn KnowledgeLibraryStore>,
        tool_registry: ToolRegistry,
        clock: Arc<dyn Clock>,
        initial_objects: Vec<MentalObject>,
    ) -> (Self, LoopHandles) {
        let mut graph = Graph::new();
        for object in initial_objects {
            graph.insert(object);
        }
        // `MemoryCoordinator::new` computes an initial `self_summary` up
        // front, so a freshly-started actor's prompts carry real Self
        // Memory content (restored from the durable store, if any) from the
        // very first call, not just from whatever gets written after the
        // first Self Memory mutation in a live run.
        let memory = MemoryCoordinator::new(graph, store, kl_store);
        let outcome_feedback_window_ms = config.outcome_feedback_window_ms;

        let (input_tx, input_rx) = mpsc::channel(64);
        let (external_agent_input_tx, external_agent_input_rx) = mpsc::channel(64);
        let (sensor_input_tx, sensor_input_rx) = mpsc::channel(64);
        let (sensor_signal_tx, sensor_signal_rx) = mpsc::channel(64);
        let (curated_answer_tx, curated_answer_rx) = mpsc::channel(16);
        let (procedure_feedback_tx, procedure_feedback_rx) = mpsc::channel(16);
        let (outcome_feedback_tx, outcome_feedback_rx) = mpsc::channel(16);
        let (room_input_tx, room_input_rx) = mpsc::channel(64);
        // Actor-generated, low-volume by construction (at most one Consult
        // per tick) - a much smaller capacity than the externally-driven
        // channels above is deliberate, not an oversight.
        let (kl_reentry_tx, kl_reentry_rx) = mpsc::channel(16);
        let (act_reentry_tx, act_reentry_rx) = mpsc::channel(16);
        // Same low-volume reasoning as the KL/Act reentry channels above -
        // at most one fresh observation is ever enqueued per tick.
        let (embedding_request_tx, embedding_request_rx) = mpsc::channel(16);
        let (embedding_reentry_tx, embedding_reentry_rx) = mpsc::channel(16);
        tokio::spawn(run_embedding_worker(embedding_request_rx, embedding_reentry_tx, embedding_client.clone()));
        let (events_tx, events_rx) = broadcast::channel(256);
        let (snapshot_tx, snapshot_rx) = watch::channel(EngineSnapshot::default());
        // `self_status` needs a receiver of this actor's own snapshot
        // channel, which doesn't exist until the line above runs - see
        // `ToolRegistry::with_self_status`'s doc comment for why this can't
        // be included in whatever registry the caller passed in - same
        // reasoning applies to `with_abstraction_status`, chained here for
        // the identical reason.
        let tool_registry = tool_registry.with_self_status(&snapshot_tx).with_abstraction_status(&snapshot_tx);

        let embedding_label = Arc::new(RwLock::new("(unconfigured)".to_string()));
        let embedding_in_flight = Arc::new(AtomicBool::new(false));
        let attention_in_flight = Arc::new(AtomicBool::new(false));
        let tier1_pool = Arc::new(tier1_pool);
        let tier2_pool = Arc::new(tier2_pool);
        let tier3_pool = Arc::new(tier3_pool);
        let tier4_pool = Arc::new(tier4_pool);
        // Cloned out for `LoopHandles` *before* being moved into the actor
        // below - see `LiveModelHandles`'s doc comment for why the API
        // layer needs its own independent handle onto this state rather
        // than only ever reading it through a published `EngineSnapshot`.
        let live_models = LiveModelHandles {
            embedding_label: embedding_label.clone(),
            embedding_in_flight: embedding_in_flight.clone(),
            attention_in_flight: attention_in_flight.clone(),
            tier1_pool: tier1_pool.clone(),
            tier2_pool: tier2_pool.clone(),
            tier3_pool: tier3_pool.clone(),
            tier4_pool: tier4_pool.clone(),
        };

        let model = ModelCoordinator {
            embedding_client,
            embedding_label,
            chat_client,
            tier1_pool,
            tier2_pool,
            tier3_pool,
            tier4_pool,
            tier4_client,
            attention_client: None,
            ticks_since_attention_reconciliation: 0,
            attention_in_flight,
            live_models: live_models.clone(),
        };

        let telemetry = TelemetrySink { pending_events: Vec::new(), log_repeat: HashMap::new(), events_tx, snapshot_tx };

        let prediction = PredictionSystem {
            channel_embeddings: HashMap::new(),
            interlocutor_embeddings: HashMap::new(),
            embedding_cache: HashMap::new(),
            embedding_cache_order: aca_util::RingBuffer::new(prediction_system::EMBEDDING_CACHE_CAPACITY),
            precision_tracker: PrecisionTracker::default(),
            affect_tracker: AffectTracker::default(),
            local_event_predictor: crate::steps::predict::LocalEventPredictor::default(),
            last_prediction_error: None,
            embedding_in_flight,
            embedding_request_tx,
            embedding_reentry_rx,
            embedding_reentry_primed: None,
        };

        let perception = PerceptionState {
            input_rx,
            input_primed: None,
            external_agent_input_rx,
            external_agent_primed: None,
            sensor_input_rx,
            sensor_primed: None,
            sensor_signal_rx,
            sensor_signal_primed: None,
            curated_answer_rx,
            curated_answer_primed: None,
            room_input_rx,
            pending_room_input: VecDeque::new(),
            kl_reentry_tx,
            kl_reentry_rx,
            kl_reentry_primed: None,
            act_reentry_tx,
            act_reentry_rx,
            act_reentry_primed: None,
            wm_empty_since: None,
            last_boredom_at: None,
            last_self_status_at: None,
        };

        let actor = Self {
            memory,
            prediction,
            executive: ExecutiveState::new(outcome_feedback_window_ms),
            drives: DriveSystem::default(),
            previous_goal_statuses: HashMap::new(),
            rng: StdRng::from_entropy(),
            clock,
            config,
            model,
            tool_registry,
            telemetry,
            perception,
            turn_latency: ForegroundTurnTracker::default(),
            procedure_feedback: ProcedureFeedbackState::new(procedure_feedback_rx, outcome_feedback_rx),
            orient_outcome_specialist: None,
            communicative_intent_specialist: None,
            cycle_seq: 0,
            last_flush_at: None,
        };

        (actor, LoopHandles { input_tx, external_agent_input_tx, sensor_input_tx, sensor_signal_tx, curated_answer_tx, procedure_feedback_tx, outcome_feedback_tx, room_input_tx, events_rx, snapshot_rx, live_models })
    }

    /// Attaches the embedding model's real identity (e.g.
    /// "mxbai-embed-large") for `active_models` - optional, so every
    /// existing caller (chiefly tests) that doesn't care what the model is
    /// called keeps working unchanged and falls back to a generic
    /// placeholder, same pattern as `TierPool::with_label`/
    /// `DivergentPool::with_labels`. Written through the shared
    /// `Arc<RwLock<_>>` rather than a plain field assignment, since
    /// `LoopHandles::live_models` (and whatever's already cloned it, e.g.
    /// the API layer) was very likely already handed out by `new` before
    /// this ever runs - see `LiveModelHandles::embedding_label`'s doc
    /// comment.
    pub fn with_embedding_label(self, label: impl Into<String>) -> Self {
        *self.model.embedding_label.write().expect("embedding_label lock poisoned") = label.into();
        self
    }

    /// Attaches the Omega Attention model's endpoint. Attachment alone is
    /// shadow-only by default: Step 6 records the specialist decision beside
    /// deterministic `rank_candidates`/`admit_top_n`, but does not let it
    /// control admission. A caller must also explicitly disable
    /// `attention_shadow_only` to permit model control. Even then, the
    /// deterministic algorithm remains the automatic fallback on timeout,
    /// client error, low confidence, or an ineligible target. Unset by
    /// default, so callers that never attach a client retain deterministic
    /// behavior without making a model request.
    pub fn with_attention_client(mut self, client: Arc<dyn AttentionClient>) -> Self {
        self.model.attention_client = Some(client);
        self
    }

    /// Attaches a held-out-gated outcome forecaster. Its type exposes no
    /// action method and the actor only emits shadow telemetry from it.
    pub fn with_orient_outcome_specialist(
        mut self,
        specialist: crate::steps::orient_outcome::OrientOutcomeSpecialist,
    ) -> Self {
        self.orient_outcome_specialist = Some(specialist);
        self
    }

    /// Attaches a held-out-gated communicative-intent classifier. It can
    /// only emit shadow comparisons against an already selected Executive
    /// operator; it has no path to propose or execute an action.
    pub fn with_communicative_intent_specialist(
        mut self,
        specialist: crate::steps::communicative_intent::CommunicativeIntentSpecialist,
    ) -> Self {
        self.communicative_intent_specialist = Some(specialist);
        self
    }

    /// The loop: `CognitiveScheduler` waits for the next real reason to
    /// tick - a due timer, an already-pending admission/room-input backlog,
    /// or fresh channel activity - then one `tick()` runs. `tick()` itself
    /// is unchanged by this: it still does its own non-blocking drain of
    /// every source every time it runs, so a timer-woken tick still
    /// opportunistically picks up anything else queued, exactly as the old
    /// unconditional busy-poll did - the only thing that changed is that
    /// `run()` no longer calls `tick()` hundreds of times a second once
    /// there's genuinely nothing to do.
    pub async fn run(mut self) -> ! {
        loop {
            CognitiveScheduler::wait_for_next_tick(&mut self).await;
            self.tick().await;
        }
    }

    /// Thin forwarding wrapper kept on the actor itself (rather than
    /// changing all 25 call sites to `self.telemetry.record(...)`) - see
    /// `TelemetrySink::record`'s doc comment for what this actually does.
    fn emit_event(&mut self, phase: CyclePhase, kind: CycleEventKind, tier: Option<aca_types::Tier>, payload: serde_json::Value) {
        self.telemetry.record(self.cycle_seq, self.clock.as_ref(), phase, kind, tier, payload);
    }

    async fn publish_snapshot(&mut self) {
        let members: Vec<WorkingMemoryMember> = self
            .memory.working_memory
            .iter()
            .filter_map(|id| self.memory.graph.get(id))
            .map(|object| WorkingMemoryMember {
                id: object.id,
                kind: object.kind,
                text: object.text.clone(),
                activation_total: object.activation.total,
                attention_score: object.workspace.attention_score,
                promotion_status: object.promotion.status,
                confidence: object.confidence,
            })
            .collect();

        // Associative edges, scoped to pairs where both endpoints are
        // currently in Working Memory - visualizable without any further
        // lookups (see EdgeSummary's doc comment for why this scope).
        let mut associative_edges = Vec::new();
        for &source_id in &self.memory.working_memory {
            if let Some(source) = self.memory.graph.get(&source_id) {
                for edge in &source.edges {
                    if self.memory.working_memory.contains(&edge.target_id) {
                        associative_edges.push(EdgeSummary {
                            source_id,
                            target_id: edge.target_id,
                            kind: edge.kind,
                            strength: edge.strength,
                        });
                    }
                }
            }
        }

        let mut memory_counts = MemoryRoleCounts::default();
        let mut goal_stack = Vec::new();
        // `(id, text, confidence, formed_at, source_object_ids)` of whichever
        // `synthesize`-produced pattern has the latest `created_at` seen so
        // far - collected as owned scalars (not a `&MentalObject`) so this
        // doesn't hold a borrow of `self.memory.graph` past the loop, where
        // `supporting_episodes`/`reinforced_count` below need to look
        // sources back up by id. See `ProvisionalAbstractionSummary`'s own
        // doc comment for why `kind == Memory && source_object_ids.len() >=
        // 2` uniquely identifies this path among every other Semantic-
        // producing one.
        let mut latest_abstraction: Option<(aca_types::MentalObjectId, String, f32, aca_util::EpochMillis, Vec<aca_types::MentalObjectId>)> = None;
        let mut active_intention_count = 0usize;
        for object in self.memory.graph.iter() {
            for role in &object.memory_roles {
                match role {
                    aca_types::MemoryRole::Working => memory_counts.working += 1,
                    aca_types::MemoryRole::Episodic => memory_counts.episodic += 1,
                    aca_types::MemoryRole::Semantic => memory_counts.semantic += 1,
                    aca_types::MemoryRole::SelfMemory => memory_counts.self_memory += 1,
                }
            }
            if object.promotion.status == aca_types::PromotionStatus::Candidate {
                memory_counts.candidates += 1;
            }
            if let Some(goal) = &object.goal {
                if matches!(goal.status, aca_types::GoalStatus::Active | aca_types::GoalStatus::Impassed) {
                    goal_stack.push(GoalSummary {
                        id: object.id,
                        text: object.text.clone(),
                        status: goal.status,
                        priority: goal.priority,
                    });
                }
                if object.kind == aca_types::MentalObjectKind::Intention && goal.status == aca_types::GoalStatus::Active {
                    active_intention_count += 1;
                }
            }
            if object.kind == aca_types::MentalObjectKind::Memory
                && object.memory_roles.contains(&aca_types::MemoryRole::Semantic)
                && object.source_object_ids.len() >= 2
                && latest_abstraction.as_ref().is_none_or(|(_, _, _, formed_at, _)| object.created_at > *formed_at)
            {
                latest_abstraction = Some((object.id, object.text.clone(), object.confidence, object.created_at, object.source_object_ids.clone()));
            }
        }
        let provisional_abstraction = latest_abstraction.map(|(id, text, confidence, formed_at, source_ids)| {
            // Reference log always seeds one entry at creation (see
            // `ActivationState::new_at`) - every entry past that first one is
            // a genuine dedup reconfirmation from `synthesize`'s own
            // near-duplicate path (`aca_graph::record_reference`), so
            // subtracting it is what makes `0` mean "formed once, never
            // reconfirmed" rather than "never referenced at all," which
            // would be true of no object in this graph.
            let reinforced_count = self.memory.graph.get(&id).map(|object| object.activation.reference_log.len().saturating_sub(1) as u32).unwrap_or(0);
            let supporting_episodes = source_ids.iter().filter_map(|source_id| self.memory.graph.get(source_id)).map(|source| source.text.clone()).collect();
            ProvisionalAbstractionSummary {
                id,
                text,
                confidence,
                formed_at,
                reinforced_count,
                source_count: source_ids.len(),
                supporting_episodes,
            }
        });

        // Built from `live_models` rather than re-deriving from
        // `self.model.tier1_pool` etc. inline - this is also exactly what the API
        // layer calls directly (see `LiveModelHandles`'s doc comment), so
        // there's one construction, not two that could drift apart.
        let tier_status = self.model.live_models.tier_status();
        let active_models = self.model.live_models.active_models();

        // Refreshed only on the write-behind flush's own cadence (see
        // `cached_kl_doc_count`'s doc comment) rather than on every publish -
        // `publish_snapshot` can run more than once per tick (once before
        // the embedding await, once at tick's end), and a real DB round trip
        // on each of those, forever, is pure waste for a field that only
        // ever changes when another household agent writes to the library
        // over MCP. Graceful degradation, not a propagated error, on the
        // ticks it does refresh: a store hiccup here shouldn't stop a
        // snapshot publish that otherwise has everything else it needs.
        if self.last_flush_at.is_none_or(|t| self.clock.now().0.saturating_sub(t.0) >= self.config.flush_interval_ms) {
            self.memory.cached_kl_doc_count = self.memory.kl_store.count_documents().await.unwrap_or(self.memory.cached_kl_doc_count);
        }
        let kl_doc_count = self.memory.cached_kl_doc_count;

        let _ = self.telemetry.snapshot_tx.send(EngineSnapshot {
            cycle_seq: self.cycle_seq,
            working_memory: members,
            associative_edges,
            memory_counts,
            goal_stack,
            tier_status,
            active_models,
            embedding_in_flight: self.model.live_models.embedding_in_flight(),
            attention_in_flight: self.model.live_models.attention_in_flight(),
            kl_doc_count,
            provisional_abstraction,
            affect_valence: self.prediction.affect_tracker.valence(),
            precision_gain: self.prediction.affect_tracker.precision_gain(),
            last_prediction_error: self.prediction.last_prediction_error,
            last_operator_proposal: self.executive.last_operator_proposal.clone(),
            last_displacement: self.executive.last_displacement.clone(),
            attended_not_ignited: self.executive.last_attended_not_ignited.clone(),
            active_intention_count,
            drive_uncertainty: self.drives.drive_state.uncertainty,
            drive_curiosity: self.drives.drive_state.curiosity,
            drive_competence: self.drives.drive_state.competence,
            drive_social_connection: self.drives.drive_state.social_connection,
            drive_resource_pressure: self.drives.drive_state.resource_pressure,
        });
    }

    /// Finishes Steps 2-3 (Compare) for one observation once its embedding
    /// has resolved (or failed) - the shared tail `tick()` used to run
    /// inline, directly inside its own cache-miss `.await`. Runs identically
    /// whether that happened synchronously this same tick (a cache hit) or
    /// asynchronously, via `embedding_worker`, on whichever later tick's
    /// reentry-drain sees the result - `observation`/`expected`/`channel`/
    /// `interlocutor_id`/`text` are exactly what `tick()` had already
    /// computed about this observation *before* the point it used to await
    /// inline, carried forward unchanged by `PendingObservation` in the
    /// deferred case. `goal_progress_reward` and `now` are deliberately
    /// **not** part of that carried state - they come from whichever tick
    /// actually calls this (the original tick on a cache hit, a later one on
    /// a worker reentry), since both are fresh, tick-scoped readings with no
    /// connection to which specific observation is being finished.
    ///
    /// Returns the old four locals plus this observation's numeric orienting
    /// result for optional post-Broadcast shadow outcome forecasting.
    /// set directly - `(new_object_id, new_surprise,
    /// current_tick_epistemic_value, current_outcome_context, orienting)` - for the
    /// caller to fold into its own Step 4 onward.
    fn apply_embedding_result(
        &mut self,
        mut observation: MentalObject,
        expected: Option<Vec<f32>>,
        channel: SourceChannel,
        interlocutor_id: Option<MentalObjectId>,
        text: &str,
        goal_progress_reward: Option<f32>,
        now: aca_util::EpochMillis,
        embedding_result: Result<Vec<f32>, aca_tiers::TierError>,
    ) -> (Option<MentalObjectId>, Option<f32>, Option<f32>, Option<(SourceChannel, Option<MentalObjectId>)>, Option<crate::steps::orient::OrientingResult>) {
        match embedding_result {
            Ok(embedding) => {
                let comparison = compare(expected.as_deref(), &embedding, channel, interlocutor_id, &mut self.prediction.precision_tracker);
                let local_prediction = self.prediction.local_event_predictor.observe(channel, self.prediction.affect_tracker.valence(), now);
                let context_source = self.memory.working_memory.iter()
                    .filter_map(|id| self.memory.graph.get(id).map(|object| (*id, object.activation.total)))
                    .max_by(|(id_a, score_a), (id_b, score_b)| score_a.total_cmp(score_b).then_with(|| id_b.cmp(id_a)))
                    .map(|(source_id, _)| source_id);
                let object_prediction = context_source.and_then(|source_id| crate::steps::predict::predict_successor_object(&self.memory.graph, source_id, &embedding, now, self.config.edge_decay_rate_per_ms));
                let goal_impact_prediction = (!self.config.ablation_config.disable_goal_impact_prediction)
                    .then(|| context_source.and_then(|source_id| crate::steps::predict::predict_goal_impact(&self.memory.graph, source_id, now, self.config.edge_decay_rate_per_ms)))
                    .flatten();
                let goal_relevance = most_salient_active_goal(&self.memory.graph)
                    .and_then(|goal| goal.embedding.as_deref())
                    .map(|goal_embedding| aca_util::cosine_similarity(goal_embedding, &embedding).max(0.0))
                    .unwrap_or(0.0)
                    .max(goal_impact_prediction.map_or(0.0, |prediction| prediction.evidence_strength));
                let trusted_signal = observation.data.get("sensor_signal").and_then(|value| value.as_bool()).unwrap_or(false);
                let threat = trusted_signal.then(|| observation.data.get("threat").and_then(|value| value.as_f64()).unwrap_or(0.0) as f32).unwrap_or(0.0);
                let urgency = trusted_signal.then(|| observation.data.get("urgency").and_then(|value| value.as_f64()).unwrap_or(0.0) as f32).unwrap_or(0.0);
                let orienting = (!self.config.ablation_config.disable_orienting).then(|| {
                    crate::steps::orient::orient(
                        &mut observation.dynamics,
                        &comparison,
                        local_prediction.source_surprise.max(object_prediction.map_or(0.0, |prediction| prediction.embedding_error * prediction.edge_strength)),
                        goal_relevance,
                        self.prediction.affect_tracker.valence(),
                        channel,
                        threat,
                        urgency,
                        &self.config.orienting_config,
                        now,
                    )
                });
                let sensor_habituated = !self.config.ablation_config.disable_sensor_habituation
                    && crate::steps::orient::should_habituate_environment(
                        channel, expected.is_some(), comparison.error_magnitude,
                        orienting, interlocutor_id.is_some(), threat, urgency,
                        self.config.sensor_habituation_threshold,
                    );

                // Metacognition/RL feedback (steps::metacognition): this
                // channel/interlocutor now has a fresh prediction error,
                // which is exactly the currency any decision keyed to it
                // was waiting to be judged by. Resolved before this tick's
                // own decisions (Steps 7-9) register anything new for *this*
                // channel - a decision made this tick should never be able
                // to grade itself against its own trigger.
                let reward = reward_from_comparison(&comparison);
                if !self.config.ablation_config.disable_eligibility_learning {
                    let credits = self.memory.eligibility_traces.apply_reward(
                        &mut self.memory.graph,
                        reward,
                        &self.config.eligibility_config,
                        now,
                    );
                    for credit in &credits {
                        self.memory.dirty_ids.insert(credit.source);
                    }
                    if !credits.is_empty() {
                        self.emit_event(
                            CyclePhase::Learn,
                            CycleEventKind::Normal,
                            None,
                            json!({
                                "eligibility_updates": credits.len(),
                                "reward": reward,
                                "mean_trace": credits.iter().map(|credit| credit.trace).sum::<f32>() / credits.len() as f32,
                            }),
                        );
                    }
                }
                // Same reward, second read: `steps::affect::AffectTracker`'s
                // whole reason for existing is to be a *slower*-moving
                // reduction of this exact number, not a second inference
                // pass - see that module's doc comment. Blended with
                // whatever goal-progress evidence (if any) the calling
                // tick's own top-of-`tick()` scan found, so a well-predicted
                // but goal-thwarting tick doesn't read as affectively good
                // just because it was expected.
                self.prediction.affect_tracker.update(reward, goal_progress_reward);
                // Raw reading alongside the smoothed trend `affect_tracker`
                // just folded this into - see `PredictionErrorSummary`'s own
                // doc comment for why both are worth keeping.
                self.prediction.last_prediction_error = Some(crate::snapshot::PredictionErrorSummary {
                    error_magnitude: comparison.error_magnitude,
                    precision: comparison.precision,
                    epistemic_value: comparison.epistemic_value,
                    reward,
                    at: now,
                });
                for resolved in self.executive.outcome_registry.resolve(channel, interlocutor_id, now) {
                    if let Some(chunk_id) = resolved.chunk_id {
                        reinforce_chunk_utility(&mut self.memory.graph, chunk_id, reward);
                        self.memory.dirty_ids.insert(chunk_id);
                        self.emit_event(
                            CyclePhase::Learn,
                            CycleEventKind::Normal,
                            None,
                            json!({"utility_update": true, "chunk_id": chunk_id.to_string(), "reward": reward}),
                        );
                    }
                    if let Some(tier) = resolved.tier {
                        self.executive.calibration_tracker.record(tier, resolved.reported_confidence, reward);
                        self.emit_event(
                            CyclePhase::Learn,
                            CycleEventKind::Normal,
                            Some(tier),
                            json!({"calibration_update": true, "reported_confidence": resolved.reported_confidence, "reward": reward}),
                        );
                    }
                }
                let current_outcome_context = Some((channel, interlocutor_id));

                observation.embedding = Some(embedding.clone());
                observation.prediction.error_magnitude = Some(comparison.error_magnitude);
                observation.prediction.precision = Some(comparison.precision);
                if let Some(orienting) = orienting {
                    if !observation.data.is_object() {
                        observation.data = json!({});
                    }
                    observation.data["orienting"] = json!({
                        "score": orienting.score,
                        "fired": orienting.fired,
                        "goal_relevance": orienting.goal_relevance,
                        "affective_salience": orienting.affective_salience,
                        "social_relevance": orienting.social_relevance,
                        "threat": orienting.threat,
                        "urgency": orienting.urgency,
                    });
                }
                if sensor_habituated {
                    if !observation.data.is_object() { observation.data = json!({}); }
                    observation.data["sensor_habituated"] = json!(true);
                }
                if !self.config.ablation_config.disable_procedural_fast_path
                    && channel == SourceChannel::ConversationInput
                    && let Some(execution) = self.memory.compiled_procedure(text)
                        .and_then(|procedure| procedure.execution(text))
                {
                    if !observation.data.is_object() {
                        observation.data = json!({});
                    }
                    match execution {
                        crate::steps::procedural::CompiledExecution::SpeakExact(response) =>
                            observation.data["compiled_response"] = json!(response),
                        crate::steps::procedural::CompiledExecution::AskExact(question) =>
                            observation.data["compiled_question"] = json!(question),
                        crate::steps::procedural::CompiledExecution::Ignore =>
                            observation.data["compiled_ignore"] = json!(true),
                    }
                }
                if !self.config.ablation_config.disable_procedural_fast_path
                    && channel == SourceChannel::ConversationInput
                    && observation.data.get("compiled_response").is_none()
                    && observation.data.get("compiled_question").is_none()
                    && observation.data.get("compiled_ignore").is_none()
                    && let Some(response) = crate::steps::procedural::innate_social_response(text)
                {
                    if !observation.data.is_object() { observation.data = json!({}); }
                    observation.data["innate_response"] = json!(response);
                }
                if let Some(interlocutor_id) = interlocutor_id {
                    self.prediction.interlocutor_embeddings.insert(interlocutor_id, embedding.clone());
                }
                self.prediction.channel_embeddings.insert(channel, embedding);

                let id = observation.id;
                self.memory.graph.insert(observation);
                self.memory.dirty_ids.insert(id);
                self.procedure_feedback.record_compared_observation(id, now);
                if sensor_habituated {
                    self.turn_latency.discard(id);
                }
                if let Some(interlocutor_id) = interlocutor_id {
                    // Hebbian link from this utterance to who it came from,
                    // plus a fresh reference on the interlocutor's own
                    // activation - see `steps::interlocutor::reinforce_link`'s
                    // doc comment for why repetition alone, not a bonus
                    // dial, is what makes familiarity emerge here.
                    crate::steps::interlocutor::reinforce_link(&mut self.memory.graph, id, interlocutor_id, now);
                    self.memory.dirty_ids.insert(interlocutor_id);
                }
                if !self.config.ablation_config.disable_spike_propagation
                    && orienting.is_some_and(|result| result.fired)
                {
                    self.memory.spike_events.schedule_from(
                        &self.memory.graph, id, now, self.config.spike_delay_ms,
                        self.config.edge_decay_rate_per_ms, self.config.spike_max_fanout,
                        self.config.spike_max_pending,
                    );
                }
                // Curiosity folded into the same surprise-like scalar
                // Coalition already ranks on (specs.md: "all reduce to the
                // same precision-weighted-error scale") rather than
                // widening `attention_score`'s signature for a second,
                // parallel channel of influence.
                // The predictive-processing account of mood/anxiety
                // (`AffectTracker::precision_gain`'s own doc comment): a
                // recent run of surprising experience amplifies how much
                // *this* tick's surprise counts toward winning Coalition; a
                // recent run of well-predicted experience dampens it
                // slightly. Neutral (gain 1.0, a no-op) until real history
                // accumulates.
                let orienting_pulse = orienting.filter(|result| result.fired).map_or(0.0, |_| self.config.orienting_config.attention_pulse);
                let new_surprise = (comparison.precision_weighted_surprise + self.config.curiosity_weight * comparison.epistemic_value)
                    * self.prediction.affect_tracker.precision_gain()
                    + orienting_pulse;

                self.emit_event(
                    CyclePhase::Compare,
                    CycleEventKind::Normal,
                    None,
                    json!({
                        "observation_id": id.to_string(),
                        "text": text,
                        "error_magnitude": comparison.error_magnitude,
                        "precision": comparison.precision,
                        "epistemic_value": comparison.epistemic_value,
                        "orienting_score": orienting.map(|result| result.score),
                        "orienting_fired": orienting.map(|result| result.fired),
                        "orienting_inputs": orienting.map(|result| json!({
                            "novelty": result.novelty,
                            "prediction_error": result.prediction_error,
                            "goal_relevance": result.goal_relevance,
                            "affective_salience": result.affective_salience,
                            "social_relevance": result.social_relevance,
                            "threat": result.threat,
                            "urgency": result.urgency,
                        })),
                        "sensor_habituated": sensor_habituated,
                        "expected_source": local_prediction.expected_source.map(|source| format!("{source:?}")),
                        "source_surprise": local_prediction.source_surprise,
                        "actual_source_probability": local_prediction.actual_source_probability,
                        "expected_interval_ms": local_prediction.expected_interval_ms,
                        "expected_interval_stddev_ms": local_prediction.expected_interval_stddev_ms,
                        "interval_error_ms": local_prediction.interval_error_ms,
                        "expected_affect": local_prediction.expected_affect,
                        "expected_affect_stddev": local_prediction.expected_affect_stddev,
                        "expected_successor_object": object_prediction.map(|prediction| prediction.expected_object.to_string()),
                        "successor_error": object_prediction.map(|prediction| prediction.embedding_error),
                        "expected_goal": goal_impact_prediction.map(|prediction| prediction.goal_id.to_string()),
                        "expected_goal_impact": goal_impact_prediction.map(|prediction| prediction.expected_impact),
                    }),
                );

                (Some(id), Some(new_surprise), Some(comparison.epistemic_value), current_outcome_context, orienting)
            }
            Err(err) => {
                self.emit_event(CyclePhase::Observe, CycleEventKind::Error, None, json!({"error": err.to_string()}));
                (None, None, None, None, None)
            }
        }
    }

    /// Applies an operator via Act, plus the bookkeeping every application
    /// needs regardless of *how* it was decided - picked outright by
    /// `select_operator`, or reached after resolving a confidence impasse.
    /// This is the organic half of impasse resolution: a person who
    /// deliberates their way out of "should I say this?" both says it *and*
    /// gets faster at the same call next time - the deliberation resolves
    /// the moment it was actually about, not only some hypothetical future
    /// recurrence of it. `chunk_resolution` already handles the second half
    /// (compiling the resolution into a learned bias, SOAR's own
    /// "practice makes it automatic" story); this is what makes sure the
    /// first half - actually doing the thing just decided - isn't skipped.
    /// The reactive "conversational presence" dial (see
    /// `LoopConfig::presence_half_life_ms` and
    /// `prompt_templates::presence_block`, the two places this value
    /// actually does something): 1.0 the instant a live
    /// `SourceChannel::ConversationInput` turn last won Broadcast, decaying
    /// exponentially toward 0.0 as idle time since then grows. Reuses
    /// `last_broadcast_by_channel` - already recorded for the attentional-
    /// refractory-window check in `tick()` - rather than tracking a second
    /// timestamp for the same event. `0.0` (fully ambient/idle framing, the
    /// only framing that existed before this dial) when no conversational
    /// turn has ever won Broadcast yet.
    fn conversational_presence(&self, now: aca_util::EpochMillis) -> f32 {
        match self.memory.last_broadcast_by_channel.get(&SourceChannel::ConversationInput) {
            Some(last) => {
                let elapsed_ms = (now.0 - last.0).max(0) as f32;
                0.5f32.powf(elapsed_ms / self.config.presence_half_life_ms as f32)
            }
            None => 0.0,
        }
    }

    async fn apply_operator(
        &mut self,
        proposal: &OperatorProposal,
        working_memory_entries: &[crate::prompt_templates::ContextEntry<'_>],
        presence: f32,
        now: aca_util::EpochMillis,
    ) -> crate::steps::act::ActOutcome {
        let target_text_before = text_preview(&self.memory.graph, proposal.target_id);
        let routine_stimulus = self
            .memory
            .graph
            .get(&proposal.target_id)
            .filter(|object| object.kind == aca_types::MentalObjectKind::Reflection)
            .and_then(|reflection| reflection.source_object_ids.first())
            .and_then(|source_id| self.memory.graph.get(source_id))
            .filter(|source| source.kind == aca_types::MentalObjectKind::Observation)
            .map(|source| (source.text.clone(), source.embedding.clone()));
        let intent_shadow = self.communicative_intent_specialist.as_ref().and_then(|specialist| {
            let actual = match proposal.operator {
                crate::steps::executive::Operator::Speak => "speak",
                crate::steps::executive::Operator::Ask => "ask",
                crate::steps::executive::Operator::Ignore => "ignore",
                _ => return None,
            };
            let reflection = self.memory.graph.get(&proposal.target_id)
                .filter(|object| object.kind == aca_types::MentalObjectKind::Reflection)?;
            let prediction = specialist.predict(&reflection.text);
            Some((actual, prediction))
        });
        if let Some((actual, prediction)) = intent_shadow {
            self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None, json!({
                "communicative_intent_shadow": true,
                "actual_operator": actual,
                "predicted_operator": prediction.label,
                "confidence": prediction.confidence,
                "agreement": prediction.label == actual,
            }));
        }
        // Grounds `cognitive_core::reflect`'s prompt (via `act`) in this
        // tick's real, verified displacement fact - but only when
        // `proposal.target_id` is itself the object `last_displacement`
        // says actually won the slot this tick. Without that `filter`, a
        // stray reflection on some *other* object still in Working Memory
        // could pick up a true-but-irrelevant fact about a different
        // object's admission, which would read to the model (and to a user)
        // as a claim about the thing actually being reflected on - not what
        // `steps::displacement::explain_release` verified at all.
        let displacement_note = self.executive.last_displacement.as_ref().filter(|d| d.entrant_id == proposal.target_id).map(|d| d.evicted_text.as_str());
        let outcome = act(
            &mut self.memory.graph,
            proposal,
            &self.model.tier1_pool,
            &self.model.tier2_pool,
            &self.model.tier3_pool,
            self.model.chat_client.as_ref(),
            self.model.embedding_client.as_ref(),
            self.memory.kl_store.as_ref(),
            &self.config.knowledge_library_config,
            &self.tool_registry,
            working_memory_entries,
            &self.memory.self_summary,
            presence,
            &self.config.memory_config,
            &self.config.confidence_revision,
            now,
            self.clock.as_ref(),
            self.config.decay_d,
            self.config.tier3_hedge_delay,
            &self.config.temperature,
            displacement_note,
        )
        .await;
        self.memory.dirty_ids.insert(proposal.target_id);
        if !self.config.ablation_config.disable_procedural_fast_path
            && let (Some((stimulus, stimulus_embedding)), crate::steps::act::ActOutcome::Spoke { text, render_path }) = (&routine_stimulus, &outcome)
            && *render_path != crate::steps::act::SpeechRenderPath::CompiledProcedure
            && let Some(skill_id) = crate::steps::procedural::record_response_with_embedding(
                &mut self.memory.graph,
                stimulus,
                text,
                stimulus_embedding.as_deref(),
                now,
                self.config.decay_d,
            )
        {
            self.memory.dirty_ids.insert(skill_id);
            self.memory.refresh_compiled_procedure(stimulus);
        }
        if let crate::steps::act::ActOutcome::ConsultedKnowledgeLibrary { result } = &outcome
            && let Some(text) = result.found.then(|| result.text.clone()).flatten()
        {
            // Self-send, deliberately `try_send` not `.await`: the actor is
            // the sole consumer of `kl_reentry_rx`, draining it only once
            // per tick via `try_recv` (see the input-resolution block
            // above). An awaited send here could self-deadlock - it can't
            // drain until the *next* tick, which can't start until this
            // send inside the *current* tick resolves. Log-and-drop on
            // `Full` is correct and sufficient at this volume (at most one
            // Consult result per tick).
            if let Err(err) = self.perception.kl_reentry_tx.try_send(text) {
                tracing::warn!(error = %err, "failed to re-enter Knowledge Library result as an Observation");
            }
        }
        if let crate::steps::act::ActOutcome::Acted { result: Ok(text), .. } = &outcome {
            // Same self-send reasoning as the Consult re-entry above:
            // try_send, actor-internal-only, log-and-drop on Full.
            if let Err(err) = self.perception.act_reentry_tx.try_send(text.clone()) {
                tracing::warn!(error = %err, "failed to re-enter a tool result as an Observation");
            }
        }
        if let crate::steps::act::ActOutcome::Acted { result, .. } = &outcome {
            self.executive.execution_tracker.record(result.is_ok());
        }
        if let crate::steps::act::ActOutcome::Remembered { outcome: MemoryFormationOutcome::NewEpisodic { id } } = &outcome {
            // Feeds `steps::synthesize`'s periodic trigger - see
            // `episodic_since_last_synthesis`'s doc comment.
            self.memory.episodic_since_last_synthesis.push(*id);
        }
        // A deliberately ignored Reflection is transient cognitive workspace,
        // not a standing stimulus. Archive and release it so activation or
        // recall cannot re-admit the same thought and execute Ignore forever.
        // Spoken/asked Reflections remain active for associative learning and
        // synthesis; raw Observations remain durable perceptual history.
        if matches!(outcome,
            crate::steps::act::ActOutcome::Silent { reason: crate::steps::act::SilentReason::Ignored })
            && self.memory.graph.get(&proposal.target_id)
                .is_some_and(|object| object.kind == aca_types::MentalObjectKind::Reflection
                    && object.produced_by_operator.as_deref() == Some("Ignore"))
        {
            if let Some(object) = self.memory.graph.get_mut(&proposal.target_id) {
                object.status = aca_types::ObjectStatus::Archived;
                object.workspace.in_working_memory = false;
            }
            self.memory.working_memory.retain(|id| *id != proposal.target_id);
            self.memory.pending_admission.remove(&proposal.target_id);
            self.memory.dirty_ids.insert(proposal.target_id);
        }
        if let crate::steps::act::ActOutcome::Reflected { reflection_id } = &outcome {
            if proposal.operator == crate::steps::executive::Operator::ContinueReflecting
                && let Some(source_id) = self.memory.graph.get(reflection_id)
                    .and_then(|reflection| (reflection.source_object_ids.len() == 1).then_some(reflection.source_object_ids[0]))
                && self.memory.graph.get(&source_id).is_some_and(|source| source.kind == aca_types::MentalObjectKind::Observation)
            {
                self.procedure_feedback.record_reflection(*reflection_id, source_id);
            }
            // The Reflection just written holds the real generated content -
            // give it a shot at Coalition next tick, or it would sit in the
            // graph forever unspoken (see `pending_admission`'s doc
            // comment).
            self.memory.pending_admission.insert(*reflection_id);
            self.memory.dirty_ids.insert(*reflection_id);
        }
        // `ReflectionFailed` is the one `Silent` reason that's a genuine
        // failure rather than a routine non-event - see
        // `act_outcome_is_failure`'s own doc comment for why this is the
        // fix for a real live gap: a failed reflection used to be
        // indistinguishable, in the console, from Omega simply choosing not
        // to respond.
        let event_kind = if act_outcome_is_failure(&outcome) { CycleEventKind::Error } else { CycleEventKind::Normal };
        let mut payload = act_outcome_payload(&outcome, proposal.operator, target_text_before);
        if matches!(outcome, crate::steps::act::ActOutcome::Spoke { .. }) {
            if let Some((source_id, elapsed_us)) = self.turn_latency.finish_action(&self.memory.graph, proposal.target_id, Instant::now()) {
                payload["foreground_observation_id"] = json!(source_id.to_string());
                payload["foreground_turn_elapsed_us"] = json!(elapsed_us);
                if matches!(proposal.operator, crate::steps::executive::Operator::Speak | crate::steps::executive::Operator::Ask)
                    && let crate::steps::act::ActOutcome::Spoke { text, render_path } = &outcome
                {
                    let reflected_terminal = self.procedure_feedback.take_reflection_source(proposal.target_id) == Some(source_id);
                    let compiled_terminal = *render_path == crate::steps::act::SpeechRenderPath::CompiledProcedure && proposal.target_id == source_id;
                    let reflected_then_spoke = proposal.operator == crate::steps::executive::Operator::Speak && reflected_terminal;
                    let compiled_spoke = proposal.operator == crate::steps::executive::Operator::Speak && compiled_terminal;
                    let reflected_then_asked = proposal.operator == crate::steps::executive::Operator::Ask && reflected_terminal;
                    let compiled_asked = proposal.operator == crate::steps::executive::Operator::Ask && compiled_terminal;
                    if (reflected_terminal || compiled_terminal)
                        && let Some(source) = self.memory.graph.get(&source_id)
                        && source.kind == aca_types::MentalObjectKind::Observation
                        && crate::steps::procedural::routine_key(&source.text).is_some()
                        && let Some(embedding) = source.embedding.as_ref()
                    {
                        self.procedure_feedback.record_attempt(source_id, ProcedureAttempt {
                            stimulus: source.text.clone(),
                            response: text.clone(),
                            stimulus_embedding: embedding.clone(),
                            reflected_then_spoke,
                            compiled_spoke,
                            reflected_then_asked,
                            compiled_asked,
                            reflected_then_ignored: false,
                            compiled_ignored: false,
                            at: now,
                        }, now);
                    }
                }
            }
        } else if proposal.operator == crate::steps::executive::Operator::Ignore
            && matches!(outcome, crate::steps::act::ActOutcome::Silent { reason: crate::steps::act::SilentReason::Ignored })
            && let Some((source_id, elapsed_us)) = self.turn_latency.finish_action(&self.memory.graph, proposal.target_id, Instant::now())
        {
            payload["foreground_observation_id"] = json!(source_id.to_string());
            payload["foreground_turn_elapsed_us"] = json!(elapsed_us);
            let reflected_then_ignored = self.procedure_feedback.take_reflection_source(proposal.target_id) == Some(source_id);
            let compiled_ignored = proposal.target_id == source_id
                && self.memory.graph.get(&source_id).is_some_and(|source|
                    source.data.get("compiled_ignore").and_then(|value| value.as_bool()) == Some(true));
            if (reflected_then_ignored || compiled_ignored)
                && let Some(source) = self.memory.graph.get(&source_id)
                && source.kind == aca_types::MentalObjectKind::Observation
                && crate::steps::procedural::routine_key(&source.text).is_some()
                && let Some(embedding) = source.embedding.as_ref()
            {
                self.procedure_feedback.record_attempt(source_id, ProcedureAttempt {
                    stimulus: source.text.clone(),
                    response: String::new(),
                    stimulus_embedding: embedding.clone(),
                    reflected_then_spoke: false,
                    compiled_spoke: false,
                    reflected_then_asked: false,
                    compiled_asked: false,
                    reflected_then_ignored,
                    compiled_ignored,
                    at: now,
                }, now);
            }
        }
        self.emit_event(CyclePhase::Act, event_kind, None, payload);
        outcome
    }

    /// Metacognitive calibration for a freshly-created Reflection: a no-op
    /// unless it's a Tier 3 reflection (Tier 1/2 already earn a trustworthy
    /// confidence structurally via cross-sample agreement - see
    /// `cognitive_core::try_tier_via_agreement` - and must not be
    /// second-guessed here). Registers a `PendingOutcome` carrying the *raw*
    /// self-reported confidence (so `calibration_tracker` learns this tier's
    /// true bias, not an already-corrected number), then overwrites the
    /// Reflection's stored confidence with the calibrated value, so every
    /// downstream reader (`propose_operators`'s
    /// `low_confidence_reflection_threshold` check, chiefly) sees the
    /// corrected number for free, with no changes needed there.
    fn calibrate_and_register_reflection(&mut self, reflection_id: MentalObjectId, outcome_context: Option<(SourceChannel, Option<MentalObjectId>)>) {
        let Some(object) = self.memory.graph.get(&reflection_id) else { return };
        if object.tier_used != Some(aca_types::Tier::T3) {
            return;
        }
        let raw_confidence = object.confidence;

        if let Some((channel, interlocutor)) = outcome_context {
            self.executive.outcome_registry.register(PendingOutcome {
                channel,
                interlocutor,
                chunk_id: None,
                tier: Some(aca_types::Tier::T3),
                reported_confidence: raw_confidence,
                created_at: self.clock.now(),
            });
        }
        let calibrated = self.executive.calibration_tracker.calibrate(aca_types::Tier::T3, raw_confidence);
        if let Some(object) = self.memory.graph.get_mut(&reflection_id) {
            object.confidence = calibrated;
        }
        self.memory.dirty_ids.insert(reflection_id);
    }

    /// Runs exactly one tick. Exposed publicly (in addition to `run`) so
    /// tests and tooling can step the cycle deterministically.
    pub async fn tick(&mut self) {
        let tick_started = Instant::now();
        self.cycle_seq += 1;
        let now = self.clock.now();
        let mut curated_answer_changed_this_tick = false;
        let mut procedure_feedback_changed_this_tick = false;
        let mut outcome_feedback_changed_this_tick = false;
        // Only previously observed spoken turns can earn verified credit.
        // Drain is bounded and serialized with the graph/index owner.
        for _ in 0..64 {
            let Some(command) = self.procedure_feedback.take_feedback() else { break; };
            let Some(attempt) = self.procedure_feedback.take_attempt(command.observation_id, now) else {
                self.emit_event(CyclePhase::Learn, CycleEventKind::Error, None,
                    json!({"procedure_feedback_rejected": "unknown_or_expired_observation"}));
                continue;
            };
            // A host's verdict on an actual spoken turn is independent
            // outcome evidence even if that turn is not an eligible
            // two-step macro credit (for example, direct compiled speech).
            if self.procedure_feedback.label_compared_observation(command.observation_id, now) {
                outcome_feedback_changed_this_tick = true;
                self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None,
                    json!({"independent_observation_outcome": true,
                        "observation_id": command.observation_id.to_string(),
                        "successful": command.successful}));
            }
            if command.successful {
                if !attempt.reflected_then_spoke && !attempt.reflected_then_asked && !attempt.reflected_then_ignored {
                    self.emit_event(CyclePhase::Learn, CycleEventKind::Error, None,
                        json!({"procedure_feedback_rejected": "no_observed_two_step_sequence"}));
                    continue;
                }
                let recorded = if attempt.reflected_then_spoke {
                    crate::steps::procedural::record_verified_sequence_success(
                        &mut self.memory.graph, command.observation_id,
                        &attempt.stimulus, &attempt.response, &attempt.stimulus_embedding,
                        now, self.config.decay_d,
                    )
                } else if attempt.reflected_then_asked {
                    crate::steps::procedural::record_verified_ask_success(
                        &mut self.memory.graph, command.observation_id,
                        &attempt.stimulus, &attempt.response, &attempt.stimulus_embedding,
                        now, self.config.decay_d,
                    )
                } else {
                    crate::steps::procedural::record_verified_ignore_success(
                        &mut self.memory.graph, command.observation_id,
                        &attempt.stimulus, &attempt.stimulus_embedding,
                        now, self.config.decay_d,
                    )
                };
                if let Some(id) = recorded {
                    self.memory.dirty_ids.insert(id);
                    procedure_feedback_changed_this_tick = true;
                    self.memory.refresh_compiled_procedure(&attempt.stimulus);
                    let successes = self.memory.graph.get(&id).and_then(|object| object.data["verified_successes"].as_u64()).unwrap_or(0);
                    let consequence = if attempt.reflected_then_spoke { "spoke_exact" }
                        else if attempt.reflected_then_asked { "asked_exact" } else { "ignored" };
                    self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None,
                        json!({"verified_operator_sequence_credited": true, "id": id.to_string(),
                            "verified_successes": successes, "expected_consequence": consequence}));
                } else {
                    self.emit_event(CyclePhase::Learn, CycleEventKind::Error, None,
                        json!({"procedure_feedback_rejected": "unsafe_or_invalid_sequence"}));
                }
            } else if attempt.reflected_then_spoke || attempt.compiled_spoke
                || attempt.reflected_then_asked || attempt.compiled_asked
                || attempt.reflected_then_ignored || attempt.compiled_ignored {
                if let Some(id) = crate::steps::procedural::record_verified_sequence_failure(
                    &mut self.memory.graph, &attempt.stimulus, now,
                ) {
                    self.memory.dirty_ids.insert(id);
                    procedure_feedback_changed_this_tick = true;
                    self.memory.refresh_compiled_procedure(&attempt.stimulus);
                    let consequence = if attempt.compiled_spoke || attempt.reflected_then_spoke { "spoke_exact" }
                        else if attempt.compiled_asked || attempt.reflected_then_asked { "asked_exact" } else { "ignored" };
                    self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None,
                        json!({"verified_operator_sequence_demoted": true, "id": id.to_string(),
                            "expected_consequence": consequence}));
                }
            }
        }
        for _ in 0..64 {
            let Some(command) = self.procedure_feedback.take_outcome_feedback() else { break; };
            if self.procedure_feedback.label_compared_observation(command.observation_id, now) {
                outcome_feedback_changed_this_tick = true;
                self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None,
                    json!({"independent_observation_outcome": true,
                        "observation_id": command.observation_id.to_string(),
                        "successful": command.successful}));
            } else {
                self.emit_event(CyclePhase::Learn, CycleEventKind::Error, None,
                    json!({"outcome_feedback_rejected": "unknown_expired_or_already_labeled_observation",
                        "observation_id": command.observation_id.to_string()}));
            }
        }
        // Curation is serialized through this actor, never performed by an
        // API worker directly mutating graph/index state. Bounded drain keeps
        // an update burst from monopolizing a foreground cognitive event.
        for _ in 0..64 {
            let Some(command) = self.perception.take_curated_answer_command() else { break; };
            match command {
                CuratedAnswerCommand::Upsert { question, answer, question_embedding } => {
                    if let Some(id) = crate::steps::known_answers::record_curated_answer(
                        &mut self.memory.graph, &question, &answer, &question_embedding,
                        now, self.config.decay_d,
                    ) {
                        self.memory.dirty_ids.insert(id);
                        curated_answer_changed_this_tick = true;
                        self.memory.curated_answers = crate::steps::known_answers::curated_answer_index(&self.memory.graph);
                        self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None,
                            json!({"curated_answer_updated": true, "id": id.to_string()}));
                    } else {
                        self.emit_event(CyclePhase::Learn, CycleEventKind::Error, None,
                            json!({"curated_answer_rejected": true}));
                    }
                }
                CuratedAnswerCommand::Revoke { question } => {
                    let ids = crate::steps::known_answers::revoke_curated_answer(&mut self.memory.graph, &question);
                    for id in &ids { self.memory.dirty_ids.insert(*id); }
                    curated_answer_changed_this_tick |= !ids.is_empty();
                    self.memory.curated_answers = crate::steps::known_answers::curated_answer_index(&self.memory.graph);
                    self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None,
                        json!({"curated_answer_revoked": ids.len()}));
                }
            }
        }
        let mut input_token_estimate = 0usize;
        let mut foreground_input_this_tick = false;
        let mut compiled_procedure_hit = false;
        let mut curated_answer_hit = false;
        let mut deferred_agenda_spawn_for_responsiveness = false;

        // Per-step wall-clock cost, folded into this tick's Telemetry event
        // as "phase_ms" - `elapsed_ms` alone can't say *which* step a slow
        // tick spent its time in (Predict/Compare/Coalition/Broadcast are
        // all cheap, in-process math; these are the only sub-blocks that can
        // actually stall on I/O or a model call). Left at `0` whenever the
        // corresponding block doesn't run this tick (cache hit, idle tick,
        // no impasse, etc.) - never `None`, so a report can sum/compare
        // across ticks without an Option check.
        let mut embedding_wait_ms = 0u64;
        let mut attention_model_ms = 0u64;
        let mut propose_operators_ms = 0u64;
        let mut act_ms = 0u64;
        let mut impasse_escalation_ms = 0u64;
        let mut synthesize_ms = 0u64;

        // Agency/normativity's reward channel, independent of Predict/
        // Observe/Compare below: has any goal transitioned since last tick's
        // snapshot, regardless of whether *this* tick also has fresh input.
        // Computed unconditionally (not nested inside the `if let
        // Some(incoming)` block below) so `previous_goal_statuses` never
        // misses a transition just because a given tick happened to be
        // idle - see `steps::metacognition::detect_goal_progress`'s doc
        // comment. Folded into `self.prediction.affect_tracker.update` only on ticks
        // that also produce a fresh epistemic reward (below); a transition
        // noticed on an otherwise-idle tick simply has nothing to blend
        // with this tick and isn't retried - a named simplification, not a
        // guarantee every transition reaches affect.
        let (goal_progress_reward, current_goal_statuses) = detect_goal_progress(&self.memory.graph, &self.previous_goal_statuses);
        if !self.config.ablation_config.disable_goal_consequence_learning {
            let recent_decision = self.executive.last_operator_proposal.as_ref()
                .filter(|proposal| proposal.operator != "Ignore" && proposal.at.elapsed_ms_until(now) <= 10_000)
                .map(|proposal| proposal.target_id);
            if let Some(source_id) = recent_decision {
                for (&goal_id, &status) in &current_goal_statuses {
                    if source_id == goal_id { continue; }
                    let Some(previous) = self.previous_goal_statuses.get(&goal_id).copied() else { continue };
                    let Some(reward) = reward_for_goal_transition(previous, status) else { continue };
                    let kind = if reward > 0.5 { EdgeKind::Supports } else { EdgeKind::Contradicts };
                    if let Some(source) = self.memory.graph.get_mut(&source_id) {
                        // Checked before reinforcing, not after: only a
                        // genuinely new Contradicts edge should cost
                        // confidence - a standing low-reward goal
                        // relationship merely being re-coactivated on a
                        // later tick must not crater it repeatedly.
                        let is_new_contradiction = kind == EdgeKind::Contradicts && !source.edges.iter().any(|edge| edge.target_id == goal_id && edge.kind == EdgeKind::Contradicts);
                        reinforce_edge(&mut source.edges, goal_id, kind, now, 0.1, DEFAULT_MAX_EDGE_STRENGTH);
                        if is_new_contradiction {
                            crate::steps::confidence_revision::apply_contradiction_penalty(source, &self.config.confidence_revision);
                        }
                        self.memory.dirty_ids.insert(source_id);
                    }
                }
            }
        }
        self.previous_goal_statuses = current_goal_statuses;

        // --- Agenda, early phase: give existing Active intentions a real
        // shot at this tick's attention before Coalition ever runs - see
        // `steps::agenda::surface_active_intentions`'s own doc comment for
        // why this must run here, unconditionally, rather than only as
        // part of the late-phase revision near the end of this function.
        let newly_surfaced_intentions = if self.config.ablation_config.disable_agenda {
            Vec::new()
        } else {
            crate::steps::agenda::surface_active_intentions(&mut self.memory.graph, &self.memory.working_memory, &mut self.memory.pending_admission, &self.config.agenda_config, now)
        };
        if !newly_surfaced_intentions.is_empty() {
            self.memory.dirty_ids.extend(newly_surfaced_intentions.iter().copied());
            self.emit_event(
                CyclePhase::Agenda,
                CycleEventKind::Normal,
                None,
                json!({"surfaced": newly_surfaced_intentions.iter().map(MentalObjectId::to_string).collect::<Vec<_>>()}),
            );
        }

        // --- Steps 1-3: Predict / Observe / Compare, only when new input
        // arrived this tick. Draining is non-blocking - an idle tick with no
        // input costs almost nothing. ---
        let mut new_object_id: Option<MentalObjectId> = None;
        let mut new_surprise: Option<f32> = None;
        // This tick's own fresh `ComparisonResult::epistemic_value`
        // reading, if Compare actually ran - `steps::drives::DriveState::
        // update`'s `uncertainty` input. Deliberately a fresh local, not a
        // read of `self.prediction.last_prediction_error` (which persists a *stale*
        // value across idle ticks) - `DriveState::update` needs to know
        // "no fresh reading this tick" precisely, not "whatever the last
        // real one happened to be."
        let mut current_tick_epistemic_value: Option<f32> = None;
        // This tick's (channel, interlocutor) - `Some` only once Compare
        // actually runs below (fresh input this tick). Consulted later, in
        // Steps 7-9, to key any new `metacognition::PendingOutcome`
        // registered for a decision made this tick - a decision with no
        // fresh-input channel to key against this tick simply isn't tracked
        // for consequence feedback (see `steps::metacognition`'s module doc
        // comment for why that scope limit is deliberate).
        let mut current_outcome_context: Option<(SourceChannel, Option<MentalObjectId>)> = None;
        let mut current_orienting_result: Option<crate::steps::orient::OrientingResult> = None;
        // Set only by the `Incoming::Boredom` arm below, when the stimulus
        // came from `steps::boredom::generate_daydream` - the dormant
        // memories it recombined, carried forward so they can be
        // Hebbian-linked to whatever new Observation id this tick actually
        // produces (see the link-back site right after `new_object_id`
        // resolves). A plain local, not read off `stimulus.source_ids`
        // again later, because `stimulus` itself doesn't survive past the
        // `match` below.
        let mut daydream_source_ids: Vec<MentalObjectId> = Vec::new();

        // A previously-enqueued embedding request completing takes priority
        // over fresh input this tick - see `embedding_worker`'s own doc
        // comment. Checked *before* draining any `Incoming` source below (via
        // the `embedding_reentry.is_none() &&` guard on that block) so that,
        // on a tick where a reentry is ready, this tick's `Incoming` sources
        // are never even drained - whatever's queued there simply waits for
        // its own `try_recv` next tick, exactly like an ordinary backlog
        // already does. This keeps "at most one new observation per tick"
        // true regardless of which of the two paths produced it.
        let embedding_reentry = self.prediction.take_embedding_reentry();

        // Seven possible origins for this tick's input, tried in priority
        // order and still non-blocking: a typed/API turn, a room-audio
        // utterance (see `RoomInput`), a turn from another agent on the
        // household network (MCP `send_input`), a Knowledge Library consult
        // result, a tool invocation result looping back to become an
        // Observation, an observation forwarded by an external sensing
        // service (a camera service, etc. - see `SensorInput`), or - only
        // when every one of those six comes up empty and Working Memory has
        // genuinely had nothing in it for a while - a self-generated
        // `steps::boredom` stimulus. At most one is taken per tick, same
        // discipline as before - this just widens *where* that one object
        // can come from. Boredom is deliberately last: real input of any
        // kind always preempts self-generated activity on any tick where
        // both are pending, which is what makes "stay reachable" hold
        // without any special-casing.
        enum Incoming {
            External(String),
            Room(RoomInput),
            FromAgent(ExternalAgentInput),
            KnowledgeReentry(String),
            ActReentry(String),
            Sensor(SensorInput),
            SensorSignal(SensorSignal),
            Boredom(crate::steps::boredom::BoredomStimulus),
        }
        // A backlog on `input_rx` (e.g. transcripts arriving faster than
        // the loop can tick through them) is folded into one "newly
        // discovered" observation for this tick rather than replayed one
        // message at a time until the queue drains - by the time this tick
        // looks, everything already queued happened "just now" from the
        // loop's perspective, so there's no reason to burn a tick per
        // backlogged message once we've fallen behind. `input_tx` is a
        // single-party channel (stdin, the local HTTP API) - there is only
        // ever one "stream" here, so blending backlog is safe in a way it
        // is not for `room_input_rx` below.
        // Every source below is `try_recv`/`pop_front` - i.e. consuming -
        // so none of it may run on a tick that isn't going to use whatever
        // it pops. Gated on the same `embedding_reentry.is_none()` check as
        // the block that actually consumes `incoming` below: if a reentry
        // is ready, this tick doesn't touch a single `Incoming` source, not
        // even to buffer-without-popping - everything queued simply waits
        // for its own look on a later tick, exactly like an ordinary
        // backlog already does. Without this, a room-audio backlog item
        // (say) could be popped off `pending_room_input` on a reentry tick,
        // computed into `incoming`, and then silently dropped when the
        // block below never consumes it - a real, confirmed bug this gate
        // exists specifically to prevent.
        let incoming = if embedding_reentry.is_some() {
            None
        } else if let Some(signal) = self.perception.take_sensor_signal() {
            // A calibrated urgency/threat event preempts ordinary backlog.
            // Do not drain or pop any other source on this tick: its queued
            // input is preserved for the next cognitive event.
            Some(Incoming::SensorSignal(signal))
        } else {
            let coalesced_input = self.perception.take_input().map(|first| {
                let mut text = first;
                while let Ok(more) = self.perception.input_rx.try_recv() {
                    text.push(' ');
                    text.push_str(&more);
                }
                text
            });
            // Room-audio backlog: drain whatever's newly queued (still
            // non-blocking) into `pending_room_input`, then coalesce only a
            // same-`stream_id` run off the *front* of that buffer - never
            // blending utterances from different speakers into one Observation.
            // Anything left in the buffer (a different stream) simply waits for
            // a later tick, which is also when it becomes the front and gets
            // its own turn - streams naturally interleave across ticks instead
            // of blending within one.
            while let Ok(item) = self.perception.room_input_rx.try_recv() {
                self.perception.pending_room_input.push_back(item);
            }
            let room_incoming = self.perception.pending_room_input.pop_front().map(|first| {
                let stream_id = first.stream_id.clone();
                let mut text = first.text;
                while self.perception.pending_room_input.front().is_some_and(|next| next.stream_id == stream_id) {
                    let next = self.perception.pending_room_input.pop_front().expect("front just matched, so pop_front cannot fail here");
                    text.push(' ');
                    text.push_str(&next.text);
                }
                RoomInput { text, speaker_label: first.speaker_label, stream_id }
            });
            let incoming = coalesced_input
                .map(Incoming::External)
                .or_else(|| room_incoming.map(Incoming::Room))
                .or_else(|| self.perception.take_external_agent_input().map(Incoming::FromAgent))
                .or_else(|| self.perception.take_kl_reentry().map(Incoming::KnowledgeReentry))
                .or_else(|| self.perception.take_act_reentry().map(Incoming::ActReentry))
                .or_else(|| self.perception.take_sensor_input().map(Incoming::Sensor));

            let boredom_config = &self.config.boredom_config;
            let self_status_available = self.tool_registry.available().iter().any(|(name, _)| *name == crate::steps::tools::SelfStatusTool::NAME);
            // Interrupt-capable self-check: unlike `boredom_eligible` below,
            // this deliberately does NOT require Working Memory to be empty
            // - see `BoredomConfig::self_status_interrupt_drive_threshold`'s
            // own doc comment for why real, sustained self-monitoring
            // pressure is allowed to put a self-status candidate up for
            // Broadcast competition even while something else already
            // occupies Working Memory, rather than waiting for the ordinary
            // idle gate. Still requires `incoming.is_none()` (real input
            // arriving this exact tick always still takes priority, per the
            // `Incoming` enum's own doc comment) and the same shared
            // `min_interval_ms` cost-control backstop `boredom_eligible`
            // uses - this is a different *gate*, not a way around the
            // existing cost control.
            let self_status_interrupt_due = !self.config.ablation_config.disable_boredom
                && incoming.is_none()
                && self_status_available
                && self.drives.drive_state.self_monitoring_pressure() >= boredom_config.self_status_interrupt_drive_threshold
                && self.perception.last_boredom_at.is_none_or(|t| now.0 - t.0 >= boredom_config.min_interval_ms);
            let boredom_eligible = !self_status_interrupt_due
                && !self.config.ablation_config.disable_boredom
                && incoming.is_none()
                && self.memory.working_memory.is_empty()
                && self.perception.wm_empty_since.is_some_and(|t| now.0 - t.0 >= boredom_config.idle_threshold_ms)
                && self.perception.last_boredom_at.is_none_or(|t| now.0 - t.0 >= boredom_config.min_interval_ms);
            if self_status_interrupt_due {
                self.perception.last_boredom_at = Some(now);
                self.perception.last_self_status_at = Some(now);
                Some(Incoming::Boredom(crate::steps::boredom::self_status_stimulus(true)))
            } else if boredom_eligible {
                self.perception.last_boredom_at = Some(now);
                let stimulus = crate::steps::boredom::generate(
                    &self.memory.graph,
                    &self.model.tier1_pool,
                    &self.tool_registry.available(),
                    self.perception.last_self_status_at,
                    &self.drives.drive_state,
                    boredom_config,
                    &self.memory.self_summary,
                    now,
                    &mut self.rng,
                    self.config.temperature.boredom,
                )
                .await;
                if let Some(stimulus) = &stimulus
                    && stimulus.requested_tool == Some(crate::steps::tools::SelfStatusTool::NAME)
                {
                    self.perception.last_self_status_at = Some(now);
                }
                stimulus.map(Incoming::Boredom)
            } else {
                incoming
            }
        };

        if let Some(incoming) = incoming {
            let actor_ingress_started = Instant::now();
            foreground_input_this_tick = matches!(&incoming, Incoming::External(_) | Incoming::Room(_) | Incoming::FromAgent(_) | Incoming::SensorSignal(_));
            let mut sensor_embedding: Option<Vec<f32>> = None;
            let (text, channel, data_tag): (String, SourceChannel, Option<serde_json::Value>) = match incoming {
                Incoming::External(text) => (text, SourceChannel::ConversationInput, None),
                Incoming::Room(input) => {
                    // `speaker_label` only ever reaches here already
                    // filtered to a named/enrolled speaker (see
                    // `RoomInput`'s doc comment) - `stream_id` is kept too,
                    // purely for the coalescing discipline above, and
                    // carries no identity meaning of its own.
                    let mut data = json!({"source": "voice", "stream_id": input.stream_id});
                    if let Some(label) = &input.speaker_label {
                        data["speaker_label"] = json!(label);
                    }
                    (input.text, SourceChannel::ConversationInput, Some(data))
                }
                Incoming::FromAgent(input) => (
                    input.text,
                    // Same channel as a human, deliberately - the tool's
                    // contract is "no privilege, full competition pipeline."
                    SourceChannel::ConversationInput,
                    Some(json!({"source": "external-agent", "agent_id": input.agent_id})),
                ),
                Incoming::KnowledgeReentry(text) => (
                    text,
                    SourceChannel::ExternalKnowledge,
                    Some(json!({"source": "knowledge-library"})),
                ),
                Incoming::ActReentry(text) => (
                    text,
                    // A tool's output is at least as reliable as a document
                    // in the Knowledge Library (it's a direct read of
                    // in-process state, not a fetched/ingested text blob),
                    // so it shares that channel's precision bucket rather
                    // than getting a fourth one for a distinction without a
                    // real difference yet.
                    SourceChannel::ExternalKnowledge,
                    Some(json!({"source": "tool"})),
                ),
                Incoming::Sensor(input) => {
                    // `speaker_label` here is deliberately the same key
                    // `Incoming::Room` sets - see `SensorInput::entity_label`'s
                    // doc comment for why sharing the key (not a parallel
                    // field/resolution path) is what lets a video-recognized
                    // identity and a voice-enrolled one land on one
                    // interlocutor node.
                    let mut data = json!({"source": input.source});
                    if let Some(label) = &input.entity_label {
                        data["speaker_label"] = json!(label);
                    }
                    (input.text, input.channel, Some(data))
                }
                Incoming::SensorSignal(input) => {
                    sensor_embedding = input.embedding.filter(|embedding| !embedding.is_empty() && embedding.iter().all(|value| value.is_finite()));
                    let threat = if input.threat.is_finite() { input.threat.clamp(0.0, 1.0) } else { 0.0 };
                    let urgency = if input.urgency.is_finite() { input.urgency.clamp(0.0, 1.0) } else { 0.0 };
                    let mut data = json!({
                        "source": input.source,
                        "sensor_signal": true,
                        "threat": threat,
                        "urgency": urgency,
                    });
                    if let Some(label) = &input.entity_label { data["speaker_label"] = json!(label); }
                    (input.text, input.channel, Some(data))
                }
                Incoming::Boredom(stimulus) => {
                    // `requested_tool`, when present, is what lets
                    // `steps::executive::propose_operators` propose
                    // `Operator::Act` for it directly next tick, without
                    // needing the invented/duty text to happen to match a
                    // reactive trigger phrase (see that function's own
                    // `requested_tool` check).
                    //
                    // `source` distinguishes a memory-grounded daydream
                    // (`steps::boredom::generate_daydream`, non-empty
                    // `source_ids`) from the standing duty or plain Tier 1
                    // invention - purely observational (nothing downstream
                    // branches on it today), the same "make the real
                    // distinction visible" reasoning as `data.source` for
                    // every other `Incoming` arm above.
                    let mut data = json!({"source": if stimulus.source_ids.is_empty() { "boredom" } else { "daydream" }});
                    if let Some(tool) = stimulus.requested_tool {
                        data["requested_tool"] = json!(tool);
                    }
                    if !stimulus.source_ids.is_empty() {
                        data["daydream_source_ids"] = json!(stimulus.source_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>());
                        // Carried forward to the link-back site after
                        // `new_object_id` resolves below - see
                        // `daydream_source_ids`'s own doc comment.
                        daydream_source_ids = stimulus.source_ids;
                    }
                    (stimulus.text, SourceChannel::SelfGeneratedThought, Some(data))
                }
            };
            input_token_estimate = estimate_token_count(&text);

            if channel == SourceChannel::ConversationInput {
                self.drives.last_conversation_input_at = Some(now);
            }

            // Resolved (find-or-created) *before* Predict/Compare - Predict
            // needs it to prefer a per-interlocutor expectation over the
            // generic per-channel one (see `interlocutor_embeddings`'s doc
            // comment), and Compare's precision lookup needs it too
            // (`PrecisionTracker::precision_for_interlocutor`). `None` for
            // everything except a room-audio utterance that resolved to a
            // named/enrolled speaker.
            let speaker_label = data_tag.as_ref().and_then(|v| v.get("speaker_label")).and_then(|v| v.as_str()).map(str::to_string);
            let interlocutor_id = speaker_label.as_deref().map(|label| crate::steps::interlocutor::find_or_create(&mut self.memory.graph, label, now, self.config.decay_d));

            let working_memory_embeddings: Vec<&[f32]> = self
                .memory.working_memory
                .iter()
                .filter_map(|id| self.memory.graph.get(id))
                .filter_map(|o| o.embedding.as_deref())
                .collect();
            let due_goal = most_salient_active_goal(&self.memory.graph);
            // "Specific overrides generic once earned," same hierarchy as
            // `PrecisionTracker::precision_for_interlocutor` - a recognized
            // interlocutor with a history of their own is predicted against
            // what *they* last said, not the blended-across-everyone
            // channel stream.
            let previous_observation_embedding = interlocutor_id
                .and_then(|id| self.prediction.interlocutor_embeddings.get(&id))
                .or_else(|| self.prediction.channel_embeddings.get(&channel))
                .map(Vec::as_slice);
            let inputs = PredictionInputs {
                previous_observation_embedding,
                working_memory_embeddings,
                due_goal_embedding: due_goal.and_then(|g| g.embedding.as_deref()),
            };
            let weights = PredictWeights::from_precision(&self.prediction.precision_tracker, channel);
            let expected = predict_expected_embedding(&inputs, &weights);
            self.emit_event(CyclePhase::Predict, CycleEventKind::Normal, None, json!({"has_expectation": expected.is_some()}));

            let mut observation = new_observation_shell(text.clone(), now, self.config.decay_d);
            if foreground_input_this_tick {
                self.turn_latency.start(observation.id, actor_ingress_started);
            }
            if let Some(tag) = data_tag {
                observation.data = tag;
            }

            // A mature routine procedure carries the embedding earned when
            // it was compiled, so exact/normalized recognition can complete
            // Observe locally too. Empty embeddings (legacy/test skills)
            // deliberately fall through to the ordinary cache/worker path.
            let compiled_procedure = (!self.config.ablation_config.disable_procedural_fast_path
                && channel == SourceChannel::ConversationInput)
                .then(|| self.memory.compiled_procedure(&text).cloned())
                .flatten()
                .filter(|procedure| !procedure.stimulus_embedding.is_empty());
            let curated_answer = (!self.config.ablation_config.disable_curated_answer_fast_path
                && channel == SourceChannel::ConversationInput)
                .then(|| self.memory.curated_answer(&text).cloned())
                .flatten();
            if let Some(answer) = curated_answer {
                curated_answer_hit = true;
                if !observation.data.is_object() { observation.data = json!({}); }
                observation.data["curated_answer"] = json!(answer.answer);
                let outcome = self.apply_embedding_result(
                    observation, expected, channel, interlocutor_id, &text,
                    goal_progress_reward, now, Ok(answer.question_embedding),
                );
                new_object_id = outcome.0;
                new_surprise = outcome.1;
                current_tick_epistemic_value = outcome.2;
                current_outcome_context = outcome.3;
                current_orienting_result = outcome.4;
            } else if let Some(procedure) = compiled_procedure {
                compiled_procedure_hit = true;
                if !observation.data.is_object() {
                    observation.data = json!({});
                }
                match procedure.execution(&text).expect("compiled index stores only condition-matching validated programs") {
                    crate::steps::procedural::CompiledExecution::SpeakExact(response) =>
                        observation.data["compiled_response"] = json!(response),
                    crate::steps::procedural::CompiledExecution::AskExact(question) =>
                        observation.data["compiled_question"] = json!(question),
                    crate::steps::procedural::CompiledExecution::Ignore =>
                        observation.data["compiled_ignore"] = json!(true),
                }
                let outcome = self.apply_embedding_result(
                    observation,
                    expected,
                    channel,
                    interlocutor_id,
                    &text,
                    goal_progress_reward,
                    now,
                    Ok(procedure.stimulus_embedding),
                );
                new_object_id = outcome.0;
                new_surprise = outcome.1;
                current_tick_epistemic_value = outcome.2;
                current_outcome_context = outcome.3;
                current_orienting_result = outcome.4;
            // Exact-text cache checked next (see `embedding_cache`'s doc
            // comment) - a hit is a real network/model call avoided
            // entirely, not just hidden latency, so it's applied inline,
            // synchronously, same as before - only a genuine cache miss goes
            // to `embedding_worker` below.
            } else if let Some(supplied) = sensor_embedding {
                let outcome = self.apply_embedding_result(observation, expected, channel, interlocutor_id, &text, goal_progress_reward, now, Ok(supplied));
                new_object_id = outcome.0;
                new_surprise = outcome.1;
                current_tick_epistemic_value = outcome.2;
                current_outcome_context = outcome.3;
                current_orienting_result = outcome.4;
            } else if let Some(cached) = self.prediction.embedding_cache.get(&observation.text) {
                let cached = cached.clone();
                let outcome = self.apply_embedding_result(observation, expected, channel, interlocutor_id, &text, goal_progress_reward, now, Ok(cached));
                new_object_id = outcome.0;
                new_surprise = outcome.1;
                current_tick_epistemic_value = outcome.2;
                current_outcome_context = outcome.3;
                current_orienting_result = outcome.4;
            } else {
                // Published *before* enqueuing, not after: a viewer polling
                // the snapshot needs to see this turn on the instant a real
                // embedding call is in flight (the watch channel is read by
                // other tasks, so this isn't blocked by the actor being
                // parked here) - same reasoning the old inline-await version
                // had, just resolved by `embedding_worker` now instead of
                // blocking this tick. Cleared by the reentry-drain above
                // once the result actually comes back, however many ticks
                // later that is.
                self.prediction.embedding_in_flight.store(true, Ordering::Relaxed);
                self.publish_snapshot().await;
                // `take` (not a plain read) - this branch means `new_object_id`
                // stays `None` this tick (the async path), so nothing here
                // would consult the outer `daydream_source_ids` local again
                // regardless, but taking it makes that explicit rather than
                // relying on that fact.
                let pending = PendingObservation { observation, expected, channel, interlocutor_id, enqueued_at: Instant::now(), daydream_source_ids: std::mem::take(&mut daydream_source_ids) };
                if self.prediction.embedding_request_tx.try_send(pending).is_err() {
                    // Worker channel full or gone - log-and-drop, same
                    // fallback shape as the KL/Act reentry self-sends
                    // elsewhere in this function: this tick's observation is
                    // lost rather than ever blocking the loop on a single
                    // slow/backed-up dependency.
                    tracing::warn!("failed to enqueue embedding request - dropping this tick's observation");
                    self.prediction.embedding_in_flight.store(false, Ordering::Relaxed);
                }
            }
        }

        if let Some(reentry) = embedding_reentry {
            self.prediction.embedding_in_flight.store(false, Ordering::Relaxed);
            embedding_wait_ms = reentry.pending.enqueued_at.elapsed().as_millis() as u64;
            // Recovered from wherever it was carried across the async
            // boundary (see `PendingObservation::daydream_source_ids`'s own
            // doc comment), *before* `reentry.pending`'s other fields get
            // moved into `apply_embedding_result` below - a clone, not a
            // move, since `reentry.pending` as a whole is still needed for
            // that call.
            daydream_source_ids = reentry.pending.daydream_source_ids.clone();
            if let Ok(embedding) = &reentry.result {
                self.prediction.cache_embedding(reentry.pending.observation.text.clone(), embedding.clone());
            }
            let text = reentry.pending.observation.text.clone();
            let outcome = self.apply_embedding_result(
                reentry.pending.observation,
                reentry.pending.expected,
                reentry.pending.channel,
                reentry.pending.interlocutor_id,
                &text,
                goal_progress_reward,
                now,
                reentry.result,
            );
            new_object_id = outcome.0;
            new_surprise = outcome.1;
            current_tick_epistemic_value = outcome.2;
            current_outcome_context = outcome.3;
            current_orienting_result = outcome.4;
        }

        // Hebbian link-back for a daydream: the recombined thought that
        // just resolved into `new_object_id` gets a real `DerivedFrom` edge
        // from each dormant memory it was recombined from, same edge shape
        // `steps::synthesize::link_sources`/`steps::interlocutor::
        // reinforce_link` already use for their own outputs. This is what
        // makes a daydream actually restructure the graph's associative
        // memory the way replay is theorized to, rather than only ever
        // producing a one-off spoken thought with no lasting trace back to
        // what it drew on. A no-op for every non-daydream observation
        // (`daydream_source_ids` stays empty), and silently skips any
        // source id that's since decayed out of the graph - same
        // tolerance `synthesize::link_sources` already has for exactly
        // this situation.
        if let Some(id) = new_object_id {
            for source_id in &daydream_source_ids {
                if let Some(source) = self.memory.graph.get_mut(source_id) {
                    reinforce_edge(&mut source.edges, id, EdgeKind::DerivedFrom, now, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH);
                    self.memory.dirty_ids.insert(*source_id);
                }
            }
        }

        // --- Step 4: Update memory dynamics --- recompute activation for
        // the new object (if any) and every current Working Memory member,
        // spreading from a snapshot of the other members' outgoing edges to
        // sidestep a self-referential mutable/immutable borrow on the same
        // graph. Only `id`/`edges` are ever read by `compute_spreading_
        // activation`, so the snapshot carries just those - not a full
        // `MentalObject` clone (text, embedding, JSON payload) per member,
        // every tick, for fields spreading activation never touches.
        let wm_snapshot: Vec<(MentalObjectId, Vec<AssociativeEdge>)> = self.memory.working_memory.iter().filter_map(|id| self.memory.graph.get(id).map(|o| (*id, o.edges.clone()))).collect();

        if let Some(id) = new_object_id {
            let sources: Vec<&[AssociativeEdge]> = wm_snapshot.iter().map(|(_, edges)| edges.as_slice()).collect();
            if let Some(object) = self.memory.graph.get_mut(&id) {
                recompute_activation(&mut object.activation, id, &sources, self.config.noise_max, self.config.edge_decay_rate_per_ms, &mut self.rng, self.clock.as_ref());
                apply_self_memory_activation_bonus(object, self.config.self_memory_activation_bonus);
            }
        }
        for &id in &self.memory.working_memory {
            let sources: Vec<&[AssociativeEdge]> = wm_snapshot.iter().filter(|(oid, _)| *oid != id).map(|(_, edges)| edges.as_slice()).collect();
            if let Some(object) = self.memory.graph.get_mut(&id) {
                recompute_activation(&mut object.activation, id, &sources, self.config.noise_max, self.config.edge_decay_rate_per_ms, &mut self.rng, self.clock.as_ref());
                apply_self_memory_activation_bonus(object, self.config.self_memory_activation_bonus);
            }
        }
        // A Reflection or subgoal created during a *previous* tick's Act
        // step never had its activation computed at all (it was inserted
        // with the all-zero default) - give it a real one now, the same way
        // `new_object_id` gets one above, before it competes in Coalition.
        for &id in &self.memory.pending_admission {
            let sources: Vec<&[AssociativeEdge]> = wm_snapshot.iter().map(|(_, edges)| edges.as_slice()).collect();
            if let Some(object) = self.memory.graph.get_mut(&id) {
                recompute_activation(&mut object.activation, id, &sources, self.config.noise_max, self.config.edge_decay_rate_per_ms, &mut self.rng, self.clock.as_ref());
                apply_self_memory_activation_bonus(object, self.config.self_memory_activation_bonus);
            }
        }

        // --- Step 5: Form coalitions --- the new object (with its surprise
        // term), every existing Working Memory member (recalled purely on
        // activation, no fresh surprise this tick), and anything pending
        // admission from a previous tick's Act step.
        let mut raw_candidates: Vec<(MentalObjectId, f32, Option<f32>)> = Vec::new();
        let mut preconscious_activity = false;
        if let Some(id) = new_object_id {
            let sensor_habituated = self.memory.graph.get(&id).is_some_and(|object|
                object.data.get("sensor_habituated").and_then(|value| value.as_bool()) == Some(true));
            // GNW's ignition-then-refractory dynamics (see
            // `LoopConfig::attentional_refractory_ms`'s doc comment): a
            // fresh candidate from a channel that just had a win is not
            // nominated this tick if still inside that channel's refractory
            // window - it isn't lost, just deferred, via the same
            // `pending_admission` one-more-honest-shot mechanism a
            // mid-tick-created Reflection already relies on.
            let channel = current_outcome_context.map(|(channel, _)| channel);
            let in_refractory = channel.is_some_and(|channel| {
                self.memory.last_broadcast_by_channel.get(&channel).is_some_and(|last| now.0 - last.0 < self.config.attentional_refractory_ms)
            });
            if sensor_habituated {
                // Memory and numerical prediction are updated above, but
                // this repeated low-salience environmental reading never
                // reaches the global-workspace/model path on its own.
            } else if in_refractory {
                self.memory.pending_admission.insert(id);
                if let Some(surprise) = new_surprise {
                    self.memory.pending_admission_surprise.insert(id, surprise);
                }
            } else if let Some(object) = self.memory.graph.get(&id) {
                raw_candidates.push((id, object.activation.total, new_surprise));
            }
        }
        for &id in &self.memory.working_memory {
            if Some(id) == new_object_id {
                continue;
            }
            if let Some(object) = self.memory.graph.get(&id) {
                // A raw Observation that's already been reflected upon has
                // handed its communicative role to that Reflection - it's
                // simply not re-nominated here anymore. This (not a manual
                // removal from `self.memory.working_memory`) is what lets it drop
                // out of Working Memory: the ordinary Broadcast "released"
                // path already handles anything that stops being nominated,
                // so this stays consistent with everything computed earlier
                // in *this* tick (Learn's co-activation credit, the
                // published snapshot's edges) still seeing it as having
                // genuinely been present.
                let superseded = object.kind == aca_types::MentalObjectKind::Observation && has_reflection_for(&self.memory.graph, id);
                if !superseded {
                    raw_candidates.push((id, object.activation.total, None));
                }
            }
        }
        // Each pending object gets exactly one honest shot at admission per
        // creation event - drained here regardless of whether it wins, so
        // it doesn't linger and get re-added (and re-scored stale) forever.
        for id in self.memory.pending_admission.drain() {
            if self.memory.working_memory.contains(&id) {
                continue;
            }
            if let Some(object) = self.memory.graph.get(&id) {
                // A plain Reflection/subgoal deferred here never had a
                // surprise term of its own (`None`, as before). A raw
                // Observation deferred instead by this tick's attentional-
                // refractory check (see Step 5 above) is the one exception -
                // without carrying its `new_surprise` forward to this, its
                // one delayed shot, it competes on bare activation alone and
                // can lose a Coalition bid a fresh, non-deferred arrival of
                // the same content would have won, silently losing the
                // input rather than merely delaying it by one tick.
                let surprise = self.memory.pending_admission_surprise.remove(&id);
                raw_candidates.push((id, object.activation.total, surprise));
            }
        }

        // An actual source firing wakes only its bounded linked neighborhood.
        // Signed pulses accumulate by target before one lazy leakage/fire
        // evaluation; a dormant object that fires earns a Coalition bid.
        if self.config.ablation_config.disable_spike_propagation {
            self.memory.spike_events.clear();
        } else {
            let due = self.memory.spike_events.drain_due(now, self.config.spike_max_events_per_tick.max(1));
            let mut target_pulses: HashMap<MentalObjectId, f32> = HashMap::new();
            for event in &due {
                if self.memory.graph.get(&event.source).is_some_and(|source| source.status == aca_types::ObjectStatus::Active)
                    && self.memory.graph.get(&event.target).is_some_and(|target| target.status == aca_types::ObjectStatus::Active)
                {
                    *target_pulses.entry(event.target).or_default() += event.pulse;
                }
            }
            let mut fired_targets = Vec::new();
            for (id, pulse) in target_pulses {
                if let Some(object) = self.memory.graph.get_mut(&id) {
                    object.dynamics.threshold = self.config.orienting_config.firing_threshold;
                    aca_graph::stimulate_dynamics(
                        &mut object.dynamics, pulse, now,
                        self.config.orienting_config.potential_tau_ms,
                        self.config.orienting_config.adaptation_tau_ms,
                    );
                    if aca_graph::try_fire_dynamics(
                        &mut object.dynamics, now,
                        self.config.orienting_config.refractory_ms,
                        self.config.orienting_config.adaptation_increment,
                    ) {
                        fired_targets.push(id);
                        if !self.memory.working_memory.contains(&id)
                            && !raw_candidates.iter().any(|(candidate_id, _, _)| *candidate_id == id)
                            && object.is_embedding_resolved()
                            && object.data.get("interlocutor_hint").is_none()
                        {
                            let sources: Vec<&[AssociativeEdge]> = wm_snapshot.iter().map(|(_, edges)| edges.as_slice()).collect();
                            recompute_activation(&mut object.activation, id, &sources, self.config.noise_max,
                                self.config.edge_decay_rate_per_ms, &mut self.rng, self.clock.as_ref());
                            apply_self_memory_activation_bonus(object, self.config.self_memory_activation_bonus);
                            raw_candidates.push((id, object.activation.total,
                                Some(self.config.orienting_config.attention_pulse)));
                        }
                    }
                }
            }
            for id in fired_targets.iter().copied() {
                self.memory.spike_events.schedule_from(
                    &self.memory.graph, id, now, self.config.spike_delay_ms,
                    self.config.edge_decay_rate_per_ms, self.config.spike_max_fanout,
                    self.config.spike_max_pending,
                );
            }
            if !due.is_empty() {
                self.emit_event(CyclePhase::Coalition, CycleEventKind::Normal, None,
                    json!({"spike_events": due.len(), "spike_firings": fired_targets.len(), "spike_pending": self.memory.spike_events.len()}));
            }
        }

        // Event-driven preconscious persistence: an attended candidate that
        // previously lost ignition gets a few bounded re-evaluations instead
        // of disappearing after one tick. The scheduler wakes specifically
        // for `next_evaluation_at`; this block never polls a not-yet-due
        // trace, and each residual surprise pulse weakens on use.
        if !self.config.ablation_config.disable_preconscious_traces {
            let trace_count_before_expiry = self.memory.preconscious_traces.len();
            self.memory.preconscious_traces.retain(|id, trace| trace.expires_at > now && self.memory.graph.get(id).is_some());
            preconscious_activity |= self.memory.preconscious_traces.len() != trace_count_before_expiry;
            let due_traces: Vec<(MentalObjectId, Option<f32>)> = self
                .memory
                .preconscious_traces
                .iter()
                .filter(|(_, trace)| trace.next_evaluation_at <= now)
                .map(|(id, trace)| (*id, trace.surprise))
                .collect();
            preconscious_activity |= !due_traces.is_empty();
            for (id, surprise) in due_traces {
                if let Some(trace) = self.memory.preconscious_traces.get_mut(&id) {
                    trace.next_evaluation_at = EpochMillis(now.0.saturating_add(self.config.preconscious_reevaluation_ms.max(1)));
                    trace.surprise = trace.surprise.map(|value| value * 0.7);
                }
                if self.memory.working_memory.contains(&id) || raw_candidates.iter().any(|(candidate_id, _, _)| *candidate_id == id) {
                    continue;
                }
                let sources: Vec<&[AssociativeEdge]> = wm_snapshot.iter().map(|(_, edges)| edges.as_slice()).collect();
                if let Some(object) = self.memory.graph.get_mut(&id) {
                    recompute_activation(&mut object.activation, id, &sources, self.config.noise_max, self.config.edge_decay_rate_per_ms, &mut self.rng, self.clock.as_ref());
                    apply_self_memory_activation_bonus(object, self.config.self_memory_activation_bonus);
                    if object.status == aca_types::ObjectStatus::Active && object.is_embedding_resolved() {
                        raw_candidates.push((id, object.activation.total, surprise));
                    }
                }
            }
        }

        // --- Step 4.5: Recall --- graph-wide multi-hop spreading
        // activation from current Working Memory (see `steps::recall`'s doc
        // comment): discovers dormant memories Step 4's single-hop spread
        // structurally cannot reach, and folds any that now clear
        // `attention_threshold` into `raw_candidates` alongside everything
        // Step 5 already nominated above - never replacing it.
        if !self.config.ablation_config.disable_recall {
            let excluded: std::collections::HashSet<MentalObjectId> = raw_candidates.iter().map(|(id, _, _)| *id).collect();
            let recalled = crate::steps::recall::recall(
                &mut self.memory.graph,
                &self.memory.working_memory,
                &excluded,
                &self.config.recall_config,
                self.config.edge_decay_rate_per_ms,
                self.config.attention_threshold,
                self.config.self_memory_activation_bonus,
                self.config.noise_max,
                &mut self.rng,
                self.clock.as_ref(),
            );
            for id in recalled {
                if let Some(object) = self.memory.graph.get(&id) {
                    raw_candidates.push((id, object.activation.total, None));
                }
            }
        }

        // --- Step 4.5b: Social recall --- a second, independently
        // ablatable recall pass anchored not at Working Memory but at one
        // specific person's own past utterances (`steps::interlocutor::
        // social_cloud_anchors`) - Phase 4's first real "specialist cloud":
        // heterogeneous not in how it judges candidates (same scoring, same
        // Coalition) but in what it can even see. Only runs on a tick where
        // a fresh utterance from a named/enrolled interlocutor actually just
        // resolved (`current_outcome_context`), since that's the only point
        // in a tick this engine actually knows *whose* cloud is relevant
        // right now - there is deliberately no "last known interlocutor"
        // fallback that would let this fire on an unrelated tick.
        if !self.config.ablation_config.disable_social_recall
            && let Some(interlocutor_id) = current_outcome_context.and_then(|(_, interlocutor_id)| interlocutor_id)
        {
            let anchors = crate::steps::interlocutor::social_cloud_anchors(&self.memory.graph, interlocutor_id);
            if !anchors.is_empty() {
                // `interlocutor_id` itself must never win this - it's a
                // pure memory anchor other objects point at, documented as
                // never a Coalition candidate (`interlocutor::find_or_create`'s
                // own doc comment) - but every anchor utterance here has a
                // real `DerivedFrom` edge straight to it, so without this
                // explicit exclusion `spread_activation_multi_hop` (which
                // follows every edge, not just `Associative` ones) would
                // happily spread activation onto it right along with
                // whatever it's actually meant to recall, and it would
                // start winning Broadcast like any other candidate.
                let mut excluded: std::collections::HashSet<MentalObjectId> = raw_candidates.iter().map(|(id, _, _)| *id).collect();
                excluded.insert(interlocutor_id);
                let recalled = crate::steps::recall::recall(
                    &mut self.memory.graph,
                    &anchors,
                    &excluded,
                    &self.config.recall_config,
                    self.config.edge_decay_rate_per_ms,
                    self.config.attention_threshold,
                    self.config.self_memory_activation_bonus,
                    self.config.noise_max,
                    &mut self.rng,
                    self.clock.as_ref(),
                );
                for id in recalled {
                    if let Some(object) = self.memory.graph.get(&id) {
                        raw_candidates.push((id, object.activation.total, None));
                    }
                }
            }
        }

        // An interlocutor anchor node (`steps::interlocutor::find_or_create`)
        // is documented as never a Coalition candidate - it's a pure memory
        // anchor other objects point at via a real `DerivedFrom` edge, not
        // something Omega "thinks about" directly. That invariant was never
        // actually enforced anywhere before this filter: any real
        // conversational turn already links its own utterance straight to
        // its speaker's interlocutor node, so the moment that utterance was
        // itself in Working Memory, ordinary Step 4.5 recall (spreading from
        // WM, no awareness of "never a candidate" nodes) could already
        // reach the interlocutor node one hop away and nominate it - a real
        // gap, not hypothetical, confirmed live while building the social
        // recall pass below (which reaches the same node just as easily,
        // one hop from any of a person's own anchored utterances). Filtered
        // once here, centrally, rather than duplicated in every recall call
        // site - this is the one point every candidate source (fresh
        // observation, WM re-nomination, pending admission, generic recall,
        // social recall) funnels through before Coalition ever sees it.
        raw_candidates.retain(|(id, _, _)| self.memory.graph.get(id).is_none_or(|object| object.data.get("interlocutor_hint").is_none()));

        let coalition = crate::steps::coalition::form_coalition(&raw_candidates, self.config.attention_threshold);
        // Divisive normalization / crowding - applied once, here, so every
        // downstream consumer of `coalition` (the deterministic path, the
        // hysteresis path, every one of Broadcast's fallback branches
        // below) sees the same crowding-adjusted field, not just one of
        // them. See `steps::coalition::apply_crowding_normalization`'s own
        // doc comment. Deliberately not applied to `raw_candidates` itself
        // or to the attention model's own JSON workspace (`build_attention_
        // workspace_json` below still reads the real, un-normalized ACT-R
        // scores) - crowding is Coalition/Broadcast's own arbitration
        // concern, not a change to what the trained model was calibrated
        // against.
        let coalition = crate::steps::coalition::apply_crowding_normalization(&coalition, self.config.crowding_strength);
        if !coalition.is_empty() {
            let top_score = coalition.iter().map(|c| c.score).fold(f32::NEG_INFINITY, f32::max);
            self.emit_event(
                CyclePhase::Coalition,
                CycleEventKind::Normal,
                None,
                json!({"candidate_count": coalition.len(), "top_score": top_score}),
            );
        }

        // --- Step 6: Broadcast --- GWT's single serial arbitration point.
        // When an attention model is configured, it replaces the
        // deterministic rank/admit-top-N decision below with its own
        // per-tick state transition (see `steps::broadcast`'s doc
        // comments) - falling back to the deterministic algorithm,
        // silently from the rest of the tick's perspective, whenever the
        // model is unconfigured, times out, errors, or answers below
        // `attention_min_confidence`. The cognitive loop's liveness never
        // depends on an unreachable local model.
        let previous_working_memory = self.memory.working_memory.clone();
        // "Current focus" for the model's decision: last tick's highest-
        // attention_score Working Memory member - `self.memory.working_memory`
        // isn't overwritten until below, so it's still valid here.
        let current_focus = previous_working_memory
            .iter()
            .filter_map(|id| self.memory.graph.get(id).map(|object| (*id, object.workspace.attention_score.unwrap_or(f32::NEG_INFINITY))))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id);

        let attention_model_available = self.model.attention_client.is_some();
        let force_reconciliation = attention_model_available && !self.config.attention_shadow_only
            && self.model.ticks_since_attention_reconciliation >= self.config.attention_reconciliation_interval;

        let admitted = if raw_candidates.is_empty() {
            // Nothing to decide about this tick - bypass the whole
            // attention-arbitration path (model call, OOD check,
            // reconciliation bookkeeping, telemetry) entirely, the same way
            // Coalition's own telemetry above is silent when there's
            // nothing to report. This remains important even with the
            // event-driven scheduler: deadlines such as maintenance can wake
            // the actor without producing an attention candidate, so calling
            // out to the model here would still be wasted work -
            // `decide_admission_deterministic` on an empty coalition is
            // already a free no-op.
            decide_admission_with_hysteresis(&coalition, &previous_working_memory, self.config.working_memory_capacity, self.config.ignition_threshold)
        } else if force_reconciliation {
            // Bounds how long the model's incremental, one-decision-per-tick
            // judgment can run without re-syncing against a full
            // deterministic re-rank of live activation values - see
            // `LoopConfig::attention_reconciliation_interval`'s doc comment.
            self.model.ticks_since_attention_reconciliation = 0;
            self.emit_event(CyclePhase::Broadcast, CycleEventKind::Normal, None, json!({"attention_forced_reconciliation": true}));
            decide_admission_with_hysteresis(&coalition, &previous_working_memory, self.config.working_memory_capacity, self.config.ignition_threshold)
        } else {
            match &self.model.attention_client {
                Some(_) if !workspace_in_distribution(&raw_candidates) => {
                    // The model was trained on a bounded synthetic
                    // distribution (candidate count 1-9, activation_total in
                    // [-10, 10] - see `attention_workspace::
                    // workspace_in_distribution`'s doc comment); this tick's
                    // real workspace falls outside it, so there's no
                    // principled basis to trust an extrapolation. Skip
                    // consulting the model entirely rather than gamble on
                    // it, but still surface this as telemetry - real-world
                    // distribution shift over a long-running graph should be
                    // observable, not silent.
                    self.model.ticks_since_attention_reconciliation = 0;
                    self.emit_event(
                        CyclePhase::Broadcast,
                        CycleEventKind::Normal,
                        None,
                        json!({"attention_out_of_distribution": true, "candidate_count": raw_candidates.len()}),
                    );
                    decide_admission_with_hysteresis(&coalition, &previous_working_memory, self.config.working_memory_capacity, self.config.ignition_threshold)
                }
                Some(client) => {
                    let workspace_json = build_attention_workspace_json(
                        &self.memory.graph,
                        &raw_candidates,
                        current_focus,
                        self.config.working_memory_capacity,
                        self.config.attention_threshold,
                        now,
                    );
                    let prompt = attention_workspace_prompt(&workspace_json);
                    self.model.attention_in_flight.store(true, Ordering::Relaxed);
                    let attention_started = Instant::now();
                    let decision = tokio::time::timeout(self.config.attention_timeout, client.suggest(prompt)).await;
                    attention_model_ms = attention_started.elapsed().as_millis() as u64;
                    self.model.attention_in_flight.store(false, Ordering::Relaxed);
                    match decision {
                        Ok(Ok(decision)) if decision.confidence >= self.config.attention_min_confidence && attention_decision_target_is_eligible(&decision, &coalition) => {
                            if !self.config.attention_shadow_only {
                                self.model.ticks_since_attention_reconciliation += 1;
                            }
                            let model_admitted = decide_admission_from_attention_model(
                                &previous_working_memory, &raw_candidates, &decision,
                                self.config.attention_threshold, self.config.working_memory_capacity,
                            );
                            // A paired policy comparison is descriptive, not
                            // outcome feedback. It lets a live run reveal
                            // when the specialist actually changes admission
                            // without copying workspace text into telemetry.
                            let deterministic_shadow = decide_admission_with_hysteresis(
                                &coalition, &previous_working_memory,
                                self.config.working_memory_capacity, self.config.ignition_threshold,
                            );
                            let model_ids: std::collections::HashSet<_> = model_admitted.iter().map(|(id, _)| *id).collect();
                            let shadow_ids: std::collections::HashSet<_> = deterministic_shadow.iter().map(|(id, _)| *id).collect();
                            self.emit_event(
                                CyclePhase::Broadcast,
                                CycleEventKind::Normal,
                                None,
                                json!({
                                    "attention_model_decision": true,
                                    "operation": format!("{:?}", decision.operation),
                                    "target_id": decision.target.map(|id| id.to_string()),
                                    "confidence": decision.confidence,
                                    "reason_code": decision.reason_code,
                                    "candidate_count": raw_candidates.len(),
                                    "model_admitted_ids": model_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
                                    "deterministic_shadow_ids": shadow_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
                                    "shadow_membership_agreement": model_ids == shadow_ids,
                                    "attention_shadow_only": self.config.attention_shadow_only,
                                }),
                            );
                            if self.config.attention_shadow_only { deterministic_shadow } else { model_admitted }
                        }
                        Ok(Ok(decision)) => {
                            self.model.ticks_since_attention_reconciliation = 0;
                            tracing::debug!(confidence = decision.confidence, "attention model vote has low confidence or an ineligible target - falling back to deterministic Broadcast this tick");
                            self.emit_event(CyclePhase::Broadcast, CycleEventKind::Normal, None,
                                json!({"attention_model_fallback": "low_confidence_or_ineligible_target",
                                    "confidence": decision.confidence,
                                    "target_present": decision.target.is_some()}));
                            decide_admission_with_hysteresis(&coalition, &previous_working_memory, self.config.working_memory_capacity, self.config.ignition_threshold)
                        }
                        Ok(Err(err)) => {
                            self.model.ticks_since_attention_reconciliation = 0;
                            tracing::debug!(error = %err, "attention model call failed - falling back to deterministic Broadcast this tick");
                            self.emit_event(CyclePhase::Broadcast, CycleEventKind::Error, None,
                                json!({"attention_model_fallback": "client_error"}));
                            decide_admission_with_hysteresis(&coalition, &previous_working_memory, self.config.working_memory_capacity, self.config.ignition_threshold)
                        }
                        Err(_) => {
                            self.model.ticks_since_attention_reconciliation = 0;
                            tracing::debug!("attention model call timed out - falling back to deterministic Broadcast this tick");
                            self.emit_event(CyclePhase::Broadcast, CycleEventKind::Error, None,
                                json!({"attention_model_fallback": "timeout",
                                    "timeout_ms": self.config.attention_timeout.as_millis() as u64}));
                            decide_admission_with_hysteresis(&coalition, &previous_working_memory, self.config.working_memory_capacity, self.config.ignition_threshold)
                        }
                    }
                }
                None => decide_admission_with_hysteresis(&coalition, &previous_working_memory, self.config.working_memory_capacity, self.config.ignition_threshold),
            }
        };
        let broadcast_result = broadcast_step(&mut self.memory.graph, &admitted, &previous_working_memory, now);
        if !self.config.ablation_config.disable_spike_propagation {
            for &id in &broadcast_result.newly_admitted {
                self.memory.spike_events.schedule_from(
                    &self.memory.graph, id, now, self.config.spike_delay_ms,
                    self.config.edge_decay_rate_per_ms, self.config.spike_max_fanout,
                    self.config.spike_max_pending,
                );
            }
        }
        self.memory.working_memory = broadcast_result.working_memory.iter().copied().collect();
        // GNW's attention/consciousness dissociation, made observable rather
        // than merely implied by two threshold numbers: every real Coalition
        // candidate this tick (attended - cleared `attention_threshold`,
        // genuinely competed) that did NOT end up in the fresh
        // `working_memory` (ignited - actually broadcast, per `steps::
        // broadcast::decide_admission_with_hysteresis`) is a real "attended
        // but not conscious" case. Computed fresh every tick from this
        // tick's real Coalition scores - `CoalitionCandidate::score`, not a
        // possibly-stale `workspace.attention_score` left over from some
        // earlier admission - see `EngineSnapshot::attended_not_ignited`'s
        // own doc comment for what this backs and why.
        let ignited_ids: HashSet<MentalObjectId> = broadcast_result.working_memory.iter().copied().collect();
        if let (Some(specialist), Some(observation_id), Some(orienting)) =
            (&self.orient_outcome_specialist, new_object_id, current_orienting_result)
        {
            let ignited = ignited_ids.contains(&observation_id);
            let probability = specialist.predict_observed_success(&orienting, ignited);
            self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None, json!({
                "orient_outcome_shadow": true,
                "observation_id": observation_id.to_string(),
                "observed_ignition": ignited,
                "predicted_success_probability": probability,
            }));
        }
        if !self.config.ablation_config.disable_lateral_inhibition {
            let changed = crate::steps::coalition::reinforce_lateral_inhibition(
                &mut self.memory.graph,
                &broadcast_result.newly_admitted,
                &coalition,
                &ignited_ids,
                now,
                self.config.lateral_inhibition_increment,
                self.config.lateral_inhibition_max_losers,
            );
            for id in changed { self.memory.dirty_ids.insert(id); }
        }
        self.executive.last_attended_not_ignited = coalition
            .iter()
            .filter(|candidate| !ignited_ids.contains(&candidate.id))
            .filter_map(|candidate| {
                self.memory.graph.get(&candidate.id).map(|object| crate::snapshot::WorkingMemoryMember {
                    id: candidate.id,
                    kind: object.kind,
                    text: object.text.clone(),
                    activation_total: object.activation.total,
                    attention_score: Some(candidate.score),
                    promotion_status: object.promotion.status,
                    confidence: object.confidence,
                })
            })
            .collect();
        if !self.config.ablation_config.disable_preconscious_traces {
            for id in &ignited_ids {
                self.memory.preconscious_traces.remove(id);
            }
            for candidate in coalition.iter().filter(|candidate| !ignited_ids.contains(&candidate.id)) {
                let surprise = raw_candidates.iter().find(|(id, _, _)| *id == candidate.id).and_then(|(_, _, surprise)| *surprise);
                self.memory.preconscious_traces.entry(candidate.id).or_insert(crate::loop_actor::memory_coordinator::PreconsciousTrace {
                    next_evaluation_at: EpochMillis(now.0.saturating_add(self.config.preconscious_reevaluation_ms.max(1))),
                    expires_at: EpochMillis(now.0.saturating_add(self.config.preconscious_trace_ms.max(1))),
                    surprise,
                });
            }
        } else {
            self.memory.preconscious_traces.clear();
        }
        // Ignition: this tick's fresh candidate (if any) actually won -
        // starts that channel's refractory window (see
        // `LoopConfig::attentional_refractory_ms`'s doc comment and this
        // field's own doc comment). Only a *fresh* candidate ignites a new
        // refractory window - an existing Working Memory member simply
        // staying admitted is not a new ignition event.
        if let (Some(id), Some((channel, _))) = (new_object_id, current_outcome_context) {
            if broadcast_result.newly_admitted.contains(&id) {
                self.memory.last_broadcast_by_channel.insert(channel, now);
            }
        }
        // `steps::boredom`'s idle gate: mark the instant WM empties out, and
        // clear that mark the instant it's occupied again - see
        // `wm_empty_since`'s own doc comment for why this (not raw-input
        // absence) is what "idle" means here.
        if self.memory.working_memory.is_empty() {
            self.perception.wm_empty_since.get_or_insert(now);
        } else {
            self.perception.wm_empty_since = None;
        }
        if !broadcast_result.newly_admitted.is_empty() || !broadcast_result.released.is_empty() {
            self.emit_event(
                CyclePhase::Broadcast,
                CycleEventKind::Normal,
                None,
                json!({
                    "admitted": broadcast_result.newly_admitted.len(),
                    "released": broadcast_result.released.len(),
                    "newly_admitted_ids": broadcast_result.newly_admitted.iter().map(MentalObjectId::to_string).collect::<Vec<_>>(),
                    "released_ids": broadcast_result.released.iter().map(MentalObjectId::to_string).collect::<Vec<_>>(),
                }),
            );
        }
        // For every real release this tick, ask `steps::displacement`
        // whether it was a genuine, counterfactually-verified competitive
        // displacement (as opposed to simply not being renominated - see
        // `ReleaseReason`'s own doc comment) and, if so, name the real
        // entrant responsible. Reset to `None` first (not left stale from a
        // previous tick) - `last_displacement` is a claim about *this*
        // tick's Broadcast, not a running memory of the last time it ever
        // happened. If several releases in the same tick are genuinely
        // displacements, the last one processed wins - an accepted
        // simplification (this is a single-field summary, not a log; the
        // full per-object detail is still in the event emitted below).
        self.executive.last_displacement = None;
        for &evicted_id in &broadcast_result.released {
            let reason = crate::steps::displacement::explain_release(&raw_candidates, &broadcast_result.newly_admitted, self.config.working_memory_capacity, evicted_id);
            if let crate::steps::displacement::ReleaseReason::Displaced(displacement) = reason {
                let entrant_text = self.memory.graph.get(&displacement.entrant).map(|object| object.text.clone()).unwrap_or_default();
                let evicted_text = self.memory.graph.get(&displacement.evicted).map(|object| object.text.clone()).unwrap_or_default();
                self.emit_event(
                    CyclePhase::Broadcast,
                    CycleEventKind::Normal,
                    None,
                    json!({"displacement": true, "entrant": displacement.entrant.to_string(), "evicted": displacement.evicted.to_string()}),
                );
                self.executive.last_displacement = Some(crate::snapshot::DisplacementSummary {
                    entrant_id: displacement.entrant,
                    entrant_text,
                    evicted_id: displacement.evicted,
                    evicted_text,
                    at: now,
                });
            }
        }
        // `working_memory` (not just `newly_admitted`/`released`) because
        // `broadcast_step` now also reinforces an associative edge between
        // every pair of co-broadcast objects (`steps::broadcast::
        // reinforce_coalescence`) - an object that simply stayed in Working
        // Memory this tick can still have gained a fresh edge, and would
        // otherwise never get flushed to the durable store.
        for id in broadcast_result.newly_admitted.iter().chain(broadcast_result.released.iter()).chain(broadcast_result.working_memory.iter()) {
            self.memory.dirty_ids.insert(*id);
        }

        // --- Steps 7-9: Execute / Act / Learn, on whatever currently holds
        // the spotlight (the top-ranked Working Memory member, if any). ---
        if let Some(&top_id) = broadcast_result.working_memory.first() {
            let surprise_for_top = if Some(top_id) == new_object_id { new_surprise } else { None };
            // Phase 3 of the GWT-parity roadmap: a real, independent second
            // consequence of this tick's broadcast winner, run before
            // `propose_operators` below rather than competing inside it -
            // see `steps::memory_formation::maybe_automatic_remember`'s own
            // doc comment for why this composes cleanly with Executive's
            // still-unchanged `Operator::Remember` proposal rather than
            // double-booking it. Not literal wall-clock concurrency (both
            // still `.await` sequentially here - see the audit doc's own
            // "what this does not establish" on that) - what actually
            // changed is that this no longer has to *win* against whatever
            // Executive separately decides below.
            let automatic_remember_outcome = crate::steps::memory_formation::maybe_automatic_remember(
                &mut self.memory.graph,
                top_id,
                surprise_for_top,
                self.config.automatic_memory_formation_surprise_threshold,
                &self.config.memory_config,
                &self.config.confidence_revision,
                &self.model.tier1_pool,
                &self.model.tier2_pool,
                &self.memory.self_summary,
                now,
                self.config.temperature.memory_formation,
            )
            .await;
            if let Some(outcome) = &automatic_remember_outcome {
                self.memory.dirty_ids.insert(top_id);
                self.emit_event(
                    CyclePhase::Learn,
                    CycleEventKind::Normal,
                    None,
                    json!({"automatic_remember": true, "outcome": format!("{outcome:?}"), "target_id": top_id.to_string()}),
                );
            }
            // Computed once for this tick's whole Execute/Act pass - after
            // `last_broadcast_by_channel` above has already been updated
            // with any fresh admission this same tick, so a live
            // conversational turn that just won Broadcast is reflected in
            // this value immediately rather than lagging by one tick.
            let presence = self.conversational_presence(now);
            // One pass over Working Memory, pairing each member's text with
            // its trust tier (`prompt_templates::ProvenanceTier`) by
            // construction - safer than two independent `HashSet` iterations
            // that happen to line up, which `working_memory_text_refs` and a
            // separately-collected tags list would otherwise rely on.
            let working_memory_texts_and_tiers: Vec<(String, crate::prompt_templates::ProvenanceTier)> = self.memory.working_memory.iter().filter_map(|id| self.memory.graph.get(id)).map(|object| (object.text.clone(), classify_provenance(object))).collect();
            let working_memory_text_refs: Vec<&str> = working_memory_texts_and_tiers.iter().map(|(text, _)| text.as_str()).collect();
            // Same members, tagged - `act`'s own reflect prompt reads this;
            // `propose_operators` just below reads the plain-text view above
            // instead, since it has no use for provenance.
            let working_memory_entries: Vec<crate::prompt_templates::ContextEntry<'_>> = working_memory_texts_and_tiers.iter().map(|(text, tier)| crate::prompt_templates::ContextEntry { text: text.as_str(), tier: *tier }).collect();
            let propose_operators_started = Instant::now();
            let mut proposals = propose_operators(
                &self.memory.graph,
                top_id,
                surprise_for_top,
                &self.config.executive_config,
                &self.model.tier1_pool,
                &working_memory_text_refs,
                &self.memory.self_summary,
                &self.tool_registry,
                now,
                self.config.agenda_config.replan_interval_ms,
                self.config.temperature.operator_proposal,
                self.config.temperature.tool_intent,
            )
            .await;
            propose_operators_ms = propose_operators_started.elapsed().as_millis() as u64;
            let learned_bias = apply_learned_bias(&self.memory.graph, &mut proposals);
            let decision = select_operator_sampled(&proposals, &self.config.executive_config, &self.config.sampling, &mut self.rng);

            match decision {
                ExecutiveDecision::Selected(proposal) => {
                    self.emit_event(
                        CyclePhase::Executive,
                        CycleEventKind::Normal,
                        None,
                        json!({
                            "operator": format!("{:?}", proposal.operator),
                            "target_text": text_preview(&self.memory.graph, proposal.target_id),
                            "preference": proposal.preference,
                            "confidence": proposal.confidence,
                        }),
                    );
                    self.executive.last_operator_proposal = Some(crate::snapshot::OperatorProposalSummary {
                        operator: format!("{:?}", proposal.operator),
                        target_id: proposal.target_id,
                        preference: proposal.preference,
                        confidence: proposal.confidence,
                        at: now,
                    });
                    // A chunk's utility bias only earns credit/blame for the
                    // consequence it actually decided - not merely for
                    // existing this tick (see `apply_learned_bias`'s doc
                    // comment).
                    if let Some((chunk_id, learned_operator)) = learned_bias {
                        if learned_operator == proposal.operator {
                            if let Some((channel, interlocutor)) = current_outcome_context {
                                self.executive.outcome_registry.register(PendingOutcome {
                                    channel,
                                    interlocutor,
                                    chunk_id: Some(chunk_id),
                                    tier: None,
                                    reported_confidence: proposal.confidence,
                                    created_at: now,
                                });
                            }
                        }
                    }
                    let act_started = Instant::now();
                    let outcome = self.apply_operator(&proposal, &working_memory_entries, presence, now).await;
                    act_ms += act_started.elapsed().as_millis() as u64;
                    if let crate::steps::act::ActOutcome::Reflected { reflection_id } = &outcome {
                        self.calibrate_and_register_reflection(*reflection_id, current_outcome_context);
                    }
                }
                ExecutiveDecision::Impasse { kind, candidates, reason } => {
                    self.emit_event(
                        CyclePhase::Executive,
                        CycleEventKind::Impasse,
                        None,
                        json!({"kind": format!("{kind:?}"), "reason": reason}),
                    );
                    match impasse_response(kind) {
                        ImpasseResponse::SpawnSubgoal => {
                            let subgoal_id = spawn_subgoal(&mut self.memory.graph, &reason, now, self.config.decay_d);
                            self.memory.dirty_ids.insert(subgoal_id);
                            // Per the build plan: "the subgoal re-enters
                            // Coalition next cycle like any other candidate."
                            self.memory.pending_admission.insert(subgoal_id);
                        }
                        ImpasseResponse::EscalateTier => {
                            let candidate_keywords: Vec<&str> = candidates.iter().map(|c| operator_keyword(c.operator)).collect();
                            let prompt = format!(
                                "Break the tie among these candidate actions for Omega: {}. Which is best given the context? {}\n\n\
                                 Respond with ONLY a JSON object of the form {{\"confidence\": <0.0-1.0>, \"response\": \"<one of: {}>\"}}.",
                                candidate_keywords.join(", "),
                                reason,
                                candidate_keywords.join(", "),
                            );
                            let tier3_escalation_started = Instant::now();
                            let tier3_resolution = resolve_confidence_impasse(
                                &self.model.tier3_pool,
                                self.model.chat_client.as_ref(),
                                GenerateRequest { prompt: prompt.clone(), temperature: self.config.temperature.impasse_resolution },
                            )
                            .await;
                            impasse_escalation_ms += tier3_escalation_started.elapsed().as_millis() as u64;
                            // Escalation along the resource axis (specs.md's
                            // Model Tiering), one rung higher than Tier 1/2's
                            // ladder in cognitive_core::reflect: Tier 3
                            // answered, but not confidently enough, so try
                            // once more with the largest model before
                            // settling. Tier 4 not answering (unconfigured,
                            // busy, or itself failed) falls back to the
                            // Tier 3 answer already in hand rather than
                            // discarding it - a real, if less confident,
                            // answer beats none.
                            let resolution = match tier3_resolution {
                                // Tier 3's raw self-report is exactly what
                                // `calibration_tracker` exists to second-
                                // guess (see `steps::metacognition`'s doc
                                // comment) - with no history yet this is a
                                // no-op, but once Tier 3 has shown a
                                // consistent bias, the threshold check below
                                // sees the corrected number, not the raw one.
                                ImpasseResolution::Escalated(ref tier3_response)
                                    if self.executive.calibration_tracker.calibrate(aca_types::Tier::T3, tier3_response.confidence)
                                        < self.config.executive_config.tier4_escalation_threshold =>
                                {
                                    let tier4_escalation_started = Instant::now();
                                    let tier4_result = resolve_confidence_impasse(&self.model.tier4_pool, self.model.tier4_client.as_ref(), GenerateRequest { prompt, temperature: self.config.temperature.impasse_resolution }).await;
                                    impasse_escalation_ms += tier4_escalation_started.elapsed().as_millis() as u64;
                                    match tier4_result {
                                        ImpasseResolution::Escalated(tier4_response) => ImpasseResolution::Escalated(tier4_response),
                                        _ => tier3_resolution,
                                    }
                                }
                                other => other,
                            };
                            match resolution {
                                ImpasseResolution::Escalated(response) => {
                                    self.emit_event(
                                        CyclePhase::Executive,
                                        CycleEventKind::Escalation,
                                        Some(response.tier),
                                        json!({"confidence": response.confidence}),
                                    );
                                    // Feeds `calibration_tracker` (whichever
                                    // tier actually answered - T3 alone, or
                                    // T4 after further escalation) with this
                                    // resolution's *raw* self-reported
                                    // confidence, not the calibrated number
                                    // used just above to decide whether to
                                    // escalate - calibration must learn from
                                    // what the tier actually claimed, not
                                    // from its own prior correction.
                                    if let Some((channel, interlocutor)) = current_outcome_context {
                                        self.executive.outcome_registry.register(PendingOutcome {
                                            channel,
                                            interlocutor,
                                            chunk_id: None,
                                            tier: Some(response.tier),
                                            reported_confidence: response.confidence,
                                            created_at: now,
                                        });
                                    }
                                    // Prefer the escalated tier's actual
                                    // chosen answer (parsed from its
                                    // response text, restricted to the
                                    // operators actually in contention) over
                                    // blindly picking whichever original
                                    // candidate happened to self-report the
                                    // highest confidence - the latter is
                                    // only a fallback for when the tier
                                    // didn't answer in the requested format,
                                    // not the primary signal.
                                    let resolved_operator = parse_operator_keyword(&response.raw_text)
                                        .filter(|op| candidates.iter().any(|c| c.operator == *op))
                                        .or_else(|| {
                                            candidates
                                                .iter()
                                                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
                                                .map(|c| c.operator)
                                        });
                                    if let Some(resolved_operator) = resolved_operator {
                                        let chunk_id = chunk_resolution(&mut self.memory.graph, &candidates, resolved_operator, response.confidence, now, self.config.decay_d);
                                        self.memory.dirty_ids.insert(chunk_id);
                                        self.emit_event(
                                            CyclePhase::Learn,
                                            CycleEventKind::Normal,
                                            None,
                                            json!({"chunked_operator": format!("{resolved_operator:?}")}),
                                        );

                                        // Resolving an impasse should settle
                                        // the thing it was actually about,
                                        // not just prime the next occurrence
                                        // of it - chunking alone would mean
                                        // deliberating your way to an answer
                                        // and then never acting on it, only
                                        // ever benefiting some future
                                        // decision instead. `response.confidence`
                                        // (the escalated tier's real, earned
                                        // confidence) replaces the original
                                        // candidate's - by definition too low
                                        // to have been selected outright, or
                                        // this wouldn't have been an impasse.
                                        if let Some(matched) = candidates.iter().find(|c| c.operator == resolved_operator) {
                                            let resolved_proposal = OperatorProposal {
                                                operator: resolved_operator,
                                                target_id: matched.target_id,
                                                preference: matched.preference,
                                                confidence: response.confidence,
                                            };
                                            let resolved_act_started = Instant::now();
                                            let resolved_outcome = self.apply_operator(&resolved_proposal, &working_memory_entries, presence, now).await;
                                            act_ms += resolved_act_started.elapsed().as_millis() as u64;
                                            if let crate::steps::act::ActOutcome::Reflected { reflection_id } = &resolved_outcome {
                                                self.calibrate_and_register_reflection(*reflection_id, current_outcome_context);
                                            }
                                        }
                                    }
                                }
                                ImpasseResolution::DeferredToHeuristic => {
                                    self.emit_event(CyclePhase::Executive, CycleEventKind::Normal, None, json!({"deferred": true}));
                                }
                                ImpasseResolution::EscalationFailed(err) => {
                                    self.emit_event(CyclePhase::Executive, CycleEventKind::Error, None, json!({"error": err.to_string()}));
                                }
                            }
                        }
                    }
                }
            }

            // General associative learning: Working Memory members that were
            // broadcast together this tick have their co-activation
            // reinforced, independent of whatever operator was selected.
            // Deliberately `broadcast_result.working_memory` (membership as
            // of Broadcast) rather than `self.memory.working_memory` read fresh
            // here - a selected ContinueReflecting releases its target from
            // Working Memory as part of Act, and that release must not
            // erase credit for having genuinely co-occurred with everything
            // else broadcast this same tick.
            let wm_ids: Vec<MentalObjectId> = broadcast_result.working_memory.clone();
            let mut reinforced_edges = 0u32;
            for i in 0..wm_ids.len() {
                for j in 0..wm_ids.len() {
                    if i != j {
                        reinforce_coactivation(&mut self.memory.graph, wm_ids[i], wm_ids[j], now);
                        if !self.config.ablation_config.disable_eligibility_learning {
                            self.memory.eligibility_traces.mark(wm_ids[i], wm_ids[j], EdgeKind::Associative, now);
                        }
                        reinforced_edges += 1;
                    }
                }
            }
            if reinforced_edges > 0 {
                self.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None, json!({"reinforced_edges": reinforced_edges}));
            }
            for &id in &wm_ids {
                self.memory.dirty_ids.insert(id);
            }
        }

        // Published once, at the true end of the tick - not right after
        // Broadcast - so it reflects everything that happened this tick,
        // including Learn's same-tick edge reinforcement (an earlier
        // ordering bug meant reinforced edges never showed up until the
        // *next* tick's snapshot).
        //
        // Skipped entirely on a genuinely idle tick - no new input, Working
        // Memory empty, nothing admitted or released - which is what was
        // previously forcing `publish_snapshot`'s full `graph.iter()` scan
        // (memory_counts/goal_stack) to run on every iteration of the
        // no-scheduler hot loop (`run()`'s `loop { tick().await; yield_now
        // ().await; }`), forever, even with nothing whatsoever happening.
        // Any tick where Working Memory holds *anything* still publishes -
        // the Executive block above runs whenever it does, and may have
        // mutated the graph - so this only skips the case that was actually
        // wasteful, not one a viewer could ever observe as staleness.
        // `cycle_seq == 1` forces one unconditional publish on the very
        // first tick regardless of activity, so a freshly-started actor's
        // *loaded* state (memory counts, goal stack restored from the
        // durable store - already populated before the first tick, not
        // something that arrives as "activity") is reported immediately
        // rather than leaving `EngineSnapshot::default()` live until
        // whatever real activity happens to come first.
        let activity_this_tick = self.cycle_seq == 1
            || new_object_id.is_some()
            || preconscious_activity
            || !broadcast_result.working_memory.is_empty()
            || !broadcast_result.newly_admitted.is_empty()
            || !broadcast_result.released.is_empty();
        if activity_this_tick {
            self.publish_snapshot().await;
        }

        // --- Pattern synthesis --- an actor-local periodic side effect,
        // same category as the write-behind flush below, not routed through
        // the Executive/Operator machinery (see `episodic_since_last_synthesis`'s
        // doc comment). Self-paced by the buffer only refilling through
        // genuine new Episodic memories; `min_interval_ms` is a cost-control
        // backstop on top.
        //
        // Complementary Learning Systems (McClelland/O'Reilly/Norman) models
        // exactly this second memory system - slow, interleaved consolidation
        // of fast hippocampal (Episodic) encoding into distributed
        // neocortical (Semantic) structure - as happening preferentially
        // *offline*, not interleaved with active encoding of new experience;
        // sleep-dependent consolidation (Diekelmann & Born) is the same
        // finding at the systems-neuroscience level. `idle_now` is the same
        // "Working Memory has genuinely nothing in it" signal
        // `steps::boredom` already gates on - synthesis prefers to run in
        // exactly the windows nothing else is competing for the tier pools.
        // `synthesis_backlog_pressure` is the backstop: a system that's
        // never idle for long enough must still eventually consolidate
        // rather than let the buffer (and the learning it represents) grow
        // without bound.
        let synthesis_config = &self.config.synthesis_config;
        // "Idle" here means "not simultaneously encoding a fresh
        // observation this exact tick" - deliberately weaker than
        // `steps::boredom`'s own "Working Memory has been empty for
        // `idle_threshold_ms`" gate. Working Memory can legitimately stay
        // occupied for a long real-world stretch (a small capacity plus a
        // long natural activation dwell time both push that way - see
        // `LoopConfig::attention_threshold`'s doc comment), so requiring
        // full emptiness here would make consolidation wait for a kind of
        // silence that may not arrive for a long time even in an actively
        // engaged system. Excluding only the exact tick new experience is
        // being encoded still captures CLS's real distinction (consolidate
        // separately from fresh encoding, not fused into the same instant).
        let idle_now = new_object_id.is_none();
        let synthesis_backlog_pressure = self.memory.episodic_since_last_synthesis.len() >= synthesis_config.max_cluster_size * 2;
        let synthesis_deferred_for_input = self.config.responsiveness_config.defer_synthesis_on_foreground_input && foreground_input_this_tick;
        let should_attempt_synthesis = !self.config.ablation_config.disable_synthesis
            && !synthesis_deferred_for_input
            && self.memory.episodic_since_last_synthesis.len() >= synthesis_config.min_new_episodic
            && self.memory.last_synthesis_at.is_none_or(|t| now.0 - t.0 >= synthesis_config.min_interval_ms)
            && (idle_now || synthesis_backlog_pressure);
        let deferred_synthesis_for_responsiveness = synthesis_deferred_for_input
            && self.memory.episodic_since_last_synthesis.len() >= synthesis_config.min_new_episodic
            && self.memory.last_synthesis_at.is_none_or(|t| now.0 - t.0 >= synthesis_config.min_interval_ms);
        if should_attempt_synthesis {
            let take_n = synthesis_config.max_cluster_size.min(self.memory.episodic_since_last_synthesis.len());
            let cluster: Vec<MentalObjectId> = rank_by_surprise(&self.memory.graph, &self.memory.episodic_since_last_synthesis).into_iter().take(take_n).collect();
            let synthesize_started = Instant::now();
            let result = crate::steps::synthesize::synthesize(
                &mut self.memory.graph,
                &cluster,
                &self.model.tier1_pool,
                &self.model.tier2_pool,
                &self.model.tier3_pool,
                self.model.chat_client.as_ref(),
                self.model.embedding_client.as_ref(),
                synthesis_config,
                &self.memory.self_summary,
                now,
                self.config.decay_d,
                self.config.tier3_hedge_delay,
                self.config.temperature.synthesis,
            )
            .await;
            synthesize_ms = synthesize_started.elapsed().as_millis() as u64;
            if let Some(pattern_id) = result {
                self.memory.pending_admission.insert(pattern_id);
                self.memory.dirty_ids.insert(pattern_id);
                self.emit_event(
                    CyclePhase::Synthesize,
                    CycleEventKind::Normal,
                    None,
                    json!({"source_count": cluster.len(), "pattern_id": pattern_id.to_string()}),
                );
            }
            // Cleared/updated unconditionally, success or failure - see
            // `last_synthesis_at`'s doc comment on why a failure must not
            // leave the trigger condition permanently true.
            self.memory.last_synthesis_at = Some(now);
            self.memory.episodic_since_last_synthesis.clear();
        }

        // --- Agenda, late phase --- run unconditionally every tick, same
        // reasoning as `steps::agenda::revise_agenda`'s own doc comment: an
        // intention's decay/spawn opportunity must not depend on the system
        // happening to be idle. `self.drives.drive_state` is folded first, from
        // signals this tick already computed elsewhere (`current_tick_
        // epistemic_value` from Compare, if any; a graph scan for
        // unresolved discrepancies; `calibration_tracker`'s existing bias
        // tracking; conversational recency; tier-pool saturation) - see
        // `steps::drives`'s own doc comment for why none of these is a new
        // signal. `revise_agenda` then reads the freshly-updated
        // `drive_state` to decide whether a new intention is warranted this
        // tick.
        let curiosity_reading = crate::steps::drives::curiosity_pressure(&self.memory.graph, self.config.executive_config.low_confidence_reflection_threshold);
        let social_connection_reading = crate::steps::drives::social_connection_pressure(self.drives.last_conversation_input_at, now);
        let resource_reading = crate::steps::drives::resource_pressure(&self.model.live_models.tier_status());
        self.drives.drive_state.update_at(
            now,
            current_tick_epistemic_value,
            curiosity_reading,
            Some(self.executive.calibration_tracker.bias_for(aca_types::Tier::T3)),
            Some(self.executive.execution_tracker.failure_rate()),
            social_connection_reading,
            resource_reading,
        );
        let agenda_outcome = if self.config.ablation_config.disable_agenda {
            crate::steps::agenda::AgendaOutcome::default()
        } else {
            let mut agenda_config = self.config.agenda_config;
            if self.config.responsiveness_config.defer_new_agenda_spawns_on_foreground_input && foreground_input_this_tick {
                agenda_config.spawn_threshold = f32::INFINITY;
                deferred_agenda_spawn_for_responsiveness = true;
            }
            crate::steps::agenda::revise_agenda(&mut self.memory.graph, &self.memory.dirty_ids, &self.drives.drive_state, &agenda_config, now, self.config.decay_d)
        };
        for id in &agenda_outcome.dirty_ids {
            self.memory.dirty_ids.insert(*id);
        }
        if let Some(spawned_id) = agenda_outcome.spawned {
            // Same "one honest shot at admission" mechanism as a freshly
            // synthesized pattern or a mid-tick Reflection - a brand-new
            // intention needs this too, or it would sit in the graph
            // entirely unnoticed until `surface_active_intentions`
            // eventually gets around to it on a later tick.
            self.memory.pending_admission.insert(spawned_id);
            self.emit_event(CyclePhase::Agenda, CycleEventKind::Normal, None, json!({"spawned": spawned_id.to_string()}));
        }
        for (id, status) in &agenda_outcome.transitions {
            self.emit_event(CyclePhase::Agenda, CycleEventKind::Normal, None, json!({"transitioned": id.to_string(), "status": format!("{status:?}")}));
        }

        // --- Self Memory summary refresh --- see
        // `MemoryCoordinator::refresh_self_summary_if_dirty`'s doc comment;
        // must run before `dirty_ids` gets cleared by the flush below.
        self.memory.refresh_self_summary_if_dirty();

        let tier_status = self.model.live_models.tier_status();
        self.emit_event(
            CyclePhase::Telemetry,
            CycleEventKind::Normal,
            None,
            json!({
                "elapsed_ms": tick_started.elapsed().as_millis() as u64,
                "elapsed_us": tick_started.elapsed().as_micros() as u64,
                "compiled_procedure_hit": compiled_procedure_hit,
                "curated_answer_hit": curated_answer_hit,
                "phase_ms": {
                    "embedding_wait": embedding_wait_ms,
                    "attention_model": attention_model_ms,
                    "propose_operators": propose_operators_ms,
                    "act": act_ms,
                    "impasse_escalation": impasse_escalation_ms,
                    "synthesize": synthesize_ms,
                },
                "input_token_estimate": input_token_estimate,
                "working_memory_count": self.memory.working_memory.len(),
                "graph_object_count": self.memory.graph.len(),
                "dirty_object_count": self.memory.dirty_ids.len(),
                "pending_event_count": self.telemetry.pending_events.len(),
                "embedding_cache_count": self.prediction.embedding_cache.len(),
                "foreground_input": foreground_input_this_tick,
                "deferred": {
                    "synthesis": deferred_synthesis_for_responsiveness,
                    "agenda_spawn": deferred_agenda_spawn_for_responsiveness,
                },
                "tiers": tier_status
                    .iter()
                    .map(|status| json!({
                        "tier": format!("{:?}", status.tier),
                        "busy": status.capacity.saturating_sub(status.available_permits),
                        "capacity": status.capacity,
                    }))
                    .collect::<Vec<_>>()
            }),
        );

        // --- Periodic write-behind flush --- never on every tick.
        let flush_interval_elapsed = self.last_flush_at.is_none_or(|t| now.0.saturating_sub(t.0) >= self.config.flush_interval_ms);
        if (curated_answer_changed_this_tick || procedure_feedback_changed_this_tick || outcome_feedback_changed_this_tick || flush_interval_elapsed)
            && (!self.memory.dirty_ids.is_empty() || !self.telemetry.pending_events.is_empty()) {
            let objects: Vec<MentalObject> = self.memory.dirty_ids.iter().filter_map(|id| self.memory.graph.get(id).cloned()).collect();
            let cycle_events = std::mem::take(&mut self.telemetry.pending_events);
            let batch = DirtyBatch { objects, cycle_events };
            if let Err(err) = self.memory.store.flush(batch).await {
                tracing::error!(error = %err, "periodic flush failed");
            } else {
                self.memory.dirty_ids.clear();
                self.last_flush_at = Some(now);
            }
        }

        // A real cooperative hand-off, not a delay: `run_embedding_worker`
        // and any other spawned background task only make progress between
        // this task's own await points. The periodic flush above used to be
        // relied on (accidentally, not by design) to provide one every few
        // ticks via its `spawn_blocking` await - confirmed live once
        // `flush_interval_ms` (wall-clock) replaced the old tick-counted
        // cadence: a fast, direct `tick()` loop (every test that bypasses
        // `CognitiveScheduler` and calls `tick()` in a tight loop) could
        // then go many ticks without a real await ever occurring, starving
        // the embedding worker of scheduling opportunities entirely -
        // exactly the failure mode `a_same_channel_arrival_inside_the_
        // refractory_window_is_deferred_not_lost` caught. An explicit,
        // unconditional yield here makes that guarantee real regardless of
        // flush cadence or which caller drives the loop.
        tokio::task::yield_now().await;
    }
}

/// specs.md's Memory Competition section: "relevance to self → elevated
/// baseline activation for Self Memory-linked nodes." Called right after
/// `recompute_activation` overwrites `activation.total` for the tick, so
/// the bonus is applied fresh every time rather than being something that
/// could be computed once and stored — a `SelfMemory`-tagged object's
/// ordinary ACT-R activation still decays exactly like any other object's;
/// this only changes how much of a head start it gets back each tick.
///
/// Withheld from a still-`PromotionStatus::Candidate` belief - see
/// `PromotionState`'s own doc comment. An unconfirmed belief should compete
/// on its bare activation like anything else, not get the "rarely lose the
/// competition" boost before anything outside its own classifier has
/// vouched for it.
pub(crate) fn apply_self_memory_activation_bonus(object: &mut MentalObject, bonus: f32) {
    if object.memory_roles.contains(&MemoryRole::SelfMemory) && object.promotion.status == aca_types::PromotionStatus::Confirmed {
        object.activation.total += bonus;
    }
}

/// Maps a Working Memory member to the `prompt_templates::ProvenanceTier`
/// it should be rendered with - see that type's own doc comment. Only
/// derives `Durable`/`StagedCandidate`/`Narrative` from what a plain
/// `MentalObject` can currently say about itself; `Authoritative` (hand-
/// seeded identity content) and `External` (Knowledge-Library-sourced) have
/// no reliable signal to detect from an ordinary Working Memory member yet
/// - honest about what isn't implemented rather than guessing.
fn classify_provenance(object: &MentalObject) -> crate::prompt_templates::ProvenanceTier {
    if object.memory_roles.iter().any(|role| matches!(role, MemoryRole::Semantic | MemoryRole::SelfMemory)) {
        match object.promotion.status {
            aca_types::PromotionStatus::Confirmed => crate::prompt_templates::ProvenanceTier::Durable,
            aca_types::PromotionStatus::Candidate => crate::prompt_templates::ProvenanceTier::StagedCandidate,
        }
    } else {
        crate::prompt_templates::ProvenanceTier::Narrative
    }
}

/// CLS-style prioritized replay ranking for pattern synthesis's cluster
/// selection (see the pattern-synthesis call site's own doc comment for the
/// theory): ranks `buffered` by each id's reconstructed precision-weighted
/// surprise (specs.md/`steps::compare::ComparisonResult::
/// precision_weighted_surprise`'s own currency), descending - reconstructed
/// from each object's own already-stored `prediction.error_magnitude` /
/// `prediction.precision` rather than needing new storage. Hippocampal
/// sharp-wave-ripple replay itself preferentially replays high-surprise
/// experience rather than whatever happened most recently (Schaul et al.'s
/// prioritized-replay finding is the same idea in an RL setting) - a pattern
/// worth generalizing is more likely to hide among what most violated
/// Omega's expectations than among an arbitrary recent run of unremarkable
/// turns, and this also spends the (Tier 1-3) cost of a synthesis attempt on
/// the buffered memories most likely to actually contain one. A stable sort,
/// so equally-surprising items keep their original buffer order rather than
/// resolving arbitrarily. An id no longer present in `graph` (discarded,
/// reclassified) sorts as zero surprise rather than panicking or being
/// dropped - `steps::synthesize::synthesize` already tolerates and skips
/// stale ids on its own.
fn rank_by_surprise(graph: &Graph, buffered: &[MentalObjectId]) -> Vec<MentalObjectId> {
    let mut ranked = buffered.to_vec();
    ranked.sort_by(|a, b| {
        let surprise_of = |id: &MentalObjectId| {
            graph
                .get(id)
                .map(|object| object.prediction.error_magnitude.unwrap_or(0.0) * object.prediction.precision.unwrap_or(0.0))
                .unwrap_or(0.0)
        };
        surprise_of(b).partial_cmp(&surprise_of(a)).unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

fn estimate_token_count(text: &str) -> usize {
    // Cheap operational estimate, not tokenizer truth: good enough to trend
    // prompt pressure in `Telemetry` without coupling the engine to any one
    // model family's tokenizer.
    text.split_whitespace().map(|word| (word.len() / 4).max(1)).sum()
}

/// A short, single-line preview of a Mental Object's text for log payloads -
/// enough to recognize *which* object an Executive/Act event is about
/// without dumping full (sometimes multi-paragraph) content into every
/// event. `None` (rendered `null`) for a target that's already gone from
/// the graph by the time this is logged, rather than a misleading empty
/// string.
const TEXT_PREVIEW_MAX_CHARS: usize = 80;

fn text_preview(graph: &Graph, id: MentalObjectId) -> Option<String> {
    graph.get(&id).map(|object| {
        let text = &object.text;
        if text.chars().count() <= TEXT_PREVIEW_MAX_CHARS {
            text.clone()
        } else {
            format!("{}…", text.chars().take(TEXT_PREVIEW_MAX_CHARS).collect::<String>())
        }
    })
}

/// `outcome`'s own shape, plus the operator that was actually *attempted*
/// and a preview of the target it was attempted on - `ActOutcome::Silent`
/// alone can't distinguish "nothing to do" from "Act was proposed but no
/// tool matched," which is exactly the distinction that mattered for
/// diagnosing a live stall (a semantically-triggered Act proposal that
/// never found a tool, re-proposed every subsequent tick because the
/// failure wasn't tagged - see the `Operator::Act` arm's own doc comment in
/// `steps::act`). Surfacing `attempted_operator`/`target_text` on every Act
/// event, not just the successful ones, is what makes that kind of stall
/// visible in the console as it happens instead of only reconstructible
/// after the fact from cycle counts.
///
/// A `Silent` outcome's own `SilentReason` (see that type's doc comment) is
/// flattened into `reason`, plus `error` when it's `ReflectionFailed` - this
/// is the one place a viewer can actually tell "Omega decided not to
/// respond" apart from "Omega tried to respond and the model call failed,"
/// which used to be visually identical. `apply_operator` is what turns
/// `ReflectionFailed` specifically into an `Error`-severity event using this
/// same `reason` field, so the two stay in lockstep by construction rather
/// than by two call sites independently agreeing on a string.
fn act_outcome_payload(outcome: &crate::steps::act::ActOutcome, attempted_operator: crate::steps::executive::Operator, target_text: Option<String>) -> serde_json::Value {
    use crate::steps::act::{ActOutcome, SilentReason};
    let mut payload = match outcome {
        ActOutcome::Spoke { text, render_path } => json!({"operator": "speak", "text": text, "render_path": format!("{render_path:?}")}),
        ActOutcome::Remembered { outcome } => json!({"operator": "remember", "outcome": format!("{outcome:?}")}),
        ActOutcome::Reflected { reflection_id } => json!({"operator": "reflect", "reflection_id": reflection_id.to_string()}),
        ActOutcome::ConsultedKnowledgeLibrary { result } => json!({"operator": "consult-knowledge-library", "found": result.found}),
        ActOutcome::Acted { tool, result } => json!({"operator": "act", "tool": tool, "ok": result.is_ok(), "error": result.as_ref().err()}),
        ActOutcome::Silent { reason } => match reason {
            SilentReason::Ignored => json!({"operator": "silent", "reason": "ignored", "attempted_operator": format!("{attempted_operator:?}")}),
            SilentReason::StaleTarget => json!({"operator": "silent", "reason": "stale-target", "attempted_operator": format!("{attempted_operator:?}")}),
            SilentReason::NoToolMatched => json!({"operator": "silent", "reason": "no-tool-matched", "attempted_operator": format!("{attempted_operator:?}")}),
            SilentReason::ReflectionFailed { error } => {
                json!({"operator": "silent", "reason": "reflection-failed", "attempted_operator": format!("{attempted_operator:?}"), "error": error})
            }
        },
    };
    if let Some(map) = payload.as_object_mut() {
        map.insert("attempted_operator".to_string(), json!(format!("{attempted_operator:?}")));
        map.insert("target_text".to_string(), json!(target_text));
    }
    payload
}

/// Whether `outcome` represents a genuine failure (something was attempted
/// and broke) rather than a deliberate or routine "nothing to do" - the one
/// case severe enough to warrant `CycleEventKind::Error` on the Act event,
/// same severity tier `Observe`/`Executive` already use for their own
/// transport/parse failures (see `apply_operator`'s `emit_event` call).
/// Every other `Silent` reason stays at `Normal` - they're not failures,
/// just non-events, and treating them as errors would drown the real ones
/// in noise.
fn act_outcome_is_failure(outcome: &crate::steps::act::ActOutcome) -> bool {
    matches!(
        outcome,
        crate::steps::act::ActOutcome::Silent {
            reason: crate::steps::act::SilentReason::ReflectionFailed { .. }
        }
    )
}

#[cfg(test)]
mod tests;
