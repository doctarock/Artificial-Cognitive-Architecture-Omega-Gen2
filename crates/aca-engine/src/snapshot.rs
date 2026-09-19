use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use aca_tiers::{DivergentPool, TierPool};
use aca_types::{EdgeKind, GoalStatus, MentalObjectId, MentalObjectKind, PromotionStatus, Tier};
use aca_util::EpochMillis;
use serde::Serialize;

/// One Working Memory member's view for external consumers (the local API,
/// a Godot client, a debug CLI). Deliberately strips the full embedding
/// vector — a viewer needs id/kind/text/activation, not a several-hundred-
/// float array per node per snapshot (see the plan).
#[derive(Debug, Clone, Serialize)]
pub struct WorkingMemoryMember {
    pub id: MentalObjectId,
    pub kind: MentalObjectKind,
    pub text: String,
    pub activation_total: f32,
    pub attention_score: Option<f32>,
    /// `MentalObject.promotion.status` - whether this memory has cleared
    /// the promotion gate (`Confirmed`) or is still an unconfirmed
    /// `Candidate` awaiting reinforcement/confirmation (`steps::
    /// memory_formation`'s promotion-gate mechanism).
    pub promotion_status: PromotionStatus,
    /// `MentalObject.confidence` at snapshot time - revisable downward by
    /// `steps::confidence_revision::apply_contradiction_penalty` when a
    /// contradiction is detected against this object.
    pub confidence: f32,
}

/// How many graph objects currently carry each `MemoryRole` tag. A count,
/// not a full dump — a visualization can show "Episodic: 42" as a growing
/// region without the engine shipping its entire memory graph every tick.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct MemoryRoleCounts {
    pub working: usize,
    pub episodic: usize,
    pub semantic: usize,
    pub self_memory: usize,
    /// Graph objects currently `PromotionStatus::Candidate` - independent
    /// of `MemoryRole`, since a candidate can carry any role while it
    /// awaits confirmation.
    pub candidates: usize,
}

/// One active or impassed entry on the SOAR goal stack.
#[derive(Debug, Clone, Serialize)]
pub struct GoalSummary {
    pub id: MentalObjectId,
    pub text: String,
    pub status: GoalStatus,
    pub priority: f32,
}

/// One learned/reinforced associative connection between two currently-
/// broadcast Mental Objects — ACT-R's `S_ki`, made visible. Scoped to pairs
/// where *both* endpoints are in Working Memory right now: a real edge
/// might point further out into long-term memory too, but visualizing
/// only WM-to-WM connections needs no additional lookups and is still
/// genuinely informative (which currently-active thoughts are associated
/// with each other).
#[derive(Debug, Clone, Serialize)]
pub struct EdgeSummary {
    pub source_id: MentalObjectId,
    pub target_id: MentalObjectId,
    pub kind: EdgeKind,
    pub strength: f32,
}

/// A model tier's live concurrency state — how many of its configured
/// slots are currently busy. Reported only for tiers actually wired up
/// (Tier 3 in this MVP); fabricating numbers for unimplemented tiers would
/// misrepresent the real system, which this snapshot exists to show
/// honestly.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TierStatus {
    pub tier: Tier,
    pub available_permits: usize,
    pub capacity: usize,
}

/// One individually-configured LLM in the reasoning chain — a single Tier
/// 1/2 divergent candidate, or the lone Tier 3/4 model — reported by real
/// identity rather than aggregated, so a viewer can show exactly which
/// model is mid-call right now rather than just a tier's busy count.
/// `id` is a stable per-slot key (e.g. "t1-1", "t3") for a viewer to track
/// across snapshots even if `label` (the human-readable model name) isn't
/// unique; unconfigured tiers simply contribute no entries.
#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub tier: Tier,
    pub label: String,
    pub in_flight: bool,
}

/// The most recently synthesized Semantic pattern (`steps::synthesize::
/// synthesize`) still present in the graph - what `self_status` couldn't
/// answer at all (see `AbstractionStatusTool`'s own doc comment for the
/// gap this closes). Identified by `kind == Memory` plus `source_object_ids.
/// len() >= 2`, which only `synthesize` ever produces: a SOAR chunk
/// (`steps::learn::chunk_resolution`) and a compiled render skill
/// (`steps::social_interface::record_render`) both leave `source_object_ids`
/// empty, and a Reflection reclassified as Semantic via `memory_formation`'s
/// `SemanticUpdate` path always carries exactly one (the single object it
/// reflected on, never two) - `synthesize` itself refuses to run on fewer
/// than two usable episodic texts, so two-or-more is a reliable, unique
/// fingerprint rather than an incidental coincidence.
///
/// "Provisional" reflects what's actually tracked, not a fixed status field:
/// `reinforced_count` is how many times `synthesize`'s own dedup path has
/// reconfirmed this exact pattern since it first formed (`0` means it's
/// only ever been synthesized once). There is deliberately no field here
/// for what would contradict the pattern - this architecture has no
/// disconfirmation mechanism, only reinforcement toward an existing match
/// (see `synthesize`'s own dedup doc comment) - so there is nothing honest
/// to report there yet.
#[derive(Debug, Clone, Serialize)]
pub struct ProvisionalAbstractionSummary {
    pub id: MentalObjectId,
    pub text: String,
    pub confidence: f32,
    pub formed_at: EpochMillis,
    /// Times `synthesize`'s dedup path has reconfirmed this exact pattern
    /// since it first formed - `0` for a pattern synthesized only once.
    pub reinforced_count: u32,
    /// `source_object_ids.len()` at formation time - the true original
    /// count, which may exceed `supporting_episodes.len()` if some sources
    /// have since decayed or been discarded from the graph.
    pub source_count: usize,
    /// Text of whichever source episodic memories are still present in the
    /// graph - may be fewer than `source_count` (see that field's doc
    /// comment), never fabricated to make up the difference.
    pub supporting_episodes: Vec<String>,
}

/// This tick's fresh prediction-error reading (`steps::compare::compare`'s
/// output, the same numbers `steps::metacognition::reward_from_comparison`
/// folds into reward and `steps::affect::AffectTracker` smooths into
/// `valence`) — published raw, not just as the smoothed trend, so a self-
/// assessment tool can point to the actual number a given tick's surprise
/// was, not only the slow-moving average it fed into. `None` until the
/// first real Observation resolves an embedding.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PredictionErrorSummary {
    pub error_magnitude: f32,
    pub precision: f32,
    pub epistemic_value: f32,
    pub reward: f32,
    pub at: EpochMillis,
}

/// The real, verified answer to "what entered my awareness because it
/// displaced what" for the most recent tick that actually had one -
/// `steps::displacement::explain_release`'s `ReleaseReason::Displaced` case,
/// republished with both objects' text so a consumer doesn't need a second
/// graph lookup to render the claim. Every field here traces back to a real
/// Broadcast competition this tick: `entrant_id` is a real newly-admitted
/// Working Memory member, `evicted_id` is a real released one, and the pair
/// is only ever populated once `steps::displacement::verify_displacement`
/// has confirmed, by actually re-running the admission algorithm with
/// `entrant_id` removed, that `evicted_id` would have kept its seat without
/// it - never on ranking order alone. `None` on any tick with no
/// competitive eviction (nothing released, or everything released simply
/// wasn't renominated rather than outcompeted - see `ReleaseReason`'s own
/// doc comment for why those are not the same thing and must not be
/// reported as a displacement).
#[derive(Debug, Clone, Serialize)]
pub struct DisplacementSummary {
    pub entrant_id: MentalObjectId,
    pub entrant_text: String,
    pub evicted_id: MentalObjectId,
    pub evicted_text: String,
    pub at: EpochMillis,
}

impl DisplacementSummary {
    /// Renders the real claim this struct backs, in the exact "X entered my
    /// awareness because it displaced Y" shape - built entirely from the
    /// verified `entrant_text`/`evicted_text` fields above, never a
    /// free-floating LLM narrative. Exists so a self-report surface and a
    /// test asserting on that surface's wording read the same construction,
    /// rather than a prompt template silently drifting from what this
    /// struct actually verified.
    pub fn claim_text(&self) -> String {
        format!("{} entered my awareness because it displaced {}", self.entrant_text, self.evicted_text)
    }
}

/// The most recently *selected* (not merely proposed) operator - `steps::
/// executive::propose_operators`'s winning `OperatorProposal`, published so
/// a self-assessment tool can report what Omega itself last actually
/// decided to do and how strongly, rather than only what it's currently
/// holding in Working Memory. `None` before the first tick that ever
/// resolves a `ExecutiveDecision::Selected`.
#[derive(Debug, Clone, Serialize)]
pub struct OperatorProposalSummary {
    pub operator: String,
    pub target_id: MentalObjectId,
    pub preference: f32,
    pub confidence: f32,
    pub at: EpochMillis,
}

/// The always-latest view of the whole engine, published over a
/// `tokio::sync::watch` channel — cheap to read, no lock contention with
/// the hot loop, no polling the actor itself. Broader than just Working
/// Memory: enough for an external viewer to render the architecture as a
/// whole (memory subsystem sizes, the goal stack, tier concurrency), not
/// only its currently-broadcast contents.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EngineSnapshot {
    pub cycle_seq: u64,
    pub working_memory: Vec<WorkingMemoryMember>,
    pub associative_edges: Vec<EdgeSummary>,
    pub memory_counts: MemoryRoleCounts,
    pub goal_stack: Vec<GoalSummary>,
    pub tier_status: Vec<TierStatus>,
    pub active_models: Vec<ModelStatus>,
    /// True for the exact span the engine is awaiting the embedding
    /// client's HTTP round trip (Observe, between Predict and Compare) -
    /// the one genuinely I/O-bound step in an otherwise-instant tick, and
    /// the one place a slow/cold model load actually stalls the cycle.
    /// Published `true` right before the await and `false` right after, so
    /// a viewer can show that specific wait as it happens rather than
    /// seeing a silent gap between two pulses.
    pub embedding_in_flight: bool,
    /// True for the exact span the engine is awaiting the attention model's
    /// HTTP round trip (Broadcast, after Coalition) - the one genuinely
    /// I/O-bound step in an otherwise-instant tick, and the one place a
    /// slow/cold attention model load actually stalls the cycle. Published
    /// `true` right before the await and `false` right after, so a viewer
    /// can show that specific wait as it happens rather than seeing a
    /// silent gap between two pulses.
    pub attention_in_flight: bool,
    /// The Knowledge Library's corpus size (`KnowledgeLibraryStore::count_documents`).
    /// specs.md's Knowledge Library section describes it as external
    /// knowledge other household agents write into over MCP - writes never
    /// pass through this actor, so without a live count here that growth
    /// would otherwise be invisible to a viewer between individually-logged
    /// consult events. `0` on a query failure (graceful degradation, same
    /// as every other store-error path in this engine) rather than
    /// panicking a snapshot publish over a reporting-only field.
    pub kl_doc_count: u64,
    /// The most recently synthesized Semantic pattern still present in the
    /// graph, if any - see `ProvisionalAbstractionSummary`'s own doc comment.
    /// `None` before `steps::synthesize::synthesize` has ever produced one.
    pub provisional_abstraction: Option<ProvisionalAbstractionSummary>,
    /// `steps::affect::AffectTracker::valence()` - the slow-moving read of
    /// how Omega's own predictive model has been doing lately. `0.0`
    /// (neutral) until real history accumulates, same as the tracker itself.
    pub affect_valence: f32,
    /// `steps::affect::AffectTracker::precision_gain()` - the attentional
    /// amplification/dampening this valence is currently producing. `1.0`
    /// (no modulation) at neutral valence.
    pub precision_gain: f32,
    /// This tick's (or the most recent tick's) raw prediction-error reading -
    /// see `PredictionErrorSummary`'s own doc comment for why this is
    /// published separately from `affect_valence`, which only ever shows the
    /// smoothed trend built from a whole series of these.
    pub last_prediction_error: Option<PredictionErrorSummary>,
    /// The most recently selected Executive operator - see
    /// `OperatorProposalSummary`'s own doc comment.
    pub last_operator_proposal: Option<OperatorProposalSummary>,
    /// The most recent tick's verified "X entered my awareness because it
    /// displaced Y" claim, if that tick had one - see
    /// `DisplacementSummary`'s own doc comment for exactly what backs it and
    /// what `None` means here (no eviction this tick, or an eviction that
    /// wasn't a genuine competitive displacement).
    pub last_displacement: Option<DisplacementSummary>,
    /// GNW's attention/consciousness dissociation (Dehaene/Changeux), made
    /// observable: this tick's real Coalition candidates that were attended
    /// (cleared `LoopConfig::attention_threshold`, genuinely competed for
    /// the workspace) but did not ignite (never cleared `LoopConfig::
    /// ignition_threshold` and win a seat in `working_memory` above -
    /// `steps::broadcast::decide_admission_with_hysteresis`). Each entry's
    /// `attention_score` is that candidate's real, fresh score this tick,
    /// not a stale leftover from some earlier admission. Empty on a tick
    /// with no such candidate - not every tick has one, and this is never
    /// padded to look otherwise.
    pub attended_not_ignited: Vec<WorkingMemoryMember>,
    /// How many `Intention`-kind objects currently have `GoalStatus::Active`
    /// - `steps::agenda`'s persistent-intention count. `0` before that
    /// module has ever spawned one.
    pub active_intention_count: usize,
    /// `steps::drives::DriveState`'s five tracked pressures, flattened -
    /// same "raw fields on the snapshot, not the tracker struct itself"
    /// shape as `affect_valence`/`precision_gain` above. See that module's
    /// own doc comment for what each one is grounded in.
    pub drive_uncertainty: f32,
    pub drive_curiosity: f32,
    pub drive_competence: f32,
    pub drive_social_connection: f32,
    pub drive_resource_pressure: f32,
}

/// Cheap `Arc`-cloned handles onto the engine's live model-tier state,
/// deliberately separate from `EngineSnapshot` itself. `EngineSnapshot`'s
/// own `active_models`/`tier_status`/`embedding_in_flight`/`attention_in_flight` are only ever
/// refreshed at the cognitive loop's own publish cadence (once right before
/// the embedding await, once right before the attention model call, once at each tick's end) - a window exactly wide
/// enough to *always* miss a Tier 1-4 candidate's in-flight span, since
/// `DivergentPool::sample`/`TierPool::run_chat` flip their busy state on
/// right before the call and back off right after, entirely inside that
/// gap. Confirmed live: a viewer polling `/snapshot/overview` only ever saw
/// Tier 0 (the embedding model) light up, never Tier 1-4, because Tier 0 is
/// the one call site that also publishes a fresh snapshot right before its
/// own await (see `CognitiveLoopActor`'s `embedding_in_flight` field doc
/// comment). These handles read the same underlying atomics/semaphores
/// directly, at whatever moment a caller asks, rather than at the loop's
/// own cadence - the local API's `/snapshot/overview` handler uses them to
/// build fresh `active_models`/`tier_status`/`embedding_in_flight`/`attention_in_flight` on every
/// request instead of trusting the possibly-stale copies baked into the
/// last-published `EngineSnapshot`.
#[derive(Clone)]
pub struct LiveModelHandles {
    /// `Arc<RwLock<_>>`, not a plain `String`: `CognitiveLoopActor::
    /// with_embedding_label` can be called after `LoopHandles` (and this
    /// bundle, cloned into it) has already been handed to the API layer -
    /// without shared interior mutability here, that later write would only
    /// ever land on the actor's own private copy, leaving every clone
    /// already handed out permanently stuck on the placeholder label.
    pub embedding_label: Arc<RwLock<String>>,
    pub embedding_in_flight: Arc<AtomicBool>,
    pub attention_in_flight: Arc<AtomicBool>,
    pub tier1_pool: Arc<DivergentPool>,
    pub tier2_pool: Arc<DivergentPool>,
    pub tier3_pool: Arc<TierPool>,
    pub tier4_pool: Arc<TierPool>,
}

impl LiveModelHandles {
    /// Every pool empty/idle - for tests that only care about wiring, not
    /// real model behavior (mirrors `EngineSnapshot::default()`'s own role).
    pub fn empty() -> Self {
        Self {
            embedding_label: Arc::new(RwLock::new("(unconfigured)".to_string())),
            embedding_in_flight: Arc::new(AtomicBool::new(false)),
            attention_in_flight: Arc::new(AtomicBool::new(false)),
            tier1_pool: Arc::new(DivergentPool::new(Tier::T1, Duration::from_secs(1), Vec::new())),
            tier2_pool: Arc::new(DivergentPool::new(Tier::T2, Duration::from_secs(1), Vec::new())),
            tier3_pool: Arc::new(TierPool::new(Tier::T3, 1, Duration::from_secs(1))),
            tier4_pool: Arc::new(TierPool::new(Tier::T4, 1, Duration::from_secs(1))),
        }
    }

    pub fn embedding_in_flight(&self) -> bool {
        self.embedding_in_flight.load(Ordering::Relaxed)
    }

    pub fn attention_in_flight(&self) -> bool {
        self.attention_in_flight.load(Ordering::Relaxed)
    }

    pub fn tier_status(&self) -> Vec<TierStatus> {
        vec![
            TierStatus {
                tier: self.tier3_pool.tier(),
                available_permits: self.tier3_pool.available_permits(),
                capacity: self.tier3_pool.capacity(),
            },
            TierStatus {
                tier: self.tier4_pool.tier(),
                available_permits: self.tier4_pool.available_permits(),
                capacity: self.tier4_pool.capacity(),
            },
        ]
    }

    /// Every individually-configured LLM in the reasoning chain, by real
    /// identity - Tier 1/2's divergent candidates plus Tier 3/4's single
    /// models - so a viewer can show exactly which model is mid-call right
    /// now instead of only a tier's aggregate busy count.
    pub fn active_models(&self) -> Vec<ModelStatus> {
        let mut active_models = Vec::new();
        active_models.push(ModelStatus {
            id: "t0".to_string(),
            tier: Tier::T0,
            label: self.embedding_label.read().expect("embedding_label lock poisoned").clone(),
            in_flight: self.embedding_in_flight(),
        });
        for (pool, prefix) in [(&self.tier1_pool, "t1"), (&self.tier2_pool, "t2")] {
            for (i, (label, in_flight)) in pool.model_statuses().into_iter().enumerate() {
                active_models.push(ModelStatus { id: format!("{prefix}-{}", i + 1), tier: pool.tier(), label, in_flight });
            }
        }
        active_models.push(ModelStatus {
            id: "t3".to_string(),
            tier: self.tier3_pool.tier(),
            label: self.tier3_pool.label().to_string(),
            in_flight: self.tier3_pool.is_busy(),
        });
        active_models.push(ModelStatus {
            id: "t4".to_string(),
            tier: self.tier4_pool.tier(),
            label: self.tier4_pool.label().to_string(),
            in_flight: self.tier4_pool.is_busy(),
        });
        active_models
    }
}
