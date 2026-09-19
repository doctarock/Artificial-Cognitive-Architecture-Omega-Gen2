use crate::steps::agenda::AgendaConfig;
use crate::steps::boredom::BoredomConfig;
use crate::steps::confidence_revision::ConfidenceRevisionConfig;
use crate::steps::eligibility::EligibilityConfig;
use crate::steps::executive::ExecutiveConfig;
use crate::steps::knowledge_library::KnowledgeLibraryConfig;
use crate::steps::memory_formation::MemoryFormationConfig;
use crate::steps::orient::OrientingConfig;
use crate::steps::recall::RecallConfig;
use crate::steps::synthesize::SynthesisConfig;
use crate::DEFAULT_WORKING_MEMORY_CAPACITY;

/// Default-off switches for ablation runs: controlled ways to remove one
/// cognitive organ from the loop while leaving the rest of the architecture
/// intact. These are experiment controls, not product defaults.
#[derive(Debug, Clone, Default)]
pub struct AblationConfig {
    pub disable_agenda: bool,
    pub disable_boredom: bool,
    pub disable_synthesis: bool,
    pub disable_recall: bool,
    /// Independent of `disable_recall` - the generic Working-Memory-anchored
    /// recall pass and the interlocutor-anchored "social cloud" pass
    /// (`steps::interlocutor::social_cloud_anchors`) are two separate
    /// mechanisms with two separate kill switches, so either can be ablated
    /// on its own to prove which one a given recalled memory actually came
    /// through.
    pub disable_social_recall: bool,
    pub disable_orienting: bool,
    pub disable_eligibility_learning: bool,
    pub disable_procedural_fast_path: bool,
    pub disable_preconscious_traces: bool,
    pub disable_lateral_inhibition: bool,
    pub disable_goal_consequence_learning: bool,
    pub disable_goal_impact_prediction: bool,
    pub disable_spike_propagation: bool,
    pub disable_sensor_habituation: bool,
    pub disable_curated_answer_fast_path: bool,
}

/// Controls how much autonomous/background cognition may run on the same
/// tick as user-facing input. The intent is responsiveness, not
/// simplification: skipped work remains eligible on later idle ticks.
#[derive(Debug, Clone)]
pub struct ResponsivenessConfig {
    pub defer_synthesis_on_foreground_input: bool,
    pub defer_new_agenda_spawns_on_foreground_input: bool,
}

impl Default for ResponsivenessConfig {
    fn default() -> Self {
        Self {
            defer_synthesis_on_foreground_input: true,
            defer_new_agenda_spawns_on_foreground_input: true,
        }
    }
}

/// Sampling temperature per model call site, previously a hardcoded literal
/// on each `GenerateRequest` (`GenerateRequest` itself only has `prompt`/
/// `temperature` - no `top_p`/`seed`/penalty knobs exist to tune here).
/// `Default` reproduces every one of those literals exactly, so adopting
/// this struct changes no behavior on its own - it only makes "more/less
/// varied" a config change instead of a source edit. Deliberately its own
/// struct rather than folded into `ExecutiveConfig`/`BoredomConfig`/
/// `MemoryFormationConfig`/`SynthesisConfig`: those already group each
/// step's *behavioral* thresholds, and temperature is an orthogonal,
/// uniform concern that cuts across all of them.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureConfig {
    /// `cognitive_core::reflect`'s Tier 1/2/3 ladder call.
    pub reflection: f32,
    /// The Tier 3/4 confidence-impasse escalation prompt in `loop_actor`'s
    /// `EscalateTier` arm.
    pub impasse_resolution: f32,
    /// `steps::boredom::generate`'s idle-initiative call.
    pub boredom: f32,
    /// `steps::executive::propose_communicative_intent`'s Speak/Ask/Ignore
    /// vote.
    pub operator_proposal: f32,
    /// `steps::executive::propose_tool_intent`'s tool-request judgement.
    pub tool_intent: f32,
    /// `steps::memory_formation::form_memory`'s classification call.
    pub memory_formation: f32,
    /// `steps::social_interface::render_speech`'s Tier 1 rendering pass.
    pub social_rendering: f32,
    /// `steps::synthesize::synthesize`'s pattern-abstraction call.
    pub synthesis: f32,
}

impl Default for TemperatureConfig {
    fn default() -> Self {
        Self {
            reflection: 0.5,
            impasse_resolution: 0.3,
            boredom: 0.6,
            operator_proposal: 0.4,
            tool_intent: 0.2,
            memory_formation: 0.3,
            social_rendering: 0.4,
            synthesis: 0.4,
        }
    }
}

/// Tuning knobs for the `CognitiveLoopActor` — thresholds and capacities,
/// not theoretical commitments (see specs.md's Model Tiering section:
/// "capacity is a tuning parameter, not a design commitment").
#[derive(Debug, Clone)]
pub struct LoopConfig {
    pub decay_d: f32,
    pub working_memory_capacity: usize,
    pub attention_threshold: f32,
    /// GNW's ignition asymmetry (Dehaene/Changeux), made literal in
    /// `steps::broadcast::decide_admission_with_hysteresis`: the stricter
    /// bar a candidate not already in Working Memory must clear to newly
    /// enter, versus the looser `attention_threshold` an already-admitted
    /// member only has to keep clearing to remain. Distinct from
    /// `attentional_refractory_ms` below - that's a *channel*-level cooldown
    /// on nominating fresh candidates at all; this is an *object*-level
    /// asymmetry in how hard entering versus staying is, applied every tick
    /// regardless of channel. Must be `>= attention_threshold` for the
    /// asymmetry to mean anything (an already-eligible object should never
    /// need to clear a *lower* bar to newly enter than to stay).
    pub ignition_threshold: f32,
    /// Divisive normalization / crowding - `steps::coalition::
    /// apply_crowding_normalization`'s own doc comment has the full design
    /// reasoning (`docs/cognitive-capability-audit.md`'s "Phase 5,
    /// revisited"). How much *else* is competing this tick becomes a real
    /// suppressive factor on every candidate's effective score, distinct
    /// from `ignition_threshold` (time/hysteresis) and from
    /// `working_memory_capacity` (a hard cap on the output, not the
    /// eligibility bar). `0.0` is an exact identity - the real,
    /// pre-normalization baseline.
    pub crowding_strength: f32,
    /// Fresh ignition learns sparse winner-to-loser inhibitory edges.
    pub lateral_inhibition_increment: f32,
    pub lateral_inhibition_max_losers: usize,
    pub noise_max: f32,
    /// Minimum wall-clock gap between periodic write-behind flushes to the
    /// durable store — never on every tick, per the build plan's concurrency
    /// model. Wall-clock rather than a tick count deliberately: with
    /// `CognitiveScheduler` free to idle-sleep between ticks
    /// (`max_idle_interval_ms`), a tick-counted cadence stretched out in
    /// real time right along with it, leaving genuinely dirty state
    /// (written just before an idle stretch began) unflushed for longer the
    /// higher that idle ceiling was raised.
    pub flush_interval_ms: i64,
    pub executive_config: ExecutiveConfig,
    pub memory_config: MemoryFormationConfig,
    /// The consumer side of `EdgeKind::Contradicts`: lowers a memory's
    /// `confidence` when a genuinely new contradiction against it is
    /// observed, rather than leaving `confidence` fixed at whatever it was
    /// stamped at creation forever. See
    /// `steps::confidence_revision::apply_contradiction_penalty`'s own doc
    /// comment for why this is distinct from ACT-R activation decay.
    pub confidence_revision: ConfidenceRevisionConfig,
    pub knowledge_library_config: KnowledgeLibraryConfig,
    /// Idle-time pattern synthesis (`steps::synthesize`): abstracting a new
    /// Semantic memory out of several recently-formed Episodic ones,
    /// self-triggered rather than externally prompted. The "balanced"
    /// defaults on `SynthesisConfig` mean: needs 3 new episodic memories
    /// since the last attempt, at least 5 minutes apart - noticeable
    /// self-directed activity without hammering the tier pools in the
    /// background on every idle tick.
    pub synthesis_config: SynthesisConfig,
    /// Graph-wide multi-hop spreading activation (`steps::recall`, Step
    /// 4.5): discovers dormant memories associatively reachable from
    /// current Working Memory that Step 4's single-hop spread structurally
    /// cannot reach on its own - specs.md's Memory Recall section describes
    /// exactly this ("the graph nodes that clear the retrieval threshold
    /// become recall candidates," unqualified - not just objects already in
    /// Working Memory). Runs on each cognitive event: it's pure in-memory
    /// computation (no tier/model call) bounded by `RecallConfig::max_hops`
    /// (frontier-limited propagation, not a full graph scan). The scheduler
    /// avoids manufacturing idle events solely to run recall.
    pub recall_config: RecallConfig,
    /// Time-decay rate applied to every `AssociativeEdge`'s *effective*
    /// strength (`aca_graph::effective_strength`, read lazily - never
    /// mutates the stored `strength`), used by both Step 4's single-hop
    /// spreading activation and `steps::recall`'s multi-hop propagation.
    /// Reasoned in the same day-scale terms `self_memory_activation_bonus`
    /// already uses below, for consistency across the two mechanisms that
    /// both govern how long dormant content stays reachable: a 14-day
    /// half-life, `ln(2) / (14 * 24 * 3600 * 1000)`. An edge at full
    /// strength retains ~50% of its strength after two weeks with zero
    /// reinforcement, decaying further from there - but `reinforce_edge`
    /// stamps a fresh `last_coactivated_at` on every co-activation, so an
    /// actively-used association never meaningfully decays at all. Not an
    /// empirically validated constant - a reasoned starting point, tunable
    /// once real usage exists.
    pub edge_decay_rate_per_ms: f64,
    /// Idle/"boredom" self-stimulus (`steps::boredom`): when Working Memory
    /// has been genuinely empty for a while, either a standing duty comes
    /// due or Tier 1 invents something worth thinking about or acting on -
    /// see that module's doc comments for the full behavior.
    pub boredom_config: BoredomConfig,
    /// How much weight `steps::compare::ComparisonResult::epistemic_value`
    /// (active inference's curiosity term - specs.md's Attention section)
    /// gets folded into a fresh observation's surprise-like attention
    /// score, alongside `precision_weighted_surprise`. `0.0` would make
    /// curiosity inert; kept modest and nonzero by default so uncertain
    /// channels get a real, but not dominant, boost toward Working Memory
    /// admission relative to genuine prediction-error surprise.
    pub curiosity_weight: f32,
    /// Cheap preconscious orienting and per-object event dynamics. Its output
    /// is a bounded attention pulse, never prose or a model call.
    pub orienting_config: OrientingConfig,
    /// Repeated low-error environment readings below this orienting score
    /// remain in memory without competing for workspace ignition.
    pub sensor_habituation_threshold: f32,
    pub eligibility_config: EligibilityConfig,
    /// Phase 3 of the GWT-parity roadmap (`docs/cognitive-capability-audit.md`'s
    /// second addendum): `steps::memory_formation::maybe_automatic_remember`'s
    /// hard gate. `precision_weighted_surprise` clamps to `[0.0, 2.0]`
    /// elsewhere in this engine (see `steps::executive::propose_operators`'s
    /// own Remember-preference scaling) - `1.0` is that range's midpoint, a
    /// reasoned "genuinely surprising, not merely nonzero" starting point,
    /// not an empirically tuned constant.
    pub automatic_memory_formation_surprise_threshold: f32,
    /// specs.md's Memory Competition section: "relevance to self → elevated
    /// baseline activation for Self Memory-linked nodes." Added to
    /// `activation.total` every tick for any object tagged
    /// `MemoryRole::SelfMemory` (see `loop_actor`'s
    /// `apply_self_memory_activation_bonus`), on top of its ordinary ACT-R
    /// base-level/spreading/noise - large enough that identity content
    /// stays a real competitor for Working Memory admission across a very
    /// long dormant stretch (specs.md's Self Memory section: "rarely lose
    /// the competition for relevance even when dormant"), not just
    /// immediately after creation/reference.
    pub self_memory_activation_bonus: f32,
    /// `steps::metacognition::OutcomeRegistry`'s expiry window: how long
    /// (wall-clock) a decision (a chunk's bias winning selection, a Tier 3/4
    /// self-report being trusted) stays eligible for reward/calibration
    /// feedback before being dropped, uncredited. ACT-R-consistent
    /// temporal-proximity credit assignment - an outcome too far removed
    /// from the decision that produced it isn't good evidence about it
    /// anymore. Wall-clock rather than a tick count deliberately - see
    /// `OutcomeRegistry`'s own doc comment on `expiry_ms` for why a
    /// tick-counted window stopped being a safe stand-in for a real
    /// duration once `CognitiveScheduler` made ticks-per-second variable.
    pub outcome_feedback_window_ms: i64,
    /// Global Neuronal Workspace's ignition-then-refractory dynamics
    /// (Dehaene/Changeux), applied per `SourceChannel`: how long after a
    /// fresh candidate from a channel wins Broadcast before that same
    /// channel's *next* fresh candidate is even nominated for Coalition -
    /// see `loop_actor::tick()`'s Step 5/Step 6. Not a delay or a dropped
    /// input: a refractory-blocked candidate still gets a full, real shot
    /// at admission on the very next tick via `pending_admission`, exactly
    /// like a Reflection created during Act. Short by design - the human
    /// attentional blink this models is on the order of a few hundred
    /// milliseconds, not `BoredomConfig::idle_threshold_ms`'s tens of
    /// seconds - long enough to suppress genuine same-tick/next-tick
    /// same-channel noise (a rapid-fire sensor, backlog replay) without
    /// ever being long enough to feel like Omega ignoring a person's very
    /// next remark in ordinary conversation.
    pub attentional_refractory_ms: i64,
    /// How long an attended-but-not-ignited candidate remains capable of
    /// influencing later competition before its subliminal trace expires.
    pub preconscious_trace_ms: i64,
    /// Event cadence for re-evaluating a retained preconscious candidate.
    pub preconscious_reevaluation_ms: i64,
    /// Bounded, delayed graph-neighborhood propagation after a local firing.
    pub spike_delay_ms: i64,
    pub spike_max_fanout: usize,
    pub spike_max_pending: usize,
    pub spike_max_events_per_tick: usize,
    /// Persistent-intention maintenance (`steps::agenda`): surfacing
    /// existing Active intentions for attention, spawning/decaying/
    /// resolving them - see that module's own doc comment for the full
    /// behavior.
    pub agenda_config: AgendaConfig,
    /// Experimental ablation switches. All default to `false`; production
    /// behavior is unchanged unless a benchmark/test explicitly opts in.
    pub ablation_config: AblationConfig,
    pub responsiveness_config: ResponsivenessConfig,
    /// How long `tick()` waits on the Omega Attention model before falling
    /// back to the deterministic Broadcast algorithm - see
    /// `steps::broadcast`'s doc comments. Short by design, unlike Tier 3/4's
    /// generous multi-minute backstops: this runs on *every* tick (a
    /// per-tick dependency, not an occasional escalation), so a hung
    /// endpoint must never be allowed to stall the whole cognitive loop.
    pub attention_timeout: std::time::Duration,
    /// The gate that decides, every tick, whether the attention model's
    /// answer is trusted over the deterministic fallback - see
    /// `tick()`'s Step 5/6 boundary. Deliberately high: this is the first
    /// mechanism in the cognitive cycle that lets a model call actively
    /// override an otherwise-purely-deterministic, well-tested algorithm,
    /// so a low-confidence answer should fail closed rather than risk a
    /// low-quality real decision.
    pub attention_min_confidence: f32,
    /// Observe a configured attention specialist without allowing its vote
    /// to change Working Memory. Synthetic confidence is not calibration;
    /// promote a model to active control only after held-out and live checks.
    pub attention_shadow_only: bool,
    /// Bounds how many consecutive ticks the attention model's judgment can
    /// run unchecked against the deterministic algorithm. The model only
    /// ever emits ONE incremental state transition per tick (unlike the
    /// deterministic algorithm, which re-derives the whole ranked Working
    /// Memory set from scratch every tick) - so a run of confident
    /// MAINTAIN/IGNORE no-ops could in principle let real, live ACT-R
    /// activation drift underneath without ever being reconsidered. Every
    /// `attention_reconciliation_interval`-th consecutive model-trusted
    /// tick is forced back through the deterministic algorithm regardless
    /// of the model's own confidence, re-synchronizing Working Memory with
    /// what the live activation values actually say and resetting the
    /// counter - bounding worst-case staleness to this many ticks rather
    /// than leaving it unbounded. `0` disables the model path from ever
    /// running at all (every tick is forced) - not a useful setting on its
    /// own, but keeps the arithmetic simple rather than special-cased.
    pub attention_reconciliation_interval: u64,
    /// How long `cognitive_core::resolve_via_tier_ladder` lets Tier 1/2
    /// race alone before it non-blockingly tries to hedge with a
    /// speculative Tier 3 call, run concurrently with whatever's left of
    /// the cheap tiers rather than only starting after they're both fully
    /// exhausted. Only ever fires once the cheap tiers have already been
    /// racing this long without a decisive winner - never on the common
    /// fast-success path - and only if the single-flight seat is free
    /// right now (`TierPool::try_acquire`); a busy Tier 3 just means this
    /// call falls back to the unchanged sequential path, exactly as
    /// before. Kept long relative to `attention_timeout`'s per-tick
    /// backstop, deliberately: this hedges a rare, already-slow escalation
    /// path, not every tick, so there's no pressure to keep it short - and
    /// firing it too eagerly would spend the scarce Tier 3 seat (shared
    /// with `resolve_confidence_impasse`'s own escalations) on reflections
    /// that were always going to resolve via the cheap tiers anyway.
    pub tier3_hedge_delay: std::time::Duration,
    /// Per-call-site sampling temperature - see `TemperatureConfig`'s own
    /// doc comment.
    pub temperature: TemperatureConfig,
    /// Opt-in softmax draw among near-tied Executive proposals - see
    /// `steps::executive::SamplingConfig`'s own doc comment. Default-off.
    pub sampling: crate::steps::executive::SamplingConfig,
    /// Decay half-life, in milliseconds, of `loop_actor::conversational_presence` -
    /// the reactive 0.0-1.0 "how awake for conversation" dial threaded into
    /// `prompt_templates::reflect_prompt`. Presence is 1.0 the instant a live
    /// `SourceChannel::ConversationInput` turn wins Broadcast and decays
    /// exponentially toward 0.0 as idle time since then grows; this is that
    /// decay's half-life, not a hard cutoff - there's no "conversation ended"
    /// event to key off, so a smooth decay is what's actually reactive to a
    /// real, ongoing back-and-forth versus a single passing remark. At the
    /// default 30s: presence is still ~0.8 ten seconds after a turn - long
    /// enough to cover ordinary think/speak latency within one exchange -
    /// but has clearly receded to ~0.06 after a full two-minute silence,
    /// back to the fully introspective register the `presence_block` doc
    /// comment describes at presence 0.
    pub presence_half_life_ms: i64,
    /// `CognitiveScheduler`'s idle-sleep ceiling: `run()` never waits longer
    /// than this before ticking again, even when nothing is due. Safe to
    /// raise on constrained hardware to cut idle wake-ups without the
    /// knock-on effects this field used to carry:
    ///
    /// - `flush_interval_ms` and `outcome_feedback_window_ms` are wall-clock
    ///   durations, not tick counts, so they hold their real-world meaning
    ///   regardless of how far apart idle ticks actually land.
    /// - `attention_reconciliation_interval` is deliberately still a tick
    ///   count, not a bug this field is coupled to: it bounds how many
    ///   consecutive attention-model *state transitions* can accumulate
    ///   unreconciled, and the model only ever emits one transition per
    ///   tick - no ticks means no fresh transitions means nothing new to
    ///   reconcile, so idle time genuinely doesn't erode this guarantee the
    ///   way it would a real duration.
    /// - `BoredomConfig::self_status_interrupt_drive_threshold`'s interrupt
    ///   has its own dedicated scheduler wake (`CognitiveScheduler::
    ///   self_status_interrupt_ready`), decoupled from this field entirely -
    ///   see that function's own doc comment.
    pub max_idle_interval_ms: u64,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            decay_d: 0.5,
            working_memory_capacity: DEFAULT_WORKING_MEMORY_CAPACITY,
            // BaseLevel(t) = -0.5*ln(t_seconds) at decay_d=0.5 crosses zero
            // at t=1s and keeps falling - a threshold near zero (e.g. the
            // previous 0.05 default) means an unreinforced object decays
            // out of Working Memory in about a second, confirmed live: a
            // spoken/remembered object appeared in the visualization and
            // was gone on the very next snapshot poll. -2.0 pushes that
            // natural dwell time out to roughly e^4 ≈ 55 seconds - long
            // enough to actually watch, still short enough that dormant
            // content genuinely fades rather than accumulating forever.
            attention_threshold: -2.0,
            // Same `BaseLevel(t) = -0.5*ln(t_seconds)` math `attention_
            // threshold`'s own comment above already grounds this file in:
            // -1.8 crosses at t = e^3.6 ≈ 37s - a real but modest gap over
            // the ~55s dwell time -2.0 gives an already-admitted object (per
            // that same comment), not the much larger gap a round number
            // like 0.0 would impose. Confirmed live: 0.0 was tried first and
            // broke this codebase's own established multi-hop recall
            // capability - a dormant memory rescued by two real associative
            // hops (`cognitive_capabilities_end_to_end.rs`'s own
            // `dormant_memory_resurfaces_via_the_live_actor` test) lands
            // around -1.3 total activation, nowhere near clearing 0.0 on its
            // first tick back. -1.8 keeps real headroom below that measured
            // value while still being strictly tighter than the -2.0 floor
            // an already-ignited object only has to sustain - entering
            // fresh is the harder case, staying is the easier one, but
            // "harder" must not mean "recall can never win admission at
            // all," which 0.0 effectively did.
            ignition_threshold: -1.8,
            // A reasoned starting point, calibrated the same way
            // `ignition_threshold` above was - not an empirically tuned
            // constant. At a handful of simultaneously-competing candidates
            // each scoring a few points (this engine's ordinary busy-tick
            // range), `crowding_strength * others_positive` should land as
            // a modest, real fraction rather than either nothing (defeats
            // the point) or a dominant force (would make ordinary
            // multi-candidate ticks unstable). 0.05 was checked against the
            // full test suite exactly as `ignition_threshold`'s own default
            // was; see that field's doc comment for what "checked" means in
            // practice here.
            crowding_strength: 0.05,
            lateral_inhibition_increment: 0.05,
            lateral_inhibition_max_losers: 8,
            noise_max: 0.01,
            // Same real-world cadence the old `flush_every_n_cycles: 5`
            // default produced at this file's `max_idle_interval_ms: 250`
            // default (5 * 250ms) - now expressed directly instead of
            // depending on ticks actually arriving every 250ms.
            flush_interval_ms: 1_250,
            executive_config: ExecutiveConfig::default(),
            memory_config: MemoryFormationConfig::default(),
            confidence_revision: ConfidenceRevisionConfig::default(),
            knowledge_library_config: KnowledgeLibraryConfig::default(),
            synthesis_config: SynthesisConfig::default(),
            recall_config: RecallConfig::default(),
            // See `edge_decay_rate_per_ms`'s own doc comment above: a
            // 14-day half-life, ln(2) / (14 * 24 * 3600 * 1000).
            edge_decay_rate_per_ms: 5.73e-10,
            boredom_config: BoredomConfig::default(),
            curiosity_weight: 0.2,
            orienting_config: OrientingConfig::default(),
            sensor_habituation_threshold: 0.2,
            eligibility_config: EligibilityConfig::default(),
            automatic_memory_formation_surprise_threshold: 1.0,
            // At decay_d=0.5, a single reference N seconds ago alone yields
            // base_level = -0.5*ln(N); a bonus of 5.0 keeps total activation
            // above the -2.0 default attention_threshold until N exceeds
            // roughly e^14 seconds (~14 days) even with zero reinforcement
            // in between - "rarely lose the competition... even when
            // dormant," not "never decays," and still just a tuning
            // parameter (see this struct's own doc comment), not a
            // theoretical commitment.
            self_memory_activation_bonus: 5.0,
            // Long enough for the next same-channel observation - including
            // one that waited on a real Tier 3 round trip - to plausibly be
            // "the reply to that," short enough that a decision's
            // credit/blame stays attributable to it rather than to whatever
            // unrelated thing the conversation moved on to.
            outcome_feedback_window_ms: 5_000,
            // ~4x a typical human attentional blink's upper bound (per this
            // field's own doc comment) - comfortably shorter than any
            // realistic human turn-taking gap, so it never reads as Omega
            // ignoring a person, while still real enough to suppress a
            // genuine same-channel burst (backlog replay, a chatty sensor)
            // arriving within a tick or two of the same channel's last win.
            attentional_refractory_ms: 400,
            preconscious_trace_ms: 1_500,
            preconscious_reevaluation_ms: 150,
            spike_delay_ms: 10,
            spike_max_fanout: 16,
            spike_max_pending: 4_096,
            spike_max_events_per_tick: 64,
            agenda_config: AgendaConfig::default(),
            ablation_config: AblationConfig::default(),
            responsiveness_config: ResponsivenessConfig::default(),
            // This runs on every tick, unlike Tier 3/4's occasional,
            // generously-timed calls - short enough that a hung endpoint
            // can never meaningfully stall the loop, comfortably long
            // enough for a local 0.5B model's real latency.
            attention_timeout: std::time::Duration::from_millis(300),
            // High by design - see this field's own doc comment above.
            attention_min_confidence: 0.6,
            attention_shadow_only: true,
            // A reasoned starting point, not an empirically validated
            // constant (same stance as this file's other tuning knobs):
            // frequent enough that live activation drift can't compound
            // unnoticed for long, without defeating the point of deferring
            // to the model most of the time.
            attention_reconciliation_interval: 20,
            // ~10s: long enough that the common case (a cheap tier
            // resolving well within a few seconds) never spends the
            // scarce Tier 3 seat speculatively, short enough that a
            // genuinely-escalating reflection doesn't pay the full
            // sequential tier1+tier2+tier3 latency chain when Tier 3
            // happens to be free to hedge with.
            tier3_hedge_delay: std::time::Duration::from_secs(10),
            temperature: TemperatureConfig::default(),
            sampling: crate::steps::executive::SamplingConfig::default(),
            // See this field's own doc comment for what this specific value
            // yields (~0.8 at 10s, ~0.06 at two minutes).
            presence_half_life_ms: 30_000,
            // See this field's own doc comment for the reasoning behind
            // this specific value.
            max_idle_interval_ms: 250,
        }
    }
}
