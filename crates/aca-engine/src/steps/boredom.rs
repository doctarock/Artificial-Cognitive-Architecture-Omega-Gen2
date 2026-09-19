use aca_graph::Graph;
use aca_tiers::{DivergentPool, GenerateRequest};
use aca_types::{MemoryRole, MentalObjectId};
use aca_util::EpochMillis;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::drives::DriveState;
use super::tools::SelfStatusTool;
use crate::prompt_templates::{daydream_prompt, idle_initiative_prompt};

/// Tuning knobs for idle/"boredom" self-stimulus — thresholds and cadence,
/// not theoretical commitments (see `LoopConfig`'s own doc comment on this
/// framing, and `SynthesisConfig`, whose shape this mirrors).
#[derive(Debug, Clone, Copy)]
pub struct BoredomConfig {
    /// How long Working Memory must have been continuously empty before a
    /// boredom stimulus is even considered.
    pub idle_threshold_ms: i64,
    /// Minimum wall-clock gap between boredom stimuli, success or failure -
    /// a cost-control backstop on top of the idle-threshold gate, same
    /// pattern as `SynthesisConfig::min_interval_ms`.
    pub min_interval_ms: i64,
    /// Cadence for the one standing duty shipped today: reviewing its own
    /// cognitive state via `SelfStatusTool`. An upper bound on how long
    /// self-status can go unchecked, not the only way it fires - see
    /// `self_status_drive_threshold`.
    pub self_status_interval_ms: i64,
    /// A self-status check is pulled forward, regardless of how recently
    /// one last ran, whenever `DriveState::self_monitoring_pressure`
    /// reaches this level. Interoceptive-inference framing (Seth/Craig's
    /// "beast machine" theory): attention turns inward when homeostasis is
    /// actually disrupted, not on a fixed schedule - a real person doesn't
    /// wait for their next calendared checkup to notice something feels
    /// wrong. High by design, same "fail closed" precedent as
    /// `LoopConfig::attention_min_confidence`: this bypasses the ordinary
    /// interval backstop, so it should only fire on genuine, sustained
    /// pressure, not routine per-tick noise. `1.1` (unreachable, since every
    /// drive is clamped to `[0, 1]`) would fully disable this early-fire
    /// path and fall back to interval-only behavior.
    pub self_status_drive_threshold: f32,
    /// A self-status check fires as a genuine interrupt - competing for
    /// Broadcast on this tick even though Working Memory is *not* empty,
    /// bypassing `loop_actor`'s ordinary idle gate entirely rather than
    /// merely pulling the interval forward the way `self_status_drive_threshold`
    /// does - once `DriveState::self_monitoring_pressure` reaches this level.
    /// Set higher than `self_status_drive_threshold`: that field only ever
    /// skips the *wait*, still inside the idle branch; this one skips the
    /// idle requirement itself, which is the more disruptive of the two, so
    /// it should only fire on real, sustained internal disruption - the
    /// interoceptive equivalent of a routine ache (pulls a checkup forward)
    /// versus something urgent enough to stop what you're doing (interrupts
    /// outright). Same "unreachable disables this path" convention as
    /// `self_status_drive_threshold` at `1.1`.
    ///
    /// `CognitiveScheduler::self_status_interrupt_ready` gives this path its
    /// own dedicated wake, independent of `LoopConfig::max_idle_interval_ms` -
    /// raising that idle-sleep ceiling does not delay this interrupt.
    pub self_status_interrupt_drive_threshold: f32,
    /// How many dormant memories a daydream recombines at once - kept
    /// small, same "tens-low-hundreds of nodes" scale reasoning as
    /// `SynthesisConfig::max_cluster_size`, deliberately smaller than that
    /// field's default: a daydream is a free-associative nudge, not a
    /// deliberate abstraction pass.
    pub daydream_sample_size: usize,
    /// Fewer than this many eligible replay candidates in the graph and
    /// `generate_daydream` gives up rather than recombining a
    /// near-empty set - falls back to the plain idle-invention path
    /// instead (see `generate`'s own doc comment).
    pub daydream_min_sources: usize,
    /// `select_replay_sources` samples from the top
    /// `daydream_sample_size * daydream_pool_multiplier` most-activated
    /// eligible memories, not the single most-activated few outright -
    /// otherwise the exact same one or two memories would recombine every
    /// idle tick, which reads as a stuck loop, not mind-wandering.
    pub daydream_pool_multiplier: usize,
    /// Minimum `max(curiosity, uncertainty)` drive pressure before a
    /// daydream is even attempted. Deliberately low - default-mode-network
    /// theory (Buckner, Andrews-Hanna) treats mind-wandering as the brain's
    /// *default* idle activity, not something that needs strong
    /// justification to occur; `daydream_resource_pressure_ceiling` below is
    /// what actually suppresses it under real constraint.
    pub daydream_min_drive: f32,
    /// A daydream is skipped (falling back to the plain idle-invention
    /// path) once `DriveState::resource_pressure` reaches this level -
    /// undirected associative exploration is the first thing to give way
    /// when the constrained Tier 3/4 pools are already under real demand,
    /// mirroring how the rest of this engine treats `resource_pressure` as
    /// a pressure specifically toward triage, not toward any single
    /// behavior.
    pub daydream_resource_pressure_ceiling: f32,
}

impl Default for BoredomConfig {
    fn default() -> Self {
        Self {
            idle_threshold_ms: 45_000,
            min_interval_ms: 120_000,
            self_status_interval_ms: 900_000,
            self_status_drive_threshold: 0.75,
            // Comfortably above `self_status_drive_threshold` - see this
            // field's own doc comment for why bypassing the idle
            // requirement outright warrants a stricter bar than merely
            // bypassing the wait.
            self_status_interrupt_drive_threshold: 0.9,
            daydream_sample_size: 3,
            daydream_min_sources: 2,
            daydream_pool_multiplier: 3,
            daydream_min_drive: 0.25,
            daydream_resource_pressure_ceiling: 0.7,
        }
    }
}

/// What `generate` decided to inject as this tick's lowest-priority
/// `Incoming` source - either a standing duty coming due, a memory-grounded
/// daydream, or something Tier 1 invented because nothing else was.
/// `requested_tool`, when present, is a name from the `available_tools` list
/// `generate` was called with; the caller tags the resulting Observation's
/// `data.requested_tool` with it so `steps::executive::propose_operators`
/// can propose `Operator::Act` for it directly, without needing the text to
/// happen to match a trigger phrase.
#[derive(Debug, Clone, PartialEq)]
pub struct BoredomStimulus {
    pub text: String,
    pub requested_tool: Option<&'static str>,
    /// The dormant Episodic/Semantic memories this stimulus was recombined
    /// from, when it came from `generate_daydream` - empty for the
    /// self-status duty and for plain Tier 1 invention, which have no
    /// source material of their own. Lets the caller tag the resulting
    /// Observation's provenance distinctly from ordinary boredom invention
    /// (see `loop_actor::tick`'s `Incoming::Boredom` handling).
    pub source_ids: Vec<MentalObjectId>,
}

/// The standing self-status duty's stimulus - one source of truth for its
/// shape, shared by `generate`'s own idle-gated due-check and
/// `loop_actor::tick`'s independent interrupt gate
/// (`BoredomConfig::self_status_interrupt_drive_threshold`), which decide
/// *when* to reach for this under genuinely different policies but must
/// never drift on *what* it actually says. `interrupt` only changes the
/// wording - both variants still request `SelfStatusTool` and carry no
/// replay provenance - but the wording difference is not cosmetic: an idle
/// routine check-in and a disruption serious enough to interrupt active
/// Working Memory content are different experiences, and the resulting
/// Observation's text is what `steps::compare`/`steps::executive` actually
/// reason over downstream.
pub fn self_status_stimulus(interrupt: bool) -> BoredomStimulus {
    let text = if interrupt {
        "Something about my own state doesn't feel right - let me check in on it now, before this goes any further."
    } else {
        "Nothing has needed my attention for a while - let me check in on my own state."
    };
    BoredomStimulus { text: text.to_string(), requested_tool: Some(SelfStatusTool::NAME), source_ids: Vec::new() }
}

/// Duty-first, daydream-second, invention-third: a standing duty coming due
/// is cheap and certain (no model call), so it always wins when due. Failing
/// that, a memory-grounded daydream is attempted whenever the graph has
/// enough dormant material and the drives favor it (see
/// `BoredomConfig::daydream_min_drive`/`daydream_resource_pressure_ceiling`);
/// only once that's unavailable or inapplicable does this fall back to
/// asking Tier 1 to invent something from nothing. Returns `None` when
/// nothing applies - no duty is due, no daydream fires, and Tier 1 isn't
/// configured to invent anything - so the caller simply skips this tick
/// rather than injecting nothing useful.
#[allow(clippy::too_many_arguments)]
pub async fn generate(
    graph: &Graph,
    tier1_pool: &DivergentPool,
    available_tools: &[(&'static str, &'static str)],
    last_self_status_at: Option<EpochMillis>,
    drive_state: &DriveState,
    config: &BoredomConfig,
    self_summary: &str,
    now: EpochMillis,
    rng: &mut StdRng,
    temperature: f32,
) -> Option<BoredomStimulus> {
    let self_status_due = last_self_status_at.is_none_or(|t| now.0 - t.0 >= config.self_status_interval_ms)
        || drive_state.self_monitoring_pressure() >= config.self_status_drive_threshold;
    if self_status_due && available_tools.iter().any(|(name, _)| *name == SelfStatusTool::NAME) {
        return Some(self_status_stimulus(false));
    }

    if tier1_pool.is_empty() {
        return None;
    }

    let daydream_ready = drive_state.curiosity.max(drive_state.uncertainty) >= config.daydream_min_drive
        && drive_state.resource_pressure < config.daydream_resource_pressure_ceiling;
    if daydream_ready
        && let Some(stimulus) = generate_daydream(graph, tier1_pool, config, self_summary, rng, temperature).await
    {
        return Some(stimulus);
    }

    let prompt = idle_initiative_prompt(self_summary, available_tools);
    let req = GenerateRequest { prompt, temperature };
    let samples = tier1_pool.sample(req, 1).await.ok()?;
    let best = samples
        .into_iter()
        .filter(|sample| !sample.raw_text.trim().is_empty())
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))?;
    Some(parse_boredom_response(&best.raw_text, available_tools))
}

/// Recombines a handful of dormant Episodic/Semantic memories into a fresh
/// thought - the constructive-episodic-simulation account of mind-wandering
/// (Schacter & Addis): imagining reuses the same machinery as remembering,
/// by recombining stored fragments rather than generating from nothing, the
/// way `idle_initiative_prompt`'s plain invention path does. Returns `None`
/// (letting the caller fall back to plain invention) whenever
/// `select_replay_sources` can't find enough eligible material, or the model
/// call itself fails/returns empty - a daydream is opportunistic, never a
/// guaranteed-to-succeed duty the way the self-status check is.
async fn generate_daydream(
    graph: &Graph,
    tier1_pool: &DivergentPool,
    config: &BoredomConfig,
    self_summary: &str,
    rng: &mut StdRng,
    temperature: f32,
) -> Option<BoredomStimulus> {
    let sources = select_replay_sources(graph, config, rng);
    if sources.len() < config.daydream_min_sources {
        return None;
    }
    let texts: Vec<&str> = sources.iter().map(|(_, text)| text.as_str()).collect();
    let prompt = daydream_prompt(self_summary, &texts);
    let req = GenerateRequest { prompt, temperature };
    let samples = tier1_pool.sample(req, 1).await.ok()?;
    let best = samples
        .into_iter()
        .filter(|sample| !sample.raw_text.trim().is_empty())
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))?;
    Some(BoredomStimulus {
        text: best.raw_text.trim().to_string(),
        requested_tool: None,
        source_ids: sources.into_iter().map(|(id, _)| id).collect(),
    })
}

/// Picks up to `daydream_sample_size` dormant memories to recombine, biased
/// toward the more-activated end of long-term memory rather than sampled
/// uniformly across everything ever stored - Foster & Wilson's finding that
/// hippocampal replay preferentially reactivates recently-encoded/
/// reward-salient experience, not an unweighted draw over the whole store.
/// Reuses `activation.total` (already computed every tick for every graph
/// object - no new signal invented) as that salience proxy, then shuffles
/// only within the resulting top-`daydream_pool_multiplier`x pool so the
/// same one or two memories don't recombine identically on every idle tick.
///
/// Draws only from `MemoryRole::Episodic`/`MemoryRole::Semantic` objects -
/// `SelfMemory` is deliberately excluded, since identity content is meant to
/// be reinforced, not treated as raw material for free recombination.
/// Callers only ever reach this while Working Memory is empty (see
/// `loop_actor`'s `boredom_eligible` gate), so every candidate here is
/// already genuinely dormant - no separate "exclude what's currently active"
/// filter is needed.
fn select_replay_sources(graph: &Graph, config: &BoredomConfig, rng: &mut StdRng) -> Vec<(MentalObjectId, String)> {
    let mut candidates: Vec<(MentalObjectId, String, f32)> = graph
        .iter()
        .filter(|object| !object.text.trim().is_empty() && object.memory_roles.iter().any(|role| matches!(role, MemoryRole::Episodic | MemoryRole::Semantic)))
        .map(|object| (object.id, object.text.clone(), object.activation.total))
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let pool_size = (config.daydream_sample_size * config.daydream_pool_multiplier).min(candidates.len());
    let mut pool: Vec<(MentalObjectId, String)> = candidates.into_iter().take(pool_size).map(|(id, text, _)| (id, text)).collect();
    pool.shuffle(rng);
    pool.truncate(config.daydream_sample_size);
    pool
}

/// Tolerant matching against the live tool list, same "forgiving of small-
/// model formatting noise" spirit as `executive::parse_operator_keyword` -
/// an exact (case-insensitive, quote/whitespace-trimmed) match against a
/// currently-available tool name is treated as "run this tool"; anything
/// else is a freeform thought, verbatim.
fn parse_boredom_response(raw_text: &str, available_tools: &[(&'static str, &'static str)]) -> BoredomStimulus {
    let normalized = raw_text.trim().trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace());
    if let Some((name, _)) = available_tools.iter().find(|(name, _)| name.eq_ignore_ascii_case(normalized)) {
        return BoredomStimulus {
            text: format!("Nothing pending - I'll run {name} on my own initiative."),
            requested_tool: Some(name),
            source_ids: Vec::new(),
        };
    }
    BoredomStimulus { text: raw_text.trim().to_string(), requested_tool: None, source_ids: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_tiers::{ChatClient, TierError, TierResponse};
    use aca_types::{MentalObject, Tier};
    use async_trait::async_trait;
    use rand::SeedableRng;
    use std::time::Duration;

    struct FixedChatClient {
        raw_text: &'static str,
        confidence: f32,
    }

    #[async_trait]
    impl ChatClient for FixedChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Ok(TierResponse { raw_text: self.raw_text.to_string(), confidence: self.confidence, tier: Tier::T1 })
        }
    }

    fn empty_tier1() -> DivergentPool {
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![])
    }

    fn configured_tier1(raw_text: &'static str) -> DivergentPool {
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![std::sync::Arc::new(FixedChatClient { raw_text, confidence: 0.8 })])
    }

    const TOOLS: &[(&str, &str)] = &[("current_time", "Reports the current time."), ("self_status", "Reports Omega's own state.")];

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn neutral_drives() -> DriveState {
        DriveState::default()
    }

    fn episodic(text: &str) -> MentalObject {
        let mut object = MentalObject::new_observation(text, EpochMillis(0), 0.5);
        object.memory_roles.push(MemoryRole::Episodic);
        object
    }

    #[tokio::test]
    async fn self_status_duty_fires_when_never_run() {
        let graph = Graph::new();
        let stimulus = generate(&graph, &empty_tier1(), TOOLS, None, &neutral_drives(), &BoredomConfig::default(), "", EpochMillis(1_000_000), &mut test_rng(), 0.6).await;
        assert_eq!(stimulus.unwrap().requested_tool, Some(SelfStatusTool::NAME));
    }

    #[tokio::test]
    async fn self_status_duty_is_skipped_when_recently_run_and_drives_are_neutral() {
        let graph = Graph::new();
        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        let last_self_status_at = Some(EpochMillis(now.0 - config.self_status_interval_ms / 2));
        // Tier 1 unconfigured too, so with the duty not due there is nothing
        // left to invent - `generate` should return `None`, not fall back to
        // proposing the duty anyway.
        let stimulus = generate(&graph, &empty_tier1(), TOOLS, last_self_status_at, &neutral_drives(), &config, "", now, &mut test_rng(), 0.6).await;
        assert!(stimulus.is_none());
    }

    #[tokio::test]
    async fn self_status_duty_is_pulled_forward_by_high_self_monitoring_pressure() {
        let graph = Graph::new();
        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        // Recently run by the clock - would ordinarily still be well within
        // its interval - but a sustained run of high uncertainty readings
        // should still pull it forward, same as real interoceptive
        // disruption bypassing a routine checkup schedule.
        let last_self_status_at = Some(EpochMillis(now.0 - config.self_status_interval_ms / 2));
        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(Some(1.0), 0.0, None, None, 0.0, 0.0);
        }
        assert!(drives.self_monitoring_pressure() >= config.self_status_drive_threshold, "test setup must actually cross the threshold");

        let stimulus = generate(&graph, &empty_tier1(), TOOLS, last_self_status_at, &drives, &config, "", now, &mut test_rng(), 0.6).await;
        assert_eq!(stimulus.unwrap().requested_tool, Some(SelfStatusTool::NAME));
    }

    #[tokio::test]
    async fn invents_a_thought_when_no_duty_is_due_and_tier1_is_configured_and_no_daydream_material_exists() {
        let graph = Graph::new();
        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        let last_self_status_at = Some(now);
        let stimulus = generate(&graph, &configured_tier1("a quiet thought about the day"), TOOLS, last_self_status_at, &neutral_drives(), &config, "", now, &mut test_rng(), 0.6).await.unwrap();
        assert_eq!(stimulus.text, "a quiet thought about the day");
        assert_eq!(stimulus.requested_tool, None);
        assert!(stimulus.source_ids.is_empty());
    }

    #[tokio::test]
    async fn invents_a_tool_call_when_tier1_names_an_available_tool() {
        let graph = Graph::new();
        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        let last_self_status_at = Some(now);
        let stimulus = generate(&graph, &configured_tier1("current_time"), TOOLS, last_self_status_at, &neutral_drives(), &config, "", now, &mut test_rng(), 0.6).await.unwrap();
        assert_eq!(stimulus.requested_tool, Some("current_time"));
    }

    #[tokio::test]
    async fn returns_none_when_no_duty_is_due_and_tier1_is_unconfigured() {
        let graph = Graph::new();
        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        let stimulus = generate(&graph, &empty_tier1(), TOOLS, Some(now), &neutral_drives(), &config, "", now, &mut test_rng(), 0.6).await;
        assert!(stimulus.is_none());
    }

    #[tokio::test]
    async fn daydreams_from_replay_sources_when_drives_favor_it_and_enough_material_exists() {
        let mut graph = Graph::new();
        graph.insert(episodic("went for a run in the rain"));
        graph.insert(episodic("skipped a run because of rain"));
        graph.insert(episodic("rescheduled a run around rain"));

        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        // Duty just ran, so it's not due; a sustained high-curiosity
        // reading (EMA needs several updates to actually saturate) clears
        // `daydream_min_drive`, and resource pressure stays untouched (and
        // therefore under the ceiling).
        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(None, 1.0, None, None, 0.0, 0.0);
        }

        let stimulus = generate(&graph, &configured_tier1("a pattern about rain and running"), TOOLS, Some(now), &drives, &config, "", now, &mut test_rng(), 0.6).await.unwrap();
        assert_eq!(stimulus.text, "a pattern about rain and running");
        assert_eq!(stimulus.requested_tool, None);
        assert_eq!(stimulus.source_ids.len(), config.daydream_sample_size);
    }

    #[tokio::test]
    async fn falls_back_to_plain_invention_when_too_few_replay_sources_exist() {
        let mut graph = Graph::new();
        graph.insert(episodic("the only dormant episode"));

        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        let mut drives = DriveState::default();
        drives.update(None, 1.0, None, None, 0.0, 0.0);

        let stimulus = generate(&graph, &configured_tier1("invented from nothing"), TOOLS, Some(now), &drives, &config, "", now, &mut test_rng(), 0.6).await.unwrap();
        assert_eq!(stimulus.text, "invented from nothing");
        assert!(stimulus.source_ids.is_empty(), "one source is below daydream_min_sources, so this must fall back to plain invention");
    }

    #[tokio::test]
    async fn falls_back_to_plain_invention_when_resource_pressure_is_too_high() {
        let mut graph = Graph::new();
        graph.insert(episodic("went for a run in the rain"));
        graph.insert(episodic("skipped a run because of rain"));
        graph.insert(episodic("rescheduled a run around rain"));

        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        // 12 updates lands resource_pressure (and curiosity, moved by the
        // same reading) at ~0.72 via the EMA - above
        // `daydream_resource_pressure_ceiling` (0.7) but still below
        // `self_status_drive_threshold` (0.75), so this isolates "daydream
        // suppressed by resource scarcity" from "self-status pulled forward
        // by it" (the latter is exercised separately, and would otherwise
        // pre-empt this path - correctly, since a resource-saturated state
        // is itself a self-monitoring disruption).
        let mut drives = DriveState::default();
        for _ in 0..12 {
            drives.update(None, 1.0, None, None, 0.0, 1.0);
        }
        assert!(drives.resource_pressure >= config.daydream_resource_pressure_ceiling, "test setup must actually clear the daydream ceiling, got {}", drives.resource_pressure);
        assert!(drives.resource_pressure < config.self_status_drive_threshold, "test setup must stay below the self-status threshold, got {}", drives.resource_pressure);

        let stimulus = generate(&graph, &configured_tier1("invented from nothing"), TOOLS, Some(now), &drives, &config, "", now, &mut test_rng(), 0.6).await.unwrap();
        assert_eq!(stimulus.text, "invented from nothing");
        assert!(stimulus.source_ids.is_empty());
    }

    #[tokio::test]
    async fn no_daydream_is_attempted_below_the_minimum_drive_threshold() {
        let mut graph = Graph::new();
        graph.insert(episodic("went for a run in the rain"));
        graph.insert(episodic("skipped a run because of rain"));
        graph.insert(episodic("rescheduled a run around rain"));

        let config = BoredomConfig::default();
        let now = EpochMillis(1_000_000);
        // Neutral drives: curiosity and uncertainty both sit at 0.0, below
        // `daydream_min_drive` - plenty of replay material exists, but
        // nothing currently motivates reaching for it.
        let stimulus = generate(&graph, &configured_tier1("invented from nothing"), TOOLS, Some(now), &neutral_drives(), &config, "", now, &mut test_rng(), 0.6).await.unwrap();
        assert_eq!(stimulus.text, "invented from nothing");
        assert!(stimulus.source_ids.is_empty());
    }

    #[test]
    fn self_status_stimulus_wording_differs_by_interrupt_but_shape_does_not() {
        let idle = self_status_stimulus(false);
        let interrupt = self_status_stimulus(true);
        assert_ne!(idle.text, interrupt.text, "an interrupt-driven check-in should read differently from a routine one");
        assert_eq!(idle.requested_tool, Some(SelfStatusTool::NAME));
        assert_eq!(interrupt.requested_tool, Some(SelfStatusTool::NAME));
        assert!(idle.source_ids.is_empty());
        assert!(interrupt.source_ids.is_empty());
    }

    #[test]
    fn select_replay_sources_excludes_self_memory_and_empty_text() {
        let mut graph = Graph::new();
        let mut identity = MentalObject::new_observation("I am Omega", EpochMillis(0), 0.5);
        identity.memory_roles = vec![MemoryRole::SelfMemory];
        graph.insert(identity);
        graph.insert(episodic("a real dormant memory"));
        let mut blank = MentalObject::new_observation("   ", EpochMillis(0), 0.5);
        blank.memory_roles.push(MemoryRole::Episodic);
        graph.insert(blank);

        let config = BoredomConfig::default();
        let sources = select_replay_sources(&graph, &config, &mut test_rng());
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].1, "a real dormant memory");
    }

    #[test]
    fn select_replay_sources_prefers_the_higher_activation_pool_over_low_activation_stragglers() {
        let mut graph = Graph::new();
        let mut low = episodic("barely activated straggler");
        low.activation.total = -10.0;
        graph.insert(low);
        for i in 0..6 {
            let mut object = episodic(&format!("salient memory {i}"));
            object.activation.total = 5.0;
            graph.insert(object);
        }

        let config = BoredomConfig { daydream_sample_size: 3, daydream_pool_multiplier: 2, ..BoredomConfig::default() };
        // Run several seeds - the low-activation straggler sits outside the
        // top pool (6 candidates already fill pool_size=6) and must never be
        // selected regardless of which shuffle the rng happens to produce.
        for seed in 0..10 {
            let sources = select_replay_sources(&graph, &config, &mut StdRng::seed_from_u64(seed));
            assert_eq!(sources.len(), 3);
            assert!(sources.iter().all(|(_, text)| text != "barely activated straggler"), "seed {seed} selected the low-activation straggler");
        }
    }

    #[test]
    fn select_replay_sources_returns_empty_with_no_eligible_candidates() {
        let graph = Graph::new();
        let config = BoredomConfig::default();
        assert!(select_replay_sources(&graph, &config, &mut test_rng()).is_empty());
    }

    #[test]
    fn parse_boredom_response_matches_tool_names_case_insensitively() {
        let stimulus = parse_boredom_response("  \"Current_Time\"  ", TOOLS);
        assert_eq!(stimulus.requested_tool, Some("current_time"));
    }

    #[test]
    fn parse_boredom_response_falls_back_to_freeform_thought() {
        let stimulus = parse_boredom_response("wondering what the weather is like", TOOLS);
        assert_eq!(stimulus.requested_tool, None);
        assert_eq!(stimulus.text, "wondering what the weather is like");
        assert!(stimulus.source_ids.is_empty());
    }
}
