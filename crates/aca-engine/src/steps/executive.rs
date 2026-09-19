use aca_tiers::{ChatClient, DivergentPool, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::MentalObjectId;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::tools::{self, ToolRegistry};
use crate::prompt_templates::{communicative_intent_prompt, tool_intent_prompt};

/// A candidate mental action the Executive can select — specs.md's
/// Communication/Executive Function sections' "possible outcomes,"
/// formalized as SOAR operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Operator {
    Speak,
    Remember,
    ContinueReflecting,
    Ignore,
    ConsultKnowledgeLibrary,
    Plan,
    Ask,
    Act,
}

/// The keyword an `operator_proposal`-style prompt asks a model to answer
/// with, and that `parse_operator_keyword` parses back - a small, explicit
/// match rather than round-tripping through `Operator`'s own kebab-case
/// serde derive, so parsing can be forgiving of the formatting variance
/// small models actually produce (underscores, stray quotes, mixed case) in
/// a way a strict `Deserialize` call wouldn't be.
pub(crate) fn operator_keyword(op: Operator) -> &'static str {
    match op {
        Operator::Speak => "speak",
        Operator::Remember => "remember",
        Operator::ContinueReflecting => "continue-reflecting",
        Operator::Ignore => "ignore",
        Operator::ConsultKnowledgeLibrary => "consult-knowledge-library",
        Operator::Plan => "plan",
        Operator::Ask => "ask",
        Operator::Act => "act",
    }
}

/// Parses a model's raw text answer into an `Operator`, tolerant of the
/// formatting noise small models routinely produce - see
/// `operator_keyword`'s doc comment. `None` on anything unrecognized; never
/// a panic or a default guess, since a wrong silent guess here would be
/// worse than falling back to whatever the caller does when nothing parses.
pub(crate) fn parse_operator_keyword(text: &str) -> Option<Operator> {
    let normalized = text.trim().trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace()).to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "speak" => Some(Operator::Speak),
        "remember" => Some(Operator::Remember),
        "continue-reflecting" => Some(Operator::ContinueReflecting),
        "ignore" => Some(Operator::Ignore),
        "consult-knowledge-library" => Some(Operator::ConsultKnowledgeLibrary),
        "plan" => Some(Operator::Plan),
        "ask" => Some(Operator::Ask),
        "act" => Some(Operator::Act),
        _ => None,
    }
}

/// One proposed action on a specific Working Memory object. `preference` is
/// the SOAR-style ordering score used to select among proposals;
/// `confidence` is how sure the proposer is that this is the right call at
/// all — a low-confidence top proposal is what triggers the "confidence"
/// impasse bucket even without a tie.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorProposal {
    pub operator: Operator,
    pub target_id: MentalObjectId,
    pub preference: f32,
    pub confidence: f32,
}

/// The two-bucket impasse classification from the plan's named
/// simplifications: `MissingInformation` (nothing to act on at all) spawns
/// a subgoal Mental Object that re-enters Coalition next cycle;
/// `Confidence` (options exist, but the ordering among them isn't decisive
/// or trustworthy) escalates the same question to a more capable model
/// tier instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpasseKind {
    MissingInformation,
    Confidence,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutiveDecision {
    Selected(OperatorProposal),
    Impasse {
        kind: ImpasseKind,
        candidates: Vec<OperatorProposal>,
        reason: String,
    },
}

/// Tuning knobs for operator selection — thresholds, not theoretical
/// commitments.
#[derive(Debug, Clone, Copy)]
pub struct ExecutiveConfig {
    /// Below this, even a lone top proposal isn't trusted enough to act on
    /// without escalating for a second opinion.
    pub confidence_threshold: f32,
    /// How close two top preference scores must be to count as a genuine
    /// tie rather than a real (if narrow) winner.
    pub preference_tie_epsilon: f32,
    /// Below this, a Tier 3 confidence-impasse resolution isn't trusted
    /// enough to stop at either - escalate once more to Tier 4 (specs.md's
    /// "reached only when a Tier 3 impasse fails to resolve with adequate
    /// confidence"). Stricter than `confidence_threshold`: reaching for the
    /// largest model should be reserved for genuinely low-confidence Tier 3
    /// answers, not merely borderline ones.
    pub tier4_escalation_threshold: f32,
    /// Below this, a Cognitive Core Reflection isn't trusted enough to
    /// speak/ask directly - see `propose_operators`' handling of it. Kept
    /// separate from `confidence_threshold` (a different question: "is this
    /// proposal's own confidence good enough to select" vs. "is this
    /// Reflection's content well-supported enough to act on at all").
    /// `reflect`'s Tier 1/2 agreement wins report real cross-validated
    /// agreement here (see `cognitive_core::try_tier_via_agreement`), always
    /// comfortably above this bar; Tier 3's own uncorroborated self-report
    /// is what this threshold is actually meant to catch.
    pub low_confidence_reflection_threshold: f32,
}

impl Default for ExecutiveConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.4,
            preference_tie_epsilon: 1e-3,
            tier4_escalation_threshold: 0.5,
            low_confidence_reflection_threshold: 0.5,
        }
    }
}

/// Step 7 - Execute: evaluates already-proposed operators by preference and
/// either selects a winner or classifies why it can't. Proposal generation
/// itself (deciding *which* operators are even worth proposing for a given
/// broadcast winner) is a separate, often semantically-loaded concern — see
/// "Where the Language Model Plugs In" in specs.md — this function only
/// does the SOAR selection mechanics on whatever proposals it's given.
pub fn select_operator(proposals: &[OperatorProposal], config: &ExecutiveConfig) -> ExecutiveDecision {
    if proposals.is_empty() {
        return ExecutiveDecision::Impasse {
            kind: ImpasseKind::MissingInformation,
            candidates: Vec::new(),
            reason: "no operator was proposed for this cycle's broadcast content".to_string(),
        };
    }

    let mut ranked = proposals.to_vec();
    ranked.sort_by(|a, b| b.preference.partial_cmp(&a.preference).unwrap_or(std::cmp::Ordering::Equal));

    let top_preference = ranked[0].preference;
    let tied: Vec<OperatorProposal> = ranked
        .iter()
        .filter(|p| (p.preference - top_preference).abs() <= config.preference_tie_epsilon)
        .cloned()
        .collect();

    if tied.len() > 1 {
        return ExecutiveDecision::Impasse {
            kind: ImpasseKind::Confidence,
            candidates: tied,
            reason: "top-preference operators are tied - no decisive winner".to_string(),
        };
    }

    let winner = ranked.into_iter().next().expect("non-empty proposals checked above");
    if winner.confidence < config.confidence_threshold {
        return ExecutiveDecision::Impasse {
            kind: ImpasseKind::Confidence,
            candidates: vec![winner],
            reason: "sole top proposal's confidence is below threshold".to_string(),
        };
    }

    ExecutiveDecision::Selected(winner)
}

/// Tuning knobs for `select_operator_sampled`'s opt-in softmax draw among
/// near-tied top proposals. `Default` disables it - `select_operator_sampled`
/// is then byte-identical to `select_operator`, so adopting this struct
/// changes no behavior on its own.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub enabled: bool,
    /// Softmax temperature over `preference` among the sampling pool -
    /// higher flattens the distribution toward uniform, lower sharpens it
    /// back toward always picking the top proposal.
    pub temperature: f32,
    /// How close a proposal's `preference` must be to the top one to enter
    /// the sampling pool at all - deliberately wider than
    /// `ExecutiveConfig::preference_tie_epsilon` (a real, if narrow, winner
    /// per that threshold can still have close-enough runners-up worth
    /// varying among). Proposals outside this window are never candidates
    /// for the draw, regardless of temperature.
    pub window: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { enabled: false, temperature: 0.2, window: 0.05 }
    }
}

/// Opt-in variant of `select_operator`: identical behavior when
/// `sampling.enabled` is `false` (the default) - the plain baseline
/// decision, untouched, including every existing tie/low-confidence
/// impasse path. When enabled, and only on the plain-winner path (a tie or
/// low-confidence impasse already means "no decisive winner," which
/// sampling has no business overriding), draws a softmax-weighted winner
/// from the proposals within `sampling.window` of the top `preference`
/// instead of always picking the top one outright.
///
/// This is deliberately the entire mechanism behind "less deterministic
/// behavior without losing coherence": it only ever reorders which
/// already-computed-this-tick proposal gets applied, inside the single-
/// owner tick that already holds every one of these proposals in hand - it
/// never touches `Graph`, Working Memory, or anything `coherence.rs`
/// governs, so there is exactly one mutator regardless of whether this is
/// enabled. The sampled winner must still clear `config.confidence_threshold`
/// - a draw that fails that bar falls back to `select_operator`'s own
/// baseline decision rather than ever selecting something less trustworthy
/// than that path already gates.
pub fn select_operator_sampled(proposals: &[OperatorProposal], config: &ExecutiveConfig, sampling: &SamplingConfig, rng: &mut impl Rng) -> ExecutiveDecision {
    let baseline = select_operator(proposals, config);
    if !sampling.enabled {
        return baseline;
    }
    let ExecutiveDecision::Selected(_) = &baseline else {
        return baseline;
    };

    let top_preference = proposals.iter().map(|p| p.preference).fold(f32::NEG_INFINITY, f32::max);
    let pool: Vec<&OperatorProposal> = proposals.iter().filter(|p| top_preference - p.preference <= sampling.window).collect();
    if pool.len() <= 1 {
        return baseline;
    }

    let weights: Vec<f32> = pool.iter().map(|p| (p.preference / sampling.temperature).exp()).collect();
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return baseline;
    }
    let mut draw = rng.r#gen::<f32>() * total;
    let mut chosen = pool[pool.len() - 1];
    for (candidate, weight) in pool.iter().zip(weights.iter()) {
        if draw < *weight {
            chosen = candidate;
            break;
        }
        draw -= weight;
    }

    if chosen.confidence < config.confidence_threshold {
        return baseline;
    }
    ExecutiveDecision::Selected(chosen.clone())
}

/// What to do about an impasse — the direct, near-trivial mapping from
/// `ImpasseKind` to action, kept as its own named function because *this*
/// mapping is the semantically important part of the design: missing
/// information means think more (a subgoal); low confidence among known
/// options means get a better-informed opinion (escalate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpasseResponse {
    SpawnSubgoal,
    EscalateTier,
}

pub fn impasse_response(kind: ImpasseKind) -> ImpasseResponse {
    match kind {
        ImpasseKind::MissingInformation => ImpasseResponse::SpawnSubgoal,
        ImpasseKind::Confidence => ImpasseResponse::EscalateTier,
    }
}

/// Spawns a subgoal Mental Object (a `Question`) to resolve a
/// `MissingInformation` impasse — the concrete mechanism behind "thought
/// that doesn't come from outside" (specs.md): the subgoal re-enters
/// Coalition next cycle like any other candidate, rather than the cycle
/// stalling because nothing was proposed. Returns the new subgoal's id so
/// the caller can log/track it.
pub fn spawn_subgoal(
    graph: &mut aca_graph::Graph,
    reason: &str,
    now: aca_util::EpochMillis,
    decay_d: f32,
) -> MentalObjectId {
    let mut subgoal = aca_types::MentalObject::new_observation(format!("subgoal: {reason}"), now, decay_d);
    subgoal.kind = aca_types::MentalObjectKind::Question;
    let id = subgoal.id;
    graph.insert(subgoal);
    id
}

/// Whether a Cognitive Core Reflection already exists for `target_id` — the
/// "has this already been thought about" check that keeps `ContinueReflecting`
/// from being proposed every tick once a real reflection has already been
/// produced. A linear scan is fine at this scale (the build plan's own
/// scale assumption: tens-low-hundreds of nodes, not a hot path).
pub(crate) fn has_reflection_for(graph: &aca_graph::Graph, target_id: MentalObjectId) -> bool {
    graph
        .iter()
        .any(|object| object.kind == aca_types::MentalObjectKind::Reflection && object.source_object_ids.contains(&target_id))
}

/// How many independent Tier 1 samples to collect before treating the
/// result as real corroboration or real disagreement - mirrors
/// `cognitive_core`'s own `MIN_SAMPLES`; a lone sample proves nothing either
/// way, and `DivergentPool::sample` makes up the shortfall from a
/// solo-configured tier by repeat-sampling.
/// Active inference's operator-selection currency, made literal (specs.md's
/// Model Tiering section already frames tier escalation as confidence-
/// driven; this is the same idea applied to *which* operator the Executive
/// should prefer among several proposed, not just which model tier answers
/// one). Expected free energy = pragmatic value (does this operator move
/// toward a preferred/goal state) + epistemic value (does it reduce
/// uncertainty) - Friston/Clark/Hohwy's active inference, the same theory
/// specs.md's Attention section already draws `epistemic_value` from
/// (`steps::compare::ComparisonResult::epistemic_value`). Kept as an
/// explicit two-argument sum, not folded silently into call sites, so every
/// operator that adopts it documents its own pragmatic/epistemic split
/// rather than producing one undifferentiated preference number.
fn expected_free_energy(pragmatic_value: f32, epistemic_value: f32) -> f32 {
    pragmatic_value + epistemic_value
}

/// `expected_free_energy`'s pragmatic-value term for `ConsultKnowledgeLibrary`
/// (see that operator's proposal site): resolving an ambiguous subgoal has
/// some inherent value even independent of how much new information it
/// turns out to yield.
const CONSULT_PRAGMATIC_VALUE: f32 = 0.35;

const COMMUNICATIVE_INTENT_SAMPLES: usize = 3;

/// specs.md's Model Tiering point 4, made literal: asks Tier 1 (many cheap,
/// concurrent models) whether an already-formed Reflection is worth
/// speaking, worth asking about, or better left unsaid - replacing
/// `propose_operators`' previous fixed by-punctuation rule for this one
/// genuinely ambiguous call. Each sample is one vote; the resulting
/// proposals are built from vote *share*, never any sample's own
/// self-reported confidence - same reasoning as
/// `cognitive_core::try_tier_via_agreement`: at this model size, self-report
/// carries no reliable signal on its own, but independent samples actually
/// landing on the same answer is real evidence.
///
/// This is what finally makes `select_operator`'s tie/low-confidence impasse
/// paths reachable through genuine ambiguity instead of numeric coincidence:
/// unanimous samples yield one proposal with vote-share confidence (clears
/// the default threshold cleanly); a majority split yields a winner that
/// still clears it; an even split yields multiple proposals at equal
/// preference - a real tie, which `select_operator` correctly refuses to
/// resolve on its own.
///
/// Returns an empty `Vec` when Tier 1 isn't configured, the pool call fails
/// outright, or not one sample parsed into `allowed` - callers fall back to
/// a deterministic rule in that case, the same "a cheap tier didn't earn an
/// answer, that's not a hard failure" spirit as `cognitive_core::reflect`.
pub(crate) async fn propose_communicative_intent(
    tier1_pool: &DivergentPool,
    target_id: MentalObjectId,
    reflection_text: &str,
    working_memory_texts: &[&str],
    self_summary: &str,
    allowed: &[Operator],
    temperature: f32,
) -> Vec<OperatorProposal> {
    if tier1_pool.is_empty() {
        return Vec::new();
    }
    let prompt = communicative_intent_prompt(self_summary, reflection_text, working_memory_texts);
    let req = GenerateRequest { prompt, temperature };
    let Ok(samples) = tier1_pool.sample(req, COMMUNICATIVE_INTENT_SAMPLES).await else {
        return Vec::new();
    };

    let votes: Vec<Operator> = samples
        .iter()
        .filter_map(|sample| parse_operator_keyword(&sample.raw_text))
        .filter(|op| allowed.contains(op))
        .collect();
    if votes.is_empty() {
        return Vec::new();
    }

    let total = votes.len() as f32;
    let mut counts: Vec<(Operator, usize)> = Vec::new();
    for op in votes {
        match counts.iter_mut().find(|(candidate, _)| *candidate == op) {
            Some((_, count)) => *count += 1,
            None => counts.push((op, 1)),
        }
    }

    counts
        .into_iter()
        .map(|(operator, count)| {
            let share = count as f32 / total;
            OperatorProposal { operator, target_id, preference: share, confidence: share }
        })
        .collect()
}

/// How many concurrent Tier 1 votes decide whether a piece of input is
/// requesting a tool - same sample count as `propose_communicative_intent`,
/// same reasoning: one sample's self-report proves nothing, independent
/// agreement is the real signal.
const TOOL_INTENT_SAMPLES: usize = 3;

/// Parses a Tier 1 sample's raw answer into one of `available_tools`' own
/// names (case-insensitive, tolerant of quoting/whitespace noise the same
/// way `parse_operator_keyword` is) - `None` for `"none"` or anything that
/// doesn't match an actually-available tool, so a hallucinated tool name
/// can't accidentally trigger a real one.
fn parse_tool_intent_keyword(text: &str, available_tools: &[(&'static str, &'static str)]) -> Option<&'static str> {
    let normalized = text.trim().trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace()).to_ascii_lowercase();
    available_tools.iter().find(|(name, _)| name.eq_ignore_ascii_case(&normalized)).map(|(name, _)| *name)
}

/// specs.md's Model Tiering point 4, made literal for tool selection - the
/// direct counterpart to `propose_communicative_intent` above, same voting
/// shape, different question ("does this input want a tool" instead of
/// "should this be spoken"). Replaces what used to be a fixed cosine-
/// similarity threshold against each tool's description embedding: that
/// mechanism measured topical relatedness, not request intent, and the two
/// are genuinely inseparable by embedding distance alone in an architecture
/// whose own subject matter is constantly self-referential/cognitive (see
/// `propose_operators`' fourth trigger's own doc comment for the live
/// evidence). Requires unanimous agreement, stricter than
/// `propose_communicative_intent`'s simple majority - confirmed live, a
/// bare majority let a small Tier 1 model's own misjudgement through (a
/// long, entirely unrelated research question routed to `current_time` on
/// a 2-of-3 vote). The asymmetry that justifies unanimity: a false negative
/// here just means the input gets reflected on normally (safe), while a
/// false positive swallows real conversational content into a tool call
/// instead of ever engaging with it (confirmed live twice over, the actual
/// failure mode this whole mechanism exists to prevent) - so this only
/// acts on agreement strong enough that noise from any one small model is
/// very unlikely to have caused it.
///
/// Falls back to `tools::match_tool_intent`'s literal keyword match - same
/// "a cheap tier didn't earn an answer, that's not a hard failure" spirit as
/// `cognitive_core::reflect` - when Tier 1 isn't configured, the pool call
/// fails outright, or no sample parsed into an actually-available tool name.
pub(crate) async fn propose_tool_intent(
    tier1_pool: &DivergentPool,
    text: &str,
    self_summary: &str,
    available_tools: &[(&'static str, &'static str)],
    temperature: f32,
) -> Option<&'static str> {
    // Nothing to ask Tier 1 about (no tools registered) or no one to ask
    // (Tier 1 unconfigured) - either way, fall straight to the keyword
    // fallback rather than giving up outright. `match_tool_intent` doesn't
    // care whether a name it returns is actually registered here - callers
    // check reachability separately (see its own doc comment) - so an empty
    // `available_tools` must not short-circuit past it.
    if tier1_pool.is_empty() || available_tools.is_empty() {
        return tools::match_tool_intent(text);
    }
    let prompt = tool_intent_prompt(self_summary, text, available_tools);
    let req = GenerateRequest { prompt, temperature };
    let Ok(samples) = tier1_pool.sample(req, TOOL_INTENT_SAMPLES).await else {
        return tools::match_tool_intent(text);
    };

    let votes: Vec<Option<&'static str>> = samples.iter().map(|sample| parse_tool_intent_keyword(&sample.raw_text, available_tools)).collect();
    if votes.is_empty() {
        return tools::match_tool_intent(text);
    }
    let total = votes.len();
    let mut counts: Vec<(Option<&'static str>, usize)> = Vec::new();
    for vote in votes {
        match counts.iter_mut().find(|(candidate, _)| *candidate == vote) {
            Some((_, count)) => *count += 1,
            None => counts.push((vote, 1)),
        }
    }
    counts.into_iter().filter(|(tool, _)| tool.is_some()).max_by_key(|(_, count)| *count).filter(|(_, count)| *count == total).and_then(|(tool, _)| tool)
}

/// A deliberately simple, rule-based operator proposer for v1. Real
/// semantic judgement about which operators are even worth proposing is
/// explicitly called out in "Where the Language Model Plugs In" (specs.md,
/// point 4) as a place a language model may need to plug in — the Speak-vs-
/// Ask-vs-Ignore decision on an already-formed Reflection (`propose_
/// communicative_intent`) and whether input wants a tool invoked
/// (`propose_tool_intent`) both now use real semantic judgement, each with
/// its previous fixed rule kept only as the fallback when Tier 1 isn't
/// configured or didn't produce a usable answer. Every other branch here
/// remains deterministic Tier-0 logic on purpose - the dedup guards
/// scattered throughout this function are safety rails against runaway
/// repetition, not ambiguous calls, and don't need - or want - a model call.
///
/// Raw conversational input (`MentalObjectKind::Observation`) is never
/// echoed verbatim: it's routed through the Cognitive Core first
/// (`Operator::ContinueReflecting` → `cognitive_core::reflect`, a real
/// Tier-3 call), and it's the resulting Reflection object — not this one —
/// that ever gets a Speak/Ask proposal. This is "communication is always
/// derived from cognition, never the reverse" (specs.md) actually holding in
/// behavior, not just in the type system: previously Speak rendered the
/// user's own input text straight back at them, and no model call ever
/// produced what got said. Once a Reflection exists for an Observation, that
/// Observation itself has nothing communicative left to do (text ending in
/// "?" still favors `Ask` over `Speak` — but now decided on the Reflection's
/// own text) — how surprising the content was (its precision-weighted
/// surprise from Compare) still favors `Remember`, independent of any of
/// this.
///
/// Crucially, this only proposes actions not already taken on this exact
/// object: communicative intent is skipped once it's already been acted on
/// (Speak/Ask for anything else, or a Reflection existing for a raw
/// Observation), and `Remember` is skipped once it's already tagged
/// Episodic. Without this, an object that isn't decaying fast enough to
/// fall out of Working Memory within a few ticks gets the *same* operator
/// re-selected and re-executed every single tick indefinitely (observed
/// directly in a live run: the same reply spoken hundreds of times) —
/// coherent behavior requires the Executive to know what it already did.
#[allow(clippy::too_many_arguments)]
pub async fn propose_operators(
    graph: &aca_graph::Graph,
    target_id: MentalObjectId,
    precision_weighted_surprise: Option<f32>,
    config: &ExecutiveConfig,
    tier1_pool: &DivergentPool,
    working_memory_texts: &[&str],
    self_summary: &str,
    tool_registry: &ToolRegistry,
    now: aca_util::EpochMillis,
    replan_interval_ms: i64,
    operator_proposal_temperature: f32,
    tool_intent_temperature: f32,
) -> Vec<OperatorProposal> {
    let Some(object) = graph.get(&target_id) else {
        return Vec::new();
    };

    // `steps::agenda`'s two object shapes are handled entirely separately,
    // before any of the Observation/Reflection/Question logic below (which
    // assumes shapes neither one actually has) even runs.
    //
    // A plan-derived Reflection (tagged by `steps::act::act`'s
    // `Operator::Plan` success arm) is agenda bookkeeping, not real
    // communicative content or a genuine ambiguity - `steps::agenda::
    // revise_agenda` is what actually consumes it (folding its text back
    // into the parent Intention) later this same tick. Checked by
    // provenance, not confidence or text, same discipline as every other
    // "don't mistake my own output for new content" guard in this function
    // (see `is_a_tools_own_result` below) - without this, a plan Reflection
    // with low self-reported confidence would otherwise match
    // `is_unresolved_low_confidence_reflection` below and get routed to
    // Consult, and a confident one would get a real Speak/Ask vote.
    if super::agenda::is_plan_reflection(object) {
        return vec![OperatorProposal { operator: Operator::Ignore, target_id, preference: 0.5, confidence: 0.9 }];
    }
    // An Intention is never raw perceptual input and never itself
    // communicative content (any actual communication happens through its
    // plan-derived Reflection, gated separately above) - so it gets its own
    // self-contained branch rather than falling through into logic written
    // for Observation/Reflection/Question. Due to (re)plan: propose
    // `Operator::Plan`, the operator wired identically to
    // `ContinueReflecting` in `steps::act::act` but dormant until this
    // trigger. Not due: `Ignore`, never an empty `Vec` - `select_operator`
    // reads an empty proposal list as a `MissingInformation` impasse, which
    // would spawn a spurious subgoal every single tick this Intention is
    // broadcast instead of just quietly waiting its turn.
    if object.kind == aca_types::MentalObjectKind::Intention {
        if super::agenda::intention_due_to_replan(object, now, replan_interval_ms) {
            return vec![OperatorProposal { operator: Operator::Plan, target_id, preference: 0.8, confidence: 0.75 }];
        }
        return vec![OperatorProposal { operator: Operator::Ignore, target_id, preference: 0.3, confidence: 0.9 }];
    }

    let already_consulted = object.produced_by_operator.as_deref() == Some("ConsultKnowledgeLibrary");
    // A missing-information subgoal (a `Question` object, per `spawn_subgoal`)
    // gets exactly one shot at consulting the Knowledge Library before
    // anything else is proposed for it - same dedup discipline as
    // Speak/Ask/Remember below, applied one operator earlier.
    let is_unconsulted_question = object.kind == aca_types::MentalObjectKind::Question && !already_consulted;
    // A second, independent trigger for the same operator: a Cognitive Core
    // Reflection whose confidence didn't clear the bar (see
    // `ExecutiveConfig::low_confidence_reflection_threshold`'s doc comment)
    // represents exactly the ambiguity specs.md's Executive Function
    // section describes ("a memory-formation decision is ambiguous... that
    // ambiguity IS an impasse") - rather than speaking an uncorroborated
    // guess or silently ignoring it, this routes it through the same
    // real-information-seeking path an explicit Question already gets,
    // reusing the identical dedup guard (`produced_by_operator`) so it
    // fires exactly once per Reflection, not every tick it stays in
    // Working Memory.
    let is_unresolved_low_confidence_reflection =
        object.kind == aca_types::MentalObjectKind::Reflection && object.confidence < config.low_confidence_reflection_threshold && !already_consulted;
    if is_unconsulted_question || is_unresolved_low_confidence_reflection {
        // Active inference's expected free energy, made literal for the one
        // operator specs.md's "Where the Language Model Plugs In" names
        // explicitly as epistemic ("Consult-Knowledge-Library has high
        // epistemic value when uncertainty is high"): `1.0 - object.confidence`
        // stands in for how much querying external knowledge is expected to
        // reduce uncertainty about this specific target, so a barely-
        // ambiguous Reflection and a wildly uncertain one no longer earn the
        // identical fixed preference a plain constant gave every candidate
        // regardless of how uncertain it actually was. `CONSULT_PRAGMATIC_VALUE`
        // is deliberately chosen so a target at the architecture's own
        // default confidence (0.5 - `MentalObject::new_observation`'s
        // baseline, and also true of a freshly spawned Question, which
        // carries no confidence signal of its own yet) reproduces the exact
        // preference this used to be a fixed constant at, so this is a
        // strict refinement of the old behavior, not a behavior change at
        // the common case.
        let epistemic_value = (1.0 - object.confidence).clamp(0.0, 1.0);
        let preference = expected_free_energy(CONSULT_PRAGMATIC_VALUE, epistemic_value).clamp(0.0, 1.0);
        return vec![OperatorProposal { operator: Operator::ConsultKnowledgeLibrary, target_id, preference, confidence: 0.7 }];
    }

    // A third trigger, same early-return shape: `data.requested_tool` names
    // a tool Omega itself decided to run - `steps::boredom`'s self-initiated
    // path (a standing duty coming due, or Tier 1 inventing something to do
    // when idle), tagged at `loop_actor`'s boredom `Incoming` arm. Checked
    // ahead of the reactive keyword match below, and via provenance rather
    // than text, since a self-issued request has nothing to infer - it
    // already says exactly which tool. Actual reachability (risk tier,
    // still registered) is resolved later, same as the keyword-matched path
    // - see `act`'s `Operator::Act` handler.
    let already_acted = object.produced_by_operator.as_deref() == Some("Act");
    if !already_acted && object.data.get("requested_tool").is_some() {
        return vec![OperatorProposal { operator: Operator::Act, target_id, preference: 0.85, confidence: 0.7 }];
    }

    let is_raw_observation = object.kind == aca_types::MentalObjectKind::Observation;

    if is_raw_observation && object.data.get("compiled_ignore").and_then(|value| value.as_bool()) == Some(true) {
        return vec![OperatorProposal { operator: Operator::Ignore, target_id, preference: 1.0, confidence: 1.0 }];
    }

    if is_raw_observation && object.data.get("compiled_question").and_then(|value| value.as_str()).is_some() {
        return vec![OperatorProposal { operator: Operator::Ask, target_id, preference: 1.0, confidence: 1.0 }];
    }

    // An exact curated static answer or repeated safe stimulus-response
    // routine is a local speech policy. Repetition alone is not independent
    // success feedback; these paths bypass model deliberation only within
    // their restrictive admission gates.
    if is_raw_observation && (object.data.get("compiled_response").and_then(|value| value.as_str()).is_some()
        || object.data.get("innate_response").and_then(|value| value.as_str()).is_some()
        || object.data.get("curated_answer").and_then(|value| value.as_str()).is_some()) {
        return vec![OperatorProposal { operator: Operator::Speak, target_id, preference: 1.0, confidence: 1.0 }];
    }

    // A fourth trigger, same early-return shape: does this input actually
    // want one of the closed set of registered tools invoked right now?
    // specs.md's "Where the Language Model Plugs In," point 4: "candidate
    // operators [that] require semantic judgement... Default Tier 1" - this
    // used to be a fixed cosine-similarity threshold against each tool's
    // description embedding, framed as "ecological psychology's direct
    // perception." It wasn't: confirmed live against the real embedding
    // model, a genuine paraphrase of a status request ("what are you
    // thinking about", 0.53 similarity) and a long, entirely unrelated
    // research question that merely happened to be *about* cognition (0.53
    // similarity) scored within noise of each other against `self_status`'s
    // description - no threshold could have separated them, because
    // embedding distance from a blurb measures topical relatedness, not
    // whether a tool is actually being requested. That distinction *is*
    // semantic judgement, the exact case point 4 calls out - so
    // `propose_tool_intent` asks Tier 1 directly instead, the same
    // real-judgement-call pattern `propose_communicative_intent` already
    // uses for Speak/Ask/Ignore. `tools::match_tool_intent`'s literal
    // keyword list remains the fallback when Tier 1 isn't configured or
    // didn't produce a usable answer, same fallback shape as everywhere
    // else in this function that calls Tier 1.
    //
    // Gated on `is_raw_observation`: only fresh perceptual input should ever
    // be read as *requesting* a tool, not Omega's own already-formed
    // Reflections - without this, a Reflection produced *by* `self_status`
    // ("Working memory holds three items...") reliably paraphrases that same
    // tool's own description closely enough to be misread as requesting it
    // again - confirmed live: a self_status -> reflect -> self_status ->
    // reflect oscillation that never once proposed Speak, because every
    // Reflection *about* Omega's own state continued to resemble the tool
    // that produced it. A real reader is less likely to make this mistake
    // than embedding distance was, but there's no reason to even ask the
    // question of Omega's own already-processed thoughts - only fresh input
    // can genuinely be *requesting* something.
    //
    // A tool's own re-entered result (`data.source == "tool"`, tagged at
    // `loop_actor`'s Act re-entry send site) is never itself eligible to
    // trigger Act again, regardless of what it says - confirmed live that
    // this matters: `CurrentTimeTool`'s first phrasing ("The current time
    // is...") accidentally contained its own trigger phrase and re-invoked
    // itself forever. Careful tool phrasing isn't a sufficient defense on
    // its own (a future tool's output could just as easily collide with a
    // *different* tool's trigger), so this checks provenance instead of
    // trusting wording to never collide.
    let is_a_tools_own_result = object.data.get("source").and_then(|v| v.as_str()) == Some("tool");
    if is_raw_observation && !already_acted && !is_a_tools_own_result {
        let affords_a_tool = propose_tool_intent(tier1_pool, &object.text, self_summary, &tool_registry.available(), tool_intent_temperature).await;
        if affords_a_tool.is_some() {
            return vec![OperatorProposal { operator: Operator::Act, target_id, preference: 0.85, confidence: 0.7 }];
        }
    }

    let already_communicated = if is_raw_observation {
        // An Act-tagged raw Observation had its informational content
        // redirected into a tool call rather than Reflection - its answer
        // arrives as a *separate*, freshly re-entered Observation (see
        // `loop_actor`'s Act re-entry send site), which goes through this
        // same pipeline on its own. Without this check, the original query
        // object would *also* get proposed for ContinueReflecting here,
        // producing a redundant second reflection on a question that
        // already has a real answer in flight.
        has_reflection_for(graph, target_id) || already_acted
    } else {
        // "Ignore" included alongside Speak/Ask/Consult: a communicative-
        // intent vote (`propose_communicative_intent`) that resolved to
        // Ignore is just as much a made decision as one that resolved to
        // Speak or Ask - see `act`'s `Operator::Ignore` arm, which tags it
        // identically, specifically so this doesn't re-sample Tier 1 for
        // the same Reflection every subsequent tick.
        matches!(object.produced_by_operator.as_deref(), Some("Speak") | Some("Ask") | Some("ConsultKnowledgeLibrary") | Some("Ignore"))
    };
    // `MemoryFormationOutcome::{NewEpisodic, SemanticUpdate, BeliefRevision}`
    // tag their stored object with `Episodic`/`Semantic`/`SelfMemory`
    // respectively (see `steps::memory_formation::form_memory`) - any one
    // of the three means "Remember already ran for this object," not just
    // Episodic specifically, or a Semantic/SelfMemory-classified object
    // would never register as processed and Remember would be re-proposed
    // for it every tick indefinitely (the same runaway-repetition failure
    // mode `remember_is_hopeless` below already guards, one classifier
    // outcome wide).
    let already_remembered = object
        .memory_roles
        .iter()
        .any(|role| matches!(role, aca_types::MemoryRole::Episodic | aca_types::MemoryRole::Semantic | aca_types::MemoryRole::SelfMemory));
    // `form_memory` discards an embedding-less candidate unconditionally,
    // every single time (it has nothing to compare/store) - proposing
    // Remember anyway would mean proposing it again next tick, and the
    // tick after that, forever. A Cognitive Core Reflection normally gets a
    // real embedding at creation time (see `cognitive_core::reflect`), but
    // a failed embed call degrades to exactly this state - treating it as
    // permanently ineligible here is what actually closes that loop rather
    // than just making it rare. Observed live: this was the runaway
    // Remember/Discarded spam that made the event feed unreadable.
    let remember_is_hopeless = object.embedding.is_none();

    if already_communicated && (already_remembered || remember_is_hopeless) {
        return vec![OperatorProposal { operator: Operator::Ignore, target_id, preference: 0.5, confidence: 0.9 }];
    }

    let mut proposals = Vec::new();
    if !already_communicated {
        if is_raw_observation {
            proposals.push(OperatorProposal { operator: Operator::ContinueReflecting, target_id, preference: 0.9, confidence: 0.7 });
        } else {
            let semantic = propose_communicative_intent(
                tier1_pool,
                target_id,
                &object.text,
                working_memory_texts,
                self_summary,
                &[Operator::Speak, Operator::Ask, Operator::Ignore],
                operator_proposal_temperature,
            )
            .await;
            if semantic.is_empty() {
                // Tier 1 unconfigured or didn't produce a usable answer -
                // fall back to the original fixed by-punctuation rule
                // rather than proposing nothing.
                let looks_like_a_question = object.text.trim_end().ends_with('?');
                proposals.push(OperatorProposal {
                    operator: if looks_like_a_question { Operator::Ask } else { Operator::Speak },
                    target_id,
                    preference: if looks_like_a_question { 0.8 } else { 0.4 },
                    confidence: 0.75,
                });
            } else {
                proposals.extend(semantic);
            }
        }
    }
    if !already_remembered && !remember_is_hopeless {
        let surprise = precision_weighted_surprise.unwrap_or(0.0).clamp(0.0, 2.0);
        proposals.push(OperatorProposal {
            operator: Operator::Remember,
            target_id,
            preference: (surprise / 2.0).clamp(0.0, 1.0),
            confidence: 0.75,
        });
    }
    proposals
}

/// What actually happened when the Executive tried to resolve a confidence
/// impasse by escalating: either a live Tier-3 (or higher) call ran and
/// produced a response, the call ran but failed in transport, or the tier
/// was busy and this tick falls back to a cheap heuristic instead — the
/// literal single-flight behavior from the build plan's concurrency model.
#[derive(Debug)]
pub enum ImpasseResolution {
    Escalated(TierResponse),
    EscalationFailed(TierError),
    DeferredToHeuristic,
}

/// Attempts to resolve a `Confidence` impasse by escalating to the tier
/// behind `pool`. Never awaits a busy pool — `try_acquire` is non-blocking,
/// so a busy Tier 3 falls back to `DeferredToHeuristic` within the same
/// tick rather than stalling it.
pub async fn resolve_confidence_impasse(
    pool: &TierPool,
    client: &dyn ChatClient,
    req: GenerateRequest,
) -> ImpasseResolution {
    match pool.try_acquire() {
        Some(permit) => match pool.run_chat_with_permit(permit, client, req).await {
            Ok(response) => ImpasseResolution::Escalated(response),
            Err(err) => ImpasseResolution::EscalationFailed(err),
        },
        None => ImpasseResolution::DeferredToHeuristic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rand::SeedableRng;
    use std::time::Duration;

    fn proposal(op: Operator, preference: f32, confidence: f32) -> OperatorProposal {
        OperatorProposal {
            operator: op,
            target_id: MentalObjectId::new(),
            preference,
            confidence,
        }
    }

    #[test]
    fn sampling_disabled_matches_select_operator_exactly() {
        let proposals = vec![proposal(Operator::Speak, 0.9, 0.9), proposal(Operator::Ask, 0.85, 0.9)];
        let config = ExecutiveConfig::default();
        let sampling = SamplingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sampled = select_operator_sampled(&proposals, &config, &sampling, &mut rng);
        assert_eq!(sampled, select_operator(&proposals, &config));
    }

    #[test]
    fn sampling_ignores_proposals_outside_the_window() {
        let proposals = vec![proposal(Operator::Speak, 0.9, 0.9), proposal(Operator::Ask, 0.3, 0.9)];
        let config = ExecutiveConfig::default();
        let sampling = SamplingConfig { enabled: true, temperature: 0.2, window: 0.05 };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        match select_operator_sampled(&proposals, &config, &sampling, &mut rng) {
            ExecutiveDecision::Selected(p) => assert_eq!(p.operator, Operator::Speak),
            other => panic!("expected a decisive winner, got {other:?}"),
        }
    }

    #[test]
    fn sampling_can_pick_a_near_tied_runner_up() {
        let proposals = vec![proposal(Operator::Speak, 0.90, 0.9), proposal(Operator::Ask, 0.88, 0.9)];
        let config = ExecutiveConfig::default();
        let sampling = SamplingConfig { enabled: true, temperature: 1.0, window: 0.05 };
        let (mut saw_speak, mut saw_ask) = (false, false);
        for seed in 0..50 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            if let ExecutiveDecision::Selected(p) = select_operator_sampled(&proposals, &config, &sampling, &mut rng) {
                match p.operator {
                    Operator::Speak => saw_speak = true,
                    Operator::Ask => saw_ask = true,
                    _ => {}
                }
            }
        }
        assert!(saw_speak && saw_ask, "a near-tied pool sampled across many draws should produce both winners at least once");
    }

    #[test]
    fn sampling_never_selects_below_the_confidence_threshold() {
        let mut config = ExecutiveConfig::default();
        config.confidence_threshold = 0.5;
        let proposals = vec![proposal(Operator::Speak, 0.9, 0.9), proposal(Operator::Ask, 0.88, 0.1)];
        let sampling = SamplingConfig { enabled: true, temperature: 1.0, window: 0.05 };
        for seed in 0..50 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            if let ExecutiveDecision::Selected(p) = select_operator_sampled(&proposals, &config, &sampling, &mut rng) {
                assert!(p.confidence >= 0.5, "must never select a proposal below confidence_threshold, got {p:?}");
            }
        }
    }

    #[test]
    fn sampling_never_overrides_a_genuine_tie_impasse() {
        let proposals = vec![proposal(Operator::Speak, 0.9, 0.9), proposal(Operator::Ask, 0.9, 0.9)];
        let config = ExecutiveConfig::default();
        let sampling = SamplingConfig { enabled: true, temperature: 1.0, window: 0.05 };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let decision = select_operator_sampled(&proposals, &config, &sampling, &mut rng);
        assert!(matches!(decision, ExecutiveDecision::Impasse { .. }));
    }

    /// An unconfigured Tier 1 - `propose_communicative_intent` treats this
    /// exactly like "not configured," so `propose_operators` falls back to
    /// its original deterministic by-punctuation rule, which is what every
    /// pre-existing test below still asserts on.
    fn empty_tier1_pool() -> DivergentPool {
        DivergentPool::new(aca_types::Tier::T1, Duration::from_secs(5), vec![])
    }

    fn empty_tool_registry() -> tools::ToolRegistry {
        tools::ToolRegistry::empty()
    }

    #[test]
    fn empty_proposals_is_a_missing_information_impasse() {
        let decision = select_operator(&[], &ExecutiveConfig::default());
        assert!(matches!(
            decision,
            ExecutiveDecision::Impasse { kind: ImpasseKind::MissingInformation, .. }
        ));
    }

    #[test]
    fn a_clear_winner_is_selected() {
        let proposals = vec![
            proposal(Operator::Speak, 0.9, 0.8),
            proposal(Operator::Ignore, 0.2, 0.9),
        ];
        let decision = select_operator(&proposals, &ExecutiveConfig::default());
        match decision {
            ExecutiveDecision::Selected(winner) => assert_eq!(winner.operator, Operator::Speak),
            other => panic!("expected a clear Selected winner, got {other:?}"),
        }
    }

    #[test]
    fn tied_top_preference_is_a_confidence_impasse() {
        let proposals = vec![
            proposal(Operator::Speak, 0.7, 0.9),
            proposal(Operator::Ask, 0.7, 0.9),
        ];
        let decision = select_operator(&proposals, &ExecutiveConfig::default());
        match decision {
            ExecutiveDecision::Impasse { kind: ImpasseKind::Confidence, candidates, .. } => {
                assert_eq!(candidates.len(), 2);
            }
            other => panic!("expected a Confidence impasse on tie, got {other:?}"),
        }
    }

    #[test]
    fn low_confidence_sole_winner_is_a_confidence_impasse() {
        let proposals = vec![proposal(Operator::Speak, 0.9, 0.1)];
        let decision = select_operator(&proposals, &ExecutiveConfig::default());
        assert!(matches!(
            decision,
            ExecutiveDecision::Impasse { kind: ImpasseKind::Confidence, .. }
        ));
    }

    #[tokio::test]
    async fn propose_operators_routes_fresh_observations_through_reflection_not_echo() {
        let mut graph = aca_graph::Graph::new();
        let object = aca_types::MentalObject::new_observation("are you there?", aca_util::EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, Some(0.2), &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(
            proposals.iter().any(|p| p.operator == Operator::ContinueReflecting),
            "raw conversational input should be proposed for reflection, not echoed"
        );
        assert!(
            !proposals.iter().any(|p| matches!(p.operator, Operator::Speak | Operator::Ask)),
            "raw conversational input should never get a direct Speak/Ask proposal"
        );
    }

    #[tokio::test]
    async fn propose_operators_proposes_act_for_a_self_issued_requested_tool_tag() {
        let mut graph = aca_graph::Graph::new();
        // Text deliberately doesn't match `match_tool_intent`'s trigger
        // phrases, and isn't a raw Observation either (so it wouldn't
        // otherwise be eligible for anything but Speak/Ask/Ignore) - only
        // `data.requested_tool` should be able to route this to Act.
        let mut object = aca_types::MentalObject::new_observation("nothing has needed my attention for a while", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        object.data = serde_json::json!({"source": "boredom", "requested_tool": "current_time"});
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert_eq!(proposals, vec![OperatorProposal { operator: Operator::Act, target_id: id, preference: 0.85, confidence: 0.7 }]);
    }

    #[tokio::test]
    async fn propose_operators_stops_proposing_reflection_once_one_exists() {
        let mut graph = aca_graph::Graph::new();
        let object = aca_types::MentalObject::new_observation("are you there?", aca_util::EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let mut reflection = aca_types::MentalObject::new_observation("yes, I'm here", aca_util::EpochMillis(1), 0.5);
        reflection.kind = aca_types::MentalObjectKind::Reflection;
        reflection.source_object_ids = vec![id];
        graph.insert(reflection);

        let proposals = propose_operators(&graph, id, Some(0.2), &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(
            !proposals.iter().any(|p| p.operator == Operator::ContinueReflecting),
            "should not propose reflecting again once a Reflection already exists for this object"
        );
    }

    #[tokio::test]
    async fn propose_operators_favors_ask_for_question_text_on_a_reflection() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("are you there?", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, Some(0.2), &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        let ask = proposals.iter().find(|p| p.operator == Operator::Ask);
        assert!(ask.is_some(), "question text on a Reflection should propose Ask");
    }

    #[tokio::test]
    async fn propose_operators_favors_speak_for_statement_text_on_a_reflection() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("the sky is blue", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, Some(0.2), &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        let speak = proposals.iter().find(|p| p.operator == Operator::Speak);
        assert!(speak.is_some(), "statement text on a Reflection should propose Speak");
    }

    #[tokio::test]
    async fn propose_operators_returns_empty_for_a_missing_target() {
        let graph = aca_graph::Graph::new();
        let proposals = propose_operators(&graph, MentalObjectId::new(), None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(proposals.is_empty());
    }

    #[tokio::test]
    async fn propose_operators_scales_remember_preference_with_surprise() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("wow", aca_util::EpochMillis(0), 0.5);
        object.embedding = Some(vec![1.0, 0.0, 0.0]);
        let id = object.id;
        graph.insert(object);

        let low_surprise = propose_operators(&graph, id, Some(0.0), &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        let high_surprise = propose_operators(&graph, id, Some(2.0), &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        let low_remember = low_surprise.iter().find(|p| p.operator == Operator::Remember).unwrap();
        let high_remember = high_surprise.iter().find(|p| p.operator == Operator::Remember).unwrap();
        assert!(high_remember.preference > low_remember.preference);
    }

    #[tokio::test]
    async fn an_embedding_less_object_is_never_proposed_for_remember() {
        // Regression test for a real runaway loop found live: `form_memory`
        // discards an embedding-less candidate unconditionally, every time -
        // proposing Remember for one anyway means proposing it again next
        // tick, forever. Reproduced with a Reflection-kind object (a failed
        // embed call on a Cognitive Core reflection is exactly how this
        // happened), not just a bare Observation.
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("a reflection with no embedding", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, Some(2.0), &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(
            !proposals.iter().any(|p| p.operator == Operator::Remember),
            "an object with no embedding should never be proposed for Remember"
        );
    }

    #[tokio::test]
    async fn a_communicated_embedding_less_object_settles_to_ignore_not_an_empty_impasse() {
        // Without the `remember_is_hopeless` check folded into the "nothing
        // left to do" branch, this exact state (already spoken, no
        // embedding) would return an *empty* proposal list forever instead -
        // which `select_operator` treats as a MissingInformation impasse,
        // spawning a fresh subgoal every single tick indefinitely. Same
        // underlying bug class, different visible symptom.
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("already handled", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        object.produced_by_operator = Some("Speak".to_string());
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].operator, Operator::Ignore);
    }

    #[tokio::test]
    async fn a_low_confidence_reflection_proposes_consult_knowledge_library_instead_of_speaking() {
        // specs.md: "a memory-formation decision is ambiguous... that
        // ambiguity IS an impasse." An uncorroborated, low-confidence
        // Reflection (see cognitive_core::try_tier_via_agreement's doc
        // comment on where this confidence value actually comes from)
        // shouldn't be spoken as if it were solid, or silently dropped -
        // it should trigger a real attempt to find out more.
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("probably something about the weather", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        object.confidence = 0.3;
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].operator, Operator::ConsultKnowledgeLibrary);
    }

    #[tokio::test]
    async fn a_high_confidence_reflection_is_not_routed_to_consult() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("the sky is blue", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        object.confidence = 0.95;
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(!proposals.iter().any(|p| p.operator == Operator::ConsultKnowledgeLibrary));
        assert!(proposals.iter().any(|p| p.operator == Operator::Speak));
    }

    #[tokio::test]
    async fn a_low_confidence_reflection_stops_proposing_consult_once_already_attempted() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("probably something about the weather", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        object.confidence = 0.3;
        object.produced_by_operator = Some("ConsultKnowledgeLibrary".to_string());
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(
            !proposals.iter().any(|p| p.operator == Operator::ConsultKnowledgeLibrary),
            "a low-confidence Reflection that already consulted should not consult again every tick"
        );
    }

    #[tokio::test]
    async fn a_tools_own_reentered_result_never_retriggers_act_even_if_the_text_matches() {
        // Regression test: this reproduces the exact live failure
        // (CurrentTimeTool's first phrasing contained its own trigger
        // phrase and looped forever) directly, independent of whatever any
        // particular tool's wording happens to be today.
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("the current time is right now", aca_util::EpochMillis(0), 0.5);
        object.data = serde_json::json!({"source": "tool"});
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(
            !proposals.iter().any(|p| p.operator == Operator::Act),
            "a tool's own re-entered result must never be treated as new grounds to invoke a tool, regardless of its wording"
        );
    }

    #[tokio::test]
    async fn propose_operators_proposes_act_for_text_matching_a_known_tool_intent() {
        let mut graph = aca_graph::Graph::new();
        let object = aca_types::MentalObject::new_observation("hey, what time is it?", aca_util::EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].operator, Operator::Act);
    }

    /// A minimal `Tool` for tests that only need a name/description to show
    /// up in `ToolRegistry::available()` - never actually invoked by
    /// `propose_operators` itself (only `steps::act::act` invokes tools),
    /// so its `invoke` body is unreachable in these tests.
    struct FakeTool {
        name: &'static str,
        description: &'static str,
    }

    #[async_trait]
    impl tools::Tool for FakeTool {
        fn name(&self) -> &'static str {
            self.name
        }
        fn description(&self) -> &'static str {
            self.description
        }
        fn risk_tier(&self) -> tools::ToolRiskTier {
            tools::ToolRiskTier::Harmless
        }
        async fn invoke(&self) -> Result<String, String> {
            unreachable!("propose_operators never invokes a tool directly")
        }
    }

    fn tool_registry_with(name: &'static str, description: &'static str) -> tools::ToolRegistry {
        tools::ToolRegistry::new(vec![std::sync::Arc::new(FakeTool { name, description })], tools::ToolRiskTier::Harmless)
    }

    #[tokio::test]
    async fn propose_operators_proposes_act_for_text_a_tier1_judges_wants_a_tool() {
        // Deliberately doesn't contain any of `tools::match_tool_intent`'s
        // TIME_PHRASES verbatim - this is exactly the phrasing gap real
        // semantic judgement (`propose_tool_intent`) is meant to close, not
        // something keyword matching could ever catch.
        let mut graph = aca_graph::Graph::new();
        let object = aca_types::MentalObject::new_observation("could you tell me what o'clock it is right now", aca_util::EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let tier1_pool = tier1_pool_with_votes(&["current_time", "current_time", "current_time"]);
        let registry = tool_registry_with("current_time", "reports the current time");
        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &tier1_pool, &[], "", &registry, aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert_eq!(proposals, vec![OperatorProposal { operator: Operator::Act, target_id: id, preference: 0.85, confidence: 0.7 }]);
    }

    #[tokio::test]
    async fn propose_operators_never_routes_a_reflection_to_act_via_tool_intent() {
        // Regression guard: confirmed live, a Reflection produced *by*
        // `self_status` ("Working memory holds three items...") reliably
        // paraphrased that same tool's own description closely enough for
        // the old embedding-affordance mechanism to route it straight back
        // into another Act instead of ever reaching Speak - a self_status ->
        // reflect -> self_status -> reflect oscillation that never once
        // proposed Speak. Only a raw Observation should ever be read as
        // *requesting* a tool - even a Tier 1 pool that would unanimously
        // vote "yes" must never be asked the question of a Reflection.
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("working memory currently holds three items about my own state", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        object.confidence = 0.9; // above low_confidence_reflection_threshold, so Consult isn't triggered instead
        let id = object.id;
        graph.insert(object);

        // Would unanimously vote "self_status" if it were ever asked - the
        // point of this test is that a Reflection must never reach that
        // question at all.
        let tier1_pool = tier1_pool_with_votes(&["self_status", "self_status", "self_status"]);
        let registry = tool_registry_with("self_status", "reports a short introspective summary of Omega's own current cognitive state");
        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &tier1_pool, &[], "", &registry, aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(
            !proposals.iter().any(|p| p.operator == Operator::Act),
            "a Reflection must never be routed back into Act via tool-intent judgement, only a raw Observation may"
        );
        assert!(proposals.iter().any(|p| p.operator == Operator::Speak), "should fall through to the ordinary communicative-intent path instead");
    }

    #[tokio::test]
    async fn propose_operators_does_not_act_when_tier1_judges_no_tool_is_wanted() {
        let mut graph = aca_graph::Graph::new();
        let object = aca_types::MentalObject::new_observation("the sky is blue today", aca_util::EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        // Unanimous "none" - genuinely unrelated content, and the text
        // matches no keyword either.
        let tier1_pool = tier1_pool_with_votes(&["none", "none", "none"]);
        let registry = tool_registry_with("current_time", "reports the current time");
        let proposals = propose_operators(&graph, id, Some(0.2), &ExecutiveConfig::default(), &tier1_pool, &[], "", &registry, aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(!proposals.iter().any(|p| p.operator == Operator::Act));
    }

    #[tokio::test]
    async fn propose_operators_stops_proposing_act_once_already_attempted() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("hey, what time is it?", aca_util::EpochMillis(0), 0.5);
        object.produced_by_operator = Some("Act".to_string());
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(
            !proposals.iter().any(|p| p.operator == Operator::Act),
            "an Observation that already acted should not act again every tick"
        );
    }

    #[tokio::test]
    async fn an_act_tagged_observation_is_not_also_proposed_for_reflection() {
        // Regression guard: without treating `produced_by_operator ==
        // Some("Act")` as "already communicated," this object (never
        // reflected on directly - its answer arrives as a separate
        // re-entered Observation) would keep getting proposed for
        // ContinueReflecting every tick even after Act already handled it.
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("hey, what time is it?", aca_util::EpochMillis(0), 0.5);
        object.produced_by_operator = Some("Act".to_string());
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert!(!proposals.iter().any(|p| p.operator == Operator::ContinueReflecting));
    }

    #[tokio::test]
    async fn propose_operators_proposes_consult_knowledge_library_for_an_unconsulted_question() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("subgoal: what is the wifi password?", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Question;
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].operator, Operator::ConsultKnowledgeLibrary);
        // Default confidence (0.5, `MentalObject::new_observation`'s
        // baseline) should reproduce the old fixed 0.85 preference exactly -
        // see `expected_free_energy`'s call site.
        assert!((proposals[0].preference - 0.85).abs() < 1e-5);
    }

    #[tokio::test]
    async fn consult_knowledge_library_preference_rises_with_the_targets_uncertainty() {
        // Active inference's epistemic value made testable: a Reflection
        // the Executive is barely unsure about should earn a lower
        // Consult preference than one it's deeply uncertain about, since
        // querying external knowledge has less to actually resolve for the
        // former.
        let config = ExecutiveConfig::default();
        let mut graph = aca_graph::Graph::new();

        let mut barely_uncertain = aca_types::MentalObject::new_observation("mostly sure about this", aca_util::EpochMillis(0), 0.5);
        barely_uncertain.kind = aca_types::MentalObjectKind::Reflection;
        barely_uncertain.confidence = config.low_confidence_reflection_threshold - 0.01;
        let barely_id = barely_uncertain.id;
        graph.insert(barely_uncertain);

        let mut deeply_uncertain = aca_types::MentalObject::new_observation("no idea about this", aca_util::EpochMillis(0), 0.5);
        deeply_uncertain.kind = aca_types::MentalObjectKind::Reflection;
        deeply_uncertain.confidence = 0.05;
        let deeply_id = deeply_uncertain.id;
        graph.insert(deeply_uncertain);

        let barely_proposals = propose_operators(&graph, barely_id, None, &config, &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        let deeply_proposals = propose_operators(&graph, deeply_id, None, &config, &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;

        let barely_preference = barely_proposals.iter().find(|p| p.operator == Operator::ConsultKnowledgeLibrary).unwrap().preference;
        let deeply_preference = deeply_proposals.iter().find(|p| p.operator == Operator::ConsultKnowledgeLibrary).unwrap().preference;
        assert!(
            deeply_preference > barely_preference,
            "deeply={deeply_preference} should exceed barely={barely_preference}"
        );
    }

    #[tokio::test]
    async fn propose_operators_stops_proposing_consult_once_already_attempted() {
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("subgoal: what is the wifi password?", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Question;
        object.produced_by_operator = Some("ConsultKnowledgeLibrary".to_string());
        let id = object.id;
        graph.insert(object);

        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &empty_tier1_pool(), &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].operator, Operator::Ignore, "a consulted Question with no embedding should settle to Ignore, not be re-proposed forever");
    }

    struct FixedOperatorClient {
        keyword: &'static str,
    }

    #[async_trait]
    impl ChatClient for FixedOperatorClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            // confidence deliberately high and identical across every
            // client - propose_communicative_intent must derive its
            // proposals' confidence from vote share, not this self-report,
            // or these tests would pass for the wrong reason.
            Ok(TierResponse { raw_text: self.keyword.to_string(), confidence: 0.99, tier: aca_types::Tier::T1 })
        }
    }

    /// One client per keyword, each answering with exactly that keyword -
    /// lets a test dictate precisely how Tier 1 "votes" on communicative
    /// intent.
    fn tier1_pool_with_votes(keywords: &[&'static str]) -> DivergentPool {
        let clients: Vec<std::sync::Arc<dyn ChatClient>> = keywords.iter().map(|k| std::sync::Arc::new(FixedOperatorClient { keyword: k }) as std::sync::Arc<dyn ChatClient>).collect();
        DivergentPool::new(aca_types::Tier::T1, Duration::from_secs(5), clients)
    }

    #[tokio::test]
    async fn propose_communicative_intent_returns_empty_when_tier1_is_unconfigured() {
        let proposals =
            propose_communicative_intent(&empty_tier1_pool(), MentalObjectId::new(), "the sky is blue", &[], "", &[Operator::Speak, Operator::Ask, Operator::Ignore], 0.4).await;
        assert!(proposals.is_empty(), "an unconfigured Tier 1 should leave the caller to fall back to a deterministic rule, not fail");
    }

    #[tokio::test]
    async fn propose_communicative_intent_unanimous_samples_yield_one_fully_confident_proposal() {
        let pool = tier1_pool_with_votes(&["speak", "speak", "speak"]);
        let proposals = propose_communicative_intent(&pool, MentalObjectId::new(), "the sky is blue", &[], "", &[Operator::Speak, Operator::Ask, Operator::Ignore], 0.4).await;

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].operator, Operator::Speak);
        assert!((proposals[0].preference - 1.0).abs() < 1e-6);
        assert!((proposals[0].confidence - 1.0).abs() < 1e-6, "unanimous real samples should report full confidence, not any sample's self-report");
    }

    #[tokio::test]
    async fn propose_communicative_intent_a_majority_split_still_yields_a_confident_winner() {
        let pool = tier1_pool_with_votes(&["speak", "speak", "ask"]);
        let proposals = propose_communicative_intent(&pool, MentalObjectId::new(), "something", &[], "", &[Operator::Speak, Operator::Ask, Operator::Ignore], 0.4).await;

        assert_eq!(proposals.len(), 2);
        let speak = proposals.iter().find(|p| p.operator == Operator::Speak).expect("majority operator should be proposed");
        let ask = proposals.iter().find(|p| p.operator == Operator::Ask).expect("minority operator should still be proposed, just at lower preference");
        assert!((speak.preference - 2.0 / 3.0).abs() < 1e-5);
        assert!((ask.preference - 1.0 / 3.0).abs() < 1e-5);
        assert!(speak.preference > ask.preference);
    }

    #[tokio::test]
    async fn propose_communicative_intent_an_even_split_yields_a_genuine_tie() {
        // This is the whole point: real Tier 1 disagreement should make
        // select_operator's Confidence-impasse path reachable through
        // actual ambiguity, not the numeric coincidence the fixed
        // by-punctuation rule required.
        let pool = tier1_pool_with_votes(&["speak", "ask", "ignore"]);
        let proposals = propose_communicative_intent(&pool, MentalObjectId::new(), "something ambiguous", &[], "", &[Operator::Speak, Operator::Ask, Operator::Ignore], 0.4).await;

        assert_eq!(proposals.len(), 3);
        let decision = select_operator(&proposals, &ExecutiveConfig::default());
        assert!(
            matches!(decision, ExecutiveDecision::Impasse { kind: ImpasseKind::Confidence, .. }),
            "an even three-way split among real disagreeing samples should be a genuine impasse, not an arbitrary pick: {decision:?}"
        );
    }

    #[tokio::test]
    async fn propose_communicative_intent_ignores_votes_outside_the_allowed_set() {
        // A sample that names a real operator, just not one of the ones
        // legal in this context, should be discarded rather than silently
        // accepted - `allowed` exists precisely to keep the vote restricted
        // to what's actually a valid choice here.
        let pool = tier1_pool_with_votes(&["speak", "remember", "speak"]);
        let proposals = propose_communicative_intent(&pool, MentalObjectId::new(), "something", &[], "", &[Operator::Speak, Operator::Ask, Operator::Ignore], 0.4).await;

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].operator, Operator::Speak);
        assert!((proposals[0].preference - 1.0).abs() < 1e-6, "the disallowed vote should not even count toward the total");
    }

    #[tokio::test]
    async fn propose_tool_intent_falls_back_to_keyword_matching_when_tier1_is_unconfigured() {
        let registry = tool_registry_with("current_time", "reports the current time");
        let result = propose_tool_intent(&empty_tier1_pool(), "what time is it", "", &registry.available(), 0.2).await;
        assert_eq!(result, Some("current_time"));
    }

    #[tokio::test]
    async fn propose_tool_intent_returns_none_when_tier1_unanimously_says_none() {
        let tier1_pool = tier1_pool_with_votes(&["none", "none", "none"]);
        let registry = tool_registry_with("current_time", "reports the current time");
        let result = propose_tool_intent(&tier1_pool, "the sky is blue today", "", &registry.available(), 0.2).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn propose_tool_intent_requires_unanimous_agreement_not_just_a_majority() {
        // Stricter than `propose_communicative_intent`'s simple majority -
        // confirmed live, a bare 2-of-3 majority let a small Tier 1 model's
        // own misjudgement through (see this function's own doc comment).
        // A false negative here is safe (falls through to ordinary
        // reflection); a false positive isn't.
        let tier1_pool = tier1_pool_with_votes(&["current_time", "current_time", "none"]);
        let registry = tool_registry_with("current_time", "reports the current time");
        let result = propose_tool_intent(&tier1_pool, "could you tell me what o'clock it is", "", &registry.available(), 0.2).await;
        assert_eq!(result, None, "a non-unanimous vote must not invoke a tool, even at 2-of-3");
    }

    #[tokio::test]
    async fn propose_tool_intent_accepts_unanimous_agreement() {
        let tier1_pool = tier1_pool_with_votes(&["current_time", "current_time", "current_time"]);
        let registry = tool_registry_with("current_time", "reports the current time");
        let result = propose_tool_intent(&tier1_pool, "could you tell me what o'clock it is", "", &registry.available(), 0.2).await;
        assert_eq!(result, Some("current_time"));
    }

    #[tokio::test]
    async fn propose_tool_intent_returns_none_on_a_genuine_three_way_split() {
        // No candidate holds a strict majority - exactly the "not actually
        // clear" case that should leave a tool un-invoked rather than
        // guessed at (see this function's own doc comment).
        let tier1_pool = tier1_pool_with_votes(&["current_time", "self_status", "none"]);
        let registry = tool_registry_with("current_time", "reports the current time");
        let result = propose_tool_intent(&tier1_pool, "ambiguous input", "", &registry.available(), 0.2).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn propose_tool_intent_ignores_a_hallucinated_tool_name() {
        // A vote for a tool name that isn't actually in `available_tools`
        // must not count - `parse_tool_intent_keyword` only accepts names
        // drawn from the list it was actually given.
        let tier1_pool = tier1_pool_with_votes(&["nonexistent_tool", "nonexistent_tool", "nonexistent_tool"]);
        let registry = tool_registry_with("current_time", "reports the current time");
        let result = propose_tool_intent(&tier1_pool, "something", "", &registry.available(), 0.2).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn propose_operators_routes_a_reflections_communicative_intent_through_real_tier1_disagreement() {
        // End-to-end through propose_operators itself (not just the
        // isolated propose_communicative_intent helper above): a Reflection
        // with no embedding never reaches the Remember branch, so this
        // should be exactly the three tied communicative-intent proposals
        // Tier 1 genuinely disagreed on.
        let mut graph = aca_graph::Graph::new();
        let mut object = aca_types::MentalObject::new_observation("something ambiguous", aca_util::EpochMillis(0), 0.5);
        object.kind = aca_types::MentalObjectKind::Reflection;
        let id = object.id;
        graph.insert(object);

        let pool = tier1_pool_with_votes(&["speak", "ask", "ignore"]);
        let proposals = propose_operators(&graph, id, None, &ExecutiveConfig::default(), &pool, &[], "", &empty_tool_registry(), aca_util::EpochMillis(0), 120_000, 0.4, 0.2).await;

        assert_eq!(proposals.len(), 3);
        let decision = select_operator(&proposals, &ExecutiveConfig::default());
        assert!(matches!(decision, ExecutiveDecision::Impasse { kind: ImpasseKind::Confidence, .. }));
    }

    #[test]
    fn impasse_response_maps_kinds_to_the_right_action() {
        assert_eq!(impasse_response(ImpasseKind::MissingInformation), ImpasseResponse::SpawnSubgoal);
        assert_eq!(impasse_response(ImpasseKind::Confidence), ImpasseResponse::EscalateTier);
    }

    #[test]
    fn spawn_subgoal_inserts_a_question_object_into_the_graph() {
        let mut graph = aca_graph::Graph::new();
        let id = spawn_subgoal(&mut graph, "nothing was proposed", aca_util::EpochMillis(1_000), 0.5);
        let subgoal = graph.get(&id).expect("subgoal should be inserted");
        assert_eq!(subgoal.kind, aca_types::MentalObjectKind::Question);
        assert!(subgoal.text.contains("nothing was proposed"));
    }

    struct SlowFakeChatClient;

    #[async_trait]
    impl ChatClient for SlowFakeChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            tokio::time::sleep(Duration::from_millis(30)).await;
            Ok(TierResponse {
                raw_text: r#"{"confidence": 0.9, "response": "resolved"}"#.to_string(),
                confidence: 0.9,
                tier: aca_types::Tier::T3,
            })
        }
    }

    fn req() -> GenerateRequest {
        GenerateRequest { prompt: "resolve the impasse".into(), temperature: 0.2 }
    }

    #[tokio::test]
    async fn two_simultaneous_confidence_impasses_yield_exactly_one_live_escalation() {
        // Tier 3 is single-flight by construction: concurrency = 1.
        let pool = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = SlowFakeChatClient;

        let (first, second) = tokio::join!(
            resolve_confidence_impasse(&pool, &client, req()),
            resolve_confidence_impasse(&pool, &client, req()),
        );

        let escalated_count = [&first, &second]
            .iter()
            .filter(|r| matches!(r, ImpasseResolution::Escalated(_)))
            .count();
        let deferred_count = [&first, &second]
            .iter()
            .filter(|r| matches!(r, ImpasseResolution::DeferredToHeuristic))
            .count();

        assert_eq!(escalated_count, 1, "exactly one of the two simultaneous impasses should get a live Tier-3 call");
        assert_eq!(deferred_count, 1, "the other must fall back to the cheap heuristic path, not wait");
    }
}
