use std::sync::Arc;

use aca_util::Clock;
use async_trait::async_trait;
use tokio::sync::watch;

use crate::snapshot::EngineSnapshot;

/// specs.md explicitly scopes "arbitrary tool execution" out of the MVP —
/// this is the deliberately narrow exception, not a reversal of that
/// scoping decision: a closed, explicitly-registered set of tools, each
/// tagged with a risk tier, gated by one trust dial
/// (`ToolRegistry::max_risk_tier`) that can be turned up over time as trust
/// is established. There is no path from `Operator::Act` to arbitrary
/// code/command execution — only to whatever's actually registered here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolRiskTier {
    /// No side effects beyond reading already-available in-process state or
    /// doing pure computation — nothing that reaches outside the process.
    /// The only tier any tool ships at today.
    Harmless,
    /// Reserved for future tools with real but easily-reversible external
    /// effects (e.g. writing a file). Not used by any tool shipped today.
    Reversible,
    /// Reserved for future tools with real, hard-to-reverse external
    /// effects (e.g. sending a message, controlling a device). Not used by
    /// any tool shipped today.
    Consequential,
}

/// One registered capability `Operator::Act` can invoke. Deliberately
/// argument-free for v1 — every tool shipped today needs no input beyond
/// "run me now"; a richer, argument-taking `Tool` contract is future work
/// once a tool that actually needs arguments motivates one.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    /// A short, human-readable summary of what this tool does — surfaced to
    /// Omega itself (see `ToolRegistry::available`) so it can decide whether
    /// running a given tool on its own initiative is worthwhile, not just
    /// react to a matched trigger phrase.
    fn description(&self) -> &'static str;
    fn risk_tier(&self) -> ToolRiskTier;
    async fn invoke(&self) -> Result<String, String>;
}

/// The closed set of tools `Operator::Act` may ever reach, plus the trust
/// dial. A tool present in `tools` but above `max_risk_tier` is exactly as
/// unreachable as one never registered at all — `find`/`allowed_names` both
/// filter on it, by construction, not by convention at each call site.
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
    max_risk_tier: ToolRiskTier,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>, max_risk_tier: ToolRiskTier) -> Self {
        Self { tools, max_risk_tier }
    }

    pub fn empty() -> Self {
        Self { tools: Vec::new(), max_risk_tier: ToolRiskTier::Harmless }
    }

    pub fn find(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|tool| tool.name() == name && tool.risk_tier() <= self.max_risk_tier)
    }

    /// Every currently-reachable tool's `(name, description)`, gated by
    /// `max_risk_tier` exactly like `find` — a tool above the trust dial is
    /// as invisible here as it is unreachable there. This is how Omega
    /// "learns" what it currently has access to: a caller (e.g. the boredom
    /// idle-initiative prompt) always re-reads the live registry, so a tool
    /// registered later becomes discoverable with no further code changes.
    pub fn available(&self) -> Vec<(&'static str, &'static str)> {
        self.tools.iter().filter(|tool| tool.risk_tier() <= self.max_risk_tier).map(|tool| (tool.name(), tool.description())).collect()
    }

    /// Registers the built-in `SelfStatusTool`, wired to `snapshot_tx`'s own
    /// live value via `.subscribe()`. Called internally by
    /// `CognitiveLoopActor::new` once its snapshot channel exists — a
    /// caller outside the actor (e.g. `omega-acad::main`) has no way to
    /// obtain that receiver *before* construction, so this specific tool
    /// can't be included in the registry handed in from outside the way
    /// `CurrentTimeTool` (needing only a `Clock`) can be. Every
    /// `CognitiveLoopActor` therefore always has `self_status` available,
    /// regardless of what registry was passed in - it's the one tool that's
    /// actor-owned rather than externally injected.
    pub(crate) fn with_self_status(mut self, snapshot_tx: &watch::Sender<EngineSnapshot>) -> Self {
        self.tools.push(Arc::new(SelfStatusTool::new(snapshot_tx.subscribe())));
        self
    }

    /// Registers the built-in `AbstractionStatusTool` - same reasoning and
    /// same actor-owned-rather-than-externally-injected wiring as
    /// `with_self_status`, just for a different slice of `EngineSnapshot`.
    pub(crate) fn with_abstraction_status(mut self, snapshot_tx: &watch::Sender<EngineSnapshot>) -> Self {
        self.tools.push(Arc::new(AbstractionStatusTool::new(snapshot_tx.subscribe())));
        self
    }
}

/// Deliberately simple keyword matching - no longer the primary tool-intent
/// mechanism (see `steps::executive::propose_tool_intent`, which asks Tier 1
/// directly per specs.md's "Where the Language Model Plugs In" point 4), only
/// the fallback for when Tier 1 isn't configured or didn't produce a usable
/// answer, same fallback role `parse_operator_keyword`'s deterministic rule
/// plays for `propose_communicative_intent`. An embedding-similarity
/// alternative to this (`match_tool_affordance`) was tried and removed -
/// confirmed live, cosine distance to a tool's description measures topical
/// relatedness, not request intent, and the two are inseparable that way in
/// an architecture whose own subject matter is constantly self-referential
/// (a real research question about cognition and a genuine status-check
/// paraphrase scored within noise of each other against `self_status`'s
/// description). Returns the tool name whose trigger phrases appear in
/// `text` — *not* whether that tool is actually reachable under the current
/// risk tier; callers check that separately via `ToolRegistry::find`, so a
/// tool can be temporarily "known but not currently allowed" without this
/// function needing to know about trust levels at all.
pub fn match_tool_intent(text: &str) -> Option<&'static str> {
    let normalized = text.to_lowercase();
    const TIME_PHRASES: &[&str] = &["what time is it", "current time", "what's the time", "what is the time", "what day is it", "what's the date", "what is the date"];
    const STATUS_PHRASES: &[&str] = &["how are you", "what are you thinking", "what's on your mind", "your status", "how do you feel"];
    // Deliberately distinct from `STATUS_PHRASES` above (working-memory/cycle
    // stats) - these ask about the synthesized pattern itself, a different
    // question `SelfStatusTool` has no data to answer at all. See
    // `AbstractionStatusTool`'s own doc comment.
    const ABSTRACTION_PHRASES: &[&str] = &[
        "provisional abstraction",
        "current abstraction",
        "current hypothesis",
        "what pattern have you",
        "what have you generalized",
        "what have you synthesized",
        "what have you learned across",
    ];
    if TIME_PHRASES.iter().any(|phrase| normalized.contains(phrase)) {
        return Some(CurrentTimeTool::NAME);
    }
    if ABSTRACTION_PHRASES.iter().any(|phrase| normalized.contains(phrase)) {
        return Some(AbstractionStatusTool::NAME);
    }
    if STATUS_PHRASES.iter().any(|phrase| normalized.contains(phrase)) {
        return Some(SelfStatusTool::NAME);
    }
    None
}

/// Reports the current wall-clock time. No external I/O — reads only the
/// injected `Clock`, the same source of "now" every other step already
/// uses, so this tool can't drift from or bypass the rest of the cycle's
/// notion of time.
pub struct CurrentTimeTool {
    clock: Arc<dyn Clock>,
}

impl CurrentTimeTool {
    pub const NAME: &'static str = "current_time";

    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self { clock }
    }
}

#[async_trait]
impl Tool for CurrentTimeTool {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> &'static str {
        "Reports the current wall-clock date and time."
    }

    fn risk_tier(&self) -> ToolRiskTier {
        ToolRiskTier::Harmless
    }

    async fn invoke(&self) -> Result<String, String> {
        let millis = self.clock.now().as_millis();
        let formatted = chrono::DateTime::from_timestamp_millis(millis)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_else(|| millis.to_string());
        // Deliberately does not contain any of TIME_PHRASES (e.g. "current
        // time") - this text re-enters the cycle as a fresh Observation
        // (see `loop_actor`'s Act re-entry send site), and if it matched
        // its own trigger it would invoke this same tool again forever.
        // Confirmed live: an earlier phrasing ("The current time is...")
        // did exactly that.
        Ok(format!("As of right now, the time is {formatted}."))
    }
}

/// Reports a short introspective summary of Omega's own current cognitive
/// state — Working Memory size, memory-role counts, cycle number. Reads
/// only the same `EngineSnapshot` `aca-api`/`aca-mcp` already expose
/// externally; this tool gives Omega a way to report on itself when asked,
/// not a second, privileged path into the graph.
pub struct SelfStatusTool {
    snapshot_rx: watch::Receiver<EngineSnapshot>,
}

impl SelfStatusTool {
    pub const NAME: &'static str = "self_status";

    pub fn new(snapshot_rx: watch::Receiver<EngineSnapshot>) -> Self {
        Self { snapshot_rx }
    }
}

#[async_trait]
impl Tool for SelfStatusTool {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> &'static str {
        "Reports a short introspective summary of Omega's own current cognitive state: Working Memory size and member IDs, memory counts, cycle number, the last prediction-error reading, current affect valence/precision gain, the last operator the Executive selected, which real object displaced which (when Working Memory competition actually forced one this tick), and which real candidates are currently attended without yet being part of conscious Working Memory."
    }

    fn risk_tier(&self) -> ToolRiskTier {
        ToolRiskTier::Harmless
    }

    async fn invoke(&self) -> Result<String, String> {
        let snapshot = self.snapshot_rx.borrow().clone();

        let member_ids = if snapshot.working_memory.is_empty() {
            "none".to_string()
        } else {
            snapshot.working_memory.iter().map(|member| member.id.to_string()).collect::<Vec<_>>().join(", ")
        };

        // Real numbers only - no field is ever fabricated when the
        // corresponding tracker hasn't produced a reading yet (same "never
        // originate content" discipline `social_interface` already applies
        // to rendered speech).
        let prediction_error = match snapshot.last_prediction_error {
            Some(p) => format!(
                "My last prediction error: magnitude {:.2}, precision {:.2}, epistemic value {:.2}, reward {:.2}.",
                p.error_magnitude, p.precision, p.epistemic_value, p.reward
            ),
            None => "I haven't resolved a prediction error yet this run.".to_string(),
        };

        let last_decision = match &snapshot.last_operator_proposal {
            Some(p) => format!(
                "The last operator I selected was {} (target {}, preference {:.2}, confidence {:.2}).",
                p.operator, p.target_id, p.preference, p.confidence
            ),
            None => "I haven't selected an operator yet this run.".to_string(),
        };

        // `DisplacementSummary::claim_text` (not a second, ad hoc rendering
        // here) - the same construction `prompt_templates::displacement_
        // block` grounds a Tier 3 reflection in, so this tool and a spoken
        // reflection never phrase the identical verified fact two different
        // ways. `None` (most ticks - see `ReleaseReason`'s doc comment for
        // how much real Working Memory turnover is *not* a competitive
        // displacement) is reported plainly, not silently omitted, matching
        // this tool's existing "real numbers only" discipline above.
        let displacement = match &snapshot.last_displacement {
            Some(d) => format!("Most recently, {}.", d.claim_text()),
            None => "Nothing has been competitively displaced from Working Memory most recently.".to_string(),
        };

        // Real GNW attention/ignition dissociation - `EngineSnapshot::
        // attended_not_ignited`'s own doc comment has the full mechanism.
        // Named plainly rather than claiming any felt quality about it (no
        // "I sense" language) - this is a real, verified fact about which
        // Coalition candidates lost this tick's ignition competition, not a
        // phenomenological report this architecture has no basis to make.
        let attended_not_ignited = if snapshot.attended_not_ignited.is_empty() {
            "Nothing is currently attended without also being conscious of it - Working Memory and Coalition attention agree right now.".to_string()
        } else {
            let texts: Vec<&str> = snapshot.attended_not_ignited.iter().map(|m| m.text.as_str()).collect();
            format!("Attended but not yet part of my conscious Working Memory: {}.", texts.join("; "))
        };

        Ok(format!(
            "Right now I'm holding {} thing(s) in Working Memory (ids: {member_ids}), on cognitive cycle {}. \
             So far I've formed {} episodic and {} semantic memories. \
             My current affect valence is {:.2} (precision gain {:.2}). \
             {prediction_error} {last_decision} {displacement} {attended_not_ignited}",
            snapshot.working_memory.len(),
            snapshot.cycle_seq,
            snapshot.memory_counts.episodic,
            snapshot.memory_counts.semantic,
            snapshot.affect_valence,
            snapshot.precision_gain,
        ))
    }
}

/// Reports Omega's current provisional abstraction - the most recently
/// synthesized general pattern (`steps::synthesize::synthesize`), its
/// confidence, how many times it's been reconfirmed, and the specific
/// episodic memories it was actually derived from. Reads only
/// `EngineSnapshot::provisional_abstraction` (computed in `loop_actor::
/// publish_snapshot` from the same graph state `self_status` and the local
/// API already read) - not a second, privileged path into the graph, same
/// discipline as `SelfStatusTool`.
///
/// This closes a real gap `self_status` never covered: which memories
/// support a given generalization, and whether it's actually been written
/// into Semantic memory yet (it always has, by the time this can report it
/// at all - `synthesize` writes the pattern the moment it forms;
/// "provisional" here means "not yet reconfirmed," not "not yet stored").
///
/// This deliberately does *not* report what would contradict the pattern:
/// this architecture has no mechanism for tracking disconfirming evidence,
/// only dedup-reinforcement toward an existing match (see `synthesize`'s own
/// doc comment). Inventing an answer to that question would be exactly the
/// kind of fabrication `steps::social_interface`'s "never originate content"
/// discipline exists to prevent elsewhere in this pipeline - so this says so
/// plainly instead of guessing.
pub struct AbstractionStatusTool {
    snapshot_rx: watch::Receiver<EngineSnapshot>,
}

impl AbstractionStatusTool {
    pub const NAME: &'static str = "abstraction_status";

    pub fn new(snapshot_rx: watch::Receiver<EngineSnapshot>) -> Self {
        Self { snapshot_rx }
    }
}

#[async_trait]
impl Tool for AbstractionStatusTool {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> &'static str {
        "Reports Omega's current provisional abstraction: the most recently synthesized general pattern, its confidence, how many times it's been reconfirmed, and the specific episodic memories that support it."
    }

    fn risk_tier(&self) -> ToolRiskTier {
        ToolRiskTier::Harmless
    }

    async fn invoke(&self) -> Result<String, String> {
        let snapshot = self.snapshot_rx.borrow().clone();
        let Some(abstraction) = snapshot.provisional_abstraction else {
            return Ok("I haven't synthesized a general pattern across my episodic memories yet - there is no provisional abstraction right now.".to_string());
        };

        let present = abstraction.supporting_episodes.len();
        let sources = if present == 0 {
            "none of its original source memories are still present in the graph to quote".to_string()
        } else {
            abstraction.supporting_episodes.iter().map(|text| format!("\"{text}\"")).collect::<Vec<_>>().join("; ")
        };
        let coverage_note = if present < abstraction.source_count {
            format!(" ({} of the original {} sources are no longer present in the graph.)", abstraction.source_count - present, abstraction.source_count)
        } else {
            String::new()
        };

        Ok(format!(
            "My current provisional abstraction is: \"{}\" (confidence {:.2}, formed from {} episodic memories, reconfirmed {} time(s) since). \
             The supporting episodic memories still present are: {sources}.{coverage_note} \
             I don't track what would count as evidence against it - this architecture has no disconfirmation mechanism yet, only reinforcement toward an existing pattern.",
            abstraction.text, abstraction.confidence, abstraction.source_count, abstraction.reinforced_count,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_util::{EpochMillis, ManualClock};

    #[test]
    fn match_tool_intent_finds_the_current_time_tool() {
        assert_eq!(match_tool_intent("hey, what time is it?"), Some(CurrentTimeTool::NAME));
        assert_eq!(match_tool_intent("What's the date today?"), Some(CurrentTimeTool::NAME));
    }

    #[test]
    fn match_tool_intent_finds_the_self_status_tool() {
        assert_eq!(match_tool_intent("Omega, how are you?"), Some(SelfStatusTool::NAME));
        assert_eq!(match_tool_intent("what's on your mind"), Some(SelfStatusTool::NAME));
    }

    #[test]
    fn match_tool_intent_finds_the_abstraction_status_tool() {
        assert_eq!(match_tool_intent("show me your current provisional abstraction"), Some(AbstractionStatusTool::NAME));
        assert_eq!(match_tool_intent("what pattern have you formed"), Some(AbstractionStatusTool::NAME));
    }

    #[test]
    fn match_tool_intent_returns_none_for_unrelated_text() {
        assert_eq!(match_tool_intent("the sky is blue today"), None);
    }

    #[tokio::test]
    async fn current_time_tool_reports_the_injected_clocks_time() {
        let clock = Arc::new(ManualClock::new(EpochMillis(1_700_000_000_000)));
        let tool = CurrentTimeTool::new(clock);
        let result = tool.invoke().await.unwrap();
        assert!(result.contains("2023"), "expected the formatted timestamp to include the year, got: {result}");
    }

    #[tokio::test]
    async fn self_status_tool_reports_the_current_snapshot() {
        let snapshot = EngineSnapshot { cycle_seq: 42, ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = SelfStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();
        assert!(result.contains("42"));
    }

    #[tokio::test]
    async fn self_status_tool_reports_a_real_displacement_when_one_happened() {
        use crate::snapshot::DisplacementSummary;
        use aca_types::MentalObjectId;

        let displacement = DisplacementSummary {
            entrant_id: MentalObjectId::new(),
            entrant_text: "someone is at the front door".to_string(),
            evicted_id: MentalObjectId::new(),
            evicted_text: "the kettle is boiling".to_string(),
            at: EpochMillis(1_000),
        };
        let snapshot = EngineSnapshot { last_displacement: Some(displacement), ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = SelfStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();

        assert!(result.contains("someone is at the front door entered my awareness because it displaced the kettle is boiling"), "self_status should report the exact same claim_text() the prompt-facing grounding uses, got: {result}");
    }

    #[tokio::test]
    async fn self_status_tool_reports_no_displacement_plainly_rather_than_omitting_it() {
        let snapshot = EngineSnapshot { last_displacement: None, ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = SelfStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();

        assert!(result.to_lowercase().contains("nothing has been competitively displaced"), "an absent displacement should be stated plainly, not silently dropped from the report, got: {result}");
    }

    #[tokio::test]
    async fn self_status_tool_reports_a_real_attended_but_not_ignited_candidate() {
        use crate::snapshot::WorkingMemoryMember;
        use aca_types::MentalObjectId;

        let member = WorkingMemoryMember {
            id: MentalObjectId::new(),
            kind: aca_types::MentalObjectKind::Observation,
            text: "a subliminal candidate".to_string(),
            activation_total: -1.9,
            attention_score: Some(-1.9),
            promotion_status: aca_types::PromotionStatus::Confirmed,
            confidence: 1.0,
        };
        let snapshot = EngineSnapshot { attended_not_ignited: vec![member], ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = SelfStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();

        assert!(result.contains("a subliminal candidate"), "a real attended-but-not-ignited candidate should be named, got: {result}");
    }

    #[tokio::test]
    async fn self_status_tool_reports_no_attended_but_not_ignited_candidates_plainly() {
        let snapshot = EngineSnapshot { attended_not_ignited: Vec::new(), ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = SelfStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();

        assert!(result.to_lowercase().contains("nothing is currently attended"), "an empty attended-but-not-ignited set should be stated plainly, not silently dropped, got: {result}");
    }

    #[tokio::test]
    async fn abstraction_status_tool_reports_none_before_anything_is_synthesized() {
        let snapshot = EngineSnapshot { provisional_abstraction: None, ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = AbstractionStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();
        assert!(result.contains("haven't synthesized"), "expected an honest 'nothing yet' report, got: {result}");
    }

    #[tokio::test]
    async fn abstraction_status_tool_reports_the_pattern_confidence_and_supporting_episodes() {
        let abstraction = crate::snapshot::ProvisionalAbstractionSummary {
            id: aca_types::MentalObjectId::new(),
            text: "weather strongly influences running habits".to_string(),
            confidence: 0.82,
            formed_at: EpochMillis(1_000),
            reinforced_count: 2,
            source_count: 3,
            supporting_episodes: vec!["went for a run in the rain".to_string(), "skipped a run because of rain".to_string(), "rescheduled a run around rain".to_string()],
        };
        let snapshot = EngineSnapshot { provisional_abstraction: Some(abstraction), ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = AbstractionStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();
        assert!(result.contains("weather strongly influences running habits"));
        assert!(result.contains("0.82"));
        assert!(result.contains("reconfirmed 2 time(s)"));
        assert!(result.contains("went for a run in the rain"));
        assert!(result.contains("skipped a run because of rain"));
        assert!(result.contains("rescheduled a run around rain"));
        assert!(result.contains("don't track what would count as evidence against it"), "must not fabricate contradicting evidence, got: {result}");
    }

    #[tokio::test]
    async fn abstraction_status_tool_notes_when_some_sources_have_decayed_out_of_the_graph() {
        let abstraction = crate::snapshot::ProvisionalAbstractionSummary {
            id: aca_types::MentalObjectId::new(),
            text: "a pattern with a missing source".to_string(),
            confidence: 0.7,
            formed_at: EpochMillis(1_000),
            reinforced_count: 0,
            source_count: 3,
            supporting_episodes: vec!["only this one is still here".to_string()],
        };
        let snapshot = EngineSnapshot { provisional_abstraction: Some(abstraction), ..Default::default() };
        let (_tx, rx) = watch::channel(snapshot);
        let tool = AbstractionStatusTool::new(rx);
        let result = tool.invoke().await.unwrap();
        assert!(result.contains("2 of the original 3 sources are no longer present"), "expected an honest coverage gap note, got: {result}");
    }

    #[test]
    fn registry_only_finds_tools_at_or_under_the_max_risk_tier() {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let tool: Arc<dyn Tool> = Arc::new(CurrentTimeTool::new(clock));
        let registry = ToolRegistry::new(vec![tool], ToolRiskTier::Harmless);
        assert!(registry.find(CurrentTimeTool::NAME).is_some());
        assert!(registry.find("not_a_real_tool").is_none());
    }

    struct ProbeTool(ToolRiskTier);

    #[async_trait]
    impl Tool for ProbeTool {
        fn name(&self) -> &'static str {
            "probe"
        }
        fn description(&self) -> &'static str {
            "a test-only probe tool"
        }
        fn risk_tier(&self) -> ToolRiskTier {
            self.0
        }
        async fn invoke(&self) -> Result<String, String> {
            Ok("probed".to_string())
        }
    }

    #[test]
    fn available_lists_names_and_descriptions_at_or_under_the_max_risk_tier() {
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(CurrentTimeTool::new(clock)), Arc::new(ProbeTool(ToolRiskTier::Reversible))];
        let registry = ToolRegistry::new(tools, ToolRiskTier::Harmless);
        let available = registry.available();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].0, CurrentTimeTool::NAME);
        assert!(!available[0].1.is_empty());
    }

    #[test]
    fn empty_registry_finds_nothing() {
        let registry = ToolRegistry::empty();
        assert!(registry.find(CurrentTimeTool::NAME).is_none());
    }

}
