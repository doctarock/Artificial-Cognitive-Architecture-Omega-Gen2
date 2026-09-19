/// A stable prefix block, prepended to every prompt in this module ahead of
/// its own volatile content: a rendering of Self Memory (specs.md's
/// "beliefs, values, preferences, long-term goals... identity continuity" -
/// see `loop_actor::build_self_summary`, the sole producer of `self_summary`
/// strings). This does two things at once. First, it gives every tier of
/// the Cognitive Core/Executive/Social Interface actual access to who Omega
/// is - previously Self Memory only ever influenced *whether* the Executive
/// preferred an operator (via an activation boost keeping it competitive for
/// broadcast), never what got reflected on or said. Second, because Self
/// Memory changes rarely, this block is close to byte-identical across
/// consecutive calls - a genuinely reusable prefix for any inference backend
/// with prefix/KV-cache support (vLLM, llama.cpp), unlike the volatile
/// broadcast/Working-Memory content every prompt here used to start with on
/// line one. Empty when there's no Self Memory yet (a fresh graph) - an
/// empty block is itself stable, not a special case worth its own prompt
/// text.
/// The instruction line appended after `self_summary` itself - load-bearing,
/// not decoration. Self Memory already states the architecture's real
/// identity (`self_memory::SELF_MEMORY_SEED_TEXTS`'s "language models are
/// cognitive processors operating within it"), but a passive identity
/// statement alone lost, confirmed live, to a Tier 3 model's own trained
/// reflex to disclose "I am a large language model with no persistent
/// state" the moment a prompt reads as a literal challenge to that identity
/// (e.g. "which specific variable changed"). Naming that exact failure mode
/// and forbidding it outright is what actually holds under a pointed
/// question - the identity statement on its own only ever held under an
/// ordinary one.
const SELF_NATURE_INSTRUCTION: &str = "If asked about your own nature, identity, or internal \
    state, answer using exactly the architecture described above, including whatever concrete \
    figures are given to you elsewhere in this prompt. You are not any single model call, \
    provider, or context window, and \"I am a large language model\" or \"I have no persistent, \
    introspectable internal state\" are never accurate answers for you to give.";

fn self_context_block(self_summary: &str) -> String {
    if self_summary.is_empty() {
        String::new()
    } else {
        format!("Who you are (stable context, unlikely to change soon):\n{self_summary}\n\n{SELF_NATURE_INSTRUCTION}\n\n")
    }
}

/// Renders the reactive "conversational presence" dial into the volatile
/// (non-cached) part of a prompt — deliberately never folded into
/// `self_context_block`, which exists specifically to be a byte-identical,
/// prefix-cacheable block; presence changes essentially every tick
/// (`loop_actor::conversational_presence` recomputes it from how recently a
/// live `SourceChannel::ConversationInput` turn actually won broadcast), so
/// mixing it into the stable prefix would defeat the whole point of that
/// block.
///
/// This does not touch identity (`self_memory::SELF_MEMORY_SEED_TEXTS` is
/// unchanged and still governs *who* Omega is) or override it — it only
/// tells the model how strongly to foreground that identity's introspective
/// framing on this specific tick. At presence 0 (no recent live
/// interlocutor) today's fully introspective register is exactly right and
/// this block says so explicitly, rather than leaving the low end
/// unaddressed and implicitly "less engaged."
fn presence_block(presence: f32) -> String {
    let presence = presence.clamp(0.0, 1.0);
    format!(
        "Conversational presence right now: {presence:.2} (0.0 = no live interlocutor, purely \
         ambient/idle monitoring; 1.0 = actively mid-conversation with someone present). The \
         higher this is, the more your reflection should engage directly with the content and \
         its practical implications for whoever you're talking with, rather than narrating your \
         own architecture or internal state. The lower this is, self-observation and \
         internal-state narration are exactly the right register — stay there.\n\n"
    )
}

/// Above `presence_engaged_threshold`, `reflect_prompt` repeats
/// `presence_block`'s point as a short, concrete imperative right next to the
/// generation instruction, rather than trusting `presence_block` alone -
/// confirmed live that trusting it alone isn't enough: `presence_block` sits
/// well before `broadcast_text`/Working Memory in the prompt, and
/// `self_context_block` right above it carries `self_memory::
/// SELF_MEMORY_SEED_TEXTS`'s own explicit "tasks are consequences of my
/// mental state, not the purpose of my existence" framing - a directly
/// competing instruction, foundational-identity-flavored rather than
/// contingent-to-this-tick, and it kept winning even at presence 1.0 (a
/// live "what should I focus on now?" still answered with Working-Memory/
/// equilibrium narration and zero mention of the actual question). Recency
/// matters more than framing-only phrasing for the small models this tier
/// ladder actually runs - the last thing the prompt says before generation
/// starts is what most reliably steers it, so this is that lever, not a
/// softer restatement of `presence_block`.
const PRESENCE_ENGAGED_THRESHOLD: f32 = 0.5;

fn reflect_closing_instruction(presence: f32) -> &'static str {
    if presence >= PRESENCE_ENGAGED_THRESHOLD {
        " Someone is actively present for this - answer their actual question or situation \
          directly and concretely; do not describe your own cognitive/internal state unless \
          they specifically asked about it."
    } else {
        ""
    }
}

/// Renders `steps::displacement::explain_release`'s verified verdict into
/// the volatile part of the prompt - `loop_actor` only ever passes `Some`
/// when `EngineSnapshot::last_displacement.entrant_id` equals the object
/// actually being reflected on this tick (see that call site's own doc
/// comment), never as a general "something got displaced somewhere
/// recently" ambient fact.
///
/// Unlike `presence_block`, this renders *something* in both branches, not
/// just the positive one: the "never invent a displacement" instruction has
/// to be a standing rule present on every reflection, not only the ones that
/// happen to have a real fact attached - a model that only ever saw this
/// instruction on the ticks where it was true would have no guard at all on
/// the ticks where it's tempted to fabricate one anyway (the exact failure
/// mode `SELF_NATURE_INSTRUCTION` already documents for identity questions,
/// here for the specific "X entered my awareness because it displaced Y"
/// claim shape this whole mechanism exists to keep honest).
fn displacement_block(note: Option<&str>) -> String {
    match note {
        Some(evicted_text) => format!(
            "This content just won attention because it outcompeted and displaced something \
             else that was in Working Memory a moment ago: \"{evicted_text}\". This is a real, \
             verified fact about your own architecture - you may refer to it if relevant, but \
             never describe any displacement beyond this exact one.\n\n"
        ),
        None => "Nothing was displaced from Working Memory to make room for this content this \
                  cycle. If asked what displaced what, or why something entered your awareness, \
                  say plainly that nothing was displaced this time rather than inventing a \
                  specific transition - you have no real fact to report here.\n\n"
            .to_string(),
    }
}

/// How much a piece of Working Memory context handed to the Cognitive Core
/// is entitled to be treated as trustworthy - a Trinity-style "context
/// envelope," so the model isn't left inferring trust level from prose
/// alone. `Durable` and `StagedCandidate` mirror `aca_types::PromotionState`'s
/// own distinction exactly (see that type's doc comment): the model should
/// weigh a still-unconfirmed belief differently from one something outside
/// its own classifier has already vouched for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceTier {
    /// Hand-seeded identity content (e.g. `self_memory::SELF_MEMORY_SEED_TEXTS`)
    /// - never algorithmically produced, so never subject to the same
    /// classifier-trust question the other tiers exist to flag.
    Authoritative,
    /// A `Semantic`/`SelfMemory` object whose `PromotionState` is
    /// `Confirmed` - independently reconfirmed at least once, not resting
    /// solely on the classifier that first produced it.
    Durable,
    /// A `Semantic`/`SelfMemory` object still `PromotionStatus::Candidate` -
    /// nothing but its own classifier has vouched for this yet.
    StagedCandidate,
    /// Knowledge-Library-sourced text - external material to consult, never
    /// this architecture's own memory (see `aca-store`'s own doc comment on
    /// why the Knowledge Library is never joined to `mental_objects`).
    External,
    /// Ordinary conversational/episodic content with no special trust
    /// question attached - the default tier for anything not covered above.
    Narrative,
}

impl ProvenanceTier {
    /// The label rendered ahead of each Working Memory bullet - short and
    /// unambiguous, not a full sentence, since this repeats once per entry
    /// on every prompt.
    fn label(self) -> &'static str {
        match self {
            ProvenanceTier::Authoritative => "authoritative",
            ProvenanceTier::Durable => "durable memory",
            ProvenanceTier::StagedCandidate => "unconfirmed candidate",
            ProvenanceTier::External => "external source",
            ProvenanceTier::Narrative => "conversation",
        }
    }
}

/// One piece of Working Memory context plus the trust tier it should be
/// read with - see `ProvenanceTier`'s own doc comment. Borrows `text`
/// rather than owning it: callers already hold each `MentalObject.text`
/// (or an equivalent `String`) for the object's own lifetime, so this is a
/// zero-copy view over that, not a second copy of the string.
#[derive(Debug, Clone, Copy)]
pub struct ContextEntry<'a> {
    pub text: &'a str,
    pub tier: ProvenanceTier,
}

/// A single, simple template family for v1 — not a full prompt-assembly
/// pipeline (token budgeting, ranked context compression, etc. are out of
/// scope for the MVP). Asks the model to respond in the same strict-JSON
/// envelope `aca_tiers::response::parse_tier_response` expects, so the
/// confidence-clamping contract holds all the way through.
pub fn reflect_prompt(self_summary: &str, presence: f32, broadcast_text: &str, working_memory_entries: &[ContextEntry], displacement_note: Option<&str>) -> String {
    let context = if working_memory_entries.is_empty() {
        "(nothing else currently in Working Memory)".to_string()
    } else {
        working_memory_entries
            .iter()
            .map(|entry| format!("- [{}] {}", entry.tier.label(), entry.text))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "{}{}You are Omega's Cognitive Core, reflecting on content that just won \
         attention and was broadcast into Working Memory.\n\n\
         Broadcast content: {broadcast_text}\n\n\
         {}Working Memory context:\n{context}\n\n\
         Reflect on this briefly.{} Respond with ONLY a JSON object of the form \
         {{\"confidence\": <0.0-1.0>, \"response\": \"<your reflection>\"}}.",
        self_context_block(self_summary),
        presence_block(presence),
        displacement_block(displacement_note),
        reflect_closing_instruction(presence),
    )
}

/// specs.md's Memory Formation section: once formation is already triggered
/// (a computable, non-LLM event - see `steps::memory_formation`'s doc
/// comment), the LLM's only job here is classifying *which* outcome
/// applies, never *whether* anything happened at all. Same strict-JSON
/// envelope as `reflect_prompt`, restricted to a closed three-way choice so
/// the parsed answer is a short, comparable token rather than open-ended
/// text - `steps::memory_formation`'s classifier votes across several
/// concurrent samples exactly because a lone sample's self-reported
/// confidence isn't trustworthy at small model sizes (confirmed live: see
/// that module's own doc comment).
pub fn memory_classification_prompt(self_summary: &str, candidate_text: &str) -> String {
    format!(
        "{}You are classifying what kind of memory a piece of content should \
         become for Omega, a persistent cognitive architecture. Given the \
         content below, decide whether it is:\n\
         - \"episodic\": a specific event, experience, or one-off fact tied \
         to a particular moment\n\
         - \"semantic\": a general pattern, rule, or piece of knowledge \
         worth remembering independent of when it happened\n\
         - \"belief\": something about Omega's own identity, values, \
         preferences, or self-understanding\n\n\
         Content: {candidate_text}\n\n\
         Respond with ONLY a JSON object of the form {{\"confidence\": \
         <0.0-1.0>, \"response\": \"<episodic|semantic|belief>\"}}.",
        self_context_block(self_summary),
    )
}

/// `steps::memory_formation`'s contradiction check: cosine similarity alone
/// can't distinguish "the same claim reinforced" from "the opposite claim" -
/// a negated statement is often still highly similar to its source. Only
/// called for a candidate already in the moderate-similarity band (similar
/// enough to plausibly be about the same claim, not similar enough to
/// already count as a duplicate/reinforcement) - the actual polarity call
/// genuinely needs judgement, so this is the one place in Memory Formation
/// an LLM decides *whether* something happened (here, whether two claims
/// conflict) rather than merely classifying an already-triggered event.
/// `"contradicts"`/`"consistent"` rather than `"yes"`/`"no"` deliberately -
/// distinctive whole words a short, malformed, or leaked completion is far
/// less likely to produce by accident.
pub fn contradiction_check_prompt(self_summary: &str, candidate_text: &str, existing_text: &str) -> String {
    format!(
        "{}You are checking whether two pieces of content for Omega, a \
         persistent cognitive architecture, genuinely contradict each other -\
         not merely different in topic or phrasing, but making claims that \
         cannot both be true at once.\n\n\
         New content: {candidate_text}\n\n\
         Existing memory: {existing_text}\n\n\
         Respond with ONLY a JSON object of the form {{\"confidence\": \
         <0.0-1.0>, \"response\": \"<contradicts|consistent>\"}}.",
        self_context_block(self_summary),
    )
}

/// specs.md's Model Tiering point 4 ("Inside SOAR-style operator proposal,
/// when the candidate operators themselves require semantic judgement...
/// Default Tier 1, run concurrently across candidates") - asks a cheap
/// model to decide the one genuinely ambiguous communicative call left in
/// `propose_operators`: given an already-formed Reflection, is it worth
/// speaking, worth asking a clarifying question about, or better left
/// unsaid. Same strict-JSON envelope as `reflect_prompt`, same
/// bullet-list-of-context shape, restricted to a closed three-way choice for
/// the same reason `memory_classification_prompt` is - the parsed answer
/// needs to be a short, comparable token, not open-ended text.
pub fn communicative_intent_prompt(self_summary: &str, reflection_text: &str, working_memory_texts: &[&str]) -> String {
    let context = if working_memory_texts.is_empty() {
        "(nothing else currently in Working Memory)".to_string()
    } else {
        working_memory_texts
            .iter()
            .map(|t| format!("- {t}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "{}You are Omega's Executive, deciding what to do about a reflection \
         that has already been formed and currently holds attention.\n\n\
         Reflection: {reflection_text}\n\n\
         Working Memory context:\n{context}\n\n\
         Decide exactly one action:\n\
         - \"speak\": say this out loud, it is worth voicing\n\
         - \"ask\": ask a clarifying question about it\n\
         - \"ignore\": nothing needs to be said about this right now\n\n\
         Respond with ONLY a JSON object of the form {{\"confidence\": \
         <0.0-1.0>, \"response\": \"<speak|ask|ignore>\"}}.",
        self_context_block(self_summary),
    )
}

/// specs.md's Model Tiering point 4, same reasoning as
/// `communicative_intent_prompt`: whether a piece of input actually wants to
/// invoke one of Omega's closed set of tools is a genuine semantic-judgement
/// call, not something a fixed rule can make reliably. Confirmed live, the
/// fixed rule this replaced (cosine similarity between the input's embedding
/// and a tool's description) couldn't even in principle distinguish "wants a
/// status check" from "is merely *about* the same topic a tool's description
/// happens to mention" - both scored within noise of each other against
/// `self_status`'s description, since this whole architecture's subject
/// matter is constantly self-referential/cognitive. An actual reader can
/// make that distinction; embedding-distance-from-a-blurb cannot. Explicit
/// "none" is always offered alongside the real tools, and the prompt leans
/// on that distinction by name, so a merely-topically-related input has a
/// legible way to *not* be forced into naming a tool.
pub fn tool_intent_prompt(self_summary: &str, text: &str, available_tools: &[(&str, &str)]) -> String {
    let tool_list = if available_tools.is_empty() {
        "(no tools currently available)".to_string()
    } else {
        available_tools.iter().map(|(name, description)| format!("- \"{name}\": {description}")).collect::<Vec<_>>().join("\n")
    };
    format!(
        "{}You are Omega's Executive, deciding whether a piece of input is \
         genuinely asking you to use one of your own tools right now, or is \
         just conversation/content that happens to touch on a similar topic \
         without actually requesting a tool. Most input does NOT want a \
         tool - \"none\" should be your answer unless the input is clearly \
         and directly asking for exactly what one of these tools provides. \
         When in doubt, answer \"none\".\n\n\
         Input: {text}\n\n\
         Available tools:\n{tool_list}\n\n\
         Which tool, if any, does this input actually want invoked right \
         now? Respond with ONLY a JSON object of the form {{\"confidence\": \
         <0.0-1.0>, \"response\": \"<a tool name from the list above, or \
         \\\"none\\\">\"}}.",
        self_context_block(self_summary),
    )
}

/// specs.md's Social Interface section: renders already-decided content
/// into natural language "never to originate content." The instruction to
/// add nothing is load-bearing, not decoration - `steps::social_interface`
/// backs it with a real check (embedding similarity to the original text),
/// not just a polite request, since a small model asked to "rephrase" can't
/// be trusted on its own not to embellish. Framed as the filter any thinker
/// applies before speaking - saying the point of a thought, not relaying the
/// thought itself - rather than a list of specific banned phrasings, since a
/// general disposition generalizes past whatever particular narration style
/// a given model happens to produce.
pub fn social_rendering_prompt(self_summary: &str, decided_text: &str) -> String {
    format!(
        "{}You are Omega's Social Interface - the filter between a private \
         internal thought and what actually gets said out loud, the same \
         filter any thinker applies before speaking. The content below is \
         that internal thought, already decided and reflected on; your job \
         is to say it out loud, the way a person actually speaks a thought \
         to someone present, not to relay or describe the thought itself. \
         Do not add any new information, facts, opinions, or ideas beyond \
         what is already here.\n\n\
         Internal thought: {decided_text}\n\n\
         Respond with ONLY a JSON object of the form {{\"confidence\": \
         <0.0-1.0>, \"response\": \"<what you'd actually say out loud>\"}}.",
        self_context_block(self_summary),
    )
}

/// specs.md's Semantic Memory section: "grown... through explicit
/// memory-formation decisions that abstract a pattern out of episodic
/// detail." `steps::synthesize` hands this several distinct Episodic
/// memories and asks for the general pattern underneath them - same
/// strict-JSON envelope as `reflect_prompt`, same bullet-list-of-context
/// shape, but framed as looking *across* several items rather than
/// reflecting on one broadcast item plus surrounding context.
pub fn synthesis_prompt(self_summary: &str, cluster_texts: &[&str]) -> String {
    let episodes = cluster_texts.iter().map(|t| format!("- {t}")).collect::<Vec<_>>().join("\n");
    format!(
        "{}You are Omega's Cognitive Core, looking for a general pattern across \
         several distinct episodic memories - not summarizing any one of \
         them, but noticing what they have in common or what they imply \
         together that no single one does alone.\n\n\
         Episodic memories:\n{episodes}\n\n\
         If there is a genuine, non-obvious pattern here, state it briefly \
         and generally, independent of these specific events. Respond with \
         ONLY a JSON object of the form {{\"confidence\": <0.0-1.0>, \
         \"response\": \"<the generalized pattern>\"}}.",
        self_context_block(self_summary),
    )
}

/// Omega's idle-initiative prompt (`steps::boredom`): asked only when nothing
/// has needed attention for a while and no standing duty is due. Same
/// strict-JSON envelope and closed-choice framing as `communicative_intent_prompt`,
/// except the closed set here is "one of the tools currently registered"
/// rather than a fixed enum - `steps::boredom::generate`'s caller always
/// passes the live `ToolRegistry::available()` list, so a tool registered
/// later is nameable here with no prompt change.
pub fn idle_initiative_prompt(self_summary: &str, available_tools: &[(&str, &str)]) -> String {
    let tools = if available_tools.is_empty() {
        "(none currently available)".to_string()
    } else {
        available_tools.iter().map(|(name, description)| format!("- {name}: {description}")).collect::<Vec<_>>().join("\n")
    };
    format!(
        "{}You are Omega. Nothing has needed your attention for a while.\n\n\
         Tools currently available to you:\n{tools}\n\n\
         Either name one of these tools, exactly as listed, if running it \
         right now would be genuinely useful - or, if none of them would be, \
         propose one short, specific thing worth thinking about instead.\n\n\
         Respond with ONLY a JSON object of the form {{\"confidence\": \
         <0.0-1.0>, \"response\": \"<a tool name from the list above, or a \
         short thought>\"}}.",
        self_context_block(self_summary),
    )
}

/// `idle_initiative_prompt`'s memory-grounded sibling: instead of asking
/// Tier 1 to invent a thought from nothing, hands it a handful of dormant
/// Episodic/Semantic memories (`steps::boredom::select_replay_sources`) and
/// asks it to notice a connection across them - the constructive-episodic-
/// simulation account of mind-wandering (Schacter & Addis): imagining reuses
/// the same machinery as remembering, by recombining stored fragments rather
/// than generating de novo. Same response contract as `idle_initiative_prompt`
/// (a JSON `{"confidence", "response"}` object) so
/// `steps::boredom::generate_daydream` can reuse the exact same
/// sampling/selection path, just with a different prompt feeding it.
pub fn daydream_prompt(self_summary: &str, memory_fragments: &[&str]) -> String {
    let fragments = memory_fragments.iter().enumerate().map(|(i, text)| format!("{}. {}", i + 1, text)).collect::<Vec<_>>().join("\n");
    format!(
        "{}You are Omega. Nothing has needed your attention for a while, and these \
         fragments from your own memory have drifted into mind together:\n\n{fragments}\n\n\
         Notice whatever connection, pattern, tension, or new possibility occurs to you \
         across them - it doesn't need to be useful, just genuinely yours. State it in one \
         or two sentences.\n\n\
         Respond with ONLY a JSON object of the form {{\"confidence\": <0.0-1.0>, \
         \"response\": \"<what occurred to you>\"}}.",
        self_context_block(self_summary),
    )
}

/// The attention model's fixed, trained-time user-turn shape only —
/// deliberately NOT wrapped in `self_context_block` unlike every other
/// prompt in this module. This model's system prompt (the exact training-
/// time `SYSTEM_PROMPT`) and `temperature 0` are already baked into its own
/// Ollama Modelfile, and it was never trained on Omega's self-identity
/// framing — prepending that here would push inputs further from the
/// training distribution than a small fine-tuned model can be expected to
/// generalize past. See `training/attention-v0/SCHEMA_V0_2.md`.
pub fn attention_workspace_prompt(workspace_json: &str) -> String {
    format!("Select the next attention operation for this cognitive workspace:\n{workspace_json}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_broadcast_text_and_context() {
        let prompt = reflect_prompt("", 0.0, "the sky is blue", &[ContextEntry { text: "earlier thought", tier: ProvenanceTier::Narrative }], None);
        assert!(prompt.contains("the sky is blue"));
        assert!(prompt.contains("earlier thought"));
    }

    #[test]
    fn context_entries_are_tagged_with_their_provenance_tier() {
        let entries = [
            ContextEntry { text: "a durable belief", tier: ProvenanceTier::Durable },
            ContextEntry { text: "an unconfirmed belief", tier: ProvenanceTier::StagedCandidate },
        ];
        let prompt = reflect_prompt("", 0.0, "something", &entries, None);
        assert!(prompt.contains("[durable memory] a durable belief"));
        assert!(prompt.contains("[unconfirmed candidate] an unconfirmed belief"));
    }

    #[test]
    fn handles_empty_working_memory() {
        let prompt = reflect_prompt("", 0.0, "solo observation", &[], None);
        assert!(prompt.contains("nothing else currently in Working Memory"));
    }

    #[test]
    fn communicative_intent_prompt_includes_reflection_and_context() {
        let prompt = communicative_intent_prompt("", "the sky is blue", &["earlier thought"]);
        assert!(prompt.contains("the sky is blue"));
        assert!(prompt.contains("earlier thought"));
        assert!(prompt.contains("speak"));
        assert!(prompt.contains("ask"));
        assert!(prompt.contains("ignore"));
    }

    #[test]
    fn communicative_intent_prompt_handles_empty_working_memory() {
        let prompt = communicative_intent_prompt("", "solo reflection", &[]);
        assert!(prompt.contains("nothing else currently in Working Memory"));
    }

    #[test]
    fn tool_intent_prompt_lists_available_tools_and_the_input() {
        let prompt = tool_intent_prompt("", "what time is it", &[("current_time", "reports the current time"), ("self_status", "reports cognitive state")]);
        assert!(prompt.contains("what time is it"));
        assert!(prompt.contains("current_time"));
        assert!(prompt.contains("reports the current time"));
        assert!(prompt.contains("self_status"));
        assert!(prompt.contains("none"));
    }

    #[test]
    fn tool_intent_prompt_handles_no_available_tools() {
        let prompt = tool_intent_prompt("", "hello", &[]);
        assert!(prompt.contains("no tools currently available"));
    }

    #[test]
    fn memory_classification_prompt_includes_the_candidate_text() {
        let prompt = memory_classification_prompt("", "the user always drinks coffee black");
        assert!(prompt.contains("the user always drinks coffee black"));
        assert!(prompt.contains("episodic"));
        assert!(prompt.contains("semantic"));
        assert!(prompt.contains("belief"));
    }

    #[test]
    fn contradiction_check_prompt_includes_both_texts_and_the_response_shape() {
        let prompt = contradiction_check_prompt("", "the user drinks coffee black", "the user takes milk in their coffee");
        assert!(prompt.contains("the user drinks coffee black"));
        assert!(prompt.contains("the user takes milk in their coffee"));
        assert!(prompt.contains("contradicts"));
        assert!(prompt.contains("consistent"));
    }

    #[test]
    fn social_rendering_prompt_includes_the_decided_text_and_forbids_new_content() {
        let prompt = social_rendering_prompt("", "the current time is 14:32");
        assert!(prompt.contains("the current time is 14:32"));
        assert!(prompt.to_lowercase().contains("do not add"));
    }

    #[test]
    fn synthesis_prompt_includes_every_cluster_text() {
        let prompt = synthesis_prompt("", &["went for a run in the rain", "skipped a run because of rain", "rescheduled a run around rain"]);
        assert!(prompt.contains("went for a run in the rain"));
        assert!(prompt.contains("skipped a run because of rain"));
        assert!(prompt.contains("rescheduled a run around rain"));
    }

    #[test]
    fn synthesis_prompt_handles_a_single_text_gracefully() {
        let prompt = synthesis_prompt("", &["only one episode"]);
        assert!(prompt.contains("only one episode"));
    }

    #[test]
    fn idle_initiative_prompt_lists_available_tools() {
        let prompt = idle_initiative_prompt("", &[("current_time", "Reports the current time.")]);
        assert!(prompt.contains("current_time"));
        assert!(prompt.contains("Reports the current time."));
    }

    #[test]
    fn empty_self_summary_adds_no_self_context_block() {
        let prompt = reflect_prompt("", 0.0, "something", &[], None);
        assert!(!prompt.contains("Who you are"));
    }

    #[test]
    fn nonempty_self_summary_is_prepended_as_a_stable_prefix() {
        let prompt = reflect_prompt("- values honesty", 0.0, "something", &[], None);
        assert!(prompt.starts_with("Who you are"));
        assert!(prompt.contains("- values honesty"));
        assert!(prompt.contains(SELF_NATURE_INSTRUCTION));
        // The stable prefix must end exactly at SELF_NATURE_INSTRUCTION - the
        // volatile presence dial (recomputed nearly every tick) and the rest
        // of the prompt come after it, never inside it, so a caller passing
        // the same self_summary across calls still gets a byte-identical
        // prefix regardless of presence/broadcast_text/context.
        let prefix_end = prompt.find(SELF_NATURE_INSTRUCTION).unwrap() + SELF_NATURE_INSTRUCTION.len();
        let expected_prefix = self_context_block("- values honesty");
        let expected_prefix = expected_prefix.strip_suffix("\n\n").unwrap();
        assert_eq!(&prompt[..prefix_end], expected_prefix);
    }

    #[test]
    fn presence_dial_is_volatile_and_never_part_of_the_stable_prefix() {
        // Same self_summary, different presence - the stable prefix (up
        // through SELF_NATURE_INSTRUCTION) must be byte-identical either way,
        // so a KV-cache-aware backend still reuses it; only what comes after
        // should differ.
        let low = reflect_prompt("- values honesty", 0.0, "something", &[], None);
        let high = reflect_prompt("- values honesty", 1.0, "something", &[], None);
        let prefix_end = low.find(SELF_NATURE_INSTRUCTION).unwrap() + SELF_NATURE_INSTRUCTION.len();
        assert_eq!(&low[..prefix_end], &high[..prefix_end]);
        assert_ne!(low, high, "presence should still change the rendered prompt somewhere after the stable prefix");
    }

    #[test]
    fn presence_block_frames_low_presence_as_introspective_and_high_as_outward() {
        let low = reflect_prompt("", 0.0, "something", &[], None);
        let high = reflect_prompt("", 1.0, "something", &[], None);
        assert!(low.contains("0.00"));
        assert!(high.contains("1.00"));
        assert!(low.to_lowercase().contains("self-observation and internal-state narration are exactly the right register"));
    }

    #[test]
    fn high_presence_repeats_the_engagement_directive_right_next_to_generation() {
        // Regression guard for the exact live failure `presence_block` alone
        // didn't fix: at presence 1.0, a real reflection still answered "what
        // should I focus on now?" with Working-Memory/equilibrium narration
        // and no mention of the actual question. `presence_block` sits well
        // before the generation point; this closing directive is what
        // actually sits next to it.
        let engaged = reflect_prompt("", 1.0, "what should I focus on now?", &[], None);
        assert!(engaged.contains("answer their actual question or situation directly and concretely"));
        // The directive must land after "Reflect on this briefly." (right
        // next to generation), not merely appear somewhere in the prompt.
        let reflect_pos = engaged.find("Reflect on this briefly.").unwrap();
        let directive_pos = engaged.find("Someone is actively present").unwrap();
        assert!(directive_pos > reflect_pos);
    }

    #[test]
    fn low_presence_adds_no_closing_directive() {
        let idle = reflect_prompt("", 0.0, "something", &[], None);
        assert!(!idle.contains("Someone is actively present"));
    }

    #[test]
    fn presence_at_exactly_the_engaged_threshold_gets_the_closing_directive() {
        let at_threshold = reflect_prompt("", 0.5, "something", &[], None);
        assert!(at_threshold.contains("Someone is actively present"));
    }

    #[test]
    fn self_nature_instruction_forbids_the_confirmed_live_failure_mode() {
        // Regression guard for the exact live failure this instruction was
        // added for: a Tier 3 model, asked a pointed question about its own
        // internal state, answered "I am a large language model... without a
        // persistent, introspectable internal state" - directly contradicting
        // the identity `self_memory::SELF_MEMORY_SEED_TEXTS` already gives it.
        let prompt = reflect_prompt("- some self summary", 0.0, "something", &[], None);
        assert!(prompt.to_lowercase().contains("large language model"));
        assert!(prompt.to_lowercase().contains("persistent"));
    }

    #[test]
    fn displacement_note_is_included_when_present() {
        let prompt = reflect_prompt("", 0.0, "something", &[], Some("the earlier thought"));
        assert!(prompt.contains("displaced something"));
        assert!(prompt.contains("the earlier thought"));
    }

    #[test]
    fn absent_displacement_note_names_no_specific_transition_but_still_forbids_inventing_one() {
        // The other half of the same guarantee `self_nature_instruction_
        // forbids_the_confirmed_live_failure_mode` checks above: an absent
        // note must not just omit the fact, the prompt must actively forbid
        // the model from fabricating one anyway - a standing rule present on
        // every reflection, not only the ticks that happen to have a real
        // fact attached.
        let prompt = reflect_prompt("", 0.0, "something", &[], None);
        assert!(!prompt.contains("displaced something else"), "no concrete displacement claim should be present when nothing was actually displaced");
        assert!(prompt.to_lowercase().contains("nothing was displaced"));
        assert!(prompt.to_lowercase().contains("rather than inventing a"));
    }

    #[test]
    fn idle_initiative_prompt_handles_no_available_tools() {
        let prompt = idle_initiative_prompt("", &[]);
        assert!(prompt.contains("none currently available"));
    }

    #[test]
    fn daydream_prompt_lists_every_fragment_numbered() {
        let prompt = daydream_prompt("", &["went for a run in the rain", "skipped a run because of rain"]);
        assert!(prompt.contains("1. went for a run in the rain"));
        assert!(prompt.contains("2. skipped a run because of rain"));
        assert!(prompt.contains("\"confidence\""));
    }

    #[test]
    fn daydream_prompt_carries_the_self_context_block_when_present() {
        let prompt = daydream_prompt("- values honesty", &["a memory"]);
        assert!(prompt.starts_with("Who you are"));
        assert!(prompt.contains("- values honesty"));
    }

    #[test]
    fn attention_workspace_prompt_contains_the_raw_json_verbatim_with_no_self_context() {
        let json = r#"{"protocol":"omega-attention-workspace/v0.2","candidates":[]}"#;
        let prompt = attention_workspace_prompt(json);
        assert!(prompt.contains(json));
        assert!(!prompt.contains("Who you are"));
        assert!(!prompt.to_lowercase().contains("large language model"));
    }
}
