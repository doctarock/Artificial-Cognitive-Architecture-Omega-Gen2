use aca_graph::Graph;
use aca_store::KnowledgeLibraryStore;
use aca_tiers::{ChatClient, DivergentPool, EmbeddingClient, TierPool};
use aca_util::{Clock, EpochMillis};

use super::confidence_revision::ConfidenceRevisionConfig;
use super::executive::{propose_tool_intent, Operator, OperatorProposal};
use super::knowledge_library::{self, KnowledgeLibraryConfig, KnowledgeLibraryResult};
use super::memory_formation::{self, MemoryFormationConfig, MemoryFormationOutcome};
use super::social_interface::{self, render_speech};
use super::tools::ToolRegistry;
use crate::cognitive_core;
use crate::config::TemperatureConfig;
use crate::prompt_templates::ContextEntry;

/// Why an operator resolved to no visible action - distinguishes a
/// deliberate decision ("nothing worth doing") from a genuine failure
/// ("tried, broke") so a caller can tell them apart instead of treating
/// every "nothing happened" identically. This is exactly the gap that made
/// a real reflection failure indistinguishable from an ordinary Ignore in
/// the live event feed - confirmed live: a Tier 3 call for a genuine user
/// question returned an empty completion, `cognitive_core::reflect`
/// propagated that as an `Err`, and the resulting silence looked, from the
/// console, identical to Omega having simply decided not to respond. See
/// `loop_actor::apply_operator`'s `emit_event` call, which uses this to
/// publish `ReflectionFailed` at `CycleEventKind::Error` (always visible to
/// the live UI) rather than the routine `Normal` severity every other
/// reason still gets.
#[derive(Debug)]
pub enum SilentReason {
    /// The Executive's own communicative-intent judgement (or the terminal
    /// "nothing left to do" branch) decided nothing should be said or done -
    /// the ordinary, most common case.
    Ignored,
    /// `proposal.target_id` was no longer present in the graph by the time
    /// Act ran (decayed/discarded between proposal and execution) - not a
    /// failure of anything, just a stale reference that should never panic
    /// the cycle.
    StaleTarget,
    /// `cognitive_core::reflect` itself returned `Err` - an empty completion
    /// or a transport failure from whichever tier actually answered (see
    /// that function's own doc comment for the two concrete ways this
    /// happens). Carries the error's own `Display` text so a viewer sees
    /// *why*, not just *that*.
    ReflectionFailed { error: String },
    /// `propose_tool_intent` (or a self-issued `requested_tool` tag)
    /// resolved to no reachable tool.
    NoToolMatched,
}

/// What Step 8 - Act actually produced. One variant per operator family;
/// `Silent` covers every case where nothing was said or done - see
/// `SilentReason` for what actually distinguishes them.
#[derive(Debug)]
pub enum ActOutcome {
    Spoke { text: String, render_path: SpeechRenderPath },
    Remembered { outcome: MemoryFormationOutcome },
    Reflected { reflection_id: aca_types::MentalObjectId },
    ConsultedKnowledgeLibrary { result: KnowledgeLibraryResult },
    /// `result` is `Err` when the matched tool ran but failed - still a
    /// real, reportable outcome, not silence (see `steps::tools::Tool::invoke`'s
    /// contract). No matched/allowed tool at all resolves to `Silent`
    /// instead, same as any other operator that found nothing to act on.
    Acted { tool: &'static str, result: Result<String, String> },
    Silent { reason: SilentReason },
}

/// Which path the Social Interface used to turn selected content into
/// speech. Exposed in Act events so optimization runs can prove how often
/// speech stayed on the fast path versus paying for Tier 1 rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechRenderPath {
    Verbatim,
    CompiledSkill,
    CompiledProcedure,
    InnateReflex,
    CuratedAnswer,
    Tier1,
}

/// Step 8 - Act: dispatches the Executive's selected operator to whichever
/// subsystem handles it. `Speak`/`Ask` render already-decided content via
/// the Social Interface; `Remember` runs Memory Formation; `ContinueReflecting`/
/// `Plan` invoke the Cognitive Core; `ConsultKnowledgeLibrary` looks outward;
/// `Act` invokes a tool from the closed, risk-tiered `ToolRegistry` (see
/// `steps::tools` - arbitrary tool/command execution is still out of
/// scope, only ever whatever's actually registered); `Ignore` resolves to
/// silence.
#[allow(clippy::too_many_arguments)]
pub async fn act(
    graph: &mut Graph,
    proposal: &OperatorProposal,
    tier1_pool: &DivergentPool,
    tier2_pool: &DivergentPool,
    tier3_pool: &TierPool,
    tier3_client: &dyn ChatClient,
    embedding_client: &dyn EmbeddingClient,
    kl_store: &dyn KnowledgeLibraryStore,
    kl_config: &KnowledgeLibraryConfig,
    tool_registry: &ToolRegistry,
    working_memory_entries: &[ContextEntry<'_>],
    self_summary: &str,
    presence: f32,
    memory_config: &MemoryFormationConfig,
    confidence_revision_config: &ConfidenceRevisionConfig,
    now: EpochMillis,
    clock: &dyn Clock,
    decay_d: f32,
    tier3_hedge_delay: std::time::Duration,
    temperature: &TemperatureConfig,
    // Forwarded verbatim to `cognitive_core::reflect` - see that
    // parameter's own doc comment. The caller (`loop_actor`) is responsible
    // for only passing `Some` when `proposal.target_id` is itself this
    // tick's real, counterfactually-confirmed displacement entrant.
    displacement_note: Option<&str>,
) -> ActOutcome {
    match proposal.operator {
        Operator::Speak | Operator::Ask => match graph.get(&proposal.target_id) {
            Some(object) => {
                let compiled_procedure = object.data.get("compiled_response").and_then(|value| value.as_str()).map(str::to_owned);
                let compiled_question = object.data.get("compiled_question").and_then(|value| value.as_str()).map(str::to_owned);
                let innate_reflex = object.data.get("innate_response").and_then(|value| value.as_str()).map(str::to_owned);
                let curated_answer = object.data.get("curated_answer").and_then(|value| value.as_str()).map(str::to_owned);
                let decided_text = curated_answer.clone().or_else(|| compiled_procedure.clone()).or_else(|| compiled_question.clone()).or_else(|| innate_reflex.clone()).unwrap_or_else(|| object.text.clone());
                let is_skill_candidate = decided_text.split_whitespace().count() <= social_interface::MAX_WORDS_FOR_SKILL_COMPILATION;
                // ACT-R-style skill compilation (see `steps::social_interface::
                // compiled_render`'s doc comment): a short, routine utterance
                // that's rendered the exact same way several times running
                // gets spoken straight from the compiled template - Tier 0,
                // no Tier 1 call at all - rather than re-earning the same
                // rendering every single time it recurs.
                let compiled = if compiled_procedure.is_none() && compiled_question.is_none() && innate_reflex.is_none() && curated_answer.is_none() && is_skill_candidate { social_interface::compiled_render(graph, &decided_text) } else { None };
                let (text, render_path) = match compiled {
                    Some(compiled_text) => (compiled_text, SpeechRenderPath::CompiledSkill),
                    None if curated_answer.is_some() => (decided_text, SpeechRenderPath::CuratedAnswer),
                    None if compiled_procedure.is_some() || compiled_question.is_some() => (decided_text, SpeechRenderPath::CompiledProcedure),
                    None if innate_reflex.is_some() => (decided_text, SpeechRenderPath::InnateReflex),
                    None => {
                        let will_skip_tier1 = decided_text.split_whitespace().count() <= social_interface::MAX_WORDS_FOR_REFLEXIVE_SPEECH || tier1_pool.is_empty();
                        let rendered = render_speech(object, tier1_pool, embedding_client, self_summary, temperature.social_rendering).await;
                        if is_skill_candidate {
                            social_interface::record_render(graph, &decided_text, &rendered, now, decay_d);
                        }
                        let render_path = if will_skip_tier1 { SpeechRenderPath::Verbatim } else { SpeechRenderPath::Tier1 };
                        (rendered, render_path)
                    }
                };
                // Mark this object as already spoken so propose_operators
                // doesn't re-propose Speak/Ask for it every subsequent tick
                // it remains in Working Memory - without this, an object
                // with no fresh input to unseat it gets spoken once per
                // tick indefinitely, which is not coherent communicative
                // behavior (observed directly in a live smoke test).
                if let Some(object) = graph.get_mut(&proposal.target_id) {
                    object.produced_by_operator = Some(format!("{:?}", proposal.operator));
                }
                ActOutcome::Spoke { text, render_path }
            }
            None => ActOutcome::Silent { reason: SilentReason::StaleTarget },
        },

        Operator::Remember => match graph.get(&proposal.target_id).cloned() {
            Some(candidate) => {
                let target_id = candidate.id;
                let outcome = memory_formation::form_memory(graph, candidate, memory_config, confidence_revision_config, tier1_pool, tier2_pool, self_summary, now, temperature.memory_formation).await;
                // `NewEpisodic`/`SemanticUpdate`/`BeliefRevision` already
                // tag the *candidate itself* with the right role inside
                // `form_memory` (it's the object that gets inserted). Only
                // `Reinforced` needs tagging here: the candidate wasn't
                // touched by `form_memory` at all (a *different*, more-
                // similar existing object was referenced instead), which
                // would otherwise leave Remember proposed for it forever.
                if let MemoryFormationOutcome::Reinforced { .. } = outcome {
                    if let Some(object) = graph.get_mut(&target_id) {
                        if !object.memory_roles.contains(&aca_types::MemoryRole::Episodic) {
                            object.memory_roles.push(aca_types::MemoryRole::Episodic);
                        }
                    }
                }
                ActOutcome::Remembered { outcome }
            }
            None => ActOutcome::Silent { reason: SilentReason::StaleTarget },
        },

        Operator::ContinueReflecting | Operator::Plan => match graph.get(&proposal.target_id).cloned() {
            Some(broadcast_object) => {
                match cognitive_core::reflect(
                    tier1_pool,
                    tier2_pool,
                    tier3_pool,
                    tier3_client,
                    embedding_client,
                    &broadcast_object,
                    working_memory_entries,
                    self_summary,
                    presence,
                    clock,
                    decay_d,
                    tier3_hedge_delay,
                    temperature.reflection,
                    displacement_note,
                )
                .await
                {
                    Ok(mut output) => {
                        // A `Plan` reflection carries its own provenance
                        // tag (`data.source = "plan"`, `data.for_intention`)
                        // before it ever enters the graph - this is what
                        // lets `steps::executive::propose_operators`'s
                        // `is_plan_reflection` keep it out of ordinary
                        // Speak/Ask/Consult consideration, and what lets
                        // `steps::agenda::revise_agenda` find and fold its
                        // content back into the parent Intention later this
                        // same tick, rather than it becoming a disconnected
                        // side-chain with its own untracked bookkeeping. A
                        // plain `ContinueReflecting` reflection is left
                        // exactly as before (no `data` tag at all).
                        if proposal.operator == Operator::Plan {
                            output.reflection.data = serde_json::json!({"source": "plan", "for_intention": proposal.target_id.to_string()});
                        }
                        let reflection_id = output.reflection.id;
                        graph.insert(output.reflection);
                        ActOutcome::Reflected { reflection_id }
                    }
                    Err(err) => {
                        // A failed reflection (empty completion, transport
                        // error, malformed response) used to collapse to the
                        // exact same "silent" outcome as a deliberate Ignore -
                        // indistinguishable from working as intended.
                        // Confirmed live: a ~60s Tier 3 call for a heard
                        // "Hello Omega." produced no Reflection object at all
                        // and no error anywhere but this `tracing::warn!`,
                        // making it look like Omega had nothing to say rather
                        // than that reflection itself had failed.
                        // `SilentReason::ReflectionFailed` (below) is what
                        // actually fixes the visibility gap now - it reaches
                        // the live event feed at `CycleEventKind::Error`, not
                        // just this process's own tracing output (see
                        // `loop_actor::apply_operator`).
                        tracing::warn!(error = %err, target_id = %proposal.target_id, "reflection failed, target will not be spoken about");
                        // Same dedup discipline as every other operator arm
                        // (see `Operator::Act`'s `None` branch's own doc
                        // comment for the identical failure mode) - without
                        // tagging the target here too, a reflection that
                        // keeps failing (confirmed live: Tier 3 repeatedly
                        // returning an empty completion for the same
                        // target) gets ContinueReflecting re-proposed for
                        // it forever, occupying the Working Memory
                        // spotlight and starving every other operator -
                        // including Speak for anything else - on every
                        // subsequent tick it wins Coalition.
                        if let Some(object) = graph.get_mut(&proposal.target_id) {
                            object.produced_by_operator = Some(format!("{:?}", proposal.operator));
                        }
                        ActOutcome::Silent { reason: SilentReason::ReflectionFailed { error: err.to_string() } }
                    }
                }
            }
            None => ActOutcome::Silent { reason: SilentReason::StaleTarget },
        },

        Operator::ConsultKnowledgeLibrary => {
            let query = graph.get(&proposal.target_id).map(|o| o.text.clone()).unwrap_or_default();
            let result = knowledge_library::consult(&query, embedding_client, kl_store, kl_config).await;
            // Tag the target so propose_operators doesn't keep proposing
            // Consult for it every subsequent tick - same discipline as the
            // Speak/Ask arm above.
            if let Some(object) = graph.get_mut(&proposal.target_id) {
                object.produced_by_operator = Some(format!("{:?}", proposal.operator));
            }
            ActOutcome::ConsultedKnowledgeLibrary { result }
        }

        Operator::Act => {
            let object = graph.get(&proposal.target_id);
            let text = object.map(|o| o.text.clone()).unwrap_or_default();
            // A self-issued request (`steps::boredom`) names its tool
            // directly via provenance rather than needing the text to match
            // a trigger phrase - see `propose_operators`'s `requested_tool`
            // check, which is what routes this arm here in the first place
            // for that case. `propose_tool_intent` (real Tier 1 semantic
            // judgement, same mechanism `executive::propose_operators`
            // already used to decide Act was worth proposing at all - see
            // that function's fourth trigger) is the fallback for
            // everything else. Deliberately re-derived here rather than
            // carried forward from the proposal: this mirrors how
            // `ContinueReflecting` below independently re-invokes
            // `cognitive_core::reflect` rather than caching a decision, and
            // keeps proposal and execution asking the identical question of
            // the identical text - the actual bug this used to have wasn't
            // "two independent calls might rarely disagree," it was that
            // execution never asked the question its own proposal was based
            // on at all (confirmed live: an Act proposal justified purely by
            // embedding similarity found no tool at execution time, because
            // this arm only ever did keyword matching).
            let requested_tool = object.and_then(|o| o.data.get("requested_tool")).and_then(|v| v.as_str());
            let tool_name = match requested_tool {
                Some(name) => Some(name),
                None => propose_tool_intent(tier1_pool, &text, self_summary, &tool_registry.available(), temperature.tool_intent).await,
            };
            match tool_name.and_then(|name| tool_registry.find(name)) {
                Some(tool) => {
                    let result = tool.invoke().await;
                    // Same dedup discipline as every other operator arm:
                    // tag the target so propose_operators doesn't keep
                    // re-proposing Act for it every subsequent tick.
                    if let Some(object) = graph.get_mut(&proposal.target_id) {
                        object.produced_by_operator = Some(format!("{:?}", proposal.operator));
                    }
                    ActOutcome::Acted { tool: tool.name(), result }
                }
                None => {
                    // Same dedup discipline as the `Some(tool)` arm above -
                    // without this, an Act proposal that reached this arm
                    // but resolved to no tool keeps re-proposing Act for the
                    // exact same object every subsequent tick forever,
                    // holding the Working Memory spotlight and starving
                    // every other operator (confirmed live: 40+ consecutive
                    // `Act` cycles for one object, blocking real input from
                    // ever reaching Speak) - the identical oscillation the
                    // `Ignore` arm below was already written to prevent for
                    // Consult, just not extended to this arm.
                    if let Some(object) = graph.get_mut(&proposal.target_id) {
                        object.produced_by_operator = Some(format!("{:?}", proposal.operator));
                    }
                    ActOutcome::Silent { reason: SilentReason::NoToolMatched }
                }
            }
        }

        Operator::Ignore => {
            // Tag it, but *only* if nothing's tagged yet - same dedup
            // discipline as every other operator arm, needed so an object
            // whose communicative-intent vote resolved to Ignore (see
            // `propose_communicative_intent`) doesn't get re-proposed - and
            // re-sample Tier 1 - every subsequent tick. Must not overwrite
            // an existing tag though: Ignore is also selected as the
            // ordinary terminal state for an object that's already been
            // Spoken/Asked/Consulted/Acted on and has nothing left to do
            // (see `propose_operators`'s "nothing left to do" branch) -
            // clobbering that tag back to "Ignore" would erase the
            // provenance `already_consulted`/`already_acted` depend on,
            // making them false again and re-triggering Consult/Act every
            // time this steady state is re-selected (confirmed live: this
            // exact oscillation, Consult/Ignore/Consult/Ignore forever).
            if let Some(object) = graph.get_mut(&proposal.target_id) {
                if object.produced_by_operator.is_none() {
                    object.produced_by_operator = Some(format!("{:?}", proposal.operator));
                }
            }
            ActOutcome::Silent { reason: SilentReason::Ignored }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tools;
    use aca_tiers::{GenerateRequest, TierError, TierResponse};
    use aca_types::{MentalObject, MentalObjectId, Tier};
    use aca_util::ManualClock;
    use async_trait::async_trait;
    use std::time::Duration;

    fn clock() -> ManualClock {
        ManualClock::new(EpochMillis(1_000))
    }

    fn embedding_client() -> aca_tiers::testing::FakeEmbeddingClient {
        aca_tiers::testing::FakeEmbeddingClient::default()
    }

    fn kl_store() -> aca_store::SqliteStore {
        aca_store::SqliteStore::open_in_memory().unwrap()
    }

    fn empty_tier1() -> DivergentPool {
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![])
    }

    fn empty_tier2() -> DivergentPool {
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![])
    }

    struct FixedChatClient {
        raw_text: &'static str,
        confidence: f32,
    }

    #[async_trait]
    impl ChatClient for FixedChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Ok(TierResponse { raw_text: self.raw_text.to_string(), confidence: self.confidence, tier: Tier::T3 })
        }
    }

    fn proposal(operator: Operator, target_id: MentalObjectId) -> OperatorProposal {
        OperatorProposal { operator, target_id, preference: 1.0, confidence: 0.9 }
    }

    #[tokio::test]
    async fn speak_renders_the_target_objects_text() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("hello world", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::Speak, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        match outcome {
            ActOutcome::Spoke { text, render_path } => {
                assert_eq!(text, "hello world");
                assert_eq!(render_path, SpeechRenderPath::Verbatim);
            }
            other => panic!("expected Spoke, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn repeated_identical_speak_content_compiles_into_a_skill_and_bypasses_tier1() {
        // ACT-R-style skill compilation (`steps::social_interface::
        // compiled_render`): once a short, routine utterance has rendered
        // the exact same way `COMPILATION_THRESHOLD` times running, further
        // Speak calls for the identical text should stop reaching Tier 1
        // entirely and return the compiled template instead - Tier 0.
        struct CountingTier1Client {
            raw_text: &'static str,
            calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        }
        #[async_trait]
        impl ChatClient for CountingTier1Client {
            async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(TierResponse { raw_text: self.raw_text.to_string(), confidence: 0.9, tier: Tier::T1 })
            }
        }

        // 7 words: above MAX_WORDS_FOR_REFLEXIVE_SPEECH (so it genuinely
        // reaches Tier 1 the first few times, not skipped as formulaic) but
        // within MAX_WORDS_FOR_SKILL_COMPILATION (so it's still eligible to
        // compile).
        let text = "checking in on my current status now";
        let mut graph = Graph::new();
        let object = MentalObject::new_observation(text, EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Returns the exact original text - a maximally faithful "rendering"
        // (cosine similarity ~1.0 against itself via FakeEmbeddingClient),
        // guaranteed to clear RENDER_SIMILARITY_THRESHOLD and be accepted.
        let tier1 = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![std::sync::Arc::new(CountingTier1Client { raw_text: text, calls: calls.clone() })]);
        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };

        // social_interface::COMPILATION_THRESHOLD is 3 (private to that
        // module) - four identical Speak calls should reach Tier 1 exactly
        // three times, with the fourth served straight from the compiled
        // skill.
        let mut observed_paths = Vec::new();
        for _ in 0..4 {
            let outcome = act(
                &mut graph,
                &proposal(Operator::Speak, id),
                &tier1,
                &empty_tier2(),
                &pool,
                &client,
                &embedding_client(),
                &kl_store(),
                &KnowledgeLibraryConfig::default(),
                &ToolRegistry::empty(),
                &[],
                "",
                0.0,
                &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
                EpochMillis(1_000),
                &clock(),
                0.5,
                Duration::from_secs(5),
                &TemperatureConfig::default(),
                None,
            )
            .await;
            match outcome {
                ActOutcome::Spoke { text: spoken, render_path } => {
                    assert_eq!(spoken, text);
                    observed_paths.push(render_path);
                }
                other => panic!("expected Spoke, got {other:?}"),
            }
        }

        assert_eq!(&observed_paths[..3], [SpeechRenderPath::Tier1, SpeechRenderPath::Tier1, SpeechRenderPath::Tier1]);
        assert_eq!(observed_paths[3], SpeechRenderPath::CompiledSkill);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "once the skill compiles, further identical Speak calls should be served Tier 0 from the compiled template, not reach Tier 1 again"
        );
    }

    #[tokio::test]
    async fn remember_stores_a_new_episodic_memory() {
        let mut graph = Graph::new();
        let mut object = MentalObject::new_observation("a surprising fact", EpochMillis(0), 0.5);
        object.embedding = Some(vec![1.0, 0.0, 0.0]);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::Remember, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        assert!(matches!(outcome, ActOutcome::Remembered { outcome: MemoryFormationOutcome::NewEpisodic { .. } }));
    }

    #[tokio::test]
    async fn continue_reflecting_invokes_the_cognitive_core_and_inserts_the_reflection() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("something to reflect on", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient {
            raw_text: "a reflection",
            confidence: 0.7,
        };
        let outcome = act(
            &mut graph,
            &proposal(Operator::ContinueReflecting, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        match outcome {
            ActOutcome::Reflected { reflection_id } => {
                let reflection = graph.get(&reflection_id).expect("reflection should be inserted into the graph");
                assert_eq!(reflection.text, "a reflection");
            }
            other => panic!("expected Reflected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_failed_reflection_still_tags_the_target() {
        // Regression guard for the same failure mode as
        // `act_resolves_to_silence_when_no_tool_matches`'s: an empty Tier 3
        // completion (confirmed live, not hypothetical - see this arm's own
        // doc comment) must still tag the target as attempted, or
        // `propose_operators`'s `already_communicated` check never sees it
        // and ContinueReflecting gets re-proposed for the same object every
        // subsequent tick it wins Coalition, forever.
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("something to reflect on", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "", confidence: 0.7 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::ContinueReflecting, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        assert!(
            matches!(outcome, ActOutcome::Silent { reason: SilentReason::ReflectionFailed { .. } }),
            "an empty Tier 3 completion must surface as a genuine failure, not the same reason as a deliberate Ignore"
        );
        if let ActOutcome::Silent { reason: SilentReason::ReflectionFailed { error } } = &outcome {
            assert!(error.contains("empty"), "expected the error text to explain what went wrong, got: {error}");
        }
        let tagged = graph.get(&id).expect("target object should still be in the graph");
        assert_eq!(tagged.produced_by_operator.as_deref(), Some("ContinueReflecting"));
    }

    #[tokio::test]
    async fn ignore_resolves_to_silence() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("irrelevant", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::Ignore, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        assert!(matches!(outcome, ActOutcome::Silent { reason: SilentReason::Ignored }));
        let tagged = graph.get(&id).expect("object should still be in the graph");
        assert_eq!(
            tagged.produced_by_operator.as_deref(),
            Some("Ignore"),
            "Ignore must tag the object, same as every other operator, or propose_operators would re-propose it (and re-sample Tier 1) every tick"
        );
    }

    #[tokio::test]
    async fn consult_knowledge_library_returns_the_stores_top_match() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("subgoal: what is the wifi password?", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let store = kl_store();
        let client = embedding_client();
        let doc_embedding = client.embed("subgoal: what is the wifi password?").await.unwrap();
        store
            .insert_document("household://wifi", "subgoal: what is the wifi password?", doc_embedding, EpochMillis(0))
            .await
            .unwrap();

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let chat_client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::ConsultKnowledgeLibrary, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &chat_client,
            &client,
            &store,
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        match outcome {
            ActOutcome::ConsultedKnowledgeLibrary { result } => {
                assert!(result.found);
                assert_eq!(result.text.as_deref(), Some("subgoal: what is the wifi password?"));
            }
            other => panic!("expected ConsultedKnowledgeLibrary, got {other:?}"),
        }
        let tagged = graph.get(&id).expect("target object should still be in the graph");
        assert_eq!(tagged.produced_by_operator.as_deref(), Some("ConsultKnowledgeLibrary"));
    }

    #[tokio::test]
    async fn consult_knowledge_library_on_an_empty_store_reports_not_found() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("subgoal: anything", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let chat_client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::ConsultKnowledgeLibrary, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &chat_client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        match outcome {
            ActOutcome::ConsultedKnowledgeLibrary { result } => assert!(!result.found),
            other => panic!("expected ConsultedKnowledgeLibrary, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn act_invokes_an_allowed_tool_and_tags_the_target() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("what time is it?", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let tool_clock: std::sync::Arc<dyn Clock> = std::sync::Arc::new(clock());
        let tool_registry = ToolRegistry::new(vec![std::sync::Arc::new(tools::CurrentTimeTool::new(tool_clock))], tools::ToolRiskTier::Harmless);
        let outcome = act(
            &mut graph,
            &proposal(Operator::Act, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &tool_registry,
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        match outcome {
            ActOutcome::Acted { tool, result } => {
                assert_eq!(tool, "current_time");
                assert!(result.is_ok());
            }
            other => panic!("expected Acted, got {other:?}"),
        }
        let tagged = graph.get(&id).expect("target object should still be in the graph");
        assert_eq!(tagged.produced_by_operator.as_deref(), Some("Act"));
    }

    #[tokio::test]
    async fn act_honors_a_self_issued_requested_tool_tag_over_text_matching() {
        let mut graph = Graph::new();
        // Text deliberately doesn't match any of `match_tool_intent`'s
        // trigger phrases - this only resolves via `data.requested_tool`.
        let mut object = MentalObject::new_observation("nothing has needed my attention for a while", EpochMillis(0), 0.5);
        object.data = serde_json::json!({"source": "boredom", "requested_tool": "current_time"});
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let tool_clock: std::sync::Arc<dyn Clock> = std::sync::Arc::new(clock());
        let tool_registry = ToolRegistry::new(vec![std::sync::Arc::new(tools::CurrentTimeTool::new(tool_clock))], tools::ToolRiskTier::Harmless);
        let outcome = act(
            &mut graph,
            &proposal(Operator::Act, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &tool_registry,
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        match outcome {
            ActOutcome::Acted { tool, result } => {
                assert_eq!(tool, "current_time");
                assert!(result.is_ok());
            }
            other => panic!("expected Acted via the requested_tool tag, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn act_resolves_to_silence_when_no_tool_matches() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("the sky is blue", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::Act, id),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        assert!(matches!(outcome, ActOutcome::Silent { reason: SilentReason::NoToolMatched }));
        // Regression guard: a failed tool match must still tag the target,
        // or `propose_operators`'s `already_acted` guard never sees it and
        // Act gets re-proposed for the same object every subsequent tick
        // forever (confirmed live - see this arm's own doc comment).
        let tagged = graph.get(&id).expect("target object should still be in the graph");
        assert_eq!(tagged.produced_by_operator.as_deref(), Some("Act"));
    }

    #[tokio::test]
    async fn act_invokes_a_tool_matched_only_via_tier1_semantic_judgement() {
        // Deliberately doesn't contain any of `tools::match_tool_intent`'s
        // trigger phrases (mirrors `executive::
        // propose_operators_proposes_act_for_text_that_only_affords_a_tool_
        // semantically`) - this only resolves via `propose_tool_intent`'s
        // real Tier 1 judgement call, exercising the same "execution must
        // ask the identical question its proposal was based on" path this
        // arm's own doc comment describes.
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("could you tell me what o'clock it is right now", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let tool_clock: std::sync::Arc<dyn Clock> = std::sync::Arc::new(clock());
        let tool_registry = ToolRegistry::new(vec![std::sync::Arc::new(tools::CurrentTimeTool::new(tool_clock))], tools::ToolRiskTier::Harmless);
        // A single-client Tier 1 pool that always answers "current_time" -
        // `DivergentPool::sample` repeat-samples a solo-configured tier, so
        // this reproduces unanimous agreement without needing three clients.
        let tier1_pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![std::sync::Arc::new(FixedChatClient { raw_text: "current_time", confidence: 0.9 })]);
        let outcome = act(
            &mut graph,
            &proposal(Operator::Act, id),
            &tier1_pool,
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &tool_registry,
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;

        match outcome {
            ActOutcome::Acted { tool, result } => {
                assert_eq!(tool, "current_time");
                assert!(result.is_ok());
            }
            other => panic!("expected Acted via the semantic affordance match, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_stale_target_id_resolves_to_silence_rather_than_panicking() {
        let mut graph = Graph::new();
        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let client = FixedChatClient { raw_text: "unused", confidence: 0.5 };
        let outcome = act(
            &mut graph,
            &proposal(Operator::Speak, MentalObjectId::new()),
            &empty_tier1(),
            &empty_tier2(),
            &pool,
            &client,
            &embedding_client(),
            &kl_store(),
            &KnowledgeLibraryConfig::default(),
            &ToolRegistry::empty(),
            &[],
            "",
            0.0,
            &MemoryFormationConfig::default(),
            &ConfidenceRevisionConfig::default(),
            EpochMillis(1_000),
            &clock(),
            0.5,
            Duration::from_secs(5),
            &TemperatureConfig::default(),
            None,
        )
        .await;
        assert!(matches!(outcome, ActOutcome::Silent { reason: SilentReason::StaleTarget }));
    }
}
