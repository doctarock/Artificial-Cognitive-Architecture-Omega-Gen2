use aca_graph::Graph;
use aca_tiers::{DivergentPool, EmbeddingClient, GenerateRequest};
use aca_types::{MemoryRole, MentalObject, MentalObjectId, MentalObjectKind};
use aca_util::{cosine_similarity, EpochMillis};
use serde::{Deserialize, Serialize};

use crate::prompt_templates::social_rendering_prompt;

/// At or below this many words, content is treated as formulaic/reflexive
/// speech and rendering is skipped entirely (Tier 0: the verbatim text
/// itself) - a backchannel, a bare "yes"/"no", a short acknowledgment. This
/// mirrors how a person's own pre-speech monitor barely engages for stock
/// replies but does engage for anything long enough to form a real clause -
/// word count, not character count, is what actually tracks "formulaic vs.
/// a genuine proposition": a short *sentence* ("Derek just left.") is still
/// a real thought that can narrate or editorialize just as easily as a long
/// one, so length alone would let exactly the content most worth filtering
/// slip through unfiltered. specs.md's own framing ("a short factual
/// acknowledgment doesn't warrant the same processor as a nuanced
/// reflective answer") is about content that's actually formulaic, which
/// this tracks more honestly than a raw character count did.
pub(crate) const MAX_WORDS_FOR_REFLEXIVE_SPEECH: usize = 3;
/// How similar a rendered candidate must stay to the original text
/// (cosine similarity of their embeddings) to be trusted as a rephrasing
/// rather than an embellishment. This is the real enforcement behind
/// "never to originate content" - a prompt instruction alone isn't
/// sufficient at small model sizes (the same lesson this codebase already
/// learned from self-reported confidence: don't trust the request, verify
/// the result).
const RENDER_SIMILARITY_THRESHOLD: f32 = 0.75;

/// At or below this many words, decided text is a candidate for skill
/// compilation (see `compiled_render`/`record_render`, below) - a wider net
/// than `MAX_WORDS_FOR_REFLEXIVE_SPEECH`'s "skip rendering entirely"
/// boundary, since a short *routine* utterance still worth an occasional
/// Tier 1 rendering pass (a status report, a standard acknowledgment) is
/// exactly the kind of content most likely to actually recur verbatim.
/// Deliberately still short: a full sentence or Reflection is unlikely to
/// ever repeat exactly, so tracking it as a skill candidate would only add
/// permanent graph bloat for content that will never be looked up again.
pub(crate) const MAX_WORDS_FOR_SKILL_COMPILATION: usize = 8;

/// How many consecutive times decided text must render to the exact same
/// output before that rendering is trusted as a compiled skill and Tier 1
/// is bypassed entirely - see `compiled_render`'s doc comment. Deliberately
/// small: this only ever governs formulaic, low-stakes content (bounded by
/// `MAX_WORDS_FOR_SKILL_COMPILATION`), where a wrong compile is cheap to
/// notice and self-correct (a differing render on the next occurrence
/// simply resets the counter - see `record_render`).
const COMPILATION_THRESHOLD: u32 = 3;

/// A compiled rendering skill (ACT-R's own procedural-utility story -
/// specs.md line 173-175's "production rules with learned utility" -
/// applied to *rendering* rather than operator selection; `steps::learn`'s
/// chunking is the same idea for the Executive). Stored as an ordinary
/// Semantic memory object, same pattern as `steps::learn::ChunkPayload`,
/// just a different payload shape and subsystem.
#[derive(Debug, Serialize, Deserialize)]
struct RenderSkillPayload {
    decided_text: String,
    compiled_render: String,
    /// Consecutive times `decided_text` has rendered to exactly
    /// `compiled_render` - see `record_render`'s doc comment for why a
    /// differing render resets this rather than being averaged in.
    consecutive_hits: u32,
}

fn find_skill(graph: &Graph, decided_text: &str) -> Option<(MentalObjectId, RenderSkillPayload)> {
    graph
        .iter()
        .filter(|object| object.memory_roles.contains(&MemoryRole::Semantic))
        .find_map(|object| {
            let payload: RenderSkillPayload = serde_json::from_value(object.data.clone()).ok()?;
            (payload.decided_text == decided_text).then_some((object.id, payload))
        })
}

/// specs.md's Model Tiering point 3 pushed to its logical endpoint: once
/// `decided_text` has rendered to the exact same output `COMPILATION_THRESHOLD`
/// times running, return that stored template directly - Tier 0, zero model
/// calls - instead of running `render_speech`'s Tier 1 pass again. This is
/// literal automaticity/proceduralization (the System-1 half of Kahneman's
/// dual-process framing, mirrored here in miniature): repeated, *consistent*
/// success at rendering the same content the same way is what "practice
/// makes it automatic" means for the Social Interface specifically, the
/// same "independent repetition landing on the same answer earns trust"
/// idea `cognitive_core::try_tier_via_agreement` already applies to Tier 1/2
/// candidate agreement. `None` when no skill is compiled yet (never seen,
/// or still below threshold) - the caller falls through to the ordinary
/// `render_speech` path.
pub fn compiled_render(graph: &Graph, decided_text: &str) -> Option<String> {
    find_skill(graph, decided_text)
        .filter(|(_, payload)| payload.consecutive_hits >= COMPILATION_THRESHOLD)
        .map(|(_, payload)| payload.compiled_render)
}

/// Records one fresh `(decided_text, actually-rendered)` pair from a real
/// (non-compiled) `render_speech` call, upserting or reinforcing a skill
/// entry - the rendering counterpart to `steps::learn::chunk_resolution`/
/// `reinforce_chunk_utility`, same "compile repeated success into a reusable
/// shortcut" idea, different subsystem, and consequently a different
/// trigger: an operator-selection chunk compiles the moment an impasse
/// resolves, but a render has no equivalent single resolving event, so this
/// is called on every eligible render instead and lets consistency
/// accumulate (or reset) naturally over repeated occurrences. A render that
/// matches the existing entry's `compiled_render` increments
/// `consecutive_hits` toward `COMPILATION_THRESHOLD`; one that *differs*
/// resets the counter to `1` with the new render - consistency, not just
/// frequency, is what earns automaticity here, since a rendering that keeps
/// changing isn't actually a stable pattern worth bypassing Tier 1 for.
/// Never called for a render that itself came from `compiled_render` - see
/// the caller in `steps::act` - a compiled skill firing doesn't need to
/// re-earn its own trust every time it's used.
pub fn record_render(graph: &mut Graph, decided_text: &str, rendered: &str, now: EpochMillis, decay_d: f32) {
    if let Some((id, mut payload)) = find_skill(graph, decided_text) {
        if payload.compiled_render == rendered {
            payload.consecutive_hits += 1;
        } else {
            payload.compiled_render = rendered.to_string();
            payload.consecutive_hits = 1;
        }
        if let Some(object) = graph.get_mut(&id) {
            object.data = serde_json::to_value(&payload).expect("RenderSkillPayload always serializes");
            aca_graph::record_reference(&mut object.activation, now);
        }
        return;
    }

    let payload = RenderSkillPayload { decided_text: decided_text.to_string(), compiled_render: rendered.to_string(), consecutive_hits: 1 };
    let mut object = MentalObject::new_observation(format!("learned rendering for: {decided_text}"), now, decay_d);
    object.kind = MentalObjectKind::Memory;
    object.memory_roles.push(MemoryRole::Semantic);
    object.data = serde_json::to_value(&payload).expect("RenderSkillPayload always serializes");
    graph.insert(object);
}

/// The Social Interface: renders already-selected communicative intent into
/// natural language. By construction this never originates content — it
/// only ever runs on a Mental Object the Executive already chose to speak,
/// so "communication is always derived from cognition, never the reverse"
/// (specs.md) holds structurally, not by convention.
///
/// specs.md's "Where the Language Model Plugs In" point 3: "Tier is chosen
/// to match the complexity of the Mental Object being rendered, not fixed."
/// Made literal here as a two-way choice: formulaic content (see
/// `MAX_WORDS_FOR_REFLEXIVE_SPEECH`) is rendered as Tier 0 (the verbatim
/// text - there's nothing to gain from paraphrasing "yes" or a short
/// timestamp); anything long enough to be a real clause gets one real Tier 1
/// attempt, regardless of how few characters it happens to take. This
/// deliberately never escalates further than Tier 1 on its own — unlike
/// `cognitive_core::reflect`, a style pass isn't worth the latency of a
/// Tier 2/3 call (confirmed live: a single Tier 3 reflection can
/// legitimately take a minute or more), and every failure mode (unconfigured
/// pool, no candidate clears `RENDER_SIMILARITY_THRESHOLD`, an empty
/// completion) falls back to the original verbatim text rather than to
/// silence — rendering can only ever improve phrasing, never block or fail a
/// Speak that would otherwise have succeeded.
pub async fn render_speech(object: &MentalObject, tier1_pool: &DivergentPool, embedding_client: &dyn EmbeddingClient, self_summary: &str, temperature: f32) -> String {
    let original = object.text.clone();
    if original.split_whitespace().count() <= MAX_WORDS_FOR_REFLEXIVE_SPEECH || tier1_pool.is_empty() {
        return original;
    }

    // `object.embedding` is already resolved for anything that reaches Speak
    // today (a Reflection - see `cognitive_core::reflect`'s own embed call) -
    // re-embedding identical text here was a pure wasted round trip to the
    // embedding server on every rendered reply. Still falls back to a fresh
    // embed for any future caller whose object doesn't carry one yet.
    let original_embedding = match object.embedding.clone() {
        Some(embedding) => embedding,
        None => match embedding_client.embed(&original).await {
            Ok(embedding) => embedding,
            Err(_) => return original,
        },
    };

    let prompt = social_rendering_prompt(self_summary, &original);
    let Ok(candidates) = tier1_pool.sample(GenerateRequest { prompt, temperature }, 1).await else {
        return original;
    };

    let mut best: Option<(String, f32)> = None;
    for candidate in candidates {
        let text = candidate.raw_text.trim();
        if text.is_empty() {
            continue;
        }
        let Ok(embedding) = embedding_client.embed(text).await else {
            continue;
        };
        let similarity = cosine_similarity(&original_embedding, &embedding);
        if similarity >= RENDER_SIMILARITY_THRESHOLD && best.as_ref().is_none_or(|(_, best_similarity)| similarity > *best_similarity) {
            best = Some((text.to_string(), similarity));
        }
    }

    best.map(|(text, _)| text).unwrap_or(original)
}

#[cfg(test)]
mod skill_compilation_tests {
    use super::*;

    #[test]
    fn no_skill_is_compiled_below_the_threshold() {
        let mut graph = Graph::new();
        record_render(&mut graph, "good morning", "Good morning!", EpochMillis(0), 0.5);
        record_render(&mut graph, "good morning", "Good morning!", EpochMillis(1_000), 0.5);
        assert_eq!(compiled_render(&graph, "good morning"), None, "two consistent hits should still be below COMPILATION_THRESHOLD");
    }

    #[test]
    fn a_skill_compiles_after_enough_consistent_renders() {
        let mut graph = Graph::new();
        for i in 0..COMPILATION_THRESHOLD {
            record_render(&mut graph, "good morning", "Good morning!", EpochMillis(i as i64 * 1_000), 0.5);
        }
        assert_eq!(compiled_render(&graph, "good morning"), Some("Good morning!".to_string()));
    }

    #[test]
    fn a_differing_render_resets_the_consecutive_hit_counter() {
        let mut graph = Graph::new();
        record_render(&mut graph, "good morning", "Good morning!", EpochMillis(0), 0.5);
        record_render(&mut graph, "good morning", "Good morning!", EpochMillis(1_000), 0.5);
        // A third, differing render should reset the streak rather than
        // pushing it over the threshold with an inconsistent third answer.
        record_render(&mut graph, "good morning", "Morning to you too!", EpochMillis(2_000), 0.5);
        assert_eq!(compiled_render(&graph, "good morning"), None, "an inconsistent render should reset the streak, not compile a skill from it");

        // Confirm it's a real reset, not just a one-tick blip: the new
        // phrasing needs its own full run of consistency from here.
        for i in 0..(COMPILATION_THRESHOLD - 1) {
            record_render(&mut graph, "good morning", "Morning to you too!", EpochMillis(3_000 + i as i64 * 1_000), 0.5);
        }
        assert_eq!(compiled_render(&graph, "good morning"), Some("Morning to you too!".to_string()));
    }

    #[test]
    fn unrelated_decided_text_is_not_affected_by_an_unrelated_skill() {
        let mut graph = Graph::new();
        for i in 0..COMPILATION_THRESHOLD {
            record_render(&mut graph, "good morning", "Good morning!", EpochMillis(i as i64 * 1_000), 0.5);
        }
        assert_eq!(compiled_render(&graph, "good afternoon"), None, "an unrelated decided_text must not match a different phrase's compiled skill");
    }

    #[test]
    fn a_skill_is_stored_as_a_semantic_memory_object() {
        let mut graph = Graph::new();
        record_render(&mut graph, "yes", "Yes.", EpochMillis(0), 0.5);
        let stored = graph.iter().find(|object| object.memory_roles.contains(&MemoryRole::Semantic));
        assert!(stored.is_some(), "a recorded render should be stored as a Semantic memory object, same as any other learned chunk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_tiers::testing::FakeEmbeddingClient;
    use aca_tiers::{ChatClient, TierError, TierResponse};
    use aca_types::Tier;
    use aca_util::EpochMillis;
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::time::Duration;

    fn empty_tier1() -> DivergentPool {
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![])
    }

    struct FixedRenderClient {
        raw_text: &'static str,
    }

    #[async_trait]
    impl ChatClient for FixedRenderClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Ok(TierResponse { raw_text: self.raw_text.to_string(), confidence: 0.9, tier: Tier::T1 })
        }
    }

    fn short_object(text: &str) -> MentalObject {
        MentalObject::new_observation(text, EpochMillis(0), 0.5)
    }

    #[tokio::test]
    async fn short_text_is_never_rendered_even_with_a_configured_pool() {
        let pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![Arc::new(FixedRenderClient { raw_text: "should never be called" })]);
        let object = short_object("yes");
        let result = render_speech(&object, &pool, &FakeEmbeddingClient::default(), "", 0.4).await;
        assert_eq!(result, "yes");
    }

    struct CountingRenderClient {
        raw_text: &'static str,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatClient for CountingRenderClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(TierResponse { raw_text: self.raw_text.to_string(), confidence: 0.9, tier: Tier::T1 })
        }
    }

    #[tokio::test]
    async fn a_short_but_multi_word_clause_still_goes_through_rendering() {
        // Regression test: a genuine short clause (not a stock reply) used
        // to slip past rendering entirely just because it fell under the
        // old character-count floor - exactly the content most likely to
        // carry unwanted narration ("reviewing what was said, Derek just
        // left" vs. the plain fact). This text is well under the old
        // 40-character floor but is five words, a real clause rather than a
        // formulaic reply, so it must actually reach Tier 1 now.
        let text = "Derek just left the room.";
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![Arc::new(CountingRenderClient { raw_text: text, calls: calls.clone() })]);
        let object = short_object(text);
        let result = render_speech(&object, &pool, &FakeEmbeddingClient::default(), "", 0.4).await;
        assert_eq!(result, text);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a short multi-word clause should reach Tier 1 rendering, not bypass it on character count alone"
        );
    }

    #[tokio::test]
    async fn long_text_falls_back_to_verbatim_when_tier1_is_unconfigured() {
        let object = short_object("this is a much longer piece of content that clears the minimum length for rendering");
        let text = object.text.clone();
        let result = render_speech(&object, &empty_tier1(), &FakeEmbeddingClient::default(), "", 0.4).await;
        assert_eq!(result, text);
    }

    #[tokio::test]
    async fn a_faithful_rephrasing_is_accepted() {
        // FakeEmbeddingClient is deterministic by text - a rephrase using
        // near-identical wording will embed near-identically too.
        let long_text = "the current time is currently right around fourteen thirty two in the afternoon";
        let pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![Arc::new(FixedRenderClient { raw_text: long_text })]);
        let object = short_object(long_text);
        let result = render_speech(&object, &pool, &FakeEmbeddingClient::default(), "", 0.4).await;
        assert_eq!(result, long_text, "an identical (maximally faithful) rendering should be accepted");
    }

    #[tokio::test]
    async fn an_empty_render_candidate_falls_back_to_verbatim() {
        let long_text = "this is a sufficiently long piece of decided content to trigger a render attempt";
        let pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![Arc::new(FixedRenderClient { raw_text: "" })]);
        let object = short_object(long_text);
        let result = render_speech(&object, &pool, &FakeEmbeddingClient::default(), "", 0.4).await;
        assert_eq!(result, long_text);
    }

    #[tokio::test]
    async fn a_wildly_divergent_candidate_is_rejected_in_favor_of_verbatim() {
        // Simulates the real risk this whole function exists to guard
        // against: a "rephrase" that actually introduces unrelated content.
        let long_text = "the meeting has been rescheduled to next Tuesday at three in the afternoon";
        let pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![Arc::new(FixedRenderClient { raw_text: "purple elephants dance slowly across the moonlit savanna" })]);
        let object = short_object(long_text);
        let result = render_speech(&object, &pool, &FakeEmbeddingClient::default(), "", 0.4).await;
        assert_eq!(result, long_text, "a candidate that diverged too far from the original must not be spoken instead of it");
    }
}
