use aca_graph::{reinforce_edge, Graph, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH};
use aca_tiers::{ChatClient, DivergentPool, EmbeddingClient, GenerateRequest, TierPool};
use aca_types::{EdgeKind, MemoryRole, MentalObject, MentalObjectId, MentalObjectKind, PromotionState};
use aca_util::{cosine_similarity, EpochMillis};

use crate::cognitive_core::resolve_via_tier_ladder;
use crate::prompt_templates::synthesis_prompt;

/// Tuning knobs for `synthesize` — thresholds and cadence, not theoretical
/// commitments (see `LoopConfig`'s own doc comment on this framing).
#[derive(Debug, Clone, Copy)]
pub struct SynthesisConfig {
    /// How many new Episodic memories must accumulate since the last
    /// synthesis attempt before another one is worth triggering.
    pub min_new_episodic: usize,
    /// The most source memories handed to a single synthesis attempt - a
    /// cap on prompt size, not a quality target.
    pub max_cluster_size: usize,
    /// Minimum wall-clock gap between synthesis attempts, success or
    /// failure - a cost-control backstop on top of the buffer's own
    /// self-pacing (it only refills through genuine new episodic memories).
    pub min_interval_ms: i64,
    /// Cosine similarity above which a freshly synthesized pattern is
    /// treated as "the same insight again" (reinforce the existing Semantic
    /// memory) rather than a genuinely new one - mirrors
    /// `MemoryFormationConfig::reinforcement_similarity_threshold`'s
    /// default exactly, same reasoning: overlapping episodic windows should
    /// not mint repeated near-identical spoken insights.
    pub dedup_similarity_threshold: f32,
    /// Whether a freshly-synthesized pattern starts as an unconfirmed
    /// `PromotionStatus::Candidate` (see that type's own doc comment) rather
    /// than immediately trusted - same reasoning as
    /// `MemoryFormationConfig::promotion_gate_enabled`: nothing but this
    /// function's own abstraction call has vouched for the pattern at the
    /// moment it's written. `false` reproduces the old unconditional-trust
    /// behavior exactly, for A/B comparison.
    pub promotion_gate_enabled: bool,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            min_new_episodic: 3,
            max_cluster_size: 8,
            min_interval_ms: 5 * 60 * 1000,
            dedup_similarity_threshold: 0.93,
            promotion_gate_enabled: true,
        }
    }
}

/// specs.md's Semantic Memory section names two growth paths: SOAR chunking
/// (`steps::learn::chunk_resolution` - compiles resolved *operator-selection
/// impasses*) and "explicit memory-formation decisions that abstract a
/// pattern out of episodic detail" - this is the second one. Given several
/// recently-formed Episodic memories, asks a model tier to abstract a
/// general pattern out of them and writes it as a new Semantic memory,
/// `DerivedFrom`-linked back to every source it actually used (mirrors
/// `chunk_resolution`'s edge-reinforcement shape exactly).
///
/// Deliberately not routed through the Executive/Operator machinery - no
/// `Operator` variant exists for this. It's called directly, on a periodic
/// actor-local trigger, the same "infrastructure, not cognition" category as
/// `loop_actor`'s periodic write-behind flush. What happens *after* this
/// returns (whether the new memory ever gets spoken about) is entirely
/// governed by the ordinary Coalition/Broadcast/Executive competition once
/// the caller gives the returned id a `pending_admission` shot, exactly like
/// a fresh Cognitive Core Reflection - this function only ever produces a
/// candidate, never speaks.
///
/// Ids in `cluster_ids` may have decayed, been discarded, or been
/// reclassified since they were buffered by the caller - only ones still
/// present in `graph` and still tagged `MemoryRole::Episodic` are usable
/// source material; fewer than two usable texts means there is nothing to
/// find a pattern across, and this returns `None` rather than fabricating
/// one from a single memory.
#[allow(clippy::too_many_arguments)]
pub async fn synthesize(
    graph: &mut Graph,
    cluster_ids: &[MentalObjectId],
    tier1_pool: &DivergentPool,
    tier2_pool: &DivergentPool,
    tier3_pool: &TierPool,
    tier3_client: &dyn ChatClient,
    embedding_client: &dyn EmbeddingClient,
    config: &SynthesisConfig,
    self_summary: &str,
    now: EpochMillis,
    decay_d: f32,
    tier3_hedge_delay: std::time::Duration,
    temperature: f32,
) -> Option<MentalObjectId> {
    let mut used_ids = Vec::new();
    let mut texts = Vec::new();
    for &id in cluster_ids {
        if let Some(object) = graph.get(&id) {
            if object.memory_roles.contains(&MemoryRole::Episodic) {
                used_ids.push(id);
                texts.push(object.text.clone());
            }
        }
    }
    if texts.len() < 2 {
        return None;
    }

    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let prompt = synthesis_prompt(self_summary, &text_refs);
    let req = GenerateRequest { prompt, temperature };
    // Empty/whitespace-only completions already fail here (see
    // `resolve_via_tier_ladder`'s own doc comment) - no separate check
    // needed on `response.raw_text` below.
    let response = resolve_via_tier_ladder(tier1_pool, tier2_pool, tier3_pool, tier3_client, embedding_client, req, tier3_hedge_delay).await.ok()?;
    let candidate_embedding = embedding_client.embed(&response.raw_text).await.ok();

    // Tier-0 de-duplication, same spirit as
    // `memory_formation::form_memory`'s near-duplicate check: prevents
    // overlapping episodic windows from minting repeated near-identical
    // spoken insights.
    if let Some(candidate_embedding) = &candidate_embedding {
        let most_similar_existing = graph
            .iter()
            .filter(|existing| existing.memory_roles.contains(&MemoryRole::Semantic))
            .filter_map(|existing| existing.embedding.as_ref().map(|emb| (existing.id, cosine_similarity(candidate_embedding, emb))))
            .filter(|(_, similarity)| *similarity >= config.dedup_similarity_threshold)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((existing_id, _similarity)) = most_similar_existing {
            if let Some(existing_object) = graph.get_mut(&existing_id) {
                aca_graph::record_reference(&mut existing_object.activation, now);
                // Same same-instant guard as `memory_formation::form_memory`'s
                // own confirmation check - only a genuinely later touch
                // confirms a still-staged candidate.
                if now.0 > existing_object.promotion.staged_at.0 {
                    existing_object.promotion.confirm(now);
                }
            }
            link_sources(graph, &used_ids, existing_id, now);
            return Some(existing_id);
        }
    }

    let mut pattern = MentalObject::new_observation(response.raw_text.clone(), now, decay_d);
    pattern.kind = MentalObjectKind::Memory;
    pattern.memory_roles.push(MemoryRole::Semantic);
    pattern.confidence = response.confidence;
    pattern.tier_used = Some(response.tier);
    pattern.source_object_ids = used_ids.clone();
    pattern.embedding = candidate_embedding;
    if config.promotion_gate_enabled {
        pattern.promotion = PromotionState::candidate(now);
    }
    let new_id = pattern.id;

    link_sources(graph, &used_ids, new_id, now);
    graph.insert(pattern);
    Some(new_id)
}

/// Reinforces a `DerivedFrom` edge from each of `source_ids` toward
/// `target_id` - mirrors `steps::learn::chunk_resolution`'s edge shape
/// exactly, shared between both the fresh-pattern and the
/// reinforce-existing-pattern paths above.
fn link_sources(graph: &mut Graph, source_ids: &[MentalObjectId], target_id: MentalObjectId, now: EpochMillis) {
    for &source_id in source_ids {
        if let Some(source) = graph.get_mut(&source_id) {
            reinforce_edge(&mut source.edges, target_id, EdgeKind::DerivedFrom, now, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_tiers::{testing::FakeEmbeddingClient, TierError, TierResponse};
    use aca_types::Tier;
    use async_trait::async_trait;
    use std::time::Duration;

    fn episodic_object(text: &str, now: EpochMillis, decay_d: f32) -> MentalObject {
        let mut object = MentalObject::new_observation(text, now, decay_d);
        object.memory_roles.push(MemoryRole::Episodic);
        object.embedding = Some(vec![1.0, 0.0, 0.0]);
        object
    }

    struct FixedChatClient {
        raw_text: &'static str,
        confidence: f32,
        tier: Tier,
    }

    #[async_trait]
    impl ChatClient for FixedChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Ok(TierResponse {
                raw_text: self.raw_text.to_string(),
                confidence: self.confidence,
                tier: self.tier,
            })
        }
    }

    fn empty_tier1() -> DivergentPool {
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![])
    }

    fn empty_tier2() -> DivergentPool {
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![])
    }

    fn disagreeing_tier2(texts: &[&'static str]) -> DivergentPool {
        let clients = texts
            .iter()
            .map(|t| std::sync::Arc::new(FixedChatClient { raw_text: t, confidence: 0.99, tier: Tier::T2 }) as std::sync::Arc<dyn ChatClient>)
            .collect();
        DivergentPool::new(Tier::T2, Duration::from_secs(5), clients)
    }

    #[tokio::test]
    async fn returns_none_with_fewer_than_two_usable_episodic_texts() {
        let mut graph = Graph::new();
        let client = FixedChatClient { raw_text: "a pattern", confidence: 0.8, tier: Tier::T3 };
        let embedding_client = FakeEmbeddingClient::default();

        // An entirely empty cluster.
        let outcome = synthesize(&mut graph, &[], &empty_tier1(), &empty_tier2(), &TierPool::new(Tier::T3, 1, Duration::from_secs(5)), &client, &embedding_client, &SynthesisConfig::default(), "", EpochMillis(0), 0.5, Duration::from_secs(5), 0.4).await;
        assert!(outcome.is_none());

        // A single usable episodic memory.
        let solo = episodic_object("only one episode", EpochMillis(0), 0.5);
        let solo_id = solo.id;
        graph.insert(solo);
        let outcome = synthesize(&mut graph, &[solo_id], &empty_tier1(), &empty_tier2(), &TierPool::new(Tier::T3, 1, Duration::from_secs(5)), &client, &embedding_client, &SynthesisConfig::default(), "", EpochMillis(0), 0.5, Duration::from_secs(5), 0.4).await;
        assert!(outcome.is_none(), "a single source has nothing to find a pattern across");
    }

    #[tokio::test]
    async fn ignores_a_buffered_id_that_was_since_discarded_or_reclassified() {
        let mut graph = Graph::new();
        let a = episodic_object("first episode", EpochMillis(0), 0.5);
        let b = episodic_object("second episode", EpochMillis(0), 0.5);
        let (a_id, b_id) = (a.id, b.id);
        graph.insert(a);
        graph.insert(b);

        // A non-episodic object (already reclassified as Semantic) mixed into the cluster.
        let mut reclassified = MentalObject::new_observation("no longer episodic", EpochMillis(0), 0.5);
        reclassified.memory_roles.push(MemoryRole::Semantic);
        let reclassified_id = reclassified.id;
        graph.insert(reclassified);

        let missing_id = MentalObjectId::new();

        let client = FixedChatClient { raw_text: "a real pattern across two episodes", confidence: 0.8, tier: Tier::T3 };
        let embedding_client = FakeEmbeddingClient::default();
        let outcome = synthesize(
            &mut graph,
            &[a_id, b_id, reclassified_id, missing_id],
            &empty_tier1(),
            &empty_tier2(),
            &TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            &client,
            &embedding_client,
            &SynthesisConfig::default(),
            "",
            EpochMillis(1_000),
            0.5,
            Duration::from_secs(5),
            0.4,
        )
        .await;

        let new_id = outcome.expect("two genuinely usable episodic sources should still be enough to synthesize");
        let pattern = graph.get(&new_id).unwrap();
        assert_eq!(pattern.source_object_ids, vec![a_id, b_id], "only the two still-Episodic ids should count as sources");
    }

    #[tokio::test]
    async fn produces_a_semantic_memory_object_with_derived_from_edges() {
        let mut graph = Graph::new();
        let a = episodic_object("went for a run in the rain", EpochMillis(0), 0.5);
        let b = episodic_object("skipped a run because of rain", EpochMillis(0), 0.5);
        let c = episodic_object("rescheduled a run around rain", EpochMillis(0), 0.5);
        let (a_id, b_id, c_id) = (a.id, b.id, c.id);
        graph.insert(a);
        graph.insert(b);
        graph.insert(c);

        let client = FixedChatClient { raw_text: "weather strongly influences running habits", confidence: 0.85, tier: Tier::T3 };
        let embedding_client = FakeEmbeddingClient::default();
        let outcome = synthesize(
            &mut graph,
            &[a_id, b_id, c_id],
            &empty_tier1(),
            &empty_tier2(),
            &TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            &client,
            &embedding_client,
            &SynthesisConfig::default(),
            "",
            EpochMillis(1_000),
            0.5,
            Duration::from_secs(5),
            0.4,
        )
        .await;

        let new_id = outcome.expect("three episodic sources should synthesize a pattern");
        let pattern = graph.get(&new_id).unwrap();
        assert_eq!(pattern.kind, MentalObjectKind::Memory);
        assert_eq!(pattern.memory_roles, vec![MemoryRole::Semantic]);
        assert_eq!(pattern.text, "weather strongly influences running habits");
        assert!(pattern.embedding.is_some());

        for source_id in [a_id, b_id, c_id] {
            let source = graph.get(&source_id).unwrap();
            assert_eq!(source.edges.len(), 1);
            assert_eq!(source.edges[0].target_id, new_id);
            assert_eq!(source.edges[0].kind, EdgeKind::DerivedFrom);
        }
    }

    #[tokio::test]
    async fn a_freshly_synthesized_pattern_starts_as_an_unconfirmed_candidate() {
        let mut graph = Graph::new();
        let a = episodic_object("went for a run in the rain", EpochMillis(0), 0.5);
        let b = episodic_object("skipped a run because of rain", EpochMillis(0), 0.5);
        let (a_id, b_id) = (a.id, b.id);
        graph.insert(a);
        graph.insert(b);

        let client = FixedChatClient { raw_text: "weather strongly influences running habits", confidence: 0.85, tier: Tier::T3 };
        let embedding_client = FakeEmbeddingClient::default();
        let outcome = synthesize(
            &mut graph,
            &[a_id, b_id],
            &empty_tier1(),
            &empty_tier2(),
            &TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            &client,
            &embedding_client,
            &SynthesisConfig::default(),
            "",
            EpochMillis(1_000),
            0.5,
            Duration::from_secs(5),
            0.4,
        )
        .await;

        let new_id = outcome.expect("two episodic sources should synthesize a pattern");
        let pattern = graph.get(&new_id).unwrap();
        assert_eq!(pattern.promotion.status, aca_types::PromotionStatus::Candidate, "nothing but this synthesis call has vouched for the pattern yet");
        assert_eq!(pattern.promotion.staged_at, EpochMillis(1_000));
    }

    #[tokio::test]
    async fn a_later_near_duplicate_confirms_an_existing_staged_pattern() {
        let mut graph = Graph::new();
        let a = episodic_object("episode one", EpochMillis(0), 0.5);
        let b = episodic_object("episode two", EpochMillis(0), 0.5);
        let (a_id, b_id) = (a.id, b.id);
        graph.insert(a);
        graph.insert(b);

        let embedding_client = FakeEmbeddingClient::default();
        let repeated_text = "a pattern already known";
        let mut existing_pattern = MentalObject::new_observation(repeated_text, EpochMillis(0), 0.5);
        existing_pattern.kind = MentalObjectKind::Memory;
        existing_pattern.memory_roles.push(MemoryRole::Semantic);
        existing_pattern.embedding = Some(embedding_client.embed(repeated_text).await.unwrap());
        existing_pattern.promotion = aca_types::PromotionState::candidate(EpochMillis(0));
        let existing_id = existing_pattern.id;
        graph.insert(existing_pattern);

        let client = FixedChatClient { raw_text: repeated_text, confidence: 0.8, tier: Tier::T3 };
        synthesize(
            &mut graph,
            &[a_id, b_id],
            &empty_tier1(),
            &empty_tier2(),
            &TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            &client,
            &embedding_client,
            &SynthesisConfig::default(),
            "",
            EpochMillis(5_000),
            0.5,
            Duration::from_secs(5),
            0.4,
        )
        .await;

        let confirmed = graph.get(&existing_id).unwrap();
        assert_eq!(confirmed.promotion.status, aca_types::PromotionStatus::Confirmed, "an independent later touch should confirm the staged pattern");
        assert_eq!(confirmed.promotion.confirming_references, 1);
    }

    #[tokio::test]
    async fn falls_through_the_tier_ladder_like_reflect_does() {
        let mut graph = Graph::new();
        let a = episodic_object("episode one", EpochMillis(0), 0.5);
        let b = episodic_object("episode two", EpochMillis(0), 0.5);
        let (a_id, b_id) = (a.id, b.id);
        graph.insert(a);
        graph.insert(b);

        let tier2 = disagreeing_tier2(&["totally different", "not at all related", "something else entirely"]);
        let tier3_client = FixedChatClient { raw_text: "the deliberate synthesized answer", confidence: 0.9, tier: Tier::T3 };
        let embedding_client = FakeEmbeddingClient::default();

        let outcome = synthesize(
            &mut graph,
            &[a_id, b_id],
            &empty_tier1(),
            &tier2,
            &TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            &tier3_client,
            &embedding_client,
            &SynthesisConfig::default(),
            "",
            EpochMillis(1_000),
            0.5,
            Duration::from_secs(5),
            0.4,
        )
        .await;

        let new_id = outcome.expect("tier 3 should back stop a disagreeing tier 2");
        let pattern = graph.get(&new_id).unwrap();
        assert_eq!(pattern.tier_used, Some(Tier::T3));
        assert_eq!(pattern.text, "the deliberate synthesized answer");
    }

    #[tokio::test]
    async fn returns_none_on_an_empty_tier3_completion() {
        let mut graph = Graph::new();
        let a = episodic_object("episode one", EpochMillis(0), 0.5);
        let b = episodic_object("episode two", EpochMillis(0), 0.5);
        let (a_id, b_id) = (a.id, b.id);
        graph.insert(a);
        graph.insert(b);

        let client = FixedChatClient { raw_text: "", confidence: 0.5, tier: Tier::T3 };
        let embedding_client = FakeEmbeddingClient::default();
        let outcome = synthesize(
            &mut graph,
            &[a_id, b_id],
            &empty_tier1(),
            &empty_tier2(),
            &TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            &client,
            &embedding_client,
            &SynthesisConfig::default(),
            "",
            EpochMillis(1_000),
            0.5,
            Duration::from_secs(5),
            0.4,
        )
        .await;

        assert!(outcome.is_none(), "an empty completion should never produce a hollow Semantic memory");
        assert_eq!(graph.len(), 2, "nothing new should be inserted");
    }

    #[tokio::test]
    async fn a_near_duplicate_pattern_reinforces_the_existing_semantic_memory_instead_of_minting_a_new_one() {
        let mut graph = Graph::new();
        let a = episodic_object("episode one", EpochMillis(0), 0.5);
        let b = episodic_object("episode two", EpochMillis(0), 0.5);
        let (a_id, b_id) = (a.id, b.id);
        graph.insert(a);
        graph.insert(b);

        let embedding_client = FakeEmbeddingClient::default();
        let repeated_text = "a pattern already known";
        let mut existing_pattern = MentalObject::new_observation(repeated_text, EpochMillis(0), 0.5);
        existing_pattern.kind = MentalObjectKind::Memory;
        existing_pattern.memory_roles.push(MemoryRole::Semantic);
        existing_pattern.embedding = Some(embedding_client.embed(repeated_text).await.unwrap());
        let existing_id = existing_pattern.id;
        let initial_reference_count = existing_pattern.activation.reference_log.len();
        graph.insert(existing_pattern);

        let client = FixedChatClient { raw_text: repeated_text, confidence: 0.8, tier: Tier::T3 };
        let outcome = synthesize(
            &mut graph,
            &[a_id, b_id],
            &empty_tier1(),
            &empty_tier2(),
            &TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            &client,
            &embedding_client,
            &SynthesisConfig::default(),
            "",
            EpochMillis(5_000),
            0.5,
            Duration::from_secs(5),
            0.4,
        )
        .await;

        assert_eq!(outcome, Some(existing_id), "a near-duplicate pattern should reinforce the existing Semantic memory, not mint a new one");
        assert_eq!(graph.len(), 3, "no new Semantic memory object should have been inserted");
        let reinforced = graph.get(&existing_id).unwrap();
        assert!(reinforced.activation.reference_log.len() > initial_reference_count);

        for source_id in [a_id, b_id] {
            let source = graph.get(&source_id).unwrap();
            assert_eq!(source.edges.len(), 1);
            assert_eq!(source.edges[0].target_id, existing_id);
        }
    }
}
