use aca_graph::{Graph, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH};
use aca_tiers::{DivergentPool, GenerateRequest};
use aca_types::{MemoryRole, MentalObject, MentalObjectId, PromotionState};
use aca_util::{cosine_similarity, EpochMillis};

use crate::prompt_templates::{contradiction_check_prompt, memory_classification_prompt};
use crate::steps::confidence_revision::{self, ConfidenceRevisionConfig};

/// What happened to a candidate Mental Object when Memory Formation ran.
/// `Discarded` covers the one case that never needs classification (no
/// embedding to compare or store); `SemanticUpdate`/`BeliefRevision` are the
/// two outcomes classified by `classify_memory_category` (see its doc
/// comment) rather than by the Tier-0 similarity check `NewEpisodic`/
/// `Reinforced` still use unchanged.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryFormationOutcome {
    NewEpisodic { id: MentalObjectId },
    Reinforced { id: MentalObjectId },
    SemanticUpdate { id: MentalObjectId },
    BeliefRevision { id: MentalObjectId },
    Discarded,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryFormationConfig {
    /// Cosine similarity above which a candidate is treated as "the same
    /// memory again" (reinforce) rather than a genuinely new one.
    pub reinforcement_similarity_threshold: f32,
    /// Whether a freshly-classified `Semantic`/`SelfBelief` memory starts as
    /// an unconfirmed `PromotionStatus::Candidate` (see that type's own doc
    /// comment) rather than immediately trusted. `true` by design: nothing
    /// but `form_memory`'s own classifier has vouched for this content at
    /// the moment it's written, and readers that treat content as durable
    /// self-knowledge (`build_self_summary`, the Self Memory activation
    /// bonus) should not trust it until something outside that classifier
    /// independently touches it again. `false` reproduces the old
    /// unconditional-trust behavior exactly, for A/B comparison.
    pub promotion_gate_enabled: bool,
    /// Cosine similarity above which a freshly-classified Semantic/SelfBelief
    /// candidate is treated as the same claim as an existing object sharing
    /// that role, rather than a genuinely new one - the confirmation path a
    /// staged `Candidate` needs (see `PromotionState`'s own doc comment):
    /// without this, a `form_memory`-classified candidate has no realistic
    /// way to ever become `Confirmed`, since nothing else in this engine
    /// re-touches Semantic/SelfMemory content by content-similarity except
    /// this check and `steps::synthesize`'s own, separately-scoped dedup.
    pub promotion_confirmation_similarity_threshold: f32,
    /// Whether a Semantic/SelfBelief candidate that lands in
    /// `contradiction_similarity_band` gets checked against the matching
    /// existing memory for a genuine contradiction. `false` skips the check
    /// entirely (and its Tier-1 call) - old unconditional-coexistence
    /// behavior, for A/B comparison. See `detect_contradiction`'s own doc
    /// comment for why this needs real judgement rather than a Tier-0
    /// heuristic.
    pub contradiction_detection_enabled: bool,
    /// `(low, high)` cosine-similarity band a candidate must land in before
    /// `detect_contradiction` is even asked: below `low`, two pieces of
    /// content are just unrelated, not worth an LLM call. At/above `high`
    /// (matching `promotion_confirmation_similarity_threshold`), the
    /// confirmation scan above already claims the candidate as the same
    /// claim reinforced before contradiction detection ever runs. The band
    /// between is "similar enough to plausibly be about the same claim, not
    /// similar enough to already count as one."
    pub contradiction_similarity_band: (f32, f32),
    /// Whether a fresh Semantic/SelfBelief candidate can be refused the
    /// staged-`Candidate` path once too many already-staged, never-yet-
    /// confirmed candidates are outstanding at once. `false` reproduces the
    /// old unbounded-staging behavior exactly, for A/B comparison.
    pub candidate_budget_enabled: bool,
    /// How many `PromotionStatus::Candidate` objects can be outstanding in
    /// the graph at once before a fresh judgement-gated candidate is
    /// refused that path and stored as a plain Episodic memory instead - the
    /// population-level half of closing the self-obsession failure mode:
    /// each individual candidate can pass its own gate (`promotion_gate_
    /// enabled`) yet a sustained, never-confirmed stream would still
    /// accumulate without bound if nothing capped the total. Small by
    /// design, matching this codebase's existing "tens-low-hundreds of
    /// nodes" scale framing (see `executive::has_reflection_for`'s doc
    /// comment) - a handful of genuinely unconfirmed beliefs/patterns is
    /// already worth investigating, not something to let grow large before
    /// noticing.
    pub max_concurrent_candidates: usize,
}

impl Default for MemoryFormationConfig {
    fn default() -> Self {
        Self {
            reinforcement_similarity_threshold: 0.93,
            promotion_gate_enabled: true,
            promotion_confirmation_similarity_threshold: 0.93,
            contradiction_detection_enabled: true,
            contradiction_similarity_band: (0.75, 0.93),
            candidate_budget_enabled: true,
            max_concurrent_candidates: 20,
        }
    }
}

/// Step 8 (the "Remember" operator's handler): given a candidate that
/// already has a resolved embedding, either reinforces the most similar
/// existing episodic memory above threshold (a fresh reference restores its
/// activation, per ACT-R) or classifies *what kind* of new memory this is
/// (see `classify_memory_category`) and stores it accordingly. A candidate
/// with no embedding yet is discarded rather than stored un-embedded and
/// unfindable by recall.
///
/// The reinforcement check is deliberately still pure Tier-0 similarity,
/// unconditional and unclassified: a near-duplicate of an existing episodic
/// memory is always a reinforcement regardless of what a classifier might
/// say about it, matching specs.md's own framing ("the trigger is now a
/// computable event... the LLM is invoked to classify *how* an experience
/// should be written... not to decide *whether* anything happened at all") -
/// classification only ever decides *which kind of new* memory this is.
#[allow(clippy::too_many_arguments)]
pub async fn form_memory(
    graph: &mut Graph,
    candidate: MentalObject,
    config: &MemoryFormationConfig,
    confidence_revision_config: &ConfidenceRevisionConfig,
    tier1_pool: &DivergentPool,
    tier2_pool: &DivergentPool,
    self_summary: &str,
    now: EpochMillis,
    temperature: f32,
) -> MemoryFormationOutcome {
    let Some(candidate_embedding) = candidate.embedding.clone() else {
        return MemoryFormationOutcome::Discarded;
    };

    let most_similar_existing = graph
        .iter()
        .filter(|existing| existing.id != candidate.id && existing.memory_roles.contains(&MemoryRole::Episodic))
        .filter_map(|existing| {
            existing
                .embedding
                .as_ref()
                .map(|emb| (existing.id, cosine_similarity(&candidate_embedding, emb)))
        })
        .filter(|(_, similarity)| *similarity >= config.reinforcement_similarity_threshold)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((existing_id, _similarity)) = most_similar_existing {
        if let Some(existing_object) = graph.get_mut(&existing_id) {
            aca_graph::record_reference(&mut existing_object.activation, now);
        }
        return MemoryFormationOutcome::Reinforced { id: existing_id };
    }

    let category = classify_memory_category(tier1_pool, tier2_pool, &candidate.text, self_summary, temperature).await;
    let mut stored = candidate;
    let (mut role, mut outcome_for, mut is_judgement_gated): (MemoryRole, fn(MentalObjectId) -> MemoryFormationOutcome, bool) = match category {
        Some(MemoryCategory::Semantic) => (MemoryRole::Semantic, |id| MemoryFormationOutcome::SemanticUpdate { id }, true),
        Some(MemoryCategory::SelfBelief) => (MemoryRole::SelfMemory, |id| MemoryFormationOutcome::BeliefRevision { id }, true),
        Some(MemoryCategory::Episodic) | None => (MemoryRole::Episodic, |id| MemoryFormationOutcome::NewEpisodic { id }, false),
    };

    // The confirmation path a staged Candidate needs (see
    // `promotion_confirmation_similarity_threshold`'s own doc comment): a
    // second, independent touch on the same claim - by role, not just raw
    // text - either reinforces an already-`Confirmed` object or promotes a
    // still-`Candidate` one, in both cases in place of minting a duplicate.
    // Run only for the two judgement-gated categories; Episodic already has
    // its own unconditional Tier-0 reinforcement check above.
    if is_judgement_gated {
        let most_similar_role_matched = graph
            .iter()
            .filter(|existing| existing.id != stored.id && existing.memory_roles.contains(&role))
            .filter_map(|existing| existing.embedding.as_ref().map(|emb| (existing.id, cosine_similarity(&candidate_embedding, emb))))
            .filter(|(_, similarity)| *similarity >= config.promotion_confirmation_similarity_threshold)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((existing_id, _similarity)) = most_similar_role_matched {
            if let Some(existing_object) = graph.get_mut(&existing_id) {
                aca_graph::record_reference(&mut existing_object.activation, now);
                // Guards against the same formation event confirming its
                // own just-staged candidate - a genuinely later touch has
                // `now` strictly after the moment that candidate was staged.
                if now.0 > existing_object.promotion.staged_at.0 {
                    existing_object.promotion.confirm(now);
                }
            }
            return outcome_for(existing_id);
        }
    }

    // Closes the self-obsession failure mode at the population level, not
    // just the individual-memory level (see `max_concurrent_candidates`'s
    // own doc comment): a stream of individually-staged, never-confirmed
    // content must not accumulate without bound just because each one
    // passes its own gate. Never blocks Remember outright - only refuses
    // the risky staged-durable path, falling back to the same safe,
    // unconditional Episodic path a low-confidence classification already
    // uses.
    if is_judgement_gated && config.candidate_budget_enabled {
        let candidate_count = graph.iter().filter(|object| object.promotion.status == aca_types::PromotionStatus::Candidate).count();
        if candidate_count >= config.max_concurrent_candidates {
            role = MemoryRole::Episodic;
            outcome_for = |id| MemoryFormationOutcome::NewEpisodic { id };
            is_judgement_gated = false;
        }
    }

    if !stored.memory_roles.contains(&role) {
        stored.memory_roles.push(role);
    }
    // Episodic formation is a computable Tier-0 event, never judgement-gated
    // (see this function's own doc comment) - only a Semantic/SelfBelief
    // classification, the two outcomes the classifier's own judgement
    // actually decided, starts life unconfirmed.
    if is_judgement_gated && config.promotion_gate_enabled {
        stored.promotion = PromotionState::candidate(now);
    }
    if is_judgement_gated && config.contradiction_detection_enabled {
        if let Some((matched_id, contradicts)) = detect_contradiction(graph, stored.id, &stored.text, &candidate_embedding, config.contradiction_similarity_band, tier1_pool, self_summary, temperature).await {
            // `stored` isn't in `graph` yet (inserted just below), so
            // whichever edge this produces is unconditionally new - no
            // "already standing edge" check needed the way the goal-
            // outcome learner's own site needs one.
            if contradicts {
                // Both sides lose confidence, symmetrically: conflicting
                // evidence casts doubt on the established memory *and* on
                // the brand-new candidate walking in at full trust. Severe,
                // sustained contradiction can also demote an already-
                // Confirmed match back to Candidate (see
                // `apply_contradiction_penalty_and_maybe_demote`'s own doc
                // comment).
                aca_graph::reinforce_edge(&mut stored.edges, matched_id, aca_types::EdgeKind::Contradicts, now, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH);
                confidence_revision::apply_contradiction_penalty_and_maybe_demote(&mut stored, confidence_revision_config, now);
                if let Some(matched) = graph.get_mut(&matched_id) {
                    confidence_revision::apply_contradiction_penalty_and_maybe_demote(matched, confidence_revision_config, now);
                }
            } else {
                // A "consistent" vote is real evidence too, just weaker
                // than the role-matched confirmation scan's own bar
                // (related, not the same claim) - it raises confidence on
                // both sides but deliberately does not auto-confirm a
                // Candidate (see `apply_corroboration_bonus`'s own doc
                // comment on why this uses the bare, non-confirming form).
                aca_graph::reinforce_edge(&mut stored.edges, matched_id, aca_types::EdgeKind::Supports, now, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH);
                confidence_revision::apply_corroboration_bonus(&mut stored, confidence_revision_config);
                if let Some(matched) = graph.get_mut(&matched_id) {
                    confidence_revision::apply_corroboration_bonus(matched, confidence_revision_config);
                }
            }
        }
    }
    let id = stored.id;
    graph.insert(stored);
    outcome_for(id)
}

/// Below `band.0`, two pieces of content are just unrelated - not worth an
/// LLM call. At/above `band.1`, `form_memory`'s own confirmation scan
/// already claims a candidate this similar as the same claim reinforced
/// before this ever runs. Cosine similarity alone can't tell "same claim
/// reinforced" from "opposite claim" within that band - a negated statement
/// is often still highly similar to its source - so a cheap Tier-1
/// majority vote (`contradiction_check_prompt`) answers the one genuinely
/// judgement-requiring question: do these two claims actually conflict.
/// Checked against every existing Semantic/SelfMemory/Episodic object in
/// the band, not just objects sharing the candidate's own role - a new
/// Semantic belief can just as easily contradict an existing Episodic
/// memory as another Semantic one. Returns the single closest in-band
/// match's id paired with the vote (`true` = contradicts, `false` =
/// consistent) once a majority actually agrees either way, or `None` (no
/// in-band candidate, or the votes didn't clear agreement in either
/// direction).
async fn detect_contradiction(
    graph: &Graph,
    candidate_id: MentalObjectId,
    candidate_text: &str,
    candidate_embedding: &[f32],
    band: (f32, f32),
    tier1_pool: &DivergentPool,
    self_summary: &str,
    temperature: f32,
) -> Option<(MentalObjectId, bool)> {
    let (low, high) = band;
    let (closest_id, _similarity) = graph
        .iter()
        .filter(|existing| existing.id != candidate_id)
        .filter(|existing| existing.memory_roles.iter().any(|role| matches!(role, MemoryRole::Semantic | MemoryRole::SelfMemory | MemoryRole::Episodic)))
        .filter_map(|existing| existing.embedding.as_ref().map(|emb| (existing.id, cosine_similarity(candidate_embedding, emb))))
        .filter(|(_, similarity)| *similarity >= low && *similarity < high)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    let existing_text = graph.get(&closest_id)?.text.clone();

    let prompt = contradiction_check_prompt(self_summary, candidate_text, &existing_text);
    let req = GenerateRequest { prompt, temperature };
    let responses = tier1_pool.sample(req, CLASSIFICATION_MIN_SAMPLES).await.ok()?;
    let votes: Vec<bool> = responses.iter().filter_map(|r| parse_contradiction_vote(&r.raw_text)).collect();
    if votes.is_empty() {
        return None;
    }
    let contradicts_votes = votes.iter().filter(|&&v| v).count();
    let contradicts_fraction = contradicts_votes as f32 / votes.len() as f32;
    if contradicts_fraction >= TIER1_CLASSIFICATION_AGREEMENT_FRACTION {
        Some((closest_id, true))
    } else if (1.0 - contradicts_fraction) >= TIER1_CLASSIFICATION_AGREEMENT_FRACTION {
        Some((closest_id, false))
    } else {
        None
    }
}

/// Same "distinctive whole word, length-guarded against leaked/echoed
/// prompts" discipline as `parse_memory_category`. `"contradicts"` checked
/// before `"consistent"` only because both should never both appear in a
/// genuine short answer - order is not otherwise load-bearing.
fn parse_contradiction_vote(raw_text: &str) -> Option<bool> {
    let normalized = raw_text.trim().to_lowercase();
    if normalized.split_whitespace().count() > MAX_WORDS_FOR_CATEGORY_MATCH {
        return None;
    }
    if normalized.contains("contradicts") {
        Some(true)
    } else if normalized.contains("consistent") {
        Some(false)
    } else {
        None
    }
}

/// Phase 3 of the GWT-parity roadmap (`docs/cognitive-capability-audit.md`'s
/// second addendum): real global broadcast makes ignited content available
/// to multiple independent consumers at once, not one linear pipeline that
/// forces every consequence through a single competitive vote. Before this,
/// `Operator::Remember` only ever ran if it out-competed Speak/Ask/
/// ContinueReflecting/etc. for the Executive's one winning slot this tick -
/// a genuinely surprising broadcast winner could be *either* remembered
/// *or* spoken about in a given tick, never both, purely because they
/// shared one arbitration. This runs automatically, unconditionally,
/// outside Executive's proposal/selection machinery entirely, whenever
/// `precision_weighted_surprise` clears `threshold` - a real, independent
/// second consequence of the same broadcast winner, alongside whatever the
/// Executive separately decides to do communicatively.
///
/// Eligibility is the identical check `steps::executive::propose_operators`'s
/// own `Operator::Remember` proposal already uses (already remembered, or
/// no embedding to store) - not a stricter or looser bar, just an
/// unconditional one. The two paths compose cleanly rather than double-
/// booking: this pass runs first (see `loop_actor::tick()`'s call site), so
/// on any tick it actually fires, the object's `memory_roles` already
/// reflect it by the time `propose_operators` runs afterward, and
/// `already_remembered` stops `Operator::Remember` from being redundantly
/// re-proposed for the same object the same tick. Content that doesn't
/// clear `threshold` still has its ordinary shot at Remember winning
/// Executive's competition, exactly as before - this only ever adds a
/// second path, never removes the first.
///
/// Same `Reinforced` role-tagging `steps::act`'s `Operator::Remember` arm
/// already applies, duplicated here rather than shared: `act`'s version is
/// entangled with `ActOutcome`/`SilentReason` bookkeeping this automatic
/// path has no use for, so a small, honest duplication reads clearer than a
/// shared helper serving two structurally different callers.
#[allow(clippy::too_many_arguments)]
pub async fn maybe_automatic_remember(
    graph: &mut Graph,
    target_id: MentalObjectId,
    precision_weighted_surprise: Option<f32>,
    threshold: f32,
    config: &MemoryFormationConfig,
    confidence_revision_config: &ConfidenceRevisionConfig,
    tier1_pool: &DivergentPool,
    tier2_pool: &DivergentPool,
    self_summary: &str,
    now: EpochMillis,
    temperature: f32,
) -> Option<MemoryFormationOutcome> {
    let candidate = graph.get(&target_id)?.clone();
    let already_remembered = candidate.memory_roles.iter().any(|role| matches!(role, MemoryRole::Episodic | MemoryRole::Semantic | MemoryRole::SelfMemory));
    let remember_is_hopeless = candidate.embedding.is_none();
    if already_remembered || remember_is_hopeless || precision_weighted_surprise.unwrap_or(0.0) < threshold {
        return None;
    }

    let outcome = form_memory(graph, candidate, config, confidence_revision_config, tier1_pool, tier2_pool, self_summary, now, temperature).await;
    if let MemoryFormationOutcome::Reinforced { .. } = outcome {
        if let Some(object) = graph.get_mut(&target_id) {
            if !object.memory_roles.contains(&MemoryRole::Episodic) {
                object.memory_roles.push(MemoryRole::Episodic);
            }
        }
    }
    Some(outcome)
}

/// specs.md's "Where the Language Model Plugs In" point 2: classifying
/// *which* new-memory outcome applies (episodic / semantic / a Self Memory
/// belief) is exactly the kind of judgement call this MVP defers to a real
/// model rather than inventing a Tier-0 heuristic for. Default Tier 1
/// triage, escalating to Tier 2 only when Tier 1's samples don't agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MemoryCategory {
    Episodic,
    Semantic,
    SelfBelief,
}

/// How many independent samples are collected before voting - mirrors
/// `cognitive_core`'s `MIN_SAMPLES`, same reasoning: a solo sample is never
/// trusted alone, `DivergentPool::sample` makes up any shortfall by
/// repeat-calling configured clients.
const CLASSIFICATION_MIN_SAMPLES: usize = 3;
/// Tier 1 needs a strong supermajority before its classification is
/// trusted - confirmed live (see this module's own testing notes) that a
/// small model can confidently mis-classify a plainly semantic fact as
/// episodic, so a bare plurality isn't enough evidence.
const TIER1_CLASSIFICATION_AGREEMENT_FRACTION: f32 = 0.66;
/// Tier 2 (~9B models) get a plain-majority bar - more reliable than Tier 1,
/// so less unanimity is required to trust their vote.
const TIER2_CLASSIFICATION_AGREEMENT_FRACTION: f32 = 0.5;

/// Above this word count, a response is treated as unparseable prose rather
/// than the single-category answer the prompt asked for - confirmed live as
/// a real, not hypothetical, failure mode: a malformed/leaked completion
/// (see `aca_tiers::response::extract_json_candidate`'s own doc comment for
/// one confirmed way `raw_text` can end up holding far more than the model's
/// actual answer) that happens to echo back the classification prompt's own
/// instruction text will contain all three category words *and* the word
/// "self" (the prompt's own wording: "self-description," "self-
/// understanding") regardless of what the model actually meant to classify.
/// The prompt's own instruction is a single JSON envelope with a one-word
/// category - a genuine answer is short by construction, so anything this
/// long is far more likely leaked/echoed content than a real classification.
const MAX_WORDS_FOR_CATEGORY_MATCH: usize = 12;

fn parse_memory_category(raw_text: &str) -> Option<MemoryCategory> {
    let normalized = raw_text.trim().to_lowercase();
    if normalized.split_whitespace().count() > MAX_WORDS_FOR_CATEGORY_MATCH {
        return None;
    }
    // Matched on the literal category words the prompt actually asks for -
    // no longer a bare `contains("self")` fallback, which matched "itself,"
    // "yourself," "self-aware," and any other incidental use of "self" in
    // ordinary prose, not just a genuine belief classification (confirmed
    // live: this misclassified real conversational fragments as Self Memory
    // beliefs, permanently polluting the "Who you are" block prepended to
    // every future prompt).
    if normalized.contains("semantic") {
        Some(MemoryCategory::Semantic)
    } else if normalized.contains("belief") {
        Some(MemoryCategory::SelfBelief)
    } else if normalized.contains("episodic") {
        Some(MemoryCategory::Episodic)
    } else {
        None
    }
}

/// Samples `pool`, parses each response into a `MemoryCategory`, and
/// returns the plurality choice only if it clears `agreement_fraction` of
/// all *parseable* votes (unparseable responses are dropped, not counted
/// against agreement, but also never inflate it). `None` when the pool is
/// unconfigured, every sample fails, nothing parses, or the plurality is too
/// weak - all treated as "didn't earn a classification here," never an
/// error.
async fn classify_via_majority(pool: &DivergentPool, prompt: &str, agreement_fraction: f32, temperature: f32) -> Option<MemoryCategory> {
    if pool.is_empty() {
        return None;
    }
    let req = GenerateRequest { prompt: prompt.to_string(), temperature };
    let responses = pool.sample(req, CLASSIFICATION_MIN_SAMPLES).await.ok()?;

    let votes: Vec<MemoryCategory> = responses.iter().filter_map(|r| parse_memory_category(&r.raw_text)).collect();
    if votes.is_empty() {
        return None;
    }

    let mut counts: std::collections::HashMap<MemoryCategory, usize> = std::collections::HashMap::new();
    for vote in &votes {
        *counts.entry(*vote).or_insert(0) += 1;
    }
    let (winner, winner_count) = counts.into_iter().max_by_key(|(_, count)| *count)?;
    if winner_count as f32 / votes.len() as f32 >= agreement_fraction {
        Some(winner)
    } else {
        None
    }
}

/// Classifies what kind of new memory `candidate_text` should become. Tries
/// Tier 1 first, then Tier 2 on disagreement; an unconfigured or
/// non-agreeing pool at either tier just falls through - the caller's
/// default (`NewEpisodic`) is always a safe fallback, exactly what happened
/// unconditionally before this classifier existed, so "no confident
/// classification" is never worse than the pre-classifier behavior.
async fn classify_memory_category(tier1_pool: &DivergentPool, tier2_pool: &DivergentPool, candidate_text: &str, self_summary: &str, temperature: f32) -> Option<MemoryCategory> {
    let prompt = memory_classification_prompt(self_summary, candidate_text);
    if let Some(category) = classify_via_majority(tier1_pool, &prompt, TIER1_CLASSIFICATION_AGREEMENT_FRACTION, temperature).await {
        return Some(category);
    }
    classify_via_majority(tier2_pool, &prompt, TIER2_CLASSIFICATION_AGREEMENT_FRACTION, temperature).await
}

/// Reinforces the associative edge between two Mental Objects that
/// co-occurred (e.g. broadcast together, or one reminding Memory Formation
/// of the other) — the Hebbian co-activation step ACT-R's `S_ki` depends
/// on. Exposed separately from `form_memory` because co-activation can
/// happen between any two objects, not only during memory formation.
pub fn reinforce_coactivation(graph: &mut Graph, from: MentalObjectId, to: MentalObjectId, now: EpochMillis) {
    if let Some(source) = graph.get_mut(&from) {
        aca_graph::reinforce_edge(
            &mut source.edges,
            to,
            aca_types::EdgeKind::Associative,
            now,
            DEFAULT_HEBBIAN_INCREMENT,
            DEFAULT_MAX_EDGE_STRENGTH,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_tiers::{ChatClient, TierError, TierResponse};
    use aca_types::{ObjectStatus, Tier};
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::time::Duration;

    fn object_with_embedding(text: &str, embedding: Vec<f32>, now: EpochMillis) -> MentalObject {
        let mut object = MentalObject::new_observation(text, now, 0.5);
        object.embedding = Some(embedding);
        object
    }

    fn empty_tier1() -> DivergentPool {
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![])
    }

    fn empty_tier2() -> DivergentPool {
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![])
    }

    /// `n` clients that all classify a candidate the same way - simulates
    /// independent samples that actually agree.
    fn agreeing_classifier_pool(tier: Tier, category_text: &str, n: usize) -> DivergentPool {
        struct FixedClassifier {
            text: String,
            tier: Tier,
        }
        #[async_trait]
        impl ChatClient for FixedClassifier {
            async fn generate(&self, _req: aca_tiers::GenerateRequest) -> Result<TierResponse, TierError> {
                Ok(TierResponse { raw_text: self.text.clone(), confidence: 0.9, tier: self.tier })
            }
        }
        let clients = (0..n).map(|_| Arc::new(FixedClassifier { text: category_text.to_string(), tier }) as Arc<dyn ChatClient>).collect();
        DivergentPool::new(tier, Duration::from_secs(5), clients)
    }

    #[tokio::test]
    async fn automatic_remember_fires_when_surprise_clears_threshold() {
        let mut graph = Graph::new();
        let object = object_with_embedding("a genuinely surprising thing", vec![0.1, 0.2, 0.3], EpochMillis(0));
        let id = object.id;
        graph.insert(object);

        let outcome = maybe_automatic_remember(&mut graph, id, Some(2.0), 1.0, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;

        assert!(matches!(outcome, Some(MemoryFormationOutcome::NewEpisodic { .. })));
        assert!(graph.get(&id).unwrap().memory_roles.contains(&MemoryRole::Episodic));
    }

    #[tokio::test]
    async fn automatic_remember_does_not_fire_below_the_surprise_threshold() {
        let mut graph = Graph::new();
        let object = object_with_embedding("only mildly interesting", vec![0.1, 0.2, 0.3], EpochMillis(0));
        let id = object.id;
        graph.insert(object);

        let outcome = maybe_automatic_remember(&mut graph, id, Some(0.5), 1.0, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;

        assert!(outcome.is_none());
        assert!(!graph.get(&id).unwrap().memory_roles.contains(&MemoryRole::Episodic), "content below the automatic threshold should be left untouched, not silently remembered anyway");
    }

    #[tokio::test]
    async fn automatic_remember_does_not_re_fire_for_an_already_remembered_object() {
        let mut graph = Graph::new();
        let mut object = object_with_embedding("already remembered earlier", vec![0.1, 0.2, 0.3], EpochMillis(0));
        object.memory_roles.push(MemoryRole::Episodic);
        let id = object.id;
        graph.insert(object);

        let outcome = maybe_automatic_remember(&mut graph, id, Some(2.0), 1.0, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;

        assert!(outcome.is_none(), "already-remembered content should never be re-processed, no matter how surprising this tick's reading is");
    }

    #[tokio::test]
    async fn automatic_remember_does_not_fire_for_a_candidate_with_no_embedding() {
        let mut graph = Graph::new();
        let object = MentalObject::new_observation("no embedding yet", EpochMillis(0), 0.5);
        let id = object.id;
        graph.insert(object);

        let outcome = maybe_automatic_remember(&mut graph, id, Some(2.0), 1.0, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;

        assert!(outcome.is_none(), "an embedding-less candidate has nothing to store or compare - same hopeless case form_memory itself discards");
    }

    #[tokio::test]
    async fn automatic_remember_is_a_graceful_no_op_for_a_missing_target() {
        let mut graph = Graph::new();
        let outcome = maybe_automatic_remember(&mut graph, MentalObjectId::new(), Some(2.0), 1.0, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert!(outcome.is_none());
    }

    #[tokio::test]
    async fn candidate_without_embedding_is_discarded() {
        let mut graph = Graph::new();
        let candidate = MentalObject::new_observation("no embedding", EpochMillis(0), 0.5);
        let outcome = form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert_eq!(outcome, MemoryFormationOutcome::Discarded);
        assert!(graph.is_empty());
    }

    #[tokio::test]
    async fn novel_candidate_becomes_a_new_episodic_memory_when_unclassified() {
        // No tier pools configured - classify_memory_category can't run, so
        // this must fall back to the pre-classifier default, exactly as it
        // always did.
        let mut graph = Graph::new();
        let candidate = object_with_embedding("first ever memory", vec![1.0, 0.0, 0.0], EpochMillis(0));
        let id = candidate.id;
        let outcome = form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert_eq!(outcome, MemoryFormationOutcome::NewEpisodic { id });
        let stored = graph.get(&id).unwrap();
        assert!(stored.memory_roles.contains(&MemoryRole::Episodic));
        assert_eq!(stored.status, ObjectStatus::Active);
    }

    #[tokio::test]
    async fn a_semantic_classification_stores_the_candidate_as_semantic_memory() {
        let mut graph = Graph::new();
        let candidate = object_with_embedding("rust panics unwind the stack by default", vec![1.0, 0.0, 0.0], EpochMillis(0));
        let id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "semantic", 3);
        let outcome = form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert_eq!(outcome, MemoryFormationOutcome::SemanticUpdate { id });
        assert!(graph.get(&id).unwrap().memory_roles.contains(&MemoryRole::Semantic));
    }

    #[tokio::test]
    async fn a_semantic_classification_starts_as_an_unconfirmed_candidate() {
        let mut graph = Graph::new();
        let candidate = object_with_embedding("rust panics unwind the stack by default", vec![1.0, 0.0, 0.0], EpochMillis(1_000));
        let id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "semantic", 3);
        form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;
        let stored = graph.get(&id).unwrap();
        assert_eq!(stored.promotion.status, aca_types::PromotionStatus::Candidate, "nothing but form_memory's own classifier has vouched for this yet");
        assert_eq!(stored.promotion.staged_at, EpochMillis(1_000));
    }

    #[tokio::test]
    async fn a_new_episodic_memory_is_confirmed_immediately() {
        // Episodic formation is a computable Tier-0 event, never
        // judgement-gated - see form_memory's own doc comment. It must not
        // start as an unconfirmed Candidate just because Semantic/SelfBelief
        // now can.
        let mut graph = Graph::new();
        let candidate = object_with_embedding("first ever memory", vec![1.0, 0.0, 0.0], EpochMillis(0));
        let id = candidate.id;
        form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert_eq!(graph.get(&id).unwrap().promotion.status, aca_types::PromotionStatus::Confirmed);
    }

    #[tokio::test]
    async fn a_later_similar_belief_confirms_an_existing_staged_candidate_instead_of_duplicating() {
        let mut graph = Graph::new();
        let tier1 = agreeing_classifier_pool(Tier::T1, "belief", 3);

        let first = object_with_embedding("I value honesty over comfort", vec![1.0, 0.0, 0.0], EpochMillis(0));
        let first_id = first.id;
        form_memory(&mut graph, first, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert_eq!(graph.get(&first_id).unwrap().promotion.status, aca_types::PromotionStatus::Candidate);

        // A near-duplicate phrasing, on a later tick.
        let second = object_with_embedding("I value honesty over comfort, still", vec![1.0, 0.0, 0.0001], EpochMillis(5_000));
        let outcome = form_memory(&mut graph, second, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(5_000), 0.3).await;

        assert_eq!(outcome, MemoryFormationOutcome::BeliefRevision { id: first_id }, "should point back at the existing candidate, not mint a duplicate");
        assert_eq!(graph.len(), 1);
        let confirmed = graph.get(&first_id).unwrap();
        assert_eq!(confirmed.promotion.status, aca_types::PromotionStatus::Confirmed, "an independent later touch should confirm the staged candidate");
        assert_eq!(confirmed.promotion.confirming_references, 1);
    }

    #[tokio::test]
    async fn a_same_tick_self_match_does_not_confirm_its_own_candidate() {
        // Regression guard: a candidate must not be able to confirm itself
        // by matching some other object staged in the very same formation
        // event - only a genuinely later touch counts.
        let mut graph = Graph::new();
        let tier1 = agreeing_classifier_pool(Tier::T1, "belief", 3);

        let first = object_with_embedding("I value honesty over comfort", vec![1.0, 0.0, 0.0], EpochMillis(1_000));
        let first_id = first.id;
        form_memory(&mut graph, first, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        // A near-duplicate arriving at the *same* instant.
        let second = object_with_embedding("I value honesty over comfort, still", vec![1.0, 0.0, 0.0001], EpochMillis(1_000));
        form_memory(&mut graph, second, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        assert_eq!(graph.get(&first_id).unwrap().promotion.status, aca_types::PromotionStatus::Candidate, "a same-instant match must not confirm");
    }

    #[tokio::test]
    async fn a_later_similar_belief_just_reinforces_an_already_confirmed_one() {
        let mut graph = Graph::new();
        let tier1 = agreeing_classifier_pool(Tier::T1, "belief", 3);

        let mut first = object_with_embedding("I value honesty over comfort", vec![1.0, 0.0, 0.0], EpochMillis(0));
        first.memory_roles.push(MemoryRole::SelfMemory);
        first.promotion = aca_types::PromotionState::confirmed(EpochMillis(0));
        let first_id = first.id;
        graph.insert(first);

        let second = object_with_embedding("I value honesty over comfort, still", vec![1.0, 0.0, 0.0001], EpochMillis(5_000));
        let outcome = form_memory(&mut graph, second, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(5_000), 0.3).await;

        assert_eq!(outcome, MemoryFormationOutcome::BeliefRevision { id: first_id });
        let reinforced = graph.get(&first_id).unwrap();
        assert_eq!(reinforced.promotion.confirming_references, 0, "already-confirmed content has nothing left to count");
        assert!(reinforced.activation.reference_log.len() > 1, "the match should still restore ACT-R activation");
    }

    #[tokio::test]
    async fn promotion_gate_disabled_reproduces_the_old_unconditional_trust_behavior() {
        let mut graph = Graph::new();
        let candidate = object_with_embedding("I value honesty over comfort", vec![1.0, 0.0, 0.0], EpochMillis(0));
        let id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "belief", 3);
        let config = MemoryFormationConfig { promotion_gate_enabled: false, ..MemoryFormationConfig::default() };
        form_memory(&mut graph, candidate, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert_eq!(graph.get(&id).unwrap().promotion.status, aca_types::PromotionStatus::Confirmed, "ablated gate must reproduce the pre-gate unconditional-trust behavior exactly");
    }

    /// `n` already-staged Candidate Semantic memories, guaranteed never to
    /// match any 3-dimensional test embedding via the confirmation or
    /// contradiction scans: `cosine_similarity` returns exactly `0.0` for
    /// mismatched vector lengths (`aca_util::cosine_similarity`'s own
    /// implementation), so a 4-dimensional filler embedding can never
    /// collide with this file's 3-dimensional test fixtures regardless of
    /// direction.
    fn n_staged_candidates(graph: &mut Graph, n: usize) {
        for i in 0..n {
            let mut object = object_with_embedding(&format!("staged candidate {i}"), vec![0.0, 0.0, 0.0, i as f32 + 1.0], EpochMillis(0));
            object.memory_roles.push(MemoryRole::Semantic);
            object.promotion = PromotionState::candidate(EpochMillis(0));
            graph.insert(object);
        }
    }

    #[tokio::test]
    async fn a_candidate_over_budget_falls_back_to_episodic_instead_of_staging() {
        let mut graph = Graph::new();
        n_staged_candidates(&mut graph, 2);
        let config = MemoryFormationConfig { max_concurrent_candidates: 2, ..MemoryFormationConfig::default() };

        let candidate = object_with_embedding("rust panics unwind the stack by default", vec![1.0, 0.0, 0.0], EpochMillis(1_000));
        let id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "semantic", 3);
        let outcome = form_memory(&mut graph, candidate, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        assert_eq!(outcome, MemoryFormationOutcome::NewEpisodic { id }, "at budget, a judgement-gated classification must fall back to the safe Episodic path");
        let stored = graph.get(&id).unwrap();
        assert!(stored.memory_roles.contains(&MemoryRole::Episodic));
        assert!(!stored.memory_roles.contains(&MemoryRole::Semantic), "must not carry the demoted role too");
        assert_eq!(stored.promotion.status, aca_types::PromotionStatus::Confirmed, "the Episodic fallback is never judgement-gated, so it starts trusted like any other Episodic memory");
    }

    #[tokio::test]
    async fn a_candidate_under_budget_stages_normally() {
        let mut graph = Graph::new();
        n_staged_candidates(&mut graph, 1);
        let config = MemoryFormationConfig { max_concurrent_candidates: 2, ..MemoryFormationConfig::default() };

        let candidate = object_with_embedding("rust panics unwind the stack by default", vec![1.0, 0.0, 0.0], EpochMillis(1_000));
        let id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "semantic", 3);
        let outcome = form_memory(&mut graph, candidate, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        assert_eq!(outcome, MemoryFormationOutcome::SemanticUpdate { id }, "below budget, the classifier's own judgement should still be trusted");
        assert_eq!(graph.get(&id).unwrap().promotion.status, aca_types::PromotionStatus::Candidate);
    }

    #[tokio::test]
    async fn confirming_a_staged_candidate_frees_a_budget_slot_for_the_next_one() {
        let mut graph = Graph::new();
        n_staged_candidates(&mut graph, 1);
        let config = MemoryFormationConfig { max_concurrent_candidates: 2, ..MemoryFormationConfig::default() };
        let tier1 = agreeing_classifier_pool(Tier::T1, "semantic", 3);

        // Fills the budget (1 pre-staged + this one = 2).
        let first = object_with_embedding("rust panics unwind the stack by default", vec![1.0, 0.0, 0.0], EpochMillis(1_000));
        let first_id = first.id;
        form_memory(&mut graph, first, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;
        assert_eq!(graph.get(&first_id).unwrap().promotion.status, aca_types::PromotionStatus::Candidate);

        // A third would be over budget and fall back to Episodic...
        let over_budget = object_with_embedding("the ocean covers most of the earth", vec![0.0, 1.0, 0.0], EpochMillis(2_000));
        let over_budget_id = over_budget.id;
        form_memory(&mut graph, over_budget, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(2_000), 0.3).await;
        assert!(graph.get(&over_budget_id).unwrap().memory_roles.contains(&MemoryRole::Episodic));

        // ...but confirming the first candidate (a later, independent touch
        // on the same claim) frees a slot for the next one.
        let confirming = object_with_embedding("rust panics unwind the stack by default, still", vec![1.0, 0.0001, 0.0], EpochMillis(3_000));
        form_memory(&mut graph, confirming, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(3_000), 0.3).await;
        assert_eq!(graph.get(&first_id).unwrap().promotion.status, aca_types::PromotionStatus::Confirmed);

        let freed_slot = object_with_embedding("water boils at 100 degrees celsius", vec![0.0, 0.0, 1.0], EpochMillis(4_000));
        let freed_slot_id = freed_slot.id;
        let outcome = form_memory(&mut graph, freed_slot, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(4_000), 0.3).await;
        assert_eq!(outcome, MemoryFormationOutcome::SemanticUpdate { id: freed_slot_id }, "confirming the earlier candidate should have freed a slot for this one to stage normally");
        assert_eq!(graph.get(&freed_slot_id).unwrap().promotion.status, aca_types::PromotionStatus::Candidate);
    }

    #[tokio::test]
    async fn candidate_budget_disabled_never_falls_back() {
        let mut graph = Graph::new();
        n_staged_candidates(&mut graph, 5);
        let config = MemoryFormationConfig { candidate_budget_enabled: false, max_concurrent_candidates: 2, ..MemoryFormationConfig::default() };

        let candidate = object_with_embedding("rust panics unwind the stack by default", vec![1.0, 0.0, 0.0], EpochMillis(1_000));
        let id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "semantic", 3);
        let outcome = form_memory(&mut graph, candidate, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        assert_eq!(outcome, MemoryFormationOutcome::SemanticUpdate { id }, "ablated budget must reproduce the old unbounded-staging behavior exactly");
    }

    #[test]
    fn parse_memory_category_no_longer_matches_incidental_uses_of_self() {
        // Regression guard for the confirmed-live contamination: prose that
        // merely contains "self" somewhere (not the model's actual "belief"
        // classification) must never be read as a Self Memory vote.
        assert_eq!(parse_memory_category("the user is describing something about themselves"), None);
        assert_eq!(parse_memory_category("this seems self-aware in tone"), None);
        assert_eq!(parse_memory_category("yourself"), None);
    }

    #[test]
    fn parse_memory_category_rejects_long_echoed_or_leaked_text() {
        // A real classification answer is a single word or short phrase by
        // construction (the prompt asks for exactly one of episodic/
        // semantic/belief). A long blob - even one that happens to contain
        // all three category words, as an echoed copy of the classification
        // prompt's own instructions would - must not be pattern-matched at
        // all; it's far more likely leaked/echoed content than a genuine
        // answer.
        let leaked = "\"episodic\": a specific event tied to a moment. \"semantic\": a general pattern. \
                       \"belief\": something about Omega's own identity, values, preferences, or self-understanding.";
        assert_eq!(parse_memory_category(leaked), None);
    }

    #[tokio::test]
    async fn a_belief_classification_stores_the_candidate_as_self_memory() {
        let mut graph = Graph::new();
        let candidate = object_with_embedding("I value honesty over comfort", vec![1.0, 0.0, 0.0], EpochMillis(0));
        let id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "belief", 3);
        let outcome = form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(0), 0.3).await;
        assert_eq!(outcome, MemoryFormationOutcome::BeliefRevision { id });
        assert!(graph.get(&id).unwrap().memory_roles.contains(&MemoryRole::SelfMemory));
    }

    #[tokio::test]
    async fn a_hung_tier1_vote_falls_through_to_tier2() {
        struct SplitVoteClient(&'static str, Tier);
        #[async_trait]
        impl ChatClient for SplitVoteClient {
            async fn generate(&self, _req: aca_tiers::GenerateRequest) -> Result<TierResponse, TierError> {
                Ok(TierResponse { raw_text: self.0.to_string(), confidence: 0.9, tier: self.1 })
            }
        }
        let mut graph = Graph::new();
        let candidate = object_with_embedding("ambiguous content", vec![1.0, 0.0, 0.0], EpochMillis(0));
        let id = candidate.id;
        // Three-way split at Tier 1 - no plurality clears the agreement bar.
        let tier1 = DivergentPool::new(
            Tier::T1,
            Duration::from_secs(5),
            vec![
                Arc::new(SplitVoteClient("episodic", Tier::T1)),
                Arc::new(SplitVoteClient("semantic", Tier::T1)),
                Arc::new(SplitVoteClient("belief", Tier::T1)),
            ],
        );
        let tier2 = agreeing_classifier_pool(Tier::T2, "semantic", 3);
        let outcome = form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &tier2, "", EpochMillis(0), 0.3).await;
        assert_eq!(outcome, MemoryFormationOutcome::SemanticUpdate { id }, "Tier 1's split vote should fall through to Tier 2's agreeing one");
    }

    #[tokio::test]
    async fn near_duplicate_candidate_reinforces_the_existing_memory_instead() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("cats are great", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Episodic);
        let existing_id = existing.id;
        let initial_reference_count = existing.activation.reference_log.len();
        graph.insert(existing);

        let candidate = object_with_embedding("cats are great", vec![1.0, 0.0, 0.0001], EpochMillis(5_000));
        let outcome = form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(5_000), 0.3).await;

        assert_eq!(outcome, MemoryFormationOutcome::Reinforced { id: existing_id });
        assert_eq!(graph.len(), 1, "reinforcement must not create a second object");
        let reinforced = graph.get(&existing_id).unwrap();
        assert!(reinforced.activation.reference_log.len() > initial_reference_count);
    }

    #[tokio::test]
    async fn dissimilar_candidate_does_not_reinforce_an_unrelated_memory() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("cats are great", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Episodic);
        graph.insert(existing);

        let candidate = object_with_embedding("the stock market fell today", vec![0.0, 1.0, 0.0], EpochMillis(1_000));
        let candidate_id = candidate.id;
        let outcome = form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &empty_tier1(), &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        assert_eq!(outcome, MemoryFormationOutcome::NewEpisodic { id: candidate_id });
        assert_eq!(graph.len(), 2, "an unrelated candidate must become its own memory");
    }

    /// `form_memory` shares one `tier1_pool` between `classify_memory_category`
    /// (parses "episodic"/"semantic"/"belief") and, once a candidate lands in
    /// the contradiction band, `detect_contradiction` (parses "contradicts"/
    /// "consistent") - a fixed single-answer pool can't serve both real
    /// prompts at once. This discriminates on prompt content so a test can
    /// drive both calls deterministically through the one pool argument, the
    /// same way a real multi-purpose model would answer each prompt on its
    /// own terms.
    struct PromptDiscriminatingClient {
        contradiction_answer: &'static str,
    }
    #[async_trait]
    impl ChatClient for PromptDiscriminatingClient {
        async fn generate(&self, req: aca_tiers::GenerateRequest) -> Result<TierResponse, TierError> {
            let raw_text = if req.prompt.contains("genuinely contradict") { self.contradiction_answer } else { "semantic" };
            Ok(TierResponse { raw_text: raw_text.to_string(), confidence: 0.9, tier: Tier::T1 })
        }
    }
    fn discriminating_pool(contradiction_answer: &'static str) -> DivergentPool {
        let clients = (0..3).map(|_| Arc::new(PromptDiscriminatingClient { contradiction_answer }) as Arc<dyn ChatClient>).collect();
        DivergentPool::new(Tier::T1, Duration::from_secs(5), clients)
    }

    #[tokio::test]
    async fn a_genuinely_opposed_candidate_in_band_produces_a_contradicts_edge() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("the kitchen light is off", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Semantic);
        let existing_id = existing.id;
        graph.insert(existing);

        // Cosine similarity with [1,0,0] is exactly 0.8 - inside the default
        // (0.75, 0.93) band: plausibly about the same claim, not similar
        // enough to already be a duplicate.
        let candidate = object_with_embedding("the kitchen light is on", vec![0.8, 0.6, 0.0], EpochMillis(1_000));
        let candidate_id = candidate.id;
        let tier1 = discriminating_pool("contradicts");
        form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        let stored = graph.get(&candidate_id).unwrap();
        assert!(stored.memory_roles.contains(&MemoryRole::Semantic));
        assert_eq!(stored.edges.len(), 1, "a genuinely opposed in-band candidate should gain a Contradicts edge");
        assert_eq!(stored.edges[0].target_id, existing_id);
        assert_eq!(stored.edges[0].kind, aca_types::EdgeKind::Contradicts);
    }

    #[tokio::test]
    async fn a_confirmed_contradiction_lowers_confidence_on_both_sides() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("the kitchen light is off", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Semantic);
        existing.confidence = 0.8;
        let existing_id = existing.id;
        graph.insert(existing);

        let candidate = object_with_embedding("the kitchen light is on", vec![0.8, 0.6, 0.0], EpochMillis(1_000));
        let candidate_id = candidate.id;
        let tier1 = discriminating_pool("contradicts");
        form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        let revision = ConfidenceRevisionConfig::default();
        let existing_confidence = graph.get(&existing_id).unwrap().confidence;
        assert!((existing_confidence - (0.8 - revision.contradiction_penalty)).abs() < 1e-6, "the established memory should lose confidence to the new contradiction, got {existing_confidence}");
        let candidate_confidence = graph.get(&candidate_id).unwrap().confidence;
        // object_with_embedding's default confidence is 0.5 (new_observation's own default).
        assert!((candidate_confidence - (0.5 - revision.contradiction_penalty)).abs() < 1e-6, "the brand-new candidate should not walk in at full trust either, got {candidate_confidence}");
    }

    #[tokio::test]
    async fn an_in_band_candidate_the_vote_calls_consistent_gets_a_supports_edge_and_a_confidence_boost() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("the kitchen light is off", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Semantic);
        existing.confidence = 0.5;
        let existing_id = existing.id;
        graph.insert(existing);

        let candidate = object_with_embedding("the kitchen light switch was replaced", vec![0.8, 0.6, 0.0], EpochMillis(1_000));
        let candidate_id = candidate.id;
        let tier1 = discriminating_pool("consistent");
        form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        let stored = graph.get(&candidate_id).unwrap();
        assert_eq!(stored.edges.len(), 1, "an in-band pair the vote calls consistent should gain a Supports edge, not none");
        assert_eq!(stored.edges[0].target_id, existing_id);
        assert_eq!(stored.edges[0].kind, aca_types::EdgeKind::Supports);

        let revision = ConfidenceRevisionConfig::default();
        let existing_confidence = graph.get(&existing_id).unwrap().confidence;
        assert!((existing_confidence - (0.5 + revision.corroboration_bonus)).abs() < 1e-6, "the established memory should gain confidence from consistent evidence, got {existing_confidence}");
        // object_with_embedding's default confidence is 0.5 (new_observation's own default).
        assert!((stored.confidence - (0.5 + revision.corroboration_bonus)).abs() < 1e-6, "the brand-new candidate should also benefit from consistent evidence, got {}", stored.confidence);
    }

    #[tokio::test]
    async fn consistent_evidence_does_not_auto_confirm_a_staged_candidate() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("the kitchen light is off", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Semantic);
        existing.promotion = PromotionState::candidate(EpochMillis(0));
        let existing_id = existing.id;
        graph.insert(existing);

        let candidate = object_with_embedding("the kitchen light switch was replaced", vec![0.8, 0.6, 0.0], EpochMillis(1_000));
        let tier1 = discriminating_pool("consistent");
        form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        assert_eq!(
            graph.get(&existing_id).unwrap().promotion.status,
            aca_types::PromotionStatus::Candidate,
            "a related-but-different-claim corroboration is weaker evidence than the confirmation scan's own bar and must not auto-confirm"
        );
    }

    #[tokio::test]
    async fn sustained_contradiction_demotes_an_existing_confirmed_object() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("the kitchen light is off", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Semantic);
        existing.confidence = 0.5;
        let existing_id = existing.id;
        graph.insert(existing);

        let tier1 = discriminating_pool("contradicts");

        // Two different in-band candidates (0.8 and ~0.85 cosine similarity
        // with `existing`, but only ~0.68 with each other - safely below
        // the band, so the second formation's own contradiction check
        // targets `existing` again rather than cross-matching the first
        // candidate).
        let candidate_a = object_with_embedding("the kitchen light is on", vec![0.8, 0.6, 0.0], EpochMillis(1_000));
        form_memory(&mut graph, candidate_a, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;
        assert_eq!(
            graph.get(&existing_id).unwrap().promotion.status,
            aca_types::PromotionStatus::Confirmed,
            "a single contradiction (0.5 -> 0.3 confidence) must not demote outright"
        );

        let candidate_b = object_with_embedding("the kitchen light switch is broken", vec![0.85, 0.0, 0.527], EpochMillis(2_000));
        form_memory(&mut graph, candidate_b, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(2_000), 0.3).await;

        assert_eq!(
            graph.get(&existing_id).unwrap().promotion.status,
            aca_types::PromotionStatus::Candidate,
            "sustained contradiction (0.3 -> 0.1 confidence, crossing demotion_confidence_threshold) should invalidate the promotion cache"
        );
    }

    #[tokio::test]
    async fn a_merely_different_candidate_below_the_band_produces_no_contradiction_check() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("the kitchen light is off", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Semantic);
        graph.insert(existing);

        // Cosine similarity with [1,0,0] is exactly 0.5 - below the band's
        // low end, so no contradiction check should even be attempted
        // (proven by using a tier1 pool that would panic-worthy-wrongly
        // answer "contradicts" for everything, yet no edge appears).
        let candidate = object_with_embedding("the stock market fell today", vec![0.5, 0.866, 0.0], EpochMillis(1_000));
        let candidate_id = candidate.id;
        let tier1 = agreeing_classifier_pool(Tier::T1, "semantic", 3);
        form_memory(&mut graph, candidate, &MemoryFormationConfig::default(), &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        let stored = graph.get(&candidate_id).unwrap();
        assert!(stored.edges.is_empty(), "a below-band candidate is unrelated enough that no contradiction check - and so no edge - should ever happen");
    }

    #[tokio::test]
    async fn contradiction_detection_disabled_never_produces_an_edge() {
        let mut graph = Graph::new();
        let mut existing = object_with_embedding("the kitchen light is off", vec![1.0, 0.0, 0.0], EpochMillis(0));
        existing.memory_roles.push(MemoryRole::Semantic);
        graph.insert(existing);

        let candidate = object_with_embedding("the kitchen light is on", vec![0.8, 0.6, 0.0], EpochMillis(1_000));
        let candidate_id = candidate.id;
        let tier1 = discriminating_pool("contradicts");
        let config = MemoryFormationConfig { contradiction_detection_enabled: false, ..MemoryFormationConfig::default() };
        form_memory(&mut graph, candidate, &config, &ConfidenceRevisionConfig::default(), &tier1, &empty_tier2(), "", EpochMillis(1_000), 0.3).await;

        assert!(graph.get(&candidate_id).unwrap().edges.is_empty(), "ablated detection must never produce an edge even when the vote would have confirmed one");
    }

    #[test]
    fn reinforce_coactivation_creates_an_associative_edge() {
        let mut graph = Graph::new();
        let a = MentalObject::new_observation("a", EpochMillis(0), 0.5);
        let b = MentalObject::new_observation("b", EpochMillis(0), 0.5);
        let (a_id, b_id) = (a.id, b.id);
        graph.insert(a);
        graph.insert(b);

        reinforce_coactivation(&mut graph, a_id, b_id, EpochMillis(1_000));

        let a_object = graph.get(&a_id).unwrap();
        assert_eq!(a_object.edges.len(), 1);
        assert_eq!(a_object.edges[0].target_id, b_id);
    }
}
