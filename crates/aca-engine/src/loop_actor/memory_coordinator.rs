use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use aca_graph::Graph;
use aca_store::{KnowledgeLibraryStore, MemoryStore};
use aca_types::{MemoryRole, MentalObject, MentalObjectId, PromotionStatus};
use aca_util::EpochMillis;

use crate::steps::compare::SourceChannel;
use crate::steps::eligibility::EligibilityTraceRegistry;
use crate::steps::known_answers::{CuratedAnswer, curated_answer_index, static_question_key};
use crate::steps::procedural::{CompiledProcedure, compiled_index, compiled_procedure, routine_key};
use crate::steps::spikes::SpikeEventQueue;

/// How many Self Memory items `build_self_summary` renders - capped, not
/// exhaustive: this string is prepended to every single Tier 1-4 prompt, so
/// its size is a real, permanent cost multiplier on every call, not a
/// one-off. A handful of the most currently-active beliefs/values/goals is
/// enough to give the Cognitive Core real identity grounding without turning
/// every prompt into a full Self Memory dump.
const SELF_SUMMARY_MAX_ITEMS: usize = 6;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PreconsciousTrace {
    pub(crate) next_evaluation_at: EpochMillis,
    pub(crate) expires_at: EpochMillis,
    pub(crate) surprise: Option<f32>,
}

/// Builds a short, deterministic summary of Self Memory content (specs.md's
/// "beliefs, values, preferences, long-term goals... identity continuity") -
/// prepended as a stable prefix block to every Tier 1-4 prompt. This closes a
/// real gap: previously Self Memory only ever influenced *whether* the
/// Executive preferred an operator (via an activation boost keeping it
/// competitive for Working Memory broadcast) - it never actually reached the
/// Cognitive Core's prompt, so a Reflection was never grounded in who Omega
/// is, only in what just won broadcast.
///
/// Doubles as a KV-cache-reuse opportunity for backends with prefix caching
/// (vLLM, llama.cpp): Self Memory changes rarely by construction, so as long
/// as this string doesn't change tick to tick, it's a genuinely stable
/// prefix. Deterministic ordering (highest activation first, id as a
/// tiebreaker so ties don't depend on iteration order) matters for exactly
/// that reason: prefix-cache reuse depends on byte-for-byte stability, not
/// just "the same facts in some order."
///
/// Excludes anything still `PromotionStatus::Candidate` - a Self Memory
/// belief `form_memory`'s own classifier just produced, with nothing outside
/// that classifier having vouched for it yet, must not appear in the "Who
/// you are" block a moment later. See `PromotionState`'s own doc comment.
fn build_self_summary(graph: &Graph, max_items: usize) -> String {
    let mut items: Vec<&MentalObject> = graph
        .iter()
        .filter(|object| object.memory_roles.contains(&MemoryRole::SelfMemory) && object.promotion.status == PromotionStatus::Confirmed)
        .collect();
    items.sort_by(|a, b| b.activation.total.partial_cmp(&a.activation.total).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
    items.truncate(max_items);
    items.iter().map(|object| format!("- {}", object.text)).collect::<Vec<_>>().join("\n")
}

/// The shared cognitive substrate: the Mental Object graph, Working Memory
/// membership, the dirty-set/persistence half of the actor, and the
/// bookkeeping that's fundamentally about *what's in memory* (pending
/// admission, pattern-synthesis backlog, the Self Memory summary, per-channel
/// broadcast refractory state). Deliberately broader than the other
/// components - `graph` and `dirty_ids` in particular are touched by nearly
/// every step in `tick()`, so splitting them further than this would be an
/// artificial boundary rather than a real one.
pub(crate) struct MemoryCoordinator {
    pub(crate) graph: Graph,
    pub(crate) working_memory: HashSet<MentalObjectId>,
    pub(crate) store: Arc<dyn MemoryStore>,
    pub(crate) kl_store: Arc<dyn KnowledgeLibraryStore>,
    pub(crate) dirty_ids: HashSet<MentalObjectId>,
    /// Objects created mid-tick (a Cognitive Core Reflection, a
    /// missing-information subgoal) that missed this tick's Coalition
    /// because Act/Learn run after it - without this, anything created
    /// during Act was write-only: inserted into the graph, never
    /// reconsidered for broadcast, so it could never actually reach Speak.
    /// Drained (and each id's activation freshly computed) at the start of
    /// the *next* tick's Coalition step, giving each one exactly one honest
    /// shot at admission before falling back to ordinary long-term-memory
    /// recall.
    pub(crate) pending_admission: HashSet<MentalObjectId>,
    /// The `new_surprise` value carried forward for an id deferred into
    /// `pending_admission` by this tick's attentional-refractory check
    /// (rather than an ordinary Reflection/subgoal deferral, which never had
    /// a surprise term of its own to carry). Entries are removed the instant
    /// they're consulted, same one-shot discipline as `pending_admission`
    /// itself.
    pub(crate) pending_admission_surprise: HashMap<MentalObjectId, f32>,
    /// New Episodic memories formed since the last pattern-synthesis attempt
    /// (`steps::synthesize`), buffered here rather than as a Mental Object -
    /// no goal-stack/due-timing mechanism exists anywhere in this engine yet.
    pub(crate) episodic_since_last_synthesis: Vec<MentalObjectId>,
    /// When synthesis was last attempted, success or failure - updated
    /// unconditionally on every attempt (not just successful ones) so a
    /// persistent failure (e.g. Tier 3 down) can't retrigger an attempt on
    /// every subsequent tick forever.
    pub(crate) last_synthesis_at: Option<EpochMillis>,
    /// Cached `kl_store.count_documents()` result - refreshed only on the
    /// same cadence as the periodic write-behind flush, not re-queried on
    /// every snapshot publish.
    pub(crate) cached_kl_doc_count: u64,
    /// A deterministic, bullet-list rendering of every `MemoryRole::SelfMemory`
    /// object currently in the graph (highest activation first) - prepended
    /// as a stable prefix block to every Tier 1-4 prompt. Recomputed only
    /// when this tick's `dirty_ids` actually touched a Self Memory object,
    /// not on every tick.
    pub(crate) self_summary: String,
    /// The instant each `SourceChannel` most recently had a *fresh*
    /// candidate win Broadcast - GNW's ignition-then-refractory dynamics
    /// (Dehaene/Changeux's successor to the plainer GWT this architecture
    /// otherwise follows), the neural correlate of the psychological
    /// "attentional blink": a channel that just ignited briefly can't
    /// immediately ignite again.
    pub(crate) last_broadcast_by_channel: HashMap<SourceChannel, EpochMillis>,
    pub(crate) eligibility_traces: EligibilityTraceRegistry,
    pub(crate) preconscious_traces: HashMap<MentalObjectId, PreconsciousTrace>,
    pub(crate) compiled_procedures: HashMap<String, CompiledProcedure>,
    pub(crate) curated_answers: HashMap<String, CuratedAnswer>,
    pub(crate) spike_events: SpikeEventQueue,
}

impl MemoryCoordinator {
    pub(crate) fn new(graph: Graph, store: Arc<dyn MemoryStore>, kl_store: Arc<dyn KnowledgeLibraryStore>) -> Self {
        let self_summary = build_self_summary(&graph, SELF_SUMMARY_MAX_ITEMS);
        let compiled_procedures = compiled_index(&graph);
        let curated_answers = curated_answer_index(&graph);
        Self {
            graph,
            working_memory: HashSet::new(),
            store,
            kl_store,
            dirty_ids: HashSet::new(),
            pending_admission: HashSet::new(),
            pending_admission_surprise: HashMap::new(),
            episodic_since_last_synthesis: Vec::new(),
            last_synthesis_at: None,
            cached_kl_doc_count: 0,
            self_summary,
            last_broadcast_by_channel: HashMap::new(),
            eligibility_traces: EligibilityTraceRegistry::default(),
            preconscious_traces: HashMap::new(),
            compiled_procedures,
            curated_answers,
            spike_events: SpikeEventQueue::default(),
        }
    }

    pub(crate) fn compiled_procedure(&self, stimulus: &str) -> Option<&CompiledProcedure> {
        routine_key(stimulus).and_then(|key| self.compiled_procedures.get(&key))
    }

    pub(crate) fn curated_answer(&self, question: &str) -> Option<&CuratedAnswer> {
        static_question_key(question).and_then(|key| self.curated_answers.get(&key))
    }

    pub(crate) fn refresh_compiled_procedure(&mut self, stimulus: &str) {
        if let Some(key) = routine_key(stimulus) {
            if let Some(procedure) = compiled_procedure(&self.graph, stimulus) {
                self.compiled_procedures.insert(key, procedure);
            } else {
                self.compiled_procedures.remove(&key);
            }
        }
    }

    /// Rebuilds `self_summary` only when this tick's `dirty_ids` actually
    /// touched a Self Memory object (a belief revision, a new long-term
    /// goal) - checked, not assumed. O(dirty_ids.len()), not a graph scan;
    /// `build_self_summary` itself is the (rare, gated) scan.
    pub(crate) fn refresh_self_summary_if_dirty(&mut self) {
        if self.dirty_ids.iter().any(|id| self.graph.get(id).is_some_and(|object| object.memory_roles.contains(&MemoryRole::SelfMemory))) {
            self.self_summary = build_self_summary(&self.graph, SELF_SUMMARY_MAX_ITEMS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::{MentalObject, PromotionState};

    fn self_memory_belief(text: &str, promotion: PromotionState) -> MentalObject {
        let mut object = MentalObject::new_observation(text, EpochMillis(0), 0.5);
        object.memory_roles.push(MemoryRole::SelfMemory);
        object.promotion = promotion;
        object
    }

    #[test]
    fn build_self_summary_excludes_a_still_staged_candidate() {
        let mut graph = Graph::new();
        graph.insert(self_memory_belief("an unconfirmed belief", PromotionState::candidate(EpochMillis(0))));
        let summary = build_self_summary(&graph, SELF_SUMMARY_MAX_ITEMS);
        assert_eq!(summary, "", "a still-Candidate belief must not appear in the block prepended to every prompt");
    }

    #[test]
    fn build_self_summary_includes_a_confirmed_belief() {
        let mut graph = Graph::new();
        graph.insert(self_memory_belief("a confirmed belief", PromotionState::confirmed(EpochMillis(0))));
        let summary = build_self_summary(&graph, SELF_SUMMARY_MAX_ITEMS);
        assert_eq!(summary, "- a confirmed belief");
    }

    #[test]
    fn build_self_summary_shows_a_belief_once_it_confirms() {
        let mut graph = Graph::new();
        let mut belief = self_memory_belief("was staged, now confirmed", PromotionState::candidate(EpochMillis(0)));
        belief.promotion.confirm(EpochMillis(1_000));
        let id = belief.id;
        graph.insert(belief);
        assert_eq!(graph.get(&id).unwrap().promotion.status, aca_types::PromotionStatus::Confirmed);
        let summary = build_self_summary(&graph, SELF_SUMMARY_MAX_ITEMS);
        assert_eq!(summary, "- was staged, now confirmed");
    }
}
