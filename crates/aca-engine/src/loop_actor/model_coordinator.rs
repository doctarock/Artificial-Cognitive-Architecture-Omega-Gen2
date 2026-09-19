use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

use aca_tiers::{AttentionClient, ChatClient, DivergentPool, EmbeddingClient, TierPool};

use crate::snapshot::LiveModelHandles;

/// Every substrate model client/pool `CognitiveLoopActor` coordinates, plus
/// the small amount of bookkeeping (`ticks_since_attention_reconciliation`)
/// that governs how the optional Omega Attention model is trusted versus
/// reconciled against the deterministic Broadcast algorithm. Extracted from
/// `CognitiveLoopActor` as the lowest-coupling cluster of its fields - every
/// other step consumes these read-mostly, none of the file's coupling-hotspot
/// methods (`apply_embedding_result`, `apply_operator`, etc.) write to more
/// than one or two of them at a time.
pub(crate) struct ModelCoordinator {
    pub(crate) embedding_client: Arc<dyn EmbeddingClient>,
    /// The embedding model's real identity (e.g. "mxbai-embed-large"), for
    /// `active_models` - the same substrate client backs Observe's per-tick
    /// embedding, Executive's escalation-answer embedding, and
    /// `steps::synthesize`'s new-memory embedding. `Arc<RwLock<_>>` (not a
    /// plain `String`) - see `LiveModelHandles::embedding_label`'s doc
    /// comment for why.
    pub(crate) embedding_label: Arc<RwLock<String>>,
    pub(crate) chat_client: Arc<dyn ChatClient>,
    pub(crate) tier1_pool: Arc<DivergentPool>,
    pub(crate) tier2_pool: Arc<DivergentPool>,
    pub(crate) tier3_pool: Arc<TierPool>,
    /// Escalation-only, per specs.md's Model Tiering section - reached only
    /// when a Tier 3 confidence-impasse resolution is itself still not
    /// confident enough. Single-flight, same as Tier 3.
    pub(crate) tier4_pool: Arc<TierPool>,
    pub(crate) tier4_client: Arc<dyn ChatClient>,
    /// The Omega Attention model's endpoint - genuinely optional, unlike
    /// every tier client above. `None` means "run Step 6 exactly as the
    /// deterministic algorithm always has."
    pub(crate) attention_client: Option<Arc<dyn AttentionClient>>,
    /// How many consecutive ticks the attention model's judgment has been
    /// trusted without a deterministic reconciliation check - see
    /// `LoopConfig::attention_reconciliation_interval`'s doc comment. Reset
    /// to `0` any tick the deterministic algorithm actually runs; incremented
    /// on every tick the model's decision was trusted instead.
    pub(crate) ticks_since_attention_reconciliation: u64,
    /// `Arc<AtomicBool>` (not a plain `bool`) - `live_models` holds its own
    /// clone, read directly by the API layer without going through a
    /// snapshot publish at all.
    pub(crate) attention_in_flight: Arc<AtomicBool>,
    /// This actor's own clone of the bundle handed out via `LoopHandles` -
    /// kept here too so `publish_snapshot` has one non-duplicated place to
    /// build `active_models`/`tier_status`/`embedding_in_flight` from.
    pub(crate) live_models: LiveModelHandles,
}
