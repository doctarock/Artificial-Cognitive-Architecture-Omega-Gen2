use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use aca_types::MentalObjectId;
use aca_util::RingBuffer;
use tokio::sync::mpsc;

use crate::embedding_worker::{EmbeddingReentry, PendingObservation};
use crate::snapshot::PredictionErrorSummary;
use crate::steps::affect::AffectTracker;
use crate::steps::compare::{PrecisionTracker, SourceChannel};
use crate::steps::predict::LocalEventPredictor;

/// `CognitiveLoopActor::prediction`'s embedding cache FIFO capacity - a
/// handful of recently-seen exact texts, enough to catch a real recurring
/// phrase (a boredom self-status line, an echoed turn) without holding an
/// unbounded or even large amount of embedding-vector memory.
pub(super) const EMBEDDING_CACHE_CAPACITY: usize = 64;

/// Everything Predict/Observe/Compare need: per-channel and per-interlocutor
/// expected-next-embedding state, the exact-text embedding cache, the
/// cross-tick async embedding-resolution plumbing, and the tracking of how
/// reliable each generative stream and the model as a whole has been lately.
pub(crate) struct PredictionSystem {
    pub(crate) channel_embeddings: HashMap<SourceChannel, Vec<f32>>,
    /// `channel_embeddings`, refined per specific recognized interlocutor -
    /// explicit theory of mind, made literal with existing plumbing rather
    /// than new mechanism: `steps::interlocutor` already gives each
    /// named/enrolled speaker their own persistent node, and
    /// `PrecisionTracker` already layers a per-interlocutor reliability
    /// estimate over the generic per-channel one (`precision_for_interlocutor`'s
    /// "specific overrides generic once earned" hierarchy). Falls back to
    /// `channel_embeddings` for anyone not yet recognized, exactly as
    /// `precision_for_interlocutor` falls back to the channel-level
    /// estimate.
    pub(crate) interlocutor_embeddings: HashMap<MentalObjectId, Vec<f32>>,
    /// Exact-text embedding cache, checked before every `resolve_embedding`
    /// call - the embedding round trip is the one genuinely I/O-bound step
    /// in an otherwise-instant tick, so a cache hit skips a real
    /// network/model call entirely rather than merely hiding its latency.
    /// Bounded by `embedding_cache_order` (a plain FIFO eviction, not LRU -
    /// simplicity over hit-rate optimality) so unbounded distinct text can
    /// never grow this without limit.
    pub(crate) embedding_cache: HashMap<String, Vec<f32>>,
    /// Insertion order for `embedding_cache`'s FIFO eviction - a `HashMap`
    /// alone has no ordering to evict by.
    pub(crate) embedding_cache_order: RingBuffer<String>,
    pub(crate) precision_tracker: PrecisionTracker,
    /// A slow-moving read of "how has Omega's own predictive model been
    /// doing lately," updated every tick real input resolves - modulates
    /// precision gain on fresh prediction error at this tick's
    /// `new_surprise` computation.
    pub(crate) affect_tracker: AffectTracker,
    pub(crate) local_event_predictor: LocalEventPredictor,
    /// This tick's (or the most recently resolved tick's) raw
    /// prediction-error reading - `affect_tracker`'s own smoothed `valence`
    /// is built from a whole series of these, but a self-assessment tool
    /// needs the actual instantaneous number too, not only the trend.
    /// Republished verbatim into `EngineSnapshot::last_prediction_error` by
    /// `publish_snapshot`.
    pub(crate) last_prediction_error: Option<PredictionErrorSummary>,
    /// `Arc<AtomicBool>` (not a plain `bool`) - `live_models` holds its own
    /// clone, read directly by the API layer without going through a
    /// snapshot publish at all.
    pub(crate) embedding_in_flight: Arc<AtomicBool>,
    /// Steps 2-3's asynchronous half - see `embedding_worker`'s own doc
    /// comment. Feeds a cache-miss observation to the worker task spawned in
    /// `new()`.
    pub(crate) embedding_request_tx: mpsc::Sender<PendingObservation>,
    /// Drained (`try_recv`, once per tick, same non-blocking discipline as
    /// every other `Incoming` source) for whichever result comes back,
    /// however many ticks later that is.
    pub(crate) embedding_reentry_rx: mpsc::Receiver<EmbeddingReentry>,
    /// Filled by `CognitiveScheduler`'s `select!` when `embedding_reentry_rx`
    /// is the channel that wakes a sleeping `run()` - see
    /// `PerceptionState::input_primed`'s doc comment for why this exists at
    /// all. `take_embedding_reentry` checks this before the channel itself.
    pub(crate) embedding_reentry_primed: Option<EmbeddingReentry>,
}

impl PredictionSystem {
    /// Inserts a freshly-resolved embedding into `embedding_cache`, evicting
    /// the oldest entry first if already at `EMBEDDING_CACHE_CAPACITY` -
    /// plain FIFO, not LRU (see that field's own doc comment). A no-op
    /// insert (the exact text is already cached, e.g. two different
    /// channels happening to produce identical text) still moves nothing in
    /// `embedding_cache_order` - fine, since a duplicate key simply
    /// overwrites in the map and the stale order entry evicts a key that's
    /// already gone, an inert no-op rather than a bug.
    pub(crate) fn cache_embedding(&mut self, text: String, embedding: Vec<f32>) {
        if let Some(evicted) = self.embedding_cache_order.push_evicting(text.clone()) {
            self.embedding_cache.remove(&evicted);
        }
        self.embedding_cache.insert(text, embedding);
    }

    /// Checks the primed slot (see `embedding_reentry_primed`'s doc comment)
    /// before the channel itself.
    pub(crate) fn take_embedding_reentry(&mut self) -> Option<EmbeddingReentry> {
        self.embedding_reentry_primed.take().or_else(|| self.embedding_reentry_rx.try_recv().ok())
    }
}
