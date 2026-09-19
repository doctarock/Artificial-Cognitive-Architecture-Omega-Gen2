use aca_types::{MentalObject, Tier};
use aca_util::EpochMillis;
use serde::{Deserialize, Serialize};

/// Retention policy for `cycle_events`. Routine `Normal` events are the
/// high-volume trace stream; abnormal events are sparse and diagnostically
/// valuable, so they get their own longer window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycleEventRetention {
    pub keep_recent_cycles: u64,
    pub keep_abnormal_cycles: u64,
}

impl Default for CycleEventRetention {
    fn default() -> Self {
        Self {
            keep_recent_cycles: 25_000,
            keep_abnormal_cycles: 250_000,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CycleEventPruneReport {
    pub deleted_normal_events: u64,
    pub deleted_abnormal_events: u64,
}

impl CycleEventPruneReport {
    pub fn deleted_total(&self) -> u64 {
        self.deleted_normal_events + self.deleted_abnormal_events
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("migration error: {0}")]
    Migration(#[from] rusqlite_migration::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("background task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("uuid parse error: {0}")]
    Uuid(#[from] uuid::Error),
    #[error("unrecognized stored value for {field}: {value:?}")]
    UnrecognizedValue { field: &'static str, value: String },
}

/// Which cycle step produced an event — mirrors the ten-step cognitive
/// cycle from specs.md exactly, so `cycle_events` reads as a direct trace of
/// the cycle rather than a separate ad hoc logging taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CyclePhase {
    Predict,
    Observe,
    Compare,
    Coalition,
    Broadcast,
    Executive,
    Act,
    Learn,
    /// Not one of specs.md's original ten steps - added for pattern
    /// synthesis (`steps::synthesize`), a self-triggered abstraction of a
    /// new Semantic memory out of several Episodic ones. Kept distinct from
    /// `Learn`'s ordinary co-activation-reinforcement pulse so it's visibly
    /// its own kind of thought in the live cognition feed, not folded into
    /// routine edge reinforcement.
    Synthesize,
    /// Not one of specs.md's original ten steps either, same category as
    /// `Synthesize` - added for `steps::agenda`, the persistent-intention
    /// maintenance step (surfacing existing intentions for attention,
    /// spawning/decaying/resolving them). Kept distinct from `Executive`
    /// (which only ever selects among this tick's already-proposed
    /// operators) so intention lifecycle events are visibly their own kind
    /// of thought in the live cognition feed.
    Agenda,
    /// Operational telemetry for optimization work: cycle latency, rough
    /// token pressure, event/write pressure, and tier utilization. Kept as
    /// a first-class phase so performance evidence can be queried without
    /// scraping tracing logs or mixing it into a cognitive step.
    Telemetry,
}

/// The live feed's event classification — this is what backs the MVP
/// acceptance bar's "visible impasses/escalations in the log."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CycleEventKind {
    Normal,
    Impasse,
    Escalation,
    Error,
}

/// One row of the live cognition trace: what happened, in which phase, on
/// which tick, optionally using which model tier. `payload` carries
/// phase-specific structured detail (kept as `serde_json::Value` for the
/// same reason `MentalObject::data` is — real usage should settle the
/// per-phase shapes before they're locked into a typed enum).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleEvent {
    pub id: uuid::Uuid,
    pub cycle_seq: u64,
    pub ts: EpochMillis,
    pub phase: CyclePhase,
    pub event_type: CycleEventKind,
    pub tier_used: Option<Tier>,
    pub payload: serde_json::Value,
}

impl CycleEvent {
    pub fn new(
        cycle_seq: u64,
        ts: EpochMillis,
        phase: CyclePhase,
        event_type: CycleEventKind,
        tier_used: Option<Tier>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id: uuid::Uuid::now_v7(),
            cycle_seq,
            ts,
            phase,
            event_type,
            tier_used,
            payload,
        }
    }
}

/// A batch of graph mutations pending durable write-behind flush: full
/// upserts for changed Mental Objects (edges included) plus any
/// `cycle_events` accumulated since the last flush. Never touches disk
/// itself — that's `MemoryStore::flush`'s job.
#[derive(Debug, Clone, Default)]
pub struct DirtyBatch {
    pub objects: Vec<MentalObject>,
    pub cycle_events: Vec<CycleEvent>,
}

impl DirtyBatch {
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.cycle_events.is_empty()
    }
}

/// Everything loaded from the durable store at boot, ready to repopulate the
/// in-memory `aca_graph::Graph`.
#[derive(Debug, Clone, Default)]
pub struct GraphSnapshotData {
    pub objects: Vec<MentalObject>,
}
