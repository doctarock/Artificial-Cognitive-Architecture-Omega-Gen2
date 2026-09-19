//! The durable write-behind layer under the in-memory activation graph.
//! Nothing here runs on the cognitive cycle's hot path — `SqliteStore` is
//! loaded once at boot and flushed periodically from a dirty-id batch the
//! `CognitiveLoopActor` accumulates, never on every tick.

mod dao;
mod knowledge_library;
mod latency;
mod schema;
mod types;
mod triage;
mod writer;

pub use knowledge_library::{KnowledgeDocId, KnowledgeLibraryStore, ScoredDocument};
pub use latency::{LatencyReport, LatencySlice, SlowCycle};
pub use triage::StoreTriageReport;
pub use types::{CycleEvent, CycleEventKind, CycleEventPruneReport, CycleEventRetention, CyclePhase, DirtyBatch, GraphSnapshotData, StoreError};
pub use writer::{MemoryStore, SqliteStore};

// Exposed for tests/tools that want to inspect the raw event log without
// going through the full `MemoryStore` trait (e.g. asserting exact recent
// events after a scripted scenario).
pub use dao::recent_cycle_events;
