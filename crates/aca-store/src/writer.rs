use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::Connection;

use crate::dao;
use crate::schema::run_migrations;
use crate::types::{CycleEvent, CycleEventPruneReport, CycleEventRetention, DirtyBatch, GraphSnapshotData, StoreError};

/// Durable read/flush surface over the graph. Implementations must never be
/// on the cognitive cycle's hot path (steps 1-6) — only the periodic
/// write-behind flush and startup load touch this.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn load_all(&self) -> Result<GraphSnapshotData, StoreError>;
    async fn flush(&self, batch: DirtyBatch) -> Result<(), StoreError>;
    async fn append_cycle_event(&self, event: CycleEvent) -> Result<(), StoreError>;
}

/// SQLite-backed `MemoryStore`. Holds the one connection behind a plain
/// `std::sync::Mutex` and only ever touches it inside
/// `tokio::task::spawn_blocking` — SQLite is inherently single-writer, so
/// this gives the same "one dedicated writer, no hot-path disk I/O"
/// guarantee the build plan calls for without a separate actor+channel
/// layer purely for the store itself (the `CognitiveLoopActor` upstream is
/// what decides *when* to call `flush`; this type only decides *how*).
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteStore {
    /// Opens (creating if absent) the database file at `path`, applies
    /// migrations, and enables WAL + NORMAL synchronous mode — WAL so a
    /// future read-only debug connection doesn't contend with the writer;
    /// NORMAL because durability window is "since the last periodic flush,"
    /// matching the write-behind design, not "every write."
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let mut conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        run_migrations(&mut conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// In-memory database, for tests: same code path, no file on disk.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let mut conn = Connection::open_in_memory()?;
        run_migrations(&mut conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Shares this store's one connection with another trait impl on the
    /// same struct (`KnowledgeLibraryStore`, in `knowledge_library.rs`) —
    /// same file/locking domain, not a second connection.
    pub(crate) fn conn_handle(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    pub async fn prune_cycle_events(&self, current_cycle_seq: u64, retention: CycleEventRetention) -> Result<CycleEventPruneReport, StoreError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite connection mutex poisoned");
            dao::prune_cycle_events(&conn, current_cycle_seq, retention)
        })
        .await?
    }

    /// Read-only diagnostic access to the events this store actually
    /// flushed, including independently supplied outcome labels.
    pub async fn recent_cycle_events(&self, limit: u32) -> Result<Vec<CycleEvent>, StoreError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite connection mutex poisoned");
            dao::recent_cycle_events(&conn, limit)
        })
        .await?
    }
}

#[async_trait]
impl MemoryStore for SqliteStore {
    async fn load_all(&self) -> Result<GraphSnapshotData, StoreError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite connection mutex poisoned");
            let objects = dao::load_all_objects(&conn)?;
            Ok(GraphSnapshotData { objects })
        })
        .await?
    }

    async fn flush(&self, batch: DirtyBatch) -> Result<(), StoreError> {
        if batch.is_empty() {
            return Ok(());
        }
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("sqlite connection mutex poisoned");
            // One commit for the whole batch, not one WAL-fsync per object -
            // a tick that reinforced several Working Memory objects (routine,
            // since activation/edges change on most active ticks) used to
            // pay N separate transaction commits here.
            let tx = conn.transaction()?;
            for object in &batch.objects {
                dao::upsert_object_in_tx(&tx, object)?;
            }
            for event in &batch.cycle_events {
                dao::append_cycle_event(&tx, event)?;
            }
            tx.commit()?;
            Ok(())
        })
        .await?
    }

    async fn append_cycle_event(&self, event: CycleEvent) -> Result<(), StoreError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite connection mutex poisoned");
            dao::append_cycle_event(&conn, &event)
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::MentalObject;
    use aca_util::EpochMillis;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn in_memory_store_round_trips_a_batch() {
        let store = SqliteStore::open_in_memory().unwrap();
        let object = MentalObject::new_observation("hello", EpochMillis(1_000), 0.5);
        let id = object.id;

        store
            .flush(DirtyBatch {
                objects: vec![object],
                cycle_events: vec![],
            })
            .await
            .unwrap();

        let snapshot = store.load_all().await.unwrap();
        assert_eq!(snapshot.objects.len(), 1);
        assert_eq!(snapshot.objects[0].id, id);
    }

    #[tokio::test]
    async fn empty_batch_flush_is_a_cheap_no_op() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.flush(DirtyBatch::default()).await.unwrap();
        let snapshot = store.load_all().await.unwrap();
        assert!(snapshot.objects.is_empty());
    }

    #[tokio::test]
    async fn file_backed_store_survives_a_reopen() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        // Drop the tempfile handle's own deletion-on-drop guard by keeping
        // `file` alive for the whole test - we just want a real path.
        let id;
        {
            let store = SqliteStore::open(&path).unwrap();
            let object = MentalObject::new_observation("persisted", EpochMillis(1_000), 0.5);
            id = object.id;
            store
                .flush(DirtyBatch {
                    objects: vec![object],
                    cycle_events: vec![],
                })
                .await
                .unwrap();
        }
        {
            let store = SqliteStore::open(&path).unwrap();
            let snapshot = store.load_all().await.unwrap();
            assert_eq!(snapshot.objects.len(), 1);
            assert_eq!(snapshot.objects[0].id, id);
        }
    }
}
