use rusqlite::Connection;
use rusqlite_migration::{Migrations, M};

/// The durable schema. `mental_object_references` and `associative_edges`
/// are separate child tables (not JSON blobs on the parent row) so ACT-R's
/// base-level sum and spreading-activation traversal have real indexed
/// adjacency lookups rather than requiring a whole-row deserialize per read
/// — the exact bottleneck flat-JSON-file storage would hit first under a
/// continuously-ticking loop.
///
/// `knowledge_library_docs` is deliberately never joined to `mental_objects`
/// — the Knowledge Library is external knowledge to consult, not memory.
fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(
            r#"
        CREATE TABLE mental_objects (
            id                  TEXT PRIMARY KEY,
            kind                TEXT NOT NULL,
            created_at          INTEGER NOT NULL,
            text                TEXT NOT NULL,
            embedding           BLOB,
            data_json           TEXT NOT NULL,
            confidence          REAL NOT NULL,
            tier_used           TEXT,
            status              TEXT NOT NULL,
            memory_roles_json   TEXT NOT NULL,
            goal_stack_id       TEXT,
            goal_parent_id      TEXT,
            goal_status         TEXT,
            goal_priority       REAL,
            base_level_cached   REAL NOT NULL,
            decay_d             REAL NOT NULL,
            last_computed_at    INTEGER NOT NULL,
            discarded_at        INTEGER
        );

        CREATE TABLE mental_object_references (
            object_id TEXT NOT NULL,
            ts        INTEGER NOT NULL
        );
        CREATE INDEX idx_mental_object_references_object_id
            ON mental_object_references(object_id);

        CREATE TABLE associative_edges (
            id                  TEXT PRIMARY KEY,
            source_id           TEXT NOT NULL,
            target_id           TEXT NOT NULL,
            kind                TEXT NOT NULL,
            strength            REAL NOT NULL,
            last_coactivated_at INTEGER NOT NULL,
            created_at          INTEGER NOT NULL,
            UNIQUE(source_id, target_id, kind)
        );
        CREATE INDEX idx_associative_edges_source_id ON associative_edges(source_id);
        CREATE INDEX idx_associative_edges_target_id ON associative_edges(target_id);

        CREATE TABLE cycle_events (
            id          TEXT PRIMARY KEY,
            cycle_seq   INTEGER NOT NULL,
            ts          INTEGER NOT NULL,
            phase       TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            tier_used   TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX idx_cycle_events_cycle_seq ON cycle_events(cycle_seq);

        CREATE TABLE knowledge_library_docs (
            id          TEXT PRIMARY KEY,
            source_uri  TEXT NOT NULL,
            text        TEXT NOT NULL,
            embedding   BLOB,
            ingested_at INTEGER NOT NULL
        );
        "#,
        ),
        // Latency reporting (`LatencyReport::from_connection`) filters
        // `cycle_events` by `phase = 'telemetry'` within a recent
        // `cycle_seq` window; the original `idx_cycle_events_cycle_seq`
        // alone can't serve that predicate without a full scan.
        M::up("CREATE INDEX idx_cycle_events_phase_cycle_seq ON cycle_events(phase, cycle_seq);"),
        // `MentalObject.produced_by_operator`/`source_object_ids` had no
        // round trip at all before this: every object's provenance (which
        // operator wrote it, which sources a synthesized memory was derived
        // from) silently reverted to empty on every restart.
        M::up(
            r#"
        ALTER TABLE mental_objects ADD COLUMN produced_by_operator TEXT;
        ALTER TABLE mental_objects ADD COLUMN source_object_ids_json TEXT NOT NULL DEFAULT '[]';
        "#,
        ),
        // `MentalObject.promotion` (`PromotionState`): whether an internally
        // classified Semantic/SelfBelief memory has been independently
        // confirmed yet, or is still only vouched for by the classifier that
        // produced it. Every pre-existing row predates this distinction and
        // defaults to already-confirmed (`'confirmed'`), matching
        // `PromotionState::default()`'s "confirmed, not demoted" contract -
        // see that type's own doc comment.
        M::up(
            r#"
        ALTER TABLE mental_objects ADD COLUMN promotion_status TEXT NOT NULL DEFAULT 'confirmed';
        ALTER TABLE mental_objects ADD COLUMN staged_at INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE mental_objects ADD COLUMN confirmed_at INTEGER;
        ALTER TABLE mental_objects ADD COLUMN confirming_references INTEGER NOT NULL DEFAULT 0;
        "#,
        ),
    ])
}

/// Runs all pending migrations against `conn`, bringing a fresh or existing
/// database file up to the latest schema.
pub fn run_migrations(conn: &mut Connection) -> Result<(), rusqlite_migration::Error> {
    migrations().to_latest(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_apply_cleanly_to_a_fresh_connection() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).expect("migrations should apply cleanly");

        let table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_rusqlite%'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 5, "expected exactly the 5 domain tables");
    }

    #[test]
    fn migrations_are_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        run_migrations(&mut conn).expect("re-running migrations must be a no-op, not an error");
    }
}
