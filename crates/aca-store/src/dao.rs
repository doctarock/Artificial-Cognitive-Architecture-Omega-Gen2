use std::collections::HashMap;
use std::str::FromStr;

use aca_types::{
    ActivationState, AssociativeEdge, EdgeKind, GoalStackId, GoalStackMembership, GoalStatus,
    MemoryRole, MentalObject, MentalObjectDynamics, MentalObjectId, MentalObjectKind, ObjectStatus,
    PredictionState, PromotionState, PromotionStatus, Tier, WorkspaceState,
};
use aca_util::{EpochMillis, RingBuffer};
use rusqlite::{params, Connection};
use serde::{de::DeserializeOwned, Serialize};

use crate::types::{CycleEvent, CycleEventKind, CycleEventPruneReport, CycleEventRetention, CyclePhase, StoreError};

const REFERENCE_LOG_LOAD_CAPACITY: usize = aca_types::DEFAULT_REFERENCE_LOG_CAPACITY;

fn enum_to_text<T: Serialize>(value: &T) -> Result<String, StoreError> {
    match serde_json::to_value(value)? {
        serde_json::Value::String(s) => Ok(s),
        other => Err(StoreError::UnrecognizedValue {
            field: "enum",
            value: other.to_string(),
        }),
    }
}

fn text_to_enum<T: DeserializeOwned>(field: &'static str, text: &str) -> Result<T, StoreError> {
    serde_json::from_value(serde_json::Value::String(text.to_string())).map_err(|_| {
        StoreError::UnrecognizedValue {
            field,
            value: text.to_string(),
        }
    })
}

fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is exactly 4 bytes")))
        .collect()
}

/// Upserts one Mental Object in full: the parent row, its reference log
/// (replace-all), and its outgoing associative edges (replace-all). Runs
/// against an already-open transaction rather than opening its own - lets a
/// caller (`SqliteStore::flush`) batch several objects (and cycle events)
/// into a single commit instead of one WAL commit per object. A crash
/// mid-write still can't leave one object's row and its edges/references
/// inconsistent with each other, since the caller's own transaction covers
/// all of it.
pub fn upsert_object_in_tx(tx: &rusqlite::Transaction, object: &MentalObject) -> Result<(), StoreError> {
    let id_text = object.id.to_string();
    let embedding_blob = object.embedding.as_deref().map(embedding_to_blob);
    let data_json = object.data.to_string();
    let memory_roles_json = serde_json::to_string(&object.memory_roles)?;
    let tier_used_text = object.tier_used.map(|t| enum_to_text(&t)).transpose()?;

    let source_object_ids_json = serde_json::to_string(&object.source_object_ids)?;
    let promotion_status_text = enum_to_text(&object.promotion.status)?;

    let (goal_stack_id, goal_parent_id, goal_status, goal_priority) = match &object.goal {
        Some(goal) => (
            Some(goal.stack_id.to_string()),
            goal.parent_goal_id.map(|id| id.to_string()),
            Some(enum_to_text(&goal.status)?),
            Some(goal.priority as f64),
        ),
        None => (None, None, None, None),
    };

    tx.execute(
        r#"
        INSERT INTO mental_objects (
            id, kind, created_at, text, embedding, data_json, confidence, tier_used, status,
            memory_roles_json, goal_stack_id, goal_parent_id, goal_status, goal_priority,
            base_level_cached, decay_d, last_computed_at, discarded_at,
            produced_by_operator, source_object_ids_json,
            promotion_status, staged_at, confirmed_at, confirming_references
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24)
        ON CONFLICT(id) DO UPDATE SET
            kind = excluded.kind, created_at = excluded.created_at, text = excluded.text,
            embedding = excluded.embedding, data_json = excluded.data_json,
            confidence = excluded.confidence, tier_used = excluded.tier_used,
            status = excluded.status, memory_roles_json = excluded.memory_roles_json,
            goal_stack_id = excluded.goal_stack_id, goal_parent_id = excluded.goal_parent_id,
            goal_status = excluded.goal_status, goal_priority = excluded.goal_priority,
            base_level_cached = excluded.base_level_cached, decay_d = excluded.decay_d,
            last_computed_at = excluded.last_computed_at, discarded_at = excluded.discarded_at,
            produced_by_operator = excluded.produced_by_operator,
            source_object_ids_json = excluded.source_object_ids_json,
            promotion_status = excluded.promotion_status, staged_at = excluded.staged_at,
            confirmed_at = excluded.confirmed_at, confirming_references = excluded.confirming_references
        "#,
        params![
            id_text,
            enum_to_text(&object.kind)?,
            object.created_at.as_millis(),
            object.text,
            embedding_blob,
            data_json,
            object.confidence as f64,
            tier_used_text,
            enum_to_text(&object.status)?,
            memory_roles_json,
            goal_stack_id,
            goal_parent_id,
            goal_status,
            goal_priority,
            object.activation.base_level as f64,
            object.activation.decay_d as f64,
            object.activation.last_computed_at.as_millis(),
            object.discarded_at.map(|t| t.as_millis()),
            object.produced_by_operator,
            source_object_ids_json,
            promotion_status_text,
            object.promotion.staged_at.as_millis(),
            object.promotion.confirmed_at.map(|t| t.as_millis()),
            object.promotion.confirming_references,
        ],
    )?;

    tx.execute(
        "DELETE FROM mental_object_references WHERE object_id = ?1",
        params![id_text],
    )?;
    {
        let mut insert_ref = tx.prepare(
            "INSERT INTO mental_object_references (object_id, ts) VALUES (?1, ?2)",
        )?;
        for ts in object.activation.reference_log.iter() {
            insert_ref.execute(params![id_text, ts.as_millis()])?;
        }
    }

    tx.execute(
        "DELETE FROM associative_edges WHERE source_id = ?1",
        params![id_text],
    )?;
    {
        let mut insert_edge = tx.prepare(
            r#"INSERT INTO associative_edges
                (id, source_id, target_id, kind, strength, last_coactivated_at, created_at)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#,
        )?;
        for edge in &object.edges {
            insert_edge.execute(params![
                uuid::Uuid::now_v7().to_string(),
                id_text,
                edge.target_id.to_string(),
                enum_to_text(&edge.kind)?,
                edge.strength as f64,
                edge.last_coactivated_at.as_millis(),
                object.activation.last_computed_at.as_millis(),
            ])?;
        }
    }

    Ok(())
}

/// Loads every Mental Object, with its reference log and outgoing edges
/// reconstructed, in three bulk queries (not N+1 per object).
pub fn load_all_objects(conn: &Connection) -> Result<Vec<MentalObject>, StoreError> {
    let mut references_by_object: HashMap<MentalObjectId, Vec<EpochMillis>> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT object_id, ts FROM mental_object_references ORDER BY object_id, ts ASC",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let object_id_text: String = row.get(0)?;
            let ts: i64 = row.get(1)?;
            let object_id = MentalObjectId::from_str(&object_id_text)?;
            references_by_object
                .entry(object_id)
                .or_default()
                .push(EpochMillis(ts));
        }
    }

    let mut edges_by_source: HashMap<MentalObjectId, Vec<AssociativeEdge>> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT source_id, target_id, kind, strength, last_coactivated_at FROM associative_edges",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let source_id_text: String = row.get(0)?;
            let target_id_text: String = row.get(1)?;
            let kind_text: String = row.get(2)?;
            let strength: f64 = row.get(3)?;
            let last_coactivated_at: i64 = row.get(4)?;

            let source_id = MentalObjectId::from_str(&source_id_text)?;
            let edge = AssociativeEdge {
                target_id: MentalObjectId::from_str(&target_id_text)?,
                kind: text_to_enum::<EdgeKind>("associative_edges.kind", &kind_text)?,
                strength: strength as f32,
                last_coactivated_at: EpochMillis(last_coactivated_at),
            };
            edges_by_source.entry(source_id).or_default().push(edge);
        }
    }

    let mut stmt = conn.prepare(
        r#"SELECT id, kind, created_at, text, embedding, data_json, confidence, tier_used, status,
                  memory_roles_json, goal_stack_id, goal_parent_id, goal_status, goal_priority,
                  base_level_cached, decay_d, last_computed_at, discarded_at,
                  produced_by_operator, source_object_ids_json,
                  promotion_status, staged_at, confirmed_at, confirming_references
           FROM mental_objects"#,
    )?;
    let mut rows = stmt.query([])?;

    let mut objects = Vec::new();
    while let Some(row) = rows.next()? {
        let id_text: String = row.get(0)?;
        let id = MentalObjectId::from_str(&id_text)?;
        let kind: String = row.get(1)?;
        let created_at: i64 = row.get(2)?;
        let text: String = row.get(3)?;
        let embedding: Option<Vec<u8>> = row.get(4)?;
        let data_json: String = row.get(5)?;
        let confidence: f64 = row.get(6)?;
        let tier_used: Option<String> = row.get(7)?;
        let status: String = row.get(8)?;
        let memory_roles_json: String = row.get(9)?;
        let goal_stack_id: Option<String> = row.get(10)?;
        let goal_parent_id: Option<String> = row.get(11)?;
        let goal_status: Option<String> = row.get(12)?;
        let goal_priority: Option<f64> = row.get(13)?;
        let base_level_cached: f64 = row.get(14)?;
        let decay_d: f64 = row.get(15)?;
        let last_computed_at: i64 = row.get(16)?;
        let discarded_at: Option<i64> = row.get(17)?;
        let produced_by_operator: Option<String> = row.get(18)?;
        let source_object_ids_json: String = row.get(19)?;
        let promotion_status: String = row.get(20)?;
        let staged_at: i64 = row.get(21)?;
        let confirmed_at: Option<i64> = row.get(22)?;
        let confirming_references: u32 = row.get(23)?;

        let mut reference_log = RingBuffer::new(REFERENCE_LOG_LOAD_CAPACITY);
        for ts in references_by_object.remove(&id).unwrap_or_default() {
            reference_log.push(ts);
        }

        let goal = match (goal_stack_id, goal_status, goal_priority) {
            (Some(stack_id), Some(status_text), Some(priority)) => Some(GoalStackMembership {
                stack_id: GoalStackId::from_str(&stack_id)?,
                parent_goal_id: goal_parent_id
                    .map(|s| MentalObjectId::from_str(&s))
                    .transpose()?,
                status: text_to_enum::<GoalStatus>("mental_objects.goal_status", &status_text)?,
                priority: priority as f32,
            }),
            _ => None,
        };

        objects.push(MentalObject {
            id,
            kind: text_to_enum::<MentalObjectKind>("mental_objects.kind", &kind)?,
            created_at: EpochMillis(created_at),
            text,
            embedding: embedding.as_deref().map(blob_to_embedding),
            data: serde_json::from_str(&data_json)?,
            activation: ActivationState {
                base_level: base_level_cached as f32,
                reference_log,
                decay_d: decay_d as f32,
                spreading: 0.0,
                noise: 0.0,
                total: base_level_cached as f32,
                last_computed_at: EpochMillis(last_computed_at),
            },
            edges: edges_by_source.remove(&id).unwrap_or_default(),
            prediction: PredictionState::default(),
            dynamics: MentalObjectDynamics::new_at(EpochMillis(last_computed_at)),
            goal,
            workspace: WorkspaceState::default(),
            memory_roles: serde_json::from_str::<Vec<MemoryRole>>(&memory_roles_json)?,
            confidence: confidence as f32,
            tier_used: tier_used
                .map(|t| text_to_enum::<Tier>("mental_objects.tier_used", &t))
                .transpose()?,
            produced_by_operator,
            source_object_ids: serde_json::from_str(&source_object_ids_json)?,
            promotion: PromotionState {
                status: text_to_enum::<PromotionStatus>("mental_objects.promotion_status", &promotion_status)?,
                staged_at: EpochMillis(staged_at),
                confirmed_at: confirmed_at.map(EpochMillis),
                confirming_references,
            },
            status: text_to_enum::<ObjectStatus>("mental_objects.status", &status)?,
            discarded_at: discarded_at.map(EpochMillis),
        });
    }

    Ok(objects)
}

pub fn append_cycle_event(conn: &Connection, event: &CycleEvent) -> Result<(), StoreError> {
    conn.execute(
        r#"INSERT INTO cycle_events (id, cycle_seq, ts, phase, event_type, tier_used, payload_json)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#,
        params![
            event.id.to_string(),
            event.cycle_seq,
            event.ts.as_millis(),
            enum_to_text(&event.phase)?,
            enum_to_text(&event.event_type)?,
            event.tier_used.map(|t| enum_to_text(&t)).transpose()?,
            event.payload.to_string(),
        ],
    )?;
    Ok(())
}

/// Used by the API/CLI to read back recent trace events, and by tests to
/// verify a specific event landed. Not on the engine's hot path.
pub fn recent_cycle_events(conn: &Connection, limit: u32) -> Result<Vec<CycleEvent>, StoreError> {
    let mut stmt = conn.prepare(
        r#"SELECT id, cycle_seq, ts, phase, event_type, tier_used, payload_json
           FROM cycle_events ORDER BY cycle_seq DESC, ts DESC LIMIT ?1"#,
    )?;
    let mut rows = stmt.query(params![limit])?;
    let mut events = Vec::new();
    while let Some(row) = rows.next()? {
        let id_text: String = row.get(0)?;
        let cycle_seq: i64 = row.get(1)?;
        let ts: i64 = row.get(2)?;
        let phase: String = row.get(3)?;
        let event_type: String = row.get(4)?;
        let tier_used: Option<String> = row.get(5)?;
        let payload_json: String = row.get(6)?;

        events.push(CycleEvent {
            id: uuid::Uuid::from_str(&id_text)?,
            cycle_seq: cycle_seq as u64,
            ts: EpochMillis(ts),
            phase: text_to_enum::<CyclePhase>("cycle_events.phase", &phase)?,
            event_type: text_to_enum::<CycleEventKind>("cycle_events.event_type", &event_type)?,
            tier_used: tier_used
                .map(|t| text_to_enum::<Tier>("cycle_events.tier_used", &t))
                .transpose()?,
            payload: serde_json::from_str(&payload_json)?,
        });
    }
    Ok(events)
}

pub fn prune_cycle_events(conn: &Connection, current_cycle_seq: u64, retention: CycleEventRetention) -> Result<CycleEventPruneReport, StoreError> {
    let normal_cutoff = current_cycle_seq.saturating_sub(retention.keep_recent_cycles) as i64;
    let abnormal_cutoff = current_cycle_seq.saturating_sub(retention.keep_abnormal_cycles) as i64;
    let normal_kind = enum_to_text(&CycleEventKind::Normal)?;

    let deleted_normal_events = conn.execute(
        "DELETE FROM cycle_events WHERE cycle_seq < ?1 AND event_type = ?2",
        params![normal_cutoff, normal_kind],
    )? as u64;
    let deleted_abnormal_events = conn.execute(
        "DELETE FROM cycle_events WHERE cycle_seq < ?1 AND event_type <> ?2",
        params![abnormal_cutoff, normal_kind],
    )? as u64;

    Ok(CycleEventPruneReport {
        deleted_normal_events,
        deleted_abnormal_events,
    })
}

/// One row of `knowledge_library_docs`, as stored — `embedding` is `None`
/// only if a row was ever inserted without one (not possible through
/// `insert_knowledge_doc` today, but the column itself is nullable in the
/// schema, so this mirrors `mental_objects.embedding`'s own optionality
/// rather than assuming every row is search-eligible).
pub struct KnowledgeDocRow {
    pub id: uuid::Uuid,
    pub source_uri: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub ingested_at: EpochMillis,
}

pub fn insert_knowledge_doc(
    conn: &Connection,
    id: uuid::Uuid,
    source_uri: &str,
    text: &str,
    embedding: &[f32],
    ingested_at: EpochMillis,
) -> Result<(), StoreError> {
    conn.execute(
        r#"INSERT INTO knowledge_library_docs (id, source_uri, text, embedding, ingested_at)
           VALUES (?1, ?2, ?3, ?4, ?5)"#,
        params![
            id.to_string(),
            source_uri,
            text,
            embedding_to_blob(embedding),
            ingested_at.as_millis(),
        ],
    )?;
    Ok(())
}

/// A cheap `COUNT(*)` for the Knowledge Library's corpus size — used only to
/// report growth to a viewer (`EngineSnapshot::kl_doc_count`); actual
/// consult/search still goes through `load_all_knowledge_docs` below.
pub fn count_knowledge_docs(conn: &Connection) -> Result<u64, StoreError> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM knowledge_library_docs", [], |row| row.get(0))?;
    Ok(count as u64)
}

/// Loads every Knowledge Library document. No paging/filtering — brute-force
/// search scores the whole corpus in memory, matching the household-scale
/// assumption the rest of this crate already makes for `mental_objects`.
pub fn load_all_knowledge_docs(conn: &Connection) -> Result<Vec<KnowledgeDocRow>, StoreError> {
    let mut stmt = conn.prepare("SELECT id, source_uri, text, embedding, ingested_at FROM knowledge_library_docs")?;
    let mut rows = stmt.query([])?;

    let mut docs = Vec::new();
    while let Some(row) = rows.next()? {
        let id_text: String = row.get(0)?;
        let source_uri: String = row.get(1)?;
        let text: String = row.get(2)?;
        let embedding: Option<Vec<u8>> = row.get(3)?;
        let ingested_at: i64 = row.get(4)?;

        docs.push(KnowledgeDocRow {
            id: uuid::Uuid::from_str(&id_text)?,
            source_uri,
            text,
            embedding: embedding.as_deref().map(blob_to_embedding),
            ingested_at: EpochMillis(ingested_at),
        });
    }
    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::run_migrations;

    fn open_test_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        conn
    }

    /// Test-only convenience wrapping a single `upsert_object_in_tx` call in
    /// its own transaction - production code always batches through an
    /// already-open transaction (`SqliteStore::flush`), so this single-shot
    /// shape only exists for these one-object-at-a-time tests.
    fn upsert_object(conn: &mut Connection, object: &MentalObject) -> Result<(), StoreError> {
        let tx = conn.transaction()?;
        upsert_object_in_tx(&tx, object)?;
        tx.commit()?;
        Ok(())
    }

    #[test]
    fn upsert_and_load_round_trips_a_plain_object() {
        let mut conn = open_test_db();
        let mut object = MentalObject::new_observation("hello world", EpochMillis(1_000), 0.5);
        object.embedding = Some(vec![0.1, 0.2, 0.3]);
        object.memory_roles = vec![MemoryRole::Working];
        let id = object.id;

        upsert_object(&mut conn, &object).unwrap();
        let loaded = load_all_objects(&conn).unwrap();
        assert_eq!(loaded.len(), 1);
        let reloaded = &loaded[0];
        assert_eq!(reloaded.id, id);
        assert_eq!(reloaded.text, "hello world");
        assert_eq!(reloaded.embedding.as_ref().unwrap().len(), 3);
        assert!((reloaded.embedding.as_ref().unwrap()[1] - 0.2).abs() < 1e-6);
        assert_eq!(reloaded.memory_roles, vec![MemoryRole::Working]);
        assert_eq!(reloaded.activation.reference_log.len(), 1);
    }

    #[test]
    fn upsert_and_load_round_trips_provenance() {
        let mut conn = open_test_db();
        let mut object = MentalObject::new_observation("a derived pattern", EpochMillis(1_000), 0.5);
        object.produced_by_operator = Some("Plan".to_string());
        object.source_object_ids = vec![MentalObjectId::new(), MentalObjectId::new()];

        upsert_object(&mut conn, &object).unwrap();
        let loaded = load_all_objects(&conn).unwrap();
        let reloaded = &loaded[0];

        assert_eq!(reloaded.produced_by_operator, Some("Plan".to_string()));
        assert_eq!(reloaded.source_object_ids, object.source_object_ids);
    }

    #[test]
    fn upsert_and_load_round_trips_a_staged_candidate() {
        let mut conn = open_test_db();
        let mut object = MentalObject::new_observation("an unconfirmed belief", EpochMillis(1_000), 0.5);
        object.promotion = PromotionState::candidate(EpochMillis(1_000));

        upsert_object(&mut conn, &object).unwrap();
        let loaded = load_all_objects(&conn).unwrap();
        let reloaded = &loaded[0];

        assert_eq!(reloaded.promotion.status, PromotionStatus::Candidate);
        assert_eq!(reloaded.promotion.staged_at, EpochMillis(1_000));
        assert_eq!(reloaded.promotion.confirmed_at, None);
    }

    #[test]
    fn upsert_and_load_round_trips_a_confirmed_promotion_with_references() {
        let mut conn = open_test_db();
        let mut object = MentalObject::new_observation("a confirmed belief", EpochMillis(1_000), 0.5);
        object.promotion = PromotionState::candidate(EpochMillis(1_000));
        object.promotion.confirm(EpochMillis(5_000));

        upsert_object(&mut conn, &object).unwrap();
        let loaded = load_all_objects(&conn).unwrap();
        let reloaded = &loaded[0];

        assert_eq!(reloaded.promotion.status, PromotionStatus::Confirmed);
        assert_eq!(reloaded.promotion.confirmed_at, Some(EpochMillis(5_000)));
        assert_eq!(reloaded.promotion.confirming_references, 1);
    }

    #[test]
    fn a_pre_migration_row_with_no_promotion_columns_loads_as_confirmed() {
        // Simulates a row written before this migration existed: insert
        // directly, bypassing upsert_object_in_tx, relying purely on the
        // schema's own column defaults - the same contract
        // `PromotionState::default()` documents.
        let conn = open_test_db();
        conn.execute(
            r#"INSERT INTO mental_objects
                (id, kind, created_at, text, data_json, confidence, status, memory_roles_json,
                 base_level_cached, decay_d, last_computed_at)
               VALUES ('00000000-0000-0000-0000-000000000001', 'observation', 0, 'legacy row', 'null', 0.5, 'active', '[]', 0.0, 0.5, 0)"#,
            [],
        ).unwrap();

        let loaded = load_all_objects(&conn).unwrap();
        let reloaded = &loaded[0];
        assert_eq!(reloaded.promotion.status, PromotionStatus::Confirmed, "a pre-migration row must load as already-confirmed, never demoted");
    }

    #[test]
    fn upsert_and_load_round_trips_edges_and_goal() {
        let mut conn = open_test_db();
        let mut object = MentalObject::new_observation("goal object", EpochMillis(2_000), 0.5);
        object.kind = MentalObjectKind::Goal;
        object.goal = Some(GoalStackMembership {
            stack_id: GoalStackId::from_str("00000000-0000-0000-0000-000000000000").unwrap(),
            parent_goal_id: None,
            status: GoalStatus::Active,
            priority: 42.0,
        });
        object.edges.push(AssociativeEdge {
            target_id: MentalObjectId::new(),
            kind: EdgeKind::Causal,
            strength: 0.7,
            last_coactivated_at: EpochMillis(2_000),
        });

        upsert_object(&mut conn, &object).unwrap();
        let loaded = load_all_objects(&conn).unwrap();
        let reloaded = &loaded[0];

        let goal = reloaded.goal.as_ref().expect("goal should round-trip");
        assert_eq!(goal.status, GoalStatus::Active);
        assert!((goal.priority - 42.0).abs() < 1e-6);
        assert_eq!(reloaded.edges.len(), 1);
        assert_eq!(reloaded.edges[0].kind, EdgeKind::Causal);
    }

    #[test]
    fn inhibitory_edge_round_trips_as_a_distinct_persistent_kind() {
        let mut conn = open_test_db();
        let mut object = MentalObject::new_observation("winner", EpochMillis(0), 0.5);
        let target = MentalObjectId::new();
        object.edges.push(AssociativeEdge { target_id: target, kind: EdgeKind::Inhibitory, strength: 0.25, last_coactivated_at: EpochMillis(0) });
        upsert_object(&mut conn, &object).unwrap();
        let loaded = load_all_objects(&conn).unwrap();
        assert_eq!(loaded[0].edges[0].kind, EdgeKind::Inhibitory);
        assert_eq!(loaded[0].edges[0].target_id, target);
    }

    #[test]
    fn upsert_replaces_edges_and_references_rather_than_accumulating() {
        let mut conn = open_test_db();
        let mut object = MentalObject::new_observation("mutable", EpochMillis(1_000), 0.5);
        object.edges.push(AssociativeEdge {
            target_id: MentalObjectId::new(),
            kind: EdgeKind::Associative,
            strength: 0.5,
            last_coactivated_at: EpochMillis(1_000),
        });
        upsert_object(&mut conn, &object).unwrap();

        // Second upsert with a different single edge - must replace, not append.
        object.edges.clear();
        object.edges.push(AssociativeEdge {
            target_id: MentalObjectId::new(),
            kind: EdgeKind::Supports,
            strength: 0.9,
            last_coactivated_at: EpochMillis(2_000),
        });
        aca_util::EpochMillis(2_000); // no-op, documents the timestamp used above
        upsert_object(&mut conn, &object).unwrap();

        let loaded = load_all_objects(&conn).unwrap();
        assert_eq!(loaded.len(), 1, "same id must upsert in place, not duplicate");
        assert_eq!(loaded[0].edges.len(), 1, "old edge must be replaced, not accumulated");
        assert_eq!(loaded[0].edges[0].kind, EdgeKind::Supports);
    }

    #[test]
    fn cycle_event_round_trips() {
        let conn = open_test_db();
        let event = CycleEvent::new(
            7,
            EpochMillis(5_000),
            CyclePhase::Executive,
            CycleEventKind::Impasse,
            Some(Tier::T3),
            serde_json::json!({ "reason": "missing-information" }),
        );
        append_cycle_event(&conn, &event).unwrap();

        let events = recent_cycle_events(&conn, 10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].cycle_seq, 7);
        assert_eq!(events[0].phase, CyclePhase::Executive);
        assert_eq!(events[0].event_type, CycleEventKind::Impasse);
        assert_eq!(events[0].tier_used, Some(Tier::T3));
        assert_eq!(events[0].payload["reason"], "missing-information");
    }

    #[test]
    fn recent_cycle_events_respects_limit_and_order() {
        let conn = open_test_db();
        for seq in 0..5u64 {
            let event = CycleEvent::new(
                seq,
                EpochMillis(1_000 + seq as i64),
                CyclePhase::Predict,
                CycleEventKind::Normal,
                None,
                serde_json::Value::Null,
            );
            append_cycle_event(&conn, &event).unwrap();
        }
        let events = recent_cycle_events(&conn, 2).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].cycle_seq, 4, "most recent first");
        assert_eq!(events[1].cycle_seq, 3);
    }

    #[test]
    fn prune_cycle_events_uses_separate_normal_and_abnormal_windows() {
        let conn = open_test_db();
        for (seq, kind) in [
            (10, CycleEventKind::Normal),
            (20, CycleEventKind::Impasse),
            (80, CycleEventKind::Normal),
            (90, CycleEventKind::Error),
        ] {
            append_cycle_event(
                &conn,
                &CycleEvent::new(seq, EpochMillis(seq as i64), CyclePhase::Telemetry, kind, None, serde_json::Value::Null),
            )
            .unwrap();
        }

        let report = prune_cycle_events(
            &conn,
            100,
            CycleEventRetention {
                keep_recent_cycles: 50,
                keep_abnormal_cycles: 70,
            },
        )
        .unwrap();

        assert_eq!(report.deleted_normal_events, 1);
        assert_eq!(report.deleted_abnormal_events, 1);
        let remaining = recent_cycle_events(&conn, 10).unwrap();
        assert_eq!(remaining.iter().map(|event| event.cycle_seq).collect::<Vec<_>>(), vec![90, 80]);
    }
}
