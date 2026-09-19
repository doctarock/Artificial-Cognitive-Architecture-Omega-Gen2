use rusqlite::Connection;
use serde::Serialize;

use crate::types::StoreError;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StoreTriageReport {
    pub page_size_bytes: u64,
    pub page_count: u64,
    pub freelist_count: u64,
    pub estimated_database_bytes: u64,
    pub estimated_free_bytes: u64,
    pub mental_objects: u64,
    pub reference_rows: u64,
    pub associative_edges: u64,
    pub cycle_events: u64,
    pub knowledge_docs: u64,
    pub embedded_mental_objects: u64,
    pub embedded_knowledge_docs: u64,
    pub mental_object_embedding_bytes: u64,
    pub knowledge_doc_embedding_bytes: u64,
    pub average_mental_object_text_bytes: f64,
    pub average_knowledge_doc_text_bytes: f64,
}

impl StoreTriageReport {
    pub fn from_connection(conn: &Connection) -> Result<Self, StoreError> {
        let page_size_bytes = pragma_u64(conn, "page_size")?;
        let page_count = pragma_u64(conn, "page_count")?;
        let freelist_count = pragma_u64(conn, "freelist_count")?;
        Ok(Self {
            page_size_bytes,
            page_count,
            freelist_count,
            estimated_database_bytes: page_size_bytes.saturating_mul(page_count),
            estimated_free_bytes: page_size_bytes.saturating_mul(freelist_count),
            mental_objects: count_rows(conn, "mental_objects")?,
            reference_rows: count_rows(conn, "mental_object_references")?,
            associative_edges: count_rows(conn, "associative_edges")?,
            cycle_events: count_rows(conn, "cycle_events")?,
            knowledge_docs: count_rows(conn, "knowledge_library_docs")?,
            embedded_mental_objects: count_where(conn, "mental_objects", "embedding IS NOT NULL")?,
            embedded_knowledge_docs: count_where(conn, "knowledge_library_docs", "embedding IS NOT NULL")?,
            mental_object_embedding_bytes: sum_blob_bytes(conn, "mental_objects", "embedding")?,
            knowledge_doc_embedding_bytes: sum_blob_bytes(conn, "knowledge_library_docs", "embedding")?,
            average_mental_object_text_bytes: avg_text_bytes(conn, "mental_objects", "text")?,
            average_knowledge_doc_text_bytes: avg_text_bytes(conn, "knowledge_library_docs", "text")?,
        })
    }
}

fn pragma_u64(conn: &Connection, name: &str) -> Result<u64, StoreError> {
    Ok(conn.query_row(&format!("PRAGMA {name}"), [], |row| row.get::<_, i64>(0))? as u64)
}

fn count_rows(conn: &Connection, table: &str) -> Result<u64, StoreError> {
    Ok(conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| row.get::<_, i64>(0))? as u64)
}

fn count_where(conn: &Connection, table: &str, predicate: &str) -> Result<u64, StoreError> {
    Ok(conn.query_row(&format!("SELECT COUNT(*) FROM {table} WHERE {predicate}"), [], |row| row.get::<_, i64>(0))? as u64)
}

fn sum_blob_bytes(conn: &Connection, table: &str, column: &str) -> Result<u64, StoreError> {
    Ok(conn.query_row(&format!("SELECT COALESCE(SUM(length({column})), 0) FROM {table}"), [], |row| row.get::<_, i64>(0))? as u64)
}

fn avg_text_bytes(conn: &Connection, table: &str, column: &str) -> Result<f64, StoreError> {
    Ok(conn.query_row(&format!("SELECT COALESCE(AVG(length({column})), 0.0) FROM {table}"), [], |row| row.get::<_, f64>(0))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::run_migrations;

    #[test]
    fn triage_report_counts_core_tables_and_storage_pressure() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        conn.execute(
            "INSERT INTO mental_objects (id, kind, created_at, text, embedding, data_json, confidence, tier_used, status, memory_roles_json, base_level_cached, decay_d, last_computed_at)
             VALUES ('m1', 'observation', 0, 'hello world', X'00000000', '{}', 1.0, NULL, 'active', '[]', 0.0, 0.5, 0)",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO mental_object_references (object_id, ts) VALUES ('m1', 0)", []).unwrap();
        conn.execute(
            "INSERT INTO knowledge_library_docs (id, source_uri, text, embedding, ingested_at) VALUES ('d1', 'memory://test', 'longer document', X'0000000000000000', 0)",
            [],
        )
        .unwrap();

        let report = StoreTriageReport::from_connection(&conn).unwrap();
        assert_eq!(report.mental_objects, 1);
        assert_eq!(report.reference_rows, 1);
        assert_eq!(report.knowledge_docs, 1);
        assert_eq!(report.embedded_mental_objects, 1);
        assert_eq!(report.mental_object_embedding_bytes, 4);
        assert_eq!(report.knowledge_doc_embedding_bytes, 8);
        assert!(report.estimated_database_bytes >= report.estimated_free_bytes);
    }
}
