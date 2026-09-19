//! Storage for the Knowledge Library: external documents Omega can consult,
//! never joined to `mental_objects` (see `schema.rs`'s own comment) — this
//! is deliberately a second, logically-separate surface over the same
//! connection `SqliteStore` already owns, not a second database.

use async_trait::async_trait;
use aca_util::EpochMillis;

use crate::dao;
use crate::types::StoreError;
use crate::writer::SqliteStore;

/// A Knowledge Library document's id. A plain `Uuid`, not the `MentalObjectId`
/// newtype — Knowledge Library documents are never mixed with Mental Objects
/// anywhere, so the type-safety motive for that newtype doesn't apply here.
pub type KnowledgeDocId = uuid::Uuid;

/// One document returned from `KnowledgeLibraryStore::search`, ranked by
/// cosine similarity to the query embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDocument {
    pub id: KnowledgeDocId,
    pub source_uri: String,
    pub text: String,
    pub score: f32,
    pub ingested_at: EpochMillis,
}

/// The Knowledge Library's storage surface: insert a document (already
/// embedded by the caller — this trait has no opinion on which embedding
/// model produced it), and brute-force cosine search over the corpus.
/// Household-scale corpus size is the explicit assumption behind "brute
/// force" here — see `aca_util::cosine_similarity`'s own doc comment for the
/// same assumption applied to `mental_objects`.
#[async_trait]
pub trait KnowledgeLibraryStore: Send + Sync {
    async fn insert_document(
        &self,
        source_uri: &str,
        text: &str,
        embedding: Vec<f32>,
        ingested_at: EpochMillis,
    ) -> Result<KnowledgeDocId, StoreError>;

    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<ScoredDocument>, StoreError>;

    /// The corpus size — reported to a viewer (`EngineSnapshot::kl_doc_count`)
    /// so growth from other agents' writes (never routed through the actor
    /// itself) is visible at all, not just individually-logged consult
    /// events.
    async fn count_documents(&self) -> Result<u64, StoreError>;
}

#[async_trait]
impl KnowledgeLibraryStore for SqliteStore {
    async fn insert_document(
        &self,
        source_uri: &str,
        text: &str,
        embedding: Vec<f32>,
        ingested_at: EpochMillis,
    ) -> Result<KnowledgeDocId, StoreError> {
        let conn = self.conn_handle();
        let id = uuid::Uuid::now_v7();
        let source_uri = source_uri.to_string();
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite connection mutex poisoned");
            dao::insert_knowledge_doc(&conn, id, &source_uri, &text, &embedding, ingested_at)?;
            Ok(id)
        })
        .await?
    }

    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<ScoredDocument>, StoreError> {
        let conn = self.conn_handle();
        let query_embedding = query_embedding.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite connection mutex poisoned");
            let rows = dao::load_all_knowledge_docs(&conn)?;
            let mut scored: Vec<ScoredDocument> = rows
                .into_iter()
                .filter_map(|row| {
                    let embedding = row.embedding?;
                    Some(ScoredDocument {
                        id: row.id,
                        source_uri: row.source_uri,
                        text: row.text,
                        score: aca_util::cosine_similarity(&query_embedding, &embedding),
                        ingested_at: row.ingested_at,
                    })
                })
                .collect();
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top_k);
            Ok(scored)
        })
        .await?
    }

    async fn count_documents(&self) -> Result<u64, StoreError> {
        let conn = self.conn_handle();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite connection mutex poisoned");
            dao::count_knowledge_docs(&conn)
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::SqliteStore;

    fn embedding(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    #[tokio::test]
    async fn insert_then_search_finds_its_own_embedding_with_near_maximal_score() {
        let store = SqliteStore::open_in_memory().unwrap();
        let id = store
            .insert_document("file://note.txt", "the sky is blue", embedding(&[1.0, 0.0, 0.0]), EpochMillis(1_000))
            .await
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
        assert_eq!(results[0].text, "the sky is blue");
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn search_ranks_by_cosine_similarity_descending() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.insert_document("a", "close match", embedding(&[1.0, 0.0]), EpochMillis(1)).await.unwrap();
        store.insert_document("b", "opposite", embedding(&[-1.0, 0.0]), EpochMillis(2)).await.unwrap();
        store.insert_document("c", "orthogonal", embedding(&[0.0, 1.0]), EpochMillis(3)).await.unwrap();

        let results = store.search(&[1.0, 0.0], 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "close match");
        assert_eq!(results[2].text, "opposite");
    }

    #[tokio::test]
    async fn search_over_an_empty_store_returns_an_empty_vec_not_an_error() {
        let store = SqliteStore::open_in_memory().unwrap();
        let results = store.search(&[1.0, 0.0], 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn count_documents_reflects_inserts() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert_eq!(store.count_documents().await.unwrap(), 0);

        store.insert_document("a", "first", embedding(&[1.0, 0.0]), EpochMillis(1)).await.unwrap();
        store.insert_document("b", "second", embedding(&[0.0, 1.0]), EpochMillis(2)).await.unwrap();

        assert_eq!(store.count_documents().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn top_k_larger_than_the_corpus_just_returns_the_whole_corpus() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.insert_document("a", "only doc", embedding(&[1.0, 0.0]), EpochMillis(1)).await.unwrap();

        let results = store.search(&[1.0, 0.0], 50).await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
