use aca_store::KnowledgeLibraryStore;
use aca_tiers::EmbeddingClient;

/// The result of consulting the Knowledge Library — external information,
/// explicitly not memory (specs.md). Results re-enter the cycle as
/// Observations through Step 2 next cycle, never injected directly into
/// Working Memory, so consultation can't bypass the same competition-for-
/// attention everything else is subject to.
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeLibraryResult {
    pub found: bool,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct KnowledgeLibraryConfig {
    pub top_k: usize,
    /// Minimum cosine similarity for a match to count as "found" — without
    /// this, any non-empty store would report `found: true` for even a
    /// completely unrelated document, once even one row exists.
    pub min_score: f32,
}

impl Default for KnowledgeLibraryConfig {
    fn default() -> Self {
        Self { top_k: 3, min_score: 0.5 }
    }
}

/// Consults the Knowledge Library: embeds `query`, searches `kl_store`, and
/// returns the best match above `config.min_score` as `found`. Embed
/// failures, store errors, and an empty/below-threshold result set all
/// degrade to `found: false` rather than propagating an error — consulting
/// external knowledge is best-effort, never something that should fail the
/// surrounding cognitive cycle.
pub async fn consult(
    query: &str,
    embedding_client: &dyn EmbeddingClient,
    kl_store: &dyn KnowledgeLibraryStore,
    config: &KnowledgeLibraryConfig,
) -> KnowledgeLibraryResult {
    let Ok(embedding) = embedding_client.embed(query).await else {
        return KnowledgeLibraryResult { found: false, text: None };
    };

    match kl_store.search(&embedding, config.top_k).await {
        Ok(matches) => match matches.into_iter().find(|candidate| candidate.score >= config.min_score) {
            Some(best) => KnowledgeLibraryResult { found: true, text: Some(best.text) },
            None => KnowledgeLibraryResult { found: false, text: None },
        },
        Err(_) => KnowledgeLibraryResult { found: false, text: None },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_store::SqliteStore;
    use aca_tiers::testing::FakeEmbeddingClient;
    use aca_util::EpochMillis;

    #[tokio::test]
    async fn a_match_above_threshold_is_found() {
        let store = SqliteStore::open_in_memory().unwrap();
        let embedding_client = FakeEmbeddingClient::default();
        let embedding = embedding_client.embed("the sky is blue").await.unwrap();
        store.insert_document("file://note.txt", "the sky is blue", embedding, EpochMillis(1_000)).await.unwrap();

        let result = consult("the sky is blue", &embedding_client, &store, &KnowledgeLibraryConfig::default()).await;
        assert!(result.found);
        assert_eq!(result.text.as_deref(), Some("the sky is blue"));
    }

    #[tokio::test]
    async fn a_match_below_threshold_counts_as_not_found() {
        let store = SqliteStore::open_in_memory().unwrap();
        let embedding_client = FakeEmbeddingClient::default();
        let embedding = embedding_client.embed("completely unrelated document").await.unwrap();
        store.insert_document("file://note.txt", "completely unrelated document", embedding, EpochMillis(1_000)).await.unwrap();

        let config = KnowledgeLibraryConfig { top_k: 3, min_score: 1.1 }; // unreachable threshold
        let result = consult("the sky is blue", &embedding_client, &store, &config).await;
        assert!(!result.found);
        assert!(result.text.is_none());
    }

    #[tokio::test]
    async fn an_empty_store_reports_not_found() {
        let store = SqliteStore::open_in_memory().unwrap();
        let embedding_client = FakeEmbeddingClient::default();

        let result = consult("anything", &embedding_client, &store, &KnowledgeLibraryConfig::default()).await;
        assert!(!result.found);
        assert!(result.text.is_none());
    }
}
