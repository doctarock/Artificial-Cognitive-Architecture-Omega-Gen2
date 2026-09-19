use aca_tiers::{EmbeddingClient, TierError};
use aca_types::MentalObject;
use aca_util::EpochMillis;

/// Step 2 - Observe, synchronous half: constructs the observation shell
/// immediately (text is always known up front) with `embedding: None` — the
/// object is excluded from Compare/Coalition until its embedding resolves
/// (see `MentalObject::is_embedding_resolved`). This split is what keeps
/// Compare entirely free of I/O despite embeddings themselves being a real
/// network call.
pub fn new_observation_shell(text: impl Into<String>, now: EpochMillis, decay_d: f32) -> MentalObject {
    MentalObject::new_observation(text, now, decay_d)
}

/// Step 2 - Observe, asynchronous companion: resolves the embedding for
/// already-known text via whichever `EmbeddingClient` is configured. The
/// caller (the future `CognitiveLoopActor`) spawns this detached and
/// delivers the result back through its inbox — this function itself never
/// touches the graph or blocks a tick.
pub async fn resolve_embedding(
    client: &dyn EmbeddingClient,
    text: &str,
) -> Result<Vec<f32>, TierError> {
    client.embed(text).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_tiers::testing::FakeEmbeddingClient;
    use aca_types::ObjectStatus;

    #[test]
    fn shell_has_no_embedding_yet() {
        let shell = new_observation_shell("hello", EpochMillis(0), 0.5);
        assert!(shell.embedding.is_none());
        assert!(!shell.is_embedding_resolved());
        assert_eq!(shell.status, ObjectStatus::Active);
    }

    #[tokio::test]
    async fn resolve_embedding_attaches_a_real_vector() {
        let client = FakeEmbeddingClient::default();
        let mut shell = new_observation_shell("hello", EpochMillis(0), 0.5);
        let embedding = resolve_embedding(&client, &shell.text).await.unwrap();
        shell.embedding = Some(embedding);
        assert!(shell.is_embedding_resolved());
    }
}
