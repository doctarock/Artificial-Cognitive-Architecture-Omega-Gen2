use std::sync::Arc;

use aca_tiers::{EmbeddingClient, TierError};
use aca_types::{MentalObject, MentalObjectId};

use crate::steps::compare::SourceChannel;
use crate::steps::observe::resolve_embedding;

/// Everything `tick()` had already computed about a fresh observation
/// *before* the point it used to await `resolve_embedding` inline - Predict's
/// expectation, the channel/interlocutor this reading is keyed against, and
/// the shell itself (`embedding: None`, `text` already set). Carried across
/// to whichever later tick's reentry-drain sees the resolved result, so that
/// tick can finish exactly the same Compare/reward/graph-insert work
/// `apply_embedding_result` already did inline, just detached from the tick
/// that originally observed the text.
#[derive(Debug)]
pub struct PendingObservation {
    pub observation: MentalObject,
    pub expected: Option<Vec<f32>>,
    pub channel: SourceChannel,
    pub interlocutor_id: Option<MentalObjectId>,
    /// When this request was enqueued - lets the reentry-drain report the
    /// real end-to-end embedding latency (`enqueued_at.elapsed()`) as this
    /// tick's `embedding_wait_ms`, even though the wait itself happened
    /// across however many ticks passed in between, not inside one blocking
    /// await.
    pub enqueued_at: std::time::Instant,
    /// The dormant memories this observation was recombined from, if it
    /// came from `steps::boredom::generate_daydream` - carried across the
    /// same async boundary as everything else here, otherwise a daydream
    /// (whose text is freshly generated, so it's essentially always an
    /// `embedding_cache` miss) would lose its provenance the instant it
    /// took this path instead of the synchronous cache-hit one, and
    /// `loop_actor::tick`'s Hebbian link-back to those sources would only
    /// ever fire for the rare exact-text-repeat case. Empty for every
    /// non-daydream observation.
    pub daydream_source_ids: Vec<MentalObjectId>,
}

/// One completed (or failed) embedding call, paired with the request it
/// answers - the embedding worker's only output.
pub struct EmbeddingReentry {
    pub pending: PendingObservation,
    pub result: Result<Vec<f32>, TierError>,
}

/// Steps 2-3's asynchronous half, detached from the tick loop: resolves
/// embeddings one request at a time (the same effective rate-limiting the
/// old inline-await path already had - only one embedding call was ever in
/// flight per tick) and delivers each result back to the actor over
/// `reentry_tx`. `.send` here is a real, awaited backpressure point (this
/// task simply waits if the actor is slow to drain) - unlike the actor's own
/// internal self-sends (`kl_reentry_tx`/`act_reentry_tx`), which must use
/// `try_send` to avoid self-deadlocking a single-threaded drain loop, this
/// is a genuinely separate task, so waiting here can never block a tick.
/// Exits cleanly once `request_rx` closes (the actor dropped, e.g. on
/// shutdown) or `reentry_tx`'s receiver is gone - nothing left to serve
/// either way.
pub async fn run_embedding_worker(mut request_rx: tokio::sync::mpsc::Receiver<PendingObservation>, reentry_tx: tokio::sync::mpsc::Sender<EmbeddingReentry>, embedding_client: Arc<dyn EmbeddingClient>) {
    while let Some(pending) = request_rx.recv().await {
        let result = resolve_embedding(embedding_client.as_ref(), &pending.observation.text).await;
        if reentry_tx.send(EmbeddingReentry { pending, result }).await.is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_tiers::testing::FakeEmbeddingClient;
    use aca_util::EpochMillis;

    #[tokio::test]
    async fn resolves_one_request_and_delivers_it_back() {
        let (request_tx, request_rx) = tokio::sync::mpsc::channel(4);
        let (reentry_tx, mut reentry_rx) = tokio::sync::mpsc::channel(4);
        let client: Arc<dyn EmbeddingClient> = Arc::new(FakeEmbeddingClient::default());
        let worker = tokio::spawn(run_embedding_worker(request_rx, reentry_tx, client));

        let observation = MentalObject::new_observation("hello", EpochMillis(0), 0.5);
        request_tx
            .send(PendingObservation { observation, expected: None, channel: SourceChannel::ConversationInput, interlocutor_id: None, enqueued_at: std::time::Instant::now(), daydream_source_ids: Vec::new() })
            .await
            .unwrap();
        drop(request_tx);

        let reentry = reentry_rx.recv().await.expect("worker should deliver exactly one result");
        assert!(reentry.result.is_ok());
        assert_eq!(reentry.pending.observation.text, "hello");
        assert!(reentry_rx.recv().await.is_none(), "worker should exit once the request channel closes");
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn exits_cleanly_when_the_reentry_receiver_is_dropped() {
        let (request_tx, request_rx) = tokio::sync::mpsc::channel(4);
        let (reentry_tx, reentry_rx) = tokio::sync::mpsc::channel(4);
        drop(reentry_rx);
        let client: Arc<dyn EmbeddingClient> = Arc::new(FakeEmbeddingClient::default());
        let worker = tokio::spawn(run_embedding_worker(request_rx, reentry_tx, client));

        let observation = MentalObject::new_observation("hello", EpochMillis(0), 0.5);
        request_tx
            .send(PendingObservation { observation, expected: None, channel: SourceChannel::ConversationInput, interlocutor_id: None, enqueued_at: std::time::Instant::now(), daydream_source_ids: Vec::new() })
            .await
            .unwrap();

        // Should return promptly (the send inside the worker fails once the
        // reentry receiver is gone) rather than hang forever.
        tokio::time::timeout(std::time::Duration::from_secs(5), worker).await.expect("worker should exit, not hang").unwrap();
    }
}
