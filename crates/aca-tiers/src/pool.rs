use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use aca_types::Tier;
use async_trait::async_trait;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::client::{ChatClient, EmbeddingClient, GenerateRequest, TierError, TierResponse};

async fn with_timeout<Fut>(timeout: Duration, tier: Tier, fut: Fut) -> Result<TierResponse, TierError>
where
    Fut: Future<Output = Result<TierResponse, TierError>>,
{
    match tokio::time::timeout(timeout, fut).await {
        Ok(result) => result,
        Err(_) => Err(TierError::Timeout {
            tier,
            elapsed_ms: timeout.as_millis() as u64,
        }),
    }
}

/// A concurrency-gated, timeout-wrapped view onto a `ChatClient` for one
/// tier. The concurrency limit *is* the tier's policy: Tier 1 = a large
/// pool ceiling, Tier 2 = 3, **Tier 3 = 1 — the literal single-flight
/// lock**, Tier 4 = 1 (rare, direct). One primitive (`tokio::sync::
/// Semaphore`), one policy knob (`concurrency`), reused across every tier.
///
/// Every call is wrapped in an explicit `tokio::time::timeout` sized per
/// tier — see `transport.rs` for why this can't be left to reqwest's
/// client-level defaults.
pub struct TierPool {
    semaphore: Arc<Semaphore>,
    timeout: Duration,
    tier: Tier,
    capacity: usize,
    label: String,
}

impl TierPool {
    pub fn new(tier: Tier, concurrency: usize, timeout: Duration) -> Self {
        let capacity = concurrency.max(1);
        Self {
            semaphore: Arc::new(Semaphore::new(capacity)),
            timeout,
            tier,
            capacity,
            label: format!("{tier:?}"),
        }
    }

    /// Attaches the real model identity (e.g. "llama3:8b") for a viewer to
    /// display - optional, so every existing caller (chiefly tests) that
    /// doesn't care what the model is called keeps working unchanged and
    /// falls back to the tier's own debug name.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// The configured concurrency ceiling - fixed at construction, useful
    /// alongside `available_permits()` for reporting "N of capacity busy"
    /// to an external observer (e.g. a live visualization).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Waits for a free slot (however long that takes), then runs the call
    /// under the configured timeout. Appropriate for Tier 1/2, where
    /// waiting briefly for a slot is fine because there are several.
    pub async fn run_chat(&self, client: &dyn ChatClient, req: GenerateRequest) -> Result<TierResponse, TierError> {
        let _permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("TierPool semaphore is never closed");
        with_timeout(self.timeout, self.tier, client.generate(req)).await
    }

    /// Non-blocking: returns `None` immediately if no slot is free right
    /// now rather than waiting for one. This is the actual mechanism behind
    /// Tier 3's single-flight rule — the SOAR Executive calls this to
    /// decide, in one tick, whether to spawn a real Tier-3 call or fall
    /// back to a cheap heuristic because the one seat is already busy.
    pub fn try_acquire(&self) -> Option<OwnedSemaphorePermit> {
        self.semaphore.clone().try_acquire_owned().ok()
    }

    /// Runs a call using a permit already obtained via `try_acquire` — the
    /// permit is held for the duration of the call and released on drop
    /// when this future completes, freeing the slot for the next attempt.
    pub async fn run_chat_with_permit(
        &self,
        _permit: OwnedSemaphorePermit,
        client: &dyn ChatClient,
        req: GenerateRequest,
    ) -> Result<TierResponse, TierError> {
        with_timeout(self.timeout, self.tier, client.generate(req)).await
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Whether the tier's one configured model is mid-call right now - a
    /// viewer's proxy for "this specific LLM is active," since every call
    /// through a `TierPool` uses the same caller-held client (see
    /// `run_chat`'s doc comment).
    pub fn is_busy(&self) -> bool {
        self.available_permits() < self.capacity
    }
}

/// Tier 1/2's shape: not a single-flight seat but "as many concurrent
/// divergent takes as are configured" (specs.md's own framing). A single
/// model being down or slow doesn't sink the whole tier - only every one of
/// them failing does. Intentionally not built on `TierPool`'s semaphore:
/// there's no shared seat to queue for here, each call fans out fresh across
/// genuinely independent clients/hosts.
pub struct DivergentPool {
    clients: Vec<Arc<dyn ChatClient>>,
    tier: Tier,
    timeout: Duration,
    /// Parallel to `clients` - a human-readable identity per candidate
    /// (e.g. its model name) for a viewer to display. Defaults to
    /// "candidate N" so every existing caller (chiefly tests) that doesn't
    /// care what the model is called keeps working unchanged; real callers
    /// override via `with_labels`.
    labels: Vec<String>,
    /// Parallel to `clients` - how many of that candidate's `generate` calls
    /// are in flight right now, incremented/decremented around each call in
    /// `sample`. A count rather than a bool: when `min_samples` exceeds
    /// `clients.len()`, `sample` round-robins multiple concurrent calls onto
    /// the same client, and a shared bool would have the first of those
    /// calls to finish clear the flag while sibling calls to that same
    /// client are still running.
    in_flight: Vec<Arc<AtomicUsize>>,
}

impl DivergentPool {
    pub fn new(tier: Tier, timeout: Duration, clients: Vec<Arc<dyn ChatClient>>) -> Self {
        let labels = (0..clients.len()).map(|i| format!("candidate {}", i + 1)).collect();
        let in_flight = (0..clients.len()).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        Self { clients, tier, timeout, labels, in_flight }
    }

    /// Overrides the default "candidate N" labels with real model identity,
    /// positionally matched to the clients passed to `new` - extra labels
    /// are ignored, missing ones keep their default.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        for (slot, label) in self.labels.iter_mut().zip(labels) {
            *slot = label;
        }
        self
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn len(&self) -> usize {
        self.clients.len()
    }

    /// An empty pool means "not configured for this tier" - callers treat
    /// it as a tier to skip past, not an error, so a fresh install with
    /// only Tier 3 configured still works exactly as before.
    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }

    /// One `(label, in_flight)` pair per configured candidate - a viewer's
    /// live view of exactly which individual models in this tier are
    /// currently mid-call.
    pub fn model_statuses(&self) -> Vec<(String, bool)> {
        self.labels.iter().cloned().zip(self.in_flight.iter().map(|count| count.load(Ordering::Relaxed) > 0)).collect()
    }

    /// Fans `req` out to `max(configured clients, min_samples)` concurrent
    /// calls (each under its own timeout, so one hung host can't stall the
    /// others) and returns every successful `TierResponse` - no picking a
    /// "winner" here, that decision needs cross-candidate agreement, which
    /// requires embeddings this crate deliberately doesn't have access to
    /// (see `EmbeddingClient`'s doc comment). When fewer clients are
    /// configured than `min_samples`, the shortfall is made up by calling
    /// configured clients again (round-robin) - a solo-model tier still
    /// gets a real multi-sample agreement check instead of only ever
    /// producing one uncorroborated answer. Errors only when every call
    /// fails.
    pub async fn sample(&self, req: GenerateRequest, min_samples: usize) -> Result<Vec<TierResponse>, TierError> {
        if self.clients.is_empty() {
            return Err(TierError::TierNotConfigured(self.tier));
        }
        let call_count = self.clients.len().max(min_samples);
        let calls = (0..call_count).map(|i| {
            let idx = i % self.clients.len();
            let client = self.clients[idx].as_ref();
            let count = self.in_flight[idx].clone();
            let req = req.clone();
            async move {
                count.fetch_add(1, Ordering::Relaxed);
                let result = with_timeout(self.timeout, self.tier, client.generate(req)).await;
                // Not RAII-guarded against `sample` itself being cancelled
                // from outside - every individual call is already bounded
                // by its own `with_timeout`, so `sample` completes in
                // bounded time on its own; matching this crate's existing
                // simplicity-over-robustness stance rather than adding a
                // Drop guard for a cancellation path nothing here exercises.
                count.fetch_sub(1, Ordering::Relaxed);
                result
            }
        });
        let results = futures::future::join_all(calls).await;
        let successes: Vec<TierResponse> = results.into_iter().filter_map(Result::ok).collect();
        if successes.is_empty() {
            return Err(TierError::AllCandidatesFailed { tier: self.tier, attempted: call_count });
        }
        Ok(successes)
    }
}

/// Wraps any `EmbeddingClient` with a bounded response timeout, transparent
/// to every caller since it implements the same trait - construct once at
/// startup around whichever real client is configured (see
/// `omega-acad::build_embedding_client`) and every consumer (Observe,
/// Executive escalation, `steps::synthesize`, the MCP server, the library
/// ingest loop) gets the protection automatically, with no call-site
/// changes anywhere.
///
/// Exists because every reasoning tier (`TierPool`/`DivergentPool`) already
/// wraps its calls in `tokio::time::timeout` - Tier 0 didn't, and `embed()`
/// was called completely unguarded at every site. `reqwest`'s client-level
/// `connect_timeout` (see `transport.rs`) only fails fast on a dead host;
/// it does nothing once a connection succeeds and the server just never
/// responds. Confirmed live: exactly that happened, and since Observe's
/// embedding call is awaited inline mid-`tick()`, a single hung request
/// froze the entire cognitive loop indefinitely - `cycle_seq` stopped
/// advancing and stayed stopped.
pub struct TimeoutEmbeddingClient {
    inner: Arc<dyn EmbeddingClient>,
    timeout: Duration,
}

impl TimeoutEmbeddingClient {
    pub fn new(inner: Arc<dyn EmbeddingClient>, timeout: Duration) -> Self {
        Self { inner, timeout }
    }
}

#[async_trait]
impl EmbeddingClient for TimeoutEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, TierError> {
        match tokio::time::timeout(self.timeout, self.inner.embed(text)).await {
            Ok(result) => result,
            Err(_) => Err(TierError::Timeout {
                tier: Tier::T0,
                elapsed_ms: self.timeout.as_millis() as u64,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    struct SlowClient {
        delay: Duration,
        in_flight: Arc<AtomicUsize>,
        max_observed_in_flight: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ChatClient for SlowClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_observed_in_flight.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(TierResponse {
                raw_text: "ok".to_string(),
                confidence: 0.5,
                tier: Tier::T2,
            })
        }
    }

    struct NeverRespondingClient {
        notify: Arc<Notify>,
    }

    #[async_trait]
    impl ChatClient for NeverRespondingClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            self.notify.notified().await; // never notified within the test's timeout window
            unreachable!("test timeout should fire first")
        }
    }

    fn req() -> GenerateRequest {
        GenerateRequest { prompt: "hi".into(), temperature: 0.3 }
    }

    #[tokio::test]
    async fn timeout_fires_cleanly_on_a_hanging_call() {
        let pool = TierPool::new(Tier::T3, 1, Duration::from_millis(20));
        let client = NeverRespondingClient { notify: Arc::new(Notify::new()) };
        let result = pool.run_chat(&client, req()).await;
        assert!(matches!(result, Err(TierError::Timeout { tier: Tier::T3, .. })));
    }

    #[tokio::test]
    async fn concurrency_limit_bounds_simultaneous_calls() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_observed = Arc::new(AtomicUsize::new(0));
        let client = SlowClient {
            delay: Duration::from_millis(30),
            in_flight: in_flight.clone(),
            max_observed_in_flight: max_observed.clone(),
        };
        let pool = TierPool::new(Tier::T2, 2, Duration::from_secs(5));

        // Poll 5 borrowing futures concurrently without detaching them into
        // independent 'static tasks - join_all is enough to prove the
        // semaphore bounds real concurrent-in-flight work.
        let futures = (0..5).map(|_| pool.run_chat(&client, req()));
        let results = futures::future::join_all(futures).await;
        for result in results {
            result.unwrap();
        }

        assert!(
            max_observed.load(Ordering::SeqCst) <= 2,
            "never more than the configured 2 concurrent slots should run at once"
        );
    }

    #[tokio::test]
    async fn try_acquire_returns_none_when_the_single_slot_is_taken() {
        let pool = TierPool::new(Tier::T3, 1, Duration::from_secs(5));
        let permit = pool.try_acquire().expect("first acquire should succeed");
        assert!(pool.try_acquire().is_none(), "single-flight: second acquire must fail while busy");
        drop(permit);
        assert!(pool.try_acquire().is_some(), "slot should free up after the permit drops");
    }

    #[tokio::test]
    async fn available_permits_reflects_pool_capacity() {
        let pool = TierPool::new(Tier::T2, 3, Duration::from_secs(5));
        assert_eq!(pool.available_permits(), 3);
        let _permit = pool.try_acquire().unwrap();
        assert_eq!(pool.available_permits(), 2);
    }

    struct FixedConfidenceClient {
        confidence: f32,
    }

    #[async_trait]
    impl ChatClient for FixedConfidenceClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Ok(TierResponse { raw_text: format!("confidence {}", self.confidence), confidence: self.confidence, tier: Tier::T2 })
        }
    }

    struct AlwaysFailsClient;

    #[async_trait]
    impl ChatClient for AlwaysFailsClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Err(TierError::TierNotConfigured(Tier::T2))
        }
    }

    #[tokio::test]
    async fn sample_collects_every_successful_candidate() {
        let pool = DivergentPool::new(
            Tier::T2,
            Duration::from_secs(5),
            vec![
                Arc::new(FixedConfidenceClient { confidence: 0.4 }),
                Arc::new(FixedConfidenceClient { confidence: 0.9 }),
                Arc::new(FixedConfidenceClient { confidence: 0.6 }),
            ],
        );
        let responses = pool.sample(req(), 3).await.unwrap();
        assert_eq!(responses.len(), 3);
    }

    #[tokio::test]
    async fn sample_ignores_individual_failures() {
        let pool = DivergentPool::new(
            Tier::T2,
            Duration::from_secs(5),
            vec![Arc::new(AlwaysFailsClient), Arc::new(FixedConfidenceClient { confidence: 0.5 })],
        );
        let responses = pool.sample(req(), 2).await.unwrap();
        assert_eq!(responses.len(), 1);
        assert!((responses[0].confidence - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn sample_errors_when_every_candidate_fails() {
        let pool = DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![Arc::new(AlwaysFailsClient), Arc::new(AlwaysFailsClient)]);
        let result = pool.sample(req(), 2).await;
        assert!(matches!(result, Err(TierError::AllCandidatesFailed { tier: Tier::T2, attempted: 2 })));
    }

    #[tokio::test]
    async fn sample_errors_cleanly_when_unconfigured() {
        let pool = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![]);
        assert!(pool.is_empty());
        let result = pool.sample(req(), 3).await;
        assert!(matches!(result, Err(TierError::TierNotConfigured(Tier::T1))));
    }

    struct CountingClient {
        confidence: f32,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ChatClient for CountingClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(TierResponse { raw_text: "ok".to_string(), confidence: self.confidence, tier: Tier::T1 })
        }
    }

    #[tokio::test]
    async fn sample_repeat_calls_a_single_configured_client_to_reach_min_samples() {
        let calls = Arc::new(AtomicUsize::new(0));
        let pool = DivergentPool::new(
            Tier::T1,
            Duration::from_secs(5),
            vec![Arc::new(CountingClient { confidence: 0.9, calls: calls.clone() })],
        );
        let responses = pool.sample(req(), 3).await.unwrap();
        assert_eq!(responses.len(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn sample_does_not_over_call_when_already_at_or_above_min_samples() {
        let pool = DivergentPool::new(
            Tier::T2,
            Duration::from_secs(5),
            vec![
                Arc::new(FixedConfidenceClient { confidence: 0.4 }),
                Arc::new(FixedConfidenceClient { confidence: 0.9 }),
                Arc::new(FixedConfidenceClient { confidence: 0.6 }),
                Arc::new(FixedConfidenceClient { confidence: 0.7 }),
            ],
        );
        let responses = pool.sample(req(), 2).await.unwrap();
        assert_eq!(responses.len(), 4, "one call per configured client when clients.len() >= min_samples");
    }
}
