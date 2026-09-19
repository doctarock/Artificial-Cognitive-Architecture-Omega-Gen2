    use super::*;
    use aca_store::SqliteStore;
    use aca_tiers::testing::FakeEmbeddingClient;
    use aca_tiers::{AttentionDecision, AttentionError, AttentionOp, EmbeddingClient, TierError, TierResponse};
    use aca_types::Tier;
    use aca_util::{EpochMillis, ManualClock, SystemClock};
    use async_trait::async_trait;
    use std::str::FromStr;
    use std::time::Duration;

    /// A tick whose fresh input misses `embedding_cache` now only *enqueues*
    /// the request to `embedding_worker` (see that module's doc comment) -
    /// the actual resolution happens on the spawned worker task, applied on
    /// whichever later tick's reentry-drain sees it. `FakeEmbeddingClient`'s
    /// `embed` has no real `.await` inside, so a single `yield_now` is
    /// enough for the current-thread test runtime to poll the spawned
    /// worker to completion (recv -> embed -> send, all synchronous once
    /// polled) before the second tick drains its result. This is the
    /// two-tick equivalent of what used to be one synchronous tick before
    /// Steps 2-3's embedding resolution moved off the blocking path -
    /// tests that need a fresh (cache-miss) observation to be fully
    /// admitted use this instead of a single `actor.tick().await`.
    async fn tick_through_embedding_resolution(actor: &mut CognitiveLoopActor) {
        actor.tick().await;
        tokio::task::yield_now().await;
        actor.tick().await;
    }

    struct FixedChatClient;

    #[async_trait]
    impl ChatClient for FixedChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Ok(TierResponse { raw_text: "a reply".to_string(), confidence: 0.9, tier: Tier::T3 })
        }
    }

    struct ConfiguredChatClient {
        text: &'static str,
        confidence: f32,
        tier: Tier,
    }

    #[async_trait]
    impl ChatClient for ConfiguredChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Ok(TierResponse { raw_text: self.text.to_string(), confidence: self.confidence, tier: self.tier })
        }
    }

    struct AlwaysFailsChatClient;

    #[async_trait]
    impl ChatClient for AlwaysFailsChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            Err(TierError::TierNotConfigured(Tier::T4))
        }
    }

    /// Reads the candidate list straight out of the real prompt
    /// (`attention_workspace_prompt`'s output) and votes SUPPRESS on
    /// whichever candidate actually shows up first - lets these tests
    /// exercise the real `tick()` wiring end to end without needing to
    /// predict a freshly-created `MentalObjectId` ahead of time.
    struct SuppressFirstCandidateClient {
        confidence: f32,
    }

    #[async_trait]
    impl AttentionClient for SuppressFirstCandidateClient {
        async fn suggest(&self, prompt: String) -> Result<AttentionDecision, AttentionError> {
            let json_start = prompt.find('{').expect("prompt should contain the workspace JSON");
            let workspace: serde_json::Value = serde_json::from_str(&prompt[json_start..]).expect("workspace JSON should parse");
            let target_str = workspace["candidates"][0]["id"].as_str().expect("at least one candidate");
            Ok(AttentionDecision {
                operation: AttentionOp::Suppress,
                target: MentalObjectId::from_str(target_str).ok(),
                confidence: self.confidence,
                reason_code: "test_suppress".to_string(),
            })
        }
    }

    struct AlwaysErrorsAttentionClient;

    #[async_trait]
    impl AttentionClient for AlwaysErrorsAttentionClient {
        async fn suggest(&self, _prompt: String) -> Result<AttentionDecision, AttentionError> {
            Err(AttentionError::MalformedResponse { reason: "test error".to_string() })
        }
    }

    struct CountingReplyClient {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatClient for CountingReplyClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(TierResponse { raw_text: "a reply".to_string(), confidence: 0.9, tier: Tier::T3 })
        }
    }

    struct InvalidTargetAttentionClient;

    #[async_trait]
    impl AttentionClient for InvalidTargetAttentionClient {
        async fn suggest(&self, _prompt: String) -> Result<AttentionDecision, AttentionError> {
            Ok(AttentionDecision {
                operation: AttentionOp::Attend,
                target: Some(MentalObjectId::new()),
                confidence: 0.99,
                reason_code: "test_invalid_target".to_string(),
            })
        }
    }

    fn empty_tier1() -> DivergentPool {
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![])
    }

    fn with_surprise(error_magnitude: f32, precision: f32) -> MentalObject {
        let mut object = MentalObject::new_observation("an episode", EpochMillis(0), 0.5);
        object.prediction.error_magnitude = Some(error_magnitude);
        object.prediction.precision = Some(precision);
        object
    }

    #[test]
    fn rank_by_surprise_orders_highest_precision_weighted_surprise_first() {
        let mut graph = Graph::new();
        let unremarkable = with_surprise(0.1, 1.0); // surprise 0.1
        let very_surprising = with_surprise(0.9, 2.0); // surprise 1.8
        let mildly_surprising = with_surprise(0.4, 1.0); // surprise 0.4
        let (unremarkable_id, very_id, mild_id) = (unremarkable.id, very_surprising.id, mildly_surprising.id);
        graph.insert(unremarkable);
        graph.insert(very_surprising);
        graph.insert(mildly_surprising);

        // Buffered in a deliberately unsorted (arrival) order.
        let buffered = vec![unremarkable_id, very_id, mild_id];
        let ranked = rank_by_surprise(&graph, &buffered);

        assert_eq!(ranked, vec![very_id, mild_id, unremarkable_id], "should be ordered by precision-weighted surprise, most surprising first");
    }

    #[test]
    fn rank_by_surprise_treats_a_stale_id_as_zero_surprise_rather_than_panicking() {
        let mut graph = Graph::new();
        let real = with_surprise(0.5, 1.0);
        let real_id = real.id;
        graph.insert(real);
        let stale_id = MentalObjectId::new(); // never inserted

        let ranked = rank_by_surprise(&graph, &[stale_id, real_id]);

        assert_eq!(ranked, vec![real_id, stale_id], "a real, surprising memory should outrank a stale id with no reconstructable surprise");
    }

    fn empty_tier2() -> DivergentPool {
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![])
    }

    #[test]
    fn log_signature_ignores_float_jitter_but_not_genuine_changes() {
        // ACT-R base-level activation and `sample_noise` both recompute
        // every tick, so a preference/confidence/score field can differ
        // slightly between two cycles that are otherwise a genuine repeat -
        // the exact noise `log_cycle_event`'s throttle exists to see past.
        let a = json!({"operator": "Ignore", "preference": 0.5000001, "confidence": 0.9});
        let b = json!({"operator": "Ignore", "preference": 0.5000009, "confidence": 0.9000004});
        assert_eq!(log_signature(&a), log_signature(&b), "float-only differences should collapse to the same signature");

        let c = json!({"operator": "Speak", "preference": 0.5000001, "confidence": 0.9});
        assert_ne!(log_signature(&a), log_signature(&c), "a genuine field change (a different operator) must still change the signature");
    }

    #[tokio::test]
    async fn emit_event_throttles_identical_normal_repeats_but_not_genuine_changes() {
        // Regression guard from the pre-scheduler loop: an idle daemon
        // re-settled on the same terminal operator hundreds of times a
        // second, and
        // logging every single repeat unthrottled filled a 435 MB log file
        // in about twenty minutes.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, _handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            vec![],
        );

        let repeats = TelemetrySink::LOG_HEARTBEAT_EVERY * 2;
        for i in 0..repeats {
            // Only the float jitters between calls - a genuine steady-state
            // repeat, not a real change.
            actor.emit_event(CyclePhase::Executive, CycleEventKind::Normal, None, json!({"operator": "Ignore", "preference": 0.5 + i as f32 * 0.0001}));
        }
        let (_signature, count) = actor.telemetry.log_repeat.get(&CyclePhase::Executive).cloned().expect("a repeat signature should be tracked after several identical events");
        assert_eq!(count, repeats, "every repeat should still be counted internally, even though only every Nth is actually logged");

        actor.emit_event(CyclePhase::Executive, CycleEventKind::Normal, None, json!({"operator": "Speak", "preference": 1.0}));
        let (_, count_after_change) = actor.telemetry.log_repeat.get(&CyclePhase::Executive).cloned().expect("a signature should still be tracked after a genuine change");
        assert_eq!(count_after_change, 1, "a genuine payload change must reset the repeat count, not accumulate onto the old streak");
    }

    fn actor_with_config(config: LoopConfig) -> CognitiveLoopActor {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (actor, _handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            vec![],
        );
        actor
    }

    #[tokio::test]
    async fn conversational_presence_is_zero_before_any_conversation_input_has_broadcast() {
        let actor = actor_with_config(LoopConfig::default());
        assert_eq!(actor.conversational_presence(EpochMillis(1_000_000)), 0.0);
    }

    #[tokio::test]
    async fn conversational_presence_is_one_the_instant_a_conversational_turn_broadcasts() {
        let mut actor = actor_with_config(LoopConfig::default());
        let now = EpochMillis(1_000_000);
        actor.memory.last_broadcast_by_channel.insert(SourceChannel::ConversationInput, now);
        assert_eq!(actor.conversational_presence(now), 1.0);
    }

    #[tokio::test]
    async fn conversational_presence_decays_by_half_at_exactly_one_half_life() {
        let config = LoopConfig { presence_half_life_ms: 30_000, ..Default::default() };
        let mut actor = actor_with_config(config);
        let broadcast_at = EpochMillis(1_000_000);
        actor.memory.last_broadcast_by_channel.insert(SourceChannel::ConversationInput, broadcast_at);
        let presence = actor.conversational_presence(EpochMillis(broadcast_at.0 + 30_000));
        assert!((presence - 0.5).abs() < 1e-4, "expected ~0.5 at exactly one half-life, got {presence}");
    }

    #[tokio::test]
    async fn conversational_presence_is_negligible_after_a_long_silence() {
        let config = LoopConfig { presence_half_life_ms: 30_000, ..Default::default() };
        let mut actor = actor_with_config(config);
        let broadcast_at = EpochMillis(1_000_000);
        actor.memory.last_broadcast_by_channel.insert(SourceChannel::ConversationInput, broadcast_at);
        // Two minutes of silence - the smoke-alarm/backup-log scenario this
        // whole mechanism exists to leave in the fully introspective register.
        let presence = actor.conversational_presence(EpochMillis(broadcast_at.0 + 120_000));
        assert!(presence < 0.1, "expected presence to have receded close to 0 after a long silence, got {presence}");
    }

    #[tokio::test]
    async fn emit_event_tracks_repeats_per_phase_not_as_one_global_slot() {
        // Regression guard for the exact bug a single-slot version had:
        // confirmed live, a real tick fires Coalition, then Executive, then
        // Act, then Learn in sequence - comparing every event only against
        // whichever event was logged immediately before it (regardless of
        // phase) meant the "is this a repeat" check could never match,
        // since Executive's signature is never equal to Coalition's. A
        // fully-settled Working Memory item spinning at ~1000 ticks/sec
        // still wrote every single line unthrottled as a result. Per-phase
        // tracking is what actually fixes it: this interleaves two phases
        // exactly like one real tick does, and both must still accumulate
        // their own repeat counts correctly.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, _handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            vec![],
        );

        for _ in 0..10 {
            actor.emit_event(CyclePhase::Coalition, CycleEventKind::Normal, None, json!({"candidate_count": 2}));
            actor.emit_event(CyclePhase::Executive, CycleEventKind::Normal, None, json!({"operator": "Ignore"}));
            actor.emit_event(CyclePhase::Act, CycleEventKind::Normal, None, json!({"operator": "silent"}));
            actor.emit_event(CyclePhase::Learn, CycleEventKind::Normal, None, json!({"reinforced_edges": 2}));
        }

        for phase in [CyclePhase::Coalition, CyclePhase::Executive, CyclePhase::Act, CyclePhase::Learn] {
            let (_, count) = actor.telemetry.log_repeat.get(&phase).cloned().unwrap_or_else(|| panic!("{phase:?} should have its own tracked repeat count"));
            assert_eq!(count, 10, "{phase:?} should count all 10 of its own repeats, undisturbed by the other phases interleaved between them");
        }
    }

    #[tokio::test]
    async fn consult_knowledge_library_fires_once_not_once_per_tick_for_a_static_question() {
        // Mirrors no_runaway_repetition.rs's speak_count == 1 regression
        // shape, one operator earlier: a Question seeded directly (bypassing
        // the normal MissingInformation-impasse path, which is unreachable
        // in the live tick loop today - see steps::executive's proposal
        // rule doc comment) should be consulted exactly once, then settle to
        // Ignore, never re-proposing Consult every subsequent tick.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut question = aca_types::MentalObject::new_observation("subgoal: what is the wifi password?", SystemClock.now(), 0.5);
        question.kind = aca_types::MentalObjectKind::Question;
        let question_id = question.id;

        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            vec![question],
        );
        // Seeded directly into the graph via `initial_objects`, but that
        // alone doesn't nominate it for Coalition - `pending_admission` is
        // what `spawn_subgoal` would normally populate for a live-created
        // subgoal (see the impasse-handling arm in `tick()`).
        actor.memory.pending_admission.insert(question_id);

        let mut consult_count = 0;
        for _ in 0..50 {
            actor.tick().await;
            while let Ok(event) = handles.events_rx.try_recv() {
                if event.payload.get("operator").and_then(|v| v.as_str()) == Some("consult-knowledge-library") {
                    consult_count += 1;
                }
            }
        }

        assert_eq!(consult_count, 1, "Consult should fire exactly once for a static Question across 50 ticks, not once per tick");
    }

    #[tokio::test]
    async fn act_fires_once_not_once_per_tick_for_a_tool_matching_question() {
        // Full live pathway, unlike the Consult test above: real input
        // through `handles.input_tx`, exercising propose_operators' Act
        // trigger and act()'s tool lookup exactly as a real turn would.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let tool_registry = ToolRegistry::new(vec![Arc::new(crate::steps::tools::CurrentTimeTool::new(Arc::new(SystemClock)))], crate::steps::tools::ToolRiskTier::Harmless);
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            tool_registry,
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("what time is it?".to_string()).await.unwrap();

        let mut act_count = 0;
        for _ in 0..50 {
            actor.tick().await;
            while let Ok(event) = handles.events_rx.try_recv() {
                if event.payload.get("operator").and_then(|v| v.as_str()) == Some("act") {
                    act_count += 1;
                }
            }
        }

        assert_eq!(act_count, 1, "Act should fire exactly once for a static tool-matching question across 50 ticks, not once per tick");
    }

    #[tokio::test]
    async fn a_single_tick_with_input_produces_a_live_event() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        actor.tick().await;

        let event = handles.events_rx.try_recv().expect("at least one event should have been emitted");
        assert_eq!(event.cycle_seq, 1);

        let snapshot = handles.snapshot_rx.borrow();
        assert_eq!(snapshot.cycle_seq, 1);
    }

    fn new_test_actor_with_config(store: Arc<SqliteStore>, config: LoopConfig) -> (CognitiveLoopActor, LoopHandles) {
        CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        )
    }

    fn new_test_actor(store: Arc<SqliteStore>) -> (CognitiveLoopActor, LoopHandles) {
        new_test_actor_with_config(store, LoopConfig::default())
    }

    #[tokio::test]
    async fn forced_reconciliation_bypasses_the_model_after_the_configured_number_of_trusted_ticks() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.attention_reconciliation_interval = 1; // force a reconciliation on every other consultation
        config.attention_shadow_only = false; // this test explicitly exercises active control
        let (actor, handles) = new_test_actor_with_config(store, config);
        let mut actor = actor.with_attention_client(Arc::new(SuppressFirstCandidateClient { confidence: 0.95 }));

        // Tick 1: counter starts at 0 (< interval 1) - model is consulted,
        // trusted, and its SUPPRESS vote empties Working Memory.
        handles.input_tx.send("hello 1".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        assert!(actor.memory.working_memory.is_empty(), "the model's high-confidence SUPPRESS should have fired on the first consultation");

        // Tick 2: counter is now 1 (>= interval 1) - forced reconciliation
        // bypasses the model entirely, so the deterministic algorithm (not
        // the always-suppressing mock) decides this tick, admitting the
        // fresh candidate it would otherwise have suppressed.
        handles.input_tx.send("hello 2".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        assert!(
            !actor.memory.working_memory.is_empty(),
            "a forced reconciliation tick should bypass the model and let the deterministic algorithm admit the fresh candidate"
        );
    }

    #[tokio::test]
    async fn unconfigured_attention_client_admits_the_fresh_observation_via_the_deterministic_algorithm() {
        // No `.with_attention_client(...)` call - `attention_client` stays
        // `None`, so `tick()`'s Step 6 must take the
        // `decide_admission_deterministic` branch exclusively.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = new_test_actor(store);

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        assert!(!actor.memory.working_memory.is_empty(), "the deterministic algorithm should still admit the fresh observation exactly as before this change");
    }

    #[tokio::test]
    async fn configured_high_confidence_suppress_vote_overrides_the_deterministic_algorithm() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.attention_shadow_only = false;
        let (actor, mut handles) = new_test_actor_with_config(store, config);
        let mut actor = actor.with_attention_client(Arc::new(SuppressFirstCandidateClient { confidence: 0.95 }));

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        assert!(
            actor.memory.working_memory.is_empty(),
            "a high-confidence SUPPRESS vote should have kept the deterministic algorithm's would-be winner out of Working Memory entirely"
        );
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let decision_event = events.iter().find(|event| event.payload["attention_model_decision"] == true)
            .expect("trusted specialist decision should emit a paired shadow record");
        assert_eq!(decision_event.payload["shadow_membership_agreement"], false);
        assert_eq!(decision_event.payload["model_admitted_ids"].as_array().unwrap().len(), 0);
        assert_eq!(decision_event.payload["deterministic_shadow_ids"].as_array().unwrap().len(), 1);
        assert!(decision_event.payload["target_id"].as_str().is_some());
    }

    #[tokio::test]
    async fn high_confidence_attention_vote_is_observed_but_cannot_override_in_shadow_mode() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (actor, mut handles) = new_test_actor(store);
        let mut actor = actor.with_attention_client(Arc::new(SuppressFirstCandidateClient { confidence: 0.95 }));
        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        assert!(!actor.memory.working_memory.is_empty(), "shadow specialist must not suppress the deterministic winner");
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let decision = events.iter().find(|event| event.payload["attention_model_decision"] == true).unwrap();
        assert_eq!(decision.payload["attention_shadow_only"], true);
        assert_eq!(decision.payload["shadow_membership_agreement"], false);
    }

    #[tokio::test]
    async fn configured_low_confidence_vote_falls_back_to_the_deterministic_algorithm() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (actor, mut handles) = new_test_actor(store);
        // Below `LoopConfig::default().attention_min_confidence` (0.6).
        let mut actor = actor.with_attention_client(Arc::new(SuppressFirstCandidateClient { confidence: 0.3 }));

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        assert!(
            !actor.memory.working_memory.is_empty(),
            "a below-threshold vote must not be trusted - the deterministic algorithm should still have admitted the fresh observation"
        );
        assert!(std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event|
            event.payload["attention_model_fallback"] == "low_confidence_or_ineligible_target"));
    }

    #[tokio::test]
    async fn configured_erroring_client_falls_back_to_the_deterministic_algorithm() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (actor, mut handles) = new_test_actor(store);
        let mut actor = actor.with_attention_client(Arc::new(AlwaysErrorsAttentionClient));

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        assert!(
            !actor.memory.working_memory.is_empty(),
            "a failed model call must not block admission - the deterministic algorithm should still have run this tick"
        );
        assert!(std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event|
            event.payload["attention_model_fallback"] == "client_error"));
    }

    #[tokio::test]
    async fn high_confidence_attention_vote_for_an_absent_target_falls_back_to_deterministic_admission() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (actor, handles) = new_test_actor(store);
        let mut actor = actor.with_attention_client(Arc::new(InvalidTargetAttentionClient));

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        assert!(!actor.memory.working_memory.is_empty(), "a confident but unresolvable model target must not suppress the normal deterministic winner");
    }

    #[tokio::test]
    async fn local_specialist_training_events_link_orienting_inputs_to_actual_admission() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = new_test_actor(store);
        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let comparison = events.iter().find(|event| event.phase == CyclePhase::Compare
            && event.payload["observation_id"].as_str().is_some())
            .expect("resolved observation should have a keyed numeric feature record");
        let observation_id = comparison.payload["observation_id"].as_str().unwrap();
        let inputs = &comparison.payload["orienting_inputs"];
        for field in ["novelty", "prediction_error", "goal_relevance", "affective_salience", "social_relevance", "threat", "urgency"] {
            assert!(inputs[field].as_f64().is_some(), "{field} must be a numeric input feature");
        }
        assert!(events.iter().any(|event| event.phase == CyclePhase::Broadcast
            && event.payload["newly_admitted_ids"].as_array().is_some_and(|ids| ids.iter().any(|id| id == observation_id))),
            "broadcast should name the actual admission so offline data can join features to outcome");
    }

    #[tokio::test]
    async fn independent_host_outcome_is_keyed_durable_and_not_self_generated() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = new_test_actor(store.clone());
        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        let comparison = std::iter::from_fn(|| handles.events_rx.try_recv().ok())
            .find(|event: &CycleEvent| event.phase == CyclePhase::Compare && event.payload["observation_id"].as_str().is_some())
            .unwrap();
        let observation_id: MentalObjectId = comparison.payload["observation_id"].as_str().unwrap().parse().unwrap();
        assert!(!store.recent_cycle_events(100).await.unwrap().iter().any(|event|
            event.payload["independent_observation_outcome"] == true),
            "normal cognitive processing must never label its own success");
        handles.outcome_feedback_tx.send(OutcomeFeedbackCommand { observation_id, successful: true }).await.unwrap();
        actor.tick().await;
        assert!(store.recent_cycle_events(100).await.unwrap().iter().any(|event|
            event.phase == CyclePhase::Learn && event.payload["independent_observation_outcome"] == true
                && event.payload["observation_id"] == observation_id.to_string()
                && event.payload["successful"] == true),
            "host label must be durably keyed to the actual Compare observation");
        while handles.events_rx.try_recv().is_ok() {} // rejection events only below
        handles.outcome_feedback_tx.send(OutcomeFeedbackCommand { observation_id, successful: false }).await.unwrap();
        actor.tick().await;
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        assert!(events.iter().any(|event| event.payload["outcome_feedback_rejected"] == "unknown_expired_or_already_labeled_observation"));
        assert!(!events.iter().any(|event| event.payload["independent_observation_outcome"] == true));
    }

    #[tokio::test]
    async fn gated_orient_outcome_specialist_emits_shadow_forecast_without_controlling_admission() {
        let artifact = serde_json::json!({
            "schema": "omega-orient-observed-outcome-logistic/v1", "status": "shadow_only",
            "features": ["novelty", "prediction_error", "goal_relevance", "affective_salience",
                "social_relevance", "threat", "urgency", "ignited"],
            "means": vec![0.0; 8], "scales": vec![1.0; 8],
            "weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], "bias": -1.0,
            "held_out_metrics": {"sample_count": 200, "balanced_accuracy": 0.8, "brier_score": 0.1}
        }).to_string();
        let specialist = crate::OrientOutcomeSpecialist::from_json(&artifact).unwrap();
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (actor, mut handles) = new_test_actor(store);
        let mut actor = actor.with_orient_outcome_specialist(specialist);
        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        assert!(!actor.memory.working_memory.is_empty(), "shadow forecaster cannot suppress normal admission");
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let forecast = events.iter().find(|event| event.payload["orient_outcome_shadow"] == true)
            .expect("fresh observed admission should receive a shadow forecast");
        assert_eq!(forecast.payload["observed_ignition"], true);
        assert!(forecast.payload["predicted_success_probability"].as_f64().is_some());
    }

    #[tokio::test]
    async fn a_failed_reflection_reaches_the_live_event_feed_as_a_genuine_error_not_silence() {
        // Regression test for the exact live failure this was built to fix:
        // a real user question reached Compare/Broadcast/Executive fine
        // (`ContinueReflecting` proposed), then Tier 3 returned an empty
        // completion, and the resulting silence was visually identical to
        // Omega deliberately choosing not to respond - the console showed
        // nothing at all where a genuine failure had actually happened. This
        // proves the fix over the same channel the UI actually consumes
        // (`events_rx`, not just `ActOutcome`/`act_outcome_payload` in
        // isolation - see the unit-level coverage for those in
        // `steps::act`'s own tests).
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(ConfiguredChatClient { text: "", confidence: 0.5, tier: Tier::T3 }),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("a real question that deserves a real answer".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut act_event = None;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.phase == CyclePhase::Act {
                act_event = Some(event);
            }
        }
        let act_event = act_event.expect("an Act event should have been emitted for the ContinueReflecting attempt");

        assert_eq!(
            act_event.event_type,
            CycleEventKind::Error,
            "a genuinely failed reflection must be Error severity, not Normal - Normal is indistinguishable from a deliberate Ignore in the UI"
        );
        assert_eq!(act_event.payload.get("reason").and_then(|v| v.as_str()), Some("reflection-failed"));
        assert_eq!(act_event.payload.get("attempted_operator").and_then(|v| v.as_str()), Some("ContinueReflecting"));
        let error_text = act_event.payload.get("error").and_then(|v| v.as_str()).expect("the payload should carry the underlying error text");
        assert!(error_text.contains("empty"), "expected the error text to explain what actually went wrong, got: {error_text}");
    }

    fn fast_boredom_config() -> crate::steps::boredom::BoredomConfig {
        crate::steps::boredom::BoredomConfig {
            idle_threshold_ms: 1_000,
            min_interval_ms: 1_000,
            // Always due, so this exercises the duty path (no Tier 1
            // needed) deterministically rather than depending on an
            // unconfigured-Tier-1 fallback.
            self_status_interval_ms: 0,
            ..crate::steps::boredom::BoredomConfig::default()
        }
    }

    #[tokio::test]
    async fn boredom_fires_the_self_status_duty_once_working_memory_has_been_idle_long_enough() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.boredom_config = fast_boredom_config();
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        // First tick: nothing incoming, WM stays empty, marks
        // `wm_empty_since` - too soon to be boredom-eligible yet.
        actor.tick().await;
        while handles.events_rx.try_recv().is_ok() {}

        // Still below `idle_threshold_ms` - should not fire.
        clock.advance(500);
        actor.tick().await;
        let fired_early = std::iter::from_fn(|| handles.events_rx.try_recv().ok())
            .any(|event| event.payload.get("operator").and_then(|v| v.as_str()) == Some("act"));
        assert!(!fired_early, "boredom should not fire before idle_threshold_ms has elapsed");

        // Past `idle_threshold_ms` now - the self-status duty should fire,
        // and (same tick, since `data.requested_tool` routes straight to
        // Act) actually invoke the tool too.
        clock.advance(1_000);
        tick_through_embedding_resolution(&mut actor).await;
        let acted_self_status = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event| {
            event.payload.get("operator").and_then(|v| v.as_str()) == Some("act")
                && event.payload.get("tool").and_then(|v| v.as_str()) == Some("self_status")
        });
        assert!(acted_self_status, "expected the idle self-status duty to fire and invoke self_status once idle long enough");
    }

    #[tokio::test]
    async fn boredom_ablation_suppresses_idle_self_stimulation() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.boredom_config = fast_boredom_config();
        config.ablation_config.disable_boredom = true;
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        actor.tick().await;
        while handles.events_rx.try_recv().is_ok() {}
        clock.advance(5_000);
        actor.tick().await;

        let acted_self_status = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event| {
            event.payload.get("operator").and_then(|v| v.as_str()) == Some("act")
                && event.payload.get("tool").and_then(|v| v.as_str()) == Some("self_status")
        });
        assert!(!acted_self_status, "disable_boredom should prevent idle self-generated tool use");
    }

    #[tokio::test]
    async fn boredom_never_fires_on_a_tick_where_real_input_is_pending() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.boredom_config = fast_boredom_config();
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        actor.tick().await;
        while handles.events_rx.try_recv().is_ok() {}
        // Comfortably past idle_threshold_ms - boredom would otherwise fire.
        clock.advance(5_000);

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut saw_conversation_compare = false;
        let mut saw_boredom_sourced_act = false;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.payload.get("text").and_then(|v| v.as_str()) == Some("hello Omega") {
                saw_conversation_compare = true;
            }
            if event.payload.get("operator").and_then(|v| v.as_str()) == Some("act")
                && event.payload.get("tool").and_then(|v| v.as_str()) == Some("self_status")
            {
                saw_boredom_sourced_act = true;
            }
        }
        assert!(saw_conversation_compare, "real input should still be processed as usual");
        assert!(!saw_boredom_sourced_act, "boredom must never fire on a tick where real input is pending, no matter how idle WM has been");
    }

    #[tokio::test]
    async fn self_status_interrupts_while_working_memory_is_non_empty_under_high_self_monitoring_pressure() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        // A low interrupt threshold, deterministically cleared below by a
        // direct field write rather than depending on many real ticks of
        // EMA convergence - `min_interval_ms`/`idle_threshold_ms` are left
        // at their real defaults on purpose, since this test is proving the
        // interrupt bypasses the *idle* gate specifically, not that a fast
        // config makes it fire.
        config.boredom_config = crate::steps::boredom::BoredomConfig { self_status_interrupt_drive_threshold: 0.5, ..crate::steps::boredom::BoredomConfig::default() };
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        // A real conversational turn lands and stays resident in Working
        // Memory - `boredom_eligible`'s own `working_memory.is_empty()`
        // requirement is never satisfied for the rest of this test, and
        // `wm_empty_since` never gets set either, so the ordinary idle duty
        // structurally cannot fire below; any self-status activity can only
        // be the interrupt path.
        handles.input_tx.send("let's talk about something".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        while handles.events_rx.try_recv().is_ok() {}
        assert!(!actor.memory.working_memory.is_empty(), "test setup must actually leave something resident in Working Memory");

        // Force self-monitoring pressure past the interrupt threshold - a
        // direct write to actor-local, non-persisted state, same shortcut
        // this test module already takes elsewhere for drive/tracker state.
        actor.drives.drive_state.resource_pressure = 0.9;

        // The interrupt stimulus's text is never cached yet (first use), so
        // - same as the idle duty's own first firing - this needs the
        // async embedding round trip, not a single `tick()`.
        tick_through_embedding_resolution(&mut actor).await;
        let saw_interrupt_compare = std::iter::from_fn(|| handles.events_rx.try_recv().ok())
            .any(|event| event.phase == CyclePhase::Compare && event.payload.get("text").and_then(|v| v.as_str()).is_some_and(|t| t.contains("doesn't feel right")));
        assert!(saw_interrupt_compare, "high self-monitoring pressure should inject the interrupt self-status stimulus even with Working Memory non-empty");
        assert!(!actor.memory.working_memory.is_empty(), "the interrupt must not have required Working Memory to empty out first");
    }

    #[tokio::test]
    async fn a_daydream_hebbian_links_its_source_memories_to_the_thought_it_produced() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.boredom_config = crate::steps::boredom::BoredomConfig {
            idle_threshold_ms: 1_000,
            min_interval_ms: 1_000,
            // The duty must not preempt the daydream this test is actually
            // exercising - `last_self_status_at` (set below) already keeps
            // its interval from being due; disabling both drive-threshold
            // early-fire paths too means only the daydream branch can fire.
            self_status_drive_threshold: 1.1,
            self_status_interrupt_drive_threshold: 1.1,
            ..crate::steps::boredom::BoredomConfig::default()
        };
        let daydream_text = "a pattern I noticed across those memories";
        let tier1 = DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![Arc::new(ConfiguredChatClient { text: daydream_text, confidence: 0.9, tier: Tier::T1 })]);
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            tier1,
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        // Three dormant Episodic memories - enough for `daydream_min_sources`
        // and small enough that `select_replay_sources`'s pool (sample_size
        // * pool_multiplier = 9, capped at 3 candidates) contains all three
        // deterministically, so this test doesn't depend on which ones the
        // rng inside `select_replay_sources` happens to pick.
        let mut source_ids = Vec::new();
        for text in ["went for a run in the rain", "skipped a run because of rain", "rescheduled a run around rain"] {
            let mut episode = MentalObject::new_observation(text, EpochMillis(0), 0.5);
            episode.memory_roles.push(MemoryRole::Episodic);
            source_ids.push(episode.id);
            actor.memory.graph.insert(episode);
        }

        // The duty's interval-due check must read false, and curiosity must
        // clear `daydream_min_drive` - both direct writes to actor-local,
        // non-persisted state, same shortcut this test module already takes
        // elsewhere.
        actor.perception.last_self_status_at = Some(EpochMillis(0));
        actor.drives.drive_state.curiosity = 1.0;

        actor.tick().await;
        while handles.events_rx.try_recv().is_ok() {}
        clock.advance(1_000);
        // The daydream text is freshly generated every time, so this is
        // always an `embedding_cache` miss - the async reentry path, not
        // the synchronous one, is what this test actually needs to prove
        // out (see `PendingObservation::daydream_source_ids`'s own doc
        // comment on why the two paths needed separate handling).
        tick_through_embedding_resolution(&mut actor).await;

        let mut linked_targets = Vec::new();
        for source_id in &source_ids {
            let source = actor.memory.graph.get(source_id).expect("source memory should still be in the graph");
            let target_id = source
                .edges
                .iter()
                .find(|edge| edge.kind == EdgeKind::DerivedFrom)
                .map(|edge| edge.target_id)
                .unwrap_or_else(|| panic!("expected a DerivedFrom edge from source {source_id:?}, got edges: {:?}", source.edges));
            linked_targets.push(target_id);
        }
        let daydream_id = linked_targets[0];
        assert!(linked_targets.iter().all(|target| *target == daydream_id), "every replay source should link to the same generated thought: {linked_targets:?}");
        let daydream_object = actor.memory.graph.get(&daydream_id).expect("the linked daydream thought should still be in the graph");
        assert_eq!(daydream_object.text, daydream_text);
    }

    struct PanicChatClient;

    #[async_trait]
    impl ChatClient for PanicChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            panic!("a compiled Tier-0 procedure must not call a generative model")
        }
    }

    struct PanicEmbeddingClient;

    #[async_trait]
    impl EmbeddingClient for PanicEmbeddingClient {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, TierError> {
            panic!("a compiled Tier-0 procedure must not call the embedding model")
        }
    }

    #[tokio::test]
    async fn a_verified_sequence_bypasses_every_generative_tier() {
        let now = EpochMillis(0);
        let mut seeded_graph = Graph::new();
        for i in 0..3 {
            crate::steps::procedural::record_verified_sequence_success(
                &mut seeded_graph,
                MentalObjectId::new(),
                "hello omega",
                "Hello Derek.",
                &[1.0, 0.0, 0.0],
                EpochMillis(i),
                0.5,
            );
        }
        let seeded_objects = seeded_graph.iter().cloned().collect();

        let mut config = LoopConfig::default();
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(PanicEmbeddingClient),
            Arc::new(PanicChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(ManualClock::new(now)),
            seeded_objects,
        );

        handles.input_tx.send("Hello Omega!".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let spoke_via_procedure = events.iter().any(|event| {
            event.phase == CyclePhase::Act
                && event.payload.get("text").and_then(|value| value.as_str()) == Some("Hello Derek.")
                && event.payload.get("render_path").and_then(|value| value.as_str()) == Some("CompiledProcedure")
                && event.payload.get("foreground_turn_elapsed_us").and_then(|value| value.as_u64()).is_some()
        });
        assert!(spoke_via_procedure, "the learned response should be spoken directly on the compiled procedure path");
        assert!(events.iter().any(|event| event.phase == CyclePhase::Telemetry
            && event.payload["compiled_procedure_hit"] == true
            && event.payload["elapsed_us"].as_u64().is_some()), "the fast path should be identifiable and measurable");
        assert!(!actor.memory.graph.iter().any(|object| object.kind == aca_types::MentalObjectKind::Reflection), "the fast path must not create a generated Reflection");
    }

    #[tokio::test]
    async fn verified_reflect_then_speak_macro_executes_on_the_actor_fast_path() {
        let mut graph = Graph::new();
        for i in 0..3 {
            crate::steps::procedural::record_verified_sequence_success(
                &mut graph, MentalObjectId::new(), "hello omega", "Hello Derek.",
                &[1.0, 0.0], EpochMillis(i), 0.5,
            ).unwrap();
        }
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(), Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store.clone(),
            ToolRegistry::empty(), Arc::new(ManualClock::new(EpochMillis(100_000))),
            graph.iter().cloned().collect(),
        );
        handles.input_tx.send("Hello Omega!".to_string()).await.unwrap();
        actor.tick().await;
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        assert!(events.iter().any(|event| event.phase == CyclePhase::Act
            && event.payload["text"] == "Hello Derek."
            && event.payload["render_path"] == "CompiledProcedure"),
            "confirmed two-step cognitive sequence should collapse into direct speech");
        assert!(!actor.memory.graph.iter().any(|object| object.kind == aca_types::MentalObjectKind::Reflection));
    }

    #[tokio::test]
    async fn verified_reflect_then_ignore_macro_executes_without_a_model_or_speech() {
        let mut graph = Graph::new();
        for i in 0..3 {
            crate::steps::procedural::record_verified_ignore_success(
                &mut graph, MentalObjectId::new(), "background hum", &[0.2, 0.8], EpochMillis(i), 0.5,
            ).unwrap();
        }
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(), Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store.clone(),
            ToolRegistry::empty(), Arc::new(ManualClock::new(EpochMillis(100_000))),
            graph.iter().cloned().collect(),
        );
        handles.input_tx.send("background hum".to_string()).await.unwrap();
        actor.tick().await;
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let ignored_event = events.iter().find(|event| event.phase == CyclePhase::Act
            && event.payload["attempted_operator"] == "Ignore"
            && event.payload["reason"] == "ignored"
            && event.payload["foreground_observation_id"].as_str().is_some())
            .unwrap_or_else(|| panic!("verified internal sequence should collapse into a directly attributable Ignore; events={events:?}"));
        assert!(!events.iter().any(|event| event.phase == CyclePhase::Act && event.payload["operator"] == "speak"));
        assert!(!actor.memory.graph.iter().any(|object| object.kind == aca_types::MentalObjectKind::Reflection));
        let source_id: MentalObjectId = ignored_event.payload["foreground_observation_id"].as_str().unwrap().parse().unwrap();
        handles.procedure_feedback_tx.send(ProcedureFeedbackCommand { observation_id: source_id, successful: false }).await.unwrap();
        actor.tick().await;
        assert!(actor.memory.compiled_procedure("background hum").is_none(),
            "negative real-world feedback must immediately demote direct Ignore too");
        let mut reloaded = Graph::new();
        for object in store.load_all().await.unwrap().objects { reloaded.insert(object); }
        assert!(crate::steps::procedural::compiled_procedure(&reloaded, "background hum").is_none(),
            "Ignore demotion must survive durable reload");
    }

    #[tokio::test]
    async fn host_feedback_compiles_three_actual_reflect_then_ignore_chains() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = new_test_actor(store);
        for i in 0..3 {
            let now = actor.clock.now();
            let mut observation = MentalObject::new_observation("background hum", now, 0.5);
            observation.embedding = Some(vec![0.2, 0.8]);
            let observation_id = observation.id;
            actor.memory.graph.insert(observation);
            actor.turn_latency.start(observation_id, Instant::now());
            let reflected = actor.apply_operator(
                &OperatorProposal { operator: crate::steps::executive::Operator::ContinueReflecting, target_id: observation_id, preference: 1.0, confidence: 1.0 },
                &[], 0.0, now,
            ).await;
            let crate::steps::act::ActOutcome::Reflected { reflection_id } = reflected else {
                panic!("the real first operator should create a reflection")
            };
            let ignored = actor.apply_operator(
                &OperatorProposal { operator: crate::steps::executive::Operator::Ignore, target_id: reflection_id, preference: 1.0, confidence: 1.0 },
                &[], 0.0, now,
            ).await;
            assert!(matches!(ignored, crate::steps::act::ActOutcome::Silent {
                reason: crate::steps::act::SilentReason::Ignored
            }));
            handles.procedure_feedback_tx.send(ProcedureFeedbackCommand {
                observation_id, successful: true,
            }).await.unwrap();
            actor.tick().await;
            if i < 2 { assert!(actor.memory.compiled_procedure("background hum").is_none()); }
        }
        let program = actor.memory.compiled_procedure("background hum").unwrap();
        assert_eq!(program.execution("background hum"), Some(crate::steps::procedural::CompiledExecution::Ignore));
    }

    #[tokio::test]
    async fn host_feedback_compiles_three_actual_reflect_then_ask_chains() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let reply_client = Arc::new(CountingReplyClient { calls });
        let clock = Arc::new(ManualClock::new(EpochMillis(100_000)));
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(), Arc::new(FakeEmbeddingClient::default()), reply_client.clone(),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            reply_client, store.clone(), store,
            ToolRegistry::empty(), clock.clone(), Vec::new(),
        );
        for i in 0..3 {
            clock.advance(1_000);
            let now = clock.now();
            let mut observation = MentalObject::new_observation("status update", now, 0.5);
            observation.embedding = Some(vec![0.4, 0.6]);
            let observation_id = observation.id;
            actor.memory.graph.insert(observation);
            actor.turn_latency.start(observation_id, Instant::now());
            let reflected = actor.apply_operator(
                &OperatorProposal { operator: crate::steps::executive::Operator::ContinueReflecting, target_id: observation_id, preference: 1.0, confidence: 1.0 },
                &[], 0.0, now,
            ).await;
            let crate::steps::act::ActOutcome::Reflected { reflection_id } = reflected else {
                panic!("the real first operator should create a reflection")
            };
            let asked = actor.apply_operator(
                &OperatorProposal { operator: crate::steps::executive::Operator::Ask, target_id: reflection_id, preference: 1.0, confidence: 1.0 },
                &[], 0.0, now,
            ).await;
            assert!(matches!(asked, crate::steps::act::ActOutcome::Spoke { ref text, .. } if text == "a reply"));
            handles.procedure_feedback_tx.send(ProcedureFeedbackCommand {
                observation_id, successful: true,
            }).await.unwrap();
            actor.tick().await;
            if i < 2 { assert!(actor.memory.compiled_procedure("status update").is_none()); }
        }
        let program = actor.memory.compiled_procedure("status update").unwrap();
        assert_eq!(program.execution("status update"),
            Some(crate::steps::procedural::CompiledExecution::AskExact("a reply")));
    }

    #[tokio::test]
    async fn verified_reflect_then_ask_executes_without_models_and_demotes() {
        let mut graph = Graph::new();
        for i in 0..3 {
            crate::steps::procedural::record_verified_ask_success(
                &mut graph, MentalObjectId::new(), "status update", "Which project do you mean?",
                &[0.4, 0.6], EpochMillis(i), 0.5,
            ).unwrap();
        }
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(), Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store.clone(),
            ToolRegistry::empty(), Arc::new(ManualClock::new(EpochMillis(100_000))),
            graph.iter().cloned().collect(),
        );
        handles.input_tx.send("status update".to_string()).await.unwrap();
        actor.tick().await;
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let asked = events.iter().find(|event| event.phase == CyclePhase::Act
            && event.payload["attempted_operator"] == "Ask"
            && event.payload["text"] == "Which project do you mean?"
            && event.payload["render_path"] == "CompiledProcedure")
            .unwrap_or_else(|| panic!("compiled Ask should execute directly; events={events:?}"));
        assert!(!actor.memory.graph.iter().any(|object| object.kind == aca_types::MentalObjectKind::Reflection));
        let source_id: MentalObjectId = asked.payload["foreground_observation_id"].as_str().unwrap().parse().unwrap();
        handles.procedure_feedback_tx.send(ProcedureFeedbackCommand { observation_id: source_id, successful: false }).await.unwrap();
        actor.tick().await;
        assert!(actor.memory.compiled_procedure("status update").is_none());
        let mut reloaded = Graph::new();
        for object in store.load_all().await.unwrap().objects { reloaded.insert(object); }
        assert!(crate::steps::procedural::compiled_procedure(&reloaded, "status update").is_none(),
            "Ask demotion must survive durable reload");
    }

    #[tokio::test]
    async fn communicative_intent_specialist_is_shadow_only_against_selected_operator() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (actor, mut handles) = new_test_actor(store);
        let artifact = include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../training/specialists/outputs/communicative_intent.json"));
        let specialist = crate::CommunicativeIntentSpecialist::from_json(artifact).unwrap();
        let mut actor = actor.with_communicative_intent_specialist(specialist);
        let now = actor.clock.now();
        let mut reflection = MentalObject::new_observation(
            "The destination folder is missing; ask which folder to use.", now, 0.5,
        );
        reflection.kind = aca_types::MentalObjectKind::Reflection;
        let reflection_id = reflection.id;
        actor.memory.graph.insert(reflection);
        let outcome = actor.apply_operator(
            &OperatorProposal { operator: crate::steps::executive::Operator::Ignore,
                target_id: reflection_id, preference: 1.0, confidence: 1.0 },
            &[], 0.0, now,
        ).await;
        assert!(matches!(outcome, crate::steps::act::ActOutcome::Silent {
            reason: crate::steps::act::SilentReason::Ignored
        }), "the shadow prediction must not replace the selected operator");
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        assert!(events.iter().any(|event| event.phase == CyclePhase::Learn
            && event.payload["communicative_intent_shadow"] == true
            && event.payload["actual_operator"] == "ignore"
            && event.payload["predicted_operator"] == "ask"
            && event.payload["agreement"] == false));
        assert_eq!(actor.memory.graph.get(&reflection_id).unwrap().status,
            aca_types::ObjectStatus::Archived,
            "an ignored Reflection must not cycle back into Executive");
        assert!(!actor.memory.working_memory.contains(&reflection_id));
    }

    #[tokio::test]
    async fn host_feedback_compiles_three_observed_reflect_then_speak_turns() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let reply_client = Arc::new(CountingReplyClient { calls: calls.clone() });
        let clock = Arc::new(ManualClock::new(EpochMillis(100_000)));
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config, Arc::new(FakeEmbeddingClient::default()), reply_client.clone(),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            reply_client, store.clone(), store.clone(),
            ToolRegistry::empty(), clock.clone(), Vec::new(),
        );

        for expected_successes in 1..=3 {
            clock.advance(1_000);
            handles.input_tx.send("nice to meet you".to_string()).await.unwrap();
            let mut spoken_source = None;
            for _ in 0..12 {
                actor.tick().await;
                tokio::task::yield_now().await;
                while let Ok(event) = handles.events_rx.try_recv() {
                    if event.phase == CyclePhase::Act && event.payload["operator"] == "speak" {
                        spoken_source = event.payload["foreground_observation_id"].as_str().and_then(|id| id.parse().ok());
                    }
                }
                if spoken_source.is_some() { break; }
            }
            let source_id: MentalObjectId = spoken_source.expect("ordinary turn should reflect then speak with an observation source");
            handles.procedure_feedback_tx.send(ProcedureFeedbackCommand {
                observation_id: source_id, successful: true,
            }).await.unwrap();
            actor.tick().await;
            assert!(store.recent_cycle_events(200).await.unwrap().iter().any(|event|
                event.payload["independent_observation_outcome"] == true
                    && event.payload["observation_id"] == source_id.to_string()
                    && event.payload["successful"] == true),
                "host-confirmed procedural success must also leave durable independent outcome evidence");
            let verified = actor.memory.graph.iter().find(|object| object.data["schema"] == "omega-verified-operator-sequence/v2")
                .expect("host feedback should create a verified macro candidate");
            assert_eq!(verified.data["verified_successes"], expected_successes);
            if expected_successes < 3 {
                assert!(actor.memory.compiled_procedure("nice to meet you").is_none());
            }
            while handles.events_rx.try_recv().is_ok() {} // next turn's events only
        }
        assert!(actor.memory.compiled_procedure("nice to meet you").is_some());
        let mut restored_graph = Graph::new();
        for object in store.load_all().await.unwrap().objects { restored_graph.insert(object); }
        assert!(crate::steps::procedural::compiled_index(&restored_graph).contains_key("nice to meet you"),
            "three host-verified successes must survive the actor's forced durable flush");
        let calls_before_fast_turn = calls.load(Ordering::SeqCst);
        clock.advance(1_000);
        handles.input_tx.send("nice to meet you".to_string()).await.unwrap();
        actor.tick().await;
        assert_eq!(calls.load(Ordering::SeqCst), calls_before_fast_turn, "mature verified macro must not call the generative client");
        let fast_events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        let fast_speech = fast_events.iter().find(|event| event.phase == CyclePhase::Act
            && event.payload["render_path"] == "CompiledProcedure")
            .expect("verified macro should speak directly");
        let fast_source: MentalObjectId = fast_speech.payload["foreground_observation_id"].as_str().unwrap().parse().unwrap();
        handles.procedure_feedback_tx.send(ProcedureFeedbackCommand {
            observation_id: fast_source, successful: false,
        }).await.unwrap();
        actor.tick().await;
        assert!(actor.memory.compiled_procedure("nice to meet you").is_none(), "negative observed feedback must demote the direct macro immediately");
        let mut demoted_graph = Graph::new();
        for object in store.load_all().await.unwrap().objects { demoted_graph.insert(object); }
        assert!(!crate::steps::procedural::compiled_index(&demoted_graph).contains_key("nice to meet you"),
            "negative feedback must also durably demote the macro");
        handles.procedure_feedback_tx.send(ProcedureFeedbackCommand {
            observation_id: fast_source, successful: true,
        }).await.unwrap();
        actor.tick().await;
        assert!(actor.memory.compiled_procedure("nice to meet you").is_none(), "replayed feedback for a consumed turn must not recredit the macro");
    }

    /// Run explicitly with `cargo test -p aca-engine local_compiled_turn_latency_probe
    /// -- --ignored --nocapture`. This is a repeatable *in-process actor*
    /// measurement, not producer-to-user delivery or LAN deployment SLO.
    #[tokio::test]
    #[ignore = "local performance probe; timings vary with host load"]
    async fn local_compiled_turn_latency_probe() {
        let mut graph = Graph::new();
        for i in 0..3 {
            crate::steps::procedural::record_verified_sequence_success(
                &mut graph, MentalObjectId::new(), "hello omega", "Hello Derek.", &[1.0, 0.0],
                EpochMillis(i), 0.5,
            );
        }
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(100_000)));
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config, Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store,
            ToolRegistry::empty(), clock.clone(), graph.iter().cloned().collect(),
        );

        let mut spoken_us = Vec::new();
        let mut actor_tick_us = Vec::new();
        for iteration in 0..110 {
            clock.advance(1_000); // outside the channel's refractory gap
            handles.input_tx.send("Hello Omega!".to_string()).await.unwrap();
            let started = Instant::now();
            actor.tick().await;
            let tick_us = started.elapsed().as_micros() as u64;
            let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
            let spoke = events.iter().find(|event| event.phase == CyclePhase::Act
                && event.payload["render_path"] == "CompiledProcedure"
                && event.payload["text"] == "Hello Derek.");
            if iteration >= 10 {
                actor_tick_us.push(tick_us);
                spoken_us.push(spoke.and_then(|event| event.payload["foreground_turn_elapsed_us"].as_u64())
                    .expect("every compiled input must produce a provenance-timed Speak event"));
            }
        }
        spoken_us.sort_unstable();
        actor_tick_us.sort_unstable();
        let p95 = |samples: &[u64]| samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
        let under_10ms = spoken_us.iter().filter(|&&us| us < 10_000).count();
        println!("local compiled actor: n={}, spoken_turn_p50_us={}, spoken_turn_p95_us={}, actor_tick_p95_us={}, spoken_under_10ms={:.1}%",
            spoken_us.len(), spoken_us[spoken_us.len() / 2], p95(&spoken_us), p95(&actor_tick_us),
            under_10ms as f64 * 100.0 / spoken_us.len() as f64);
    }

    #[tokio::test]
    async fn a_curated_static_answer_bypasses_embedding_and_chat_without_generalizing_the_question() {
        let mut graph = Graph::new();
        crate::steps::known_answers::record_curated_answer(
            &mut graph, "What is the capital of France?", "Paris.",
            &[1.0, 0.0], EpochMillis(0), 0.5,
        ).expect("explicit curated answer should be admitted");
        let initial_objects = graph.iter().cloned().collect();
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(), Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store,
            ToolRegistry::empty(), Arc::new(ManualClock::new(EpochMillis(0))), initial_objects,
        );
        assert!(actor.memory.curated_answer("What is the capital of Germany?").is_none());
        handles.input_tx.send("WHAT is the capital of France ?".to_string()).await.unwrap();
        actor.tick().await;
        let events: Vec<_> = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).collect();
        assert!(events.iter().any(|event| event.phase == CyclePhase::Act
            && event.payload["text"] == "Paris."
            && event.payload["render_path"] == "CuratedAnswer"
            && event.payload["foreground_turn_elapsed_us"].as_u64().is_some()));
        assert!(events.iter().any(|event| event.phase == CyclePhase::Telemetry
            && event.payload["curated_answer_hit"] == true));
        assert!(!actor.memory.graph.iter().any(|object| object.kind == aca_types::MentalObjectKind::Reflection));
    }

    #[tokio::test]
    async fn host_curated_answer_updates_and_revocations_are_serialized_through_the_actor() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config, Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store.clone(),
            ToolRegistry::empty(), Arc::new(ManualClock::new(EpochMillis(0))), Vec::new(),
        );
        handles.curated_answer_tx.send(CuratedAnswerCommand::Upsert {
            question: "What is the capital of France?".to_string(),
            answer: "Paris.".to_string(), question_embedding: vec![1.0, 0.0],
        }).await.unwrap();
        handles.curated_answer_tx.send(CuratedAnswerCommand::Upsert {
            question: "What is the capital of France?".to_string(),
            answer: "Paris, France.".to_string(), question_embedding: vec![1.0, 0.0],
        }).await.unwrap();
        actor.tick().await;
        assert_eq!(actor.memory.curated_answer("What is the capital of France?").unwrap().answer, "Paris, France.",
            "two revisions in one tick must apply in host command order");
        assert!(store.load_all().await.unwrap().objects.iter().any(|object|
            object.data["schema"] == "omega-curated-static-answer/v1"
                && object.status == aca_types::ObjectStatus::Active),
            "curation upsert should flush at the end of its tick");
        handles.input_tx.send("What is the capital of France?".to_string()).await.unwrap();
        actor.tick().await;
        assert!(std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event|
            event.phase == CyclePhase::Act && event.payload["render_path"] == "CuratedAnswer"));
        handles.curated_answer_tx.send(CuratedAnswerCommand::Revoke {
            question: "What is the capital of France?".to_string(),
        }).await.unwrap();
        actor.tick().await;
        assert!(actor.memory.curated_answer("What is the capital of France?").is_none());
        assert!(actor.memory.graph.iter().any(|object|
            object.data["schema"] == "omega-curated-static-answer/v1"
                && object.status == aca_types::ObjectStatus::Discarded));
        assert!(store.load_all().await.unwrap().objects.iter().any(|object|
            object.data["schema"] == "omega-curated-static-answer/v1"
                && object.status == aca_types::ObjectStatus::Discarded),
            "revocation should flush at the end of its tick");
    }

    #[tokio::test]
    async fn a_calibrated_sensor_signal_orients_without_embedding_io() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ignition_threshold = f32::INFINITY;
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;
        let (mut actor, handles) = CognitiveLoopActor::new(
            config,
            Arc::new(PanicEmbeddingClient),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(ManualClock::new(EpochMillis(0))),
            Vec::new(),
        );
        handles.input_tx.send("hello".to_string()).await.unwrap();
        handles.sensor_signal_tx.send(SensorSignal {
            text: "fall detected".to_string(),
            channel: SourceChannel::Environment,
            source: "camera-fall-classifier",
            entity_label: None,
            threat: 1.0,
            urgency: 1.0,
            embedding: Some(vec![1.0, 0.0, 0.0]),
        }).await.unwrap();
        CognitiveScheduler::wait_for_next_tick(&mut actor).await;
        actor.tick().await;
        let observation = actor.memory.graph.iter().find(|object| object.text == "fall detected").expect("sensor signal should resolve in one tick");
        assert!(!actor.memory.graph.iter().any(|object| object.text == "hello"), "urgent sensor signal should preempt queued conversation without consuming it");
        assert_eq!(actor.perception.take_input().as_deref(), Some("hello"), "preempted conversation should remain queued or primed for the next event");
        assert!(observation.data["orienting"]["fired"].as_bool().unwrap_or(false));
        assert_eq!(observation.data["orienting"]["threat"], 1.0);
        assert_eq!(observation.data["orienting"]["urgency"], 1.0);
        assert!(observation.data.get("supplied_embedding").is_none(), "the supplied vector belongs in the embedding field, not a duplicate data blob");
    }

    #[tokio::test]
    async fn a_repeated_quiet_environment_signal_updates_prediction_without_workspace_nomination() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        config.ablation_config.disable_recall = true;
        config.ignition_threshold = f32::INFINITY;
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config, Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store,
            ToolRegistry::empty(), clock.clone(), Vec::new(),
        );
        for _ in 0..2 {
            handles.sensor_signal_tx.send(SensorSignal {
                text: "quiet kitchen".to_string(), channel: SourceChannel::Environment,
                source: "camera", entity_label: None, threat: 0.0, urgency: 0.0,
                embedding: Some(vec![1.0, 0.0]),
            }).await.unwrap();
            actor.tick().await;
            clock.advance(100);
        }
        let observations: Vec<_> = actor.memory.graph.iter().filter(|object| object.text == "quiet kitchen").collect();
        assert_eq!(observations.len(), 2, "habituation must not drop observation or prediction history");
        assert_eq!(observations.iter().filter(|object| object.data["sensor_habituated"] == true).count(), 1);
        actor.config.ablation_config.disable_sensor_habituation = true;
        handles.sensor_signal_tx.send(SensorSignal {
            text: "quiet kitchen".to_string(), channel: SourceChannel::Environment,
            source: "camera", entity_label: None, threat: 0.0, urgency: 0.0,
            embedding: Some(vec![1.0, 0.0]),
        }).await.unwrap();
        actor.tick().await;
        let latest = actor.memory.graph.iter().filter(|object| object.text == "quiet kitchen")
            .max_by_key(|object| object.created_at).unwrap();
        assert_ne!(latest.data["sensor_habituated"], true, "ablation must restore ordinary Coalition eligibility");
        assert!(std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event|
            event.phase == CyclePhase::Compare && event.payload["sensor_habituated"] == true));
    }

    #[tokio::test]
    async fn a_first_time_greeting_bypasses_chat_generation() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(PanicChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(ManualClock::new(EpochMillis(0))),
            Vec::new(),
        );
        handles.input_tx.send("Hi Omega!".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        let greeted = std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event| {
            event.phase == CyclePhase::Act
                && event.payload["text"] == "Hello."
                && event.payload["render_path"] == "InnateReflex"
        });
        assert!(greeted, "an isolated greeting should become a direct Tier-0 speech action on its first occurrence");
        assert!(!actor.memory.graph.iter().any(|object| object.kind == aca_types::MentalObjectKind::Reflection));
    }

    #[tokio::test]
    async fn an_observed_goal_outcome_learns_a_signed_local_consequence() {
        let now = EpochMillis(1_000);
        let decision = MentalObject::new_observation("decision", now, 0.5);
        let mut goal = MentalObject::new_observation("goal", now, 0.5);
        goal.kind = aca_types::MentalObjectKind::Goal;
        goal.goal = Some(aca_types::GoalStackMembership {
            stack_id: aca_types::GoalStackId::new(), parent_goal_id: None,
            status: aca_types::GoalStatus::Satisfied, priority: 1.0,
        });
        let (decision_id, goal_id) = (decision.id, goal.id);
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        config.ignition_threshold = f32::INFINITY;
        let (mut actor, _handles) = CognitiveLoopActor::new(
            config,
            Arc::new(PanicEmbeddingClient),
            Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store,
            ToolRegistry::empty(), Arc::new(ManualClock::new(now)),
            vec![decision, goal],
        );
        actor.previous_goal_statuses.insert(goal_id, aca_types::GoalStatus::Active);
        actor.executive.last_operator_proposal = Some(crate::snapshot::OperatorProposalSummary {
            operator: "Act".to_string(), target_id: decision_id,
            preference: 1.0, confidence: 1.0, at: now,
        });
        actor.tick().await;
        let learned = actor.memory.graph.get(&decision_id).unwrap().edges.iter()
            .find(|edge| edge.target_id == goal_id && edge.kind == EdgeKind::Supports);
        assert!(learned.is_some(), "goal success should stamp a supportive consequence link to the recent decision");
    }

    #[tokio::test]
    async fn a_due_edge_spike_fires_a_dormant_neighbor_and_nominates_it() {
        let now = EpochMillis(1_000);
        let mut source = MentalObject::new_observation("source", now, 0.5);
        let mut target = MentalObject::new_observation("target", now, 0.5);
        source.embedding = Some(vec![1.0, 0.0]);
        target.embedding = Some(vec![0.0, 1.0]);
        let source_id = source.id;
        let target_id = target.id;
        reinforce_edge(&mut source.edges, target_id, EdgeKind::Associative, now, 1.0, 1.0);
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        config.ablation_config.disable_recall = true;
        config.orienting_config.firing_threshold = 0.2;
        config.ignition_threshold = f32::INFINITY;
        let clock = Arc::new(ManualClock::new(now));
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config, Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store,
            ToolRegistry::empty(), clock.clone(), vec![source, target],
        );
        assert_eq!(actor.memory.spike_events.schedule_from(&actor.memory.graph, source_id, now,
            10, 0.0, 4, 16), 1);
        clock.advance(10);
        actor.tick().await;
        assert_eq!(actor.memory.graph.get(&target_id).unwrap().dynamics.last_fired_at, Some(EpochMillis(1_010)));
        assert!(std::iter::from_fn(|| handles.events_rx.try_recv().ok()).any(|event|
            event.phase == CyclePhase::Coalition && event.payload["spike_firings"] == 1));
    }

    #[tokio::test]
    async fn an_inhibitory_edge_spike_reduces_target_potential_without_firing() {
        let now = EpochMillis(1_000);
        let mut source = MentalObject::new_observation("source", now, 0.5);
        let mut target = MentalObject::new_observation("target", now, 0.5);
        source.embedding = Some(vec![1.0, 0.0]);
        target.embedding = Some(vec![0.0, 1.0]);
        target.dynamics.potential = 0.45;
        let (source_id, target_id) = (source.id, target.id);
        reinforce_edge(&mut source.edges, target_id, EdgeKind::Inhibitory, now, 1.0, 1.0);
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        config.ablation_config.disable_recall = true;
        config.ignition_threshold = f32::INFINITY;
        let clock = Arc::new(ManualClock::new(now));
        let (mut actor, _handles) = CognitiveLoopActor::new(
            config, Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store,
            ToolRegistry::empty(), clock.clone(), vec![source, target],
        );
        actor.memory.spike_events.schedule_from(&actor.memory.graph, source_id, now, 10, 0.0, 4, 16);
        clock.advance(10);
        actor.tick().await;
        let dynamics = &actor.memory.graph.get(&target_id).unwrap().dynamics;
        assert!(dynamics.potential < 0.45);
        assert_eq!(dynamics.last_fired_at, None);
    }

    #[tokio::test]
    async fn spike_ablation_drops_queued_events_without_stimulating_targets() {
        let now = EpochMillis(1_000);
        let mut source = MentalObject::new_observation("source", now, 0.5);
        let target = MentalObject::new_observation("target", now, 0.5);
        let (source_id, target_id) = (source.id, target.id);
        reinforce_edge(&mut source.edges, target_id, EdgeKind::Associative, now, 1.0, 1.0);
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let mut config = LoopConfig::default();
        config.ablation_config.disable_agenda = true;
        config.ablation_config.disable_boredom = true;
        config.ablation_config.disable_synthesis = true;
        config.ablation_config.disable_recall = true;
        config.ablation_config.disable_spike_propagation = true;
        config.ignition_threshold = f32::INFINITY;
        let clock = Arc::new(ManualClock::new(now));
        let (mut actor, _handles) = CognitiveLoopActor::new(
            config, Arc::new(PanicEmbeddingClient), Arc::new(PanicChatClient),
            empty_tier1(), empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(PanicChatClient), store.clone(), store,
            ToolRegistry::empty(), clock.clone(), vec![source, target],
        );
        actor.memory.spike_events.schedule_from(&actor.memory.graph, source_id, now, 10, 0.0, 4, 16);
        clock.advance(10);
        actor.tick().await;
        assert_eq!(actor.memory.spike_events.len(), 0);
        assert_eq!(actor.memory.graph.get(&target_id).unwrap().dynamics.last_fired_at, None);
    }

    #[tokio::test]
    async fn a_backlog_of_queued_input_is_coalesced_into_one_observation_not_replayed_per_tick() {
        // Proves the fix for transcripts falling behind: if several
        // messages piled up on `input_rx` before the loop got a chance to
        // tick (e.g. a burst of speech utterances), the next tick should
        // pick up all of them at once as a single "newly discovered"
        // observation, not drain the queue one tick per message.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("hello".to_string()).await.unwrap();
        handles.input_tx.send("Omega".to_string()).await.unwrap();
        handles.input_tx.send("are you there".to_string()).await.unwrap();

        tick_through_embedding_resolution(&mut actor).await;

        let mut compare_events = 0u32;
        let mut compared_text = String::new();
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.phase == aca_store::CyclePhase::Compare {
                compare_events += 1;
                compared_text = event.payload.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            }
        }
        assert_eq!(compare_events, 1, "a queued backlog should produce exactly one Compare this tick, not one per queued message");
        assert_eq!(compared_text, "hello Omega are you there");

        // The queue should be fully drained by that one tick - nothing left
        // over to trickle out one-by-one on subsequent ticks.
        assert!(handles.input_tx.try_reserve().is_ok(), "channel should have free capacity after the coalesced drain");
        actor.tick().await;
        let mut second_tick_compare_events = 0u32;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.phase == aca_store::CyclePhase::Compare {
                second_tick_compare_events += 1;
            }
        }
        assert_eq!(second_tick_compare_events, 0, "the backlog was already fully consumed by the first tick");
    }

    #[tokio::test]
    async fn room_input_coalesces_within_a_stream_but_never_blends_across_streams() {
        // The multi-speaker plumbing's core fix: unlike plain `input_tx`
        // (a single-party channel, safe to coalesce wholesale), room-audio
        // backlog from *different* speakers must never be merged into one
        // Observation - two same-tick utterances from different streams
        // should surface as two separate Compare events across two ticks,
        // while same-stream backlog still coalesces exactly like before.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.room_input_tx.send(RoomInput { text: "hello".to_string(), speaker_label: None, stream_id: "track-a".to_string() }).await.unwrap();
        handles.room_input_tx.send(RoomInput { text: "Omega".to_string(), speaker_label: None, stream_id: "track-a".to_string() }).await.unwrap();
        handles.room_input_tx.send(RoomInput { text: "unrelated".to_string(), speaker_label: None, stream_id: "track-b".to_string() }).await.unwrap();

        tick_through_embedding_resolution(&mut actor).await;
        let mut first_tick_texts = Vec::new();
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.phase == aca_store::CyclePhase::Compare {
                first_tick_texts.push(event.payload.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string());
            }
        }
        assert_eq!(first_tick_texts, vec!["hello Omega"], "same-stream backlog should still coalesce into one Observation, and the other stream must not be blended in");

        tick_through_embedding_resolution(&mut actor).await;
        let mut second_tick_texts = Vec::new();
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.phase == aca_store::CyclePhase::Compare {
                second_tick_texts.push(event.payload.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string());
            }
        }
        assert_eq!(second_tick_texts, vec!["unrelated"], "the other stream's backlog should get its own tick and its own Observation, never blended with the first");
    }

    #[tokio::test]
    async fn repeated_room_utterances_from_a_named_speaker_grow_that_interlocutors_activation_not_a_new_node_each_time() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        handles
            .room_input_tx
            .send(RoomInput { text: "hello Omega".to_string(), speaker_label: Some("derek".to_string()), stream_id: "track-derek".to_string() })
            .await
            .unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let interlocutor_id = actor
            .memory.graph
            .iter()
            .find(|object| object.data.get("interlocutor_hint").and_then(|v| v.as_str()) == Some("derek"))
            .map(|object| object.id)
            .expect("an interlocutor node for 'derek' should have been created");
        let references_after_one = actor.memory.graph.get(&interlocutor_id).unwrap().activation.reference_log.len();

        clock.advance(1_000);
        handles
            .room_input_tx
            .send(RoomInput { text: "how are you".to_string(), speaker_label: Some("derek".to_string()), stream_id: "track-derek".to_string() })
            .await
            .unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let interlocutor_count = actor.memory.graph.iter().filter(|object| object.data.get("interlocutor_hint").is_some()).count();
        assert_eq!(interlocutor_count, 1, "a second utterance from the same recognized speaker must reinforce the existing node, not mint a second one");

        let references_after_two = actor.memory.graph.get(&interlocutor_id).unwrap().activation.reference_log.len();
        assert!(
            references_after_two > references_after_one,
            "a second utterance from the same recognized speaker should add another reference (familiarity via repetition, not a bonus dial)"
        );
    }

    #[tokio::test]
    async fn repeated_identical_text_reuses_the_cached_embedding_instead_of_a_fresh_call() {
        // Per-text call counts, not a single global counter: a real tick
        // also embeds the resulting Reflection's own (different) text via
        // `cognitive_core::reflect`, so a global counter would climb every
        // tick regardless of whether *this* text's embedding was cached.
        struct CountingEmbeddingClient {
            inner: FakeEmbeddingClient,
            calls: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, usize>>>,
        }
        #[async_trait]
        impl EmbeddingClient for CountingEmbeddingClient {
            async fn embed(&self, text: &str) -> Result<Vec<f32>, TierError> {
                *self.calls.lock().unwrap().entry(text.to_string()).or_insert(0) += 1;
                self.inner.embed(text).await
            }
        }

        let calls = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(CountingEmbeddingClient { inner: FakeEmbeddingClient::default(), calls: calls.clone() }),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );
        let text = "the same message every time";

        handles.input_tx.send(text.to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        assert_eq!(calls.lock().unwrap().get(text).copied(), Some(1), "the first arrival of this text should genuinely call the embedding client");

        clock.advance(1_000);
        handles.input_tx.send(text.to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        assert_eq!(
            calls.lock().unwrap().get(text).copied(),
            Some(1),
            "an exact-text repeat should be served from embedding_cache, not issue a second real call for this text"
        );
    }

    #[tokio::test]
    async fn each_recognized_interlocutor_gets_their_own_last_seen_embedding() {
        // Explicit theory of mind, made testable: two different recognized
        // speakers must each be predicted against their own last utterance,
        // not a single embedding shared across everyone on the conversation
        // channel - see `interlocutor_embeddings`'s doc comment.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        handles
            .room_input_tx
            .send(RoomInput { text: "alice talks about the ocean".to_string(), speaker_label: Some("alice".to_string()), stream_id: "track-alice".to_string() })
            .await
            .unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        clock.advance(1_000);
        handles
            .room_input_tx
            .send(RoomInput { text: "bob talks about tax law".to_string(), speaker_label: Some("bob".to_string()), stream_id: "track-bob".to_string() })
            .await
            .unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let alice_id = actor
            .memory.graph
            .iter()
            .find(|object| object.data.get("interlocutor_hint").and_then(|v| v.as_str()) == Some("alice"))
            .map(|object| object.id)
            .expect("an interlocutor node for 'alice' should have been created");
        let bob_id = actor
            .memory.graph
            .iter()
            .find(|object| object.data.get("interlocutor_hint").and_then(|v| v.as_str()) == Some("bob"))
            .map(|object| object.id)
            .expect("an interlocutor node for 'bob' should have been created");

        let alice_embedding = actor.prediction.interlocutor_embeddings.get(&alice_id).expect("alice should have her own tracked last-seen embedding");
        let bob_embedding = actor.prediction.interlocutor_embeddings.get(&bob_id).expect("bob should have his own tracked last-seen embedding");
        assert_ne!(alice_embedding, bob_embedding, "each interlocutor's own last-seen embedding should be tracked distinctly, not shared");
    }

    #[tokio::test]
    async fn an_environment_sensor_observation_flows_through_the_same_pipeline_as_conversation() {
        // Proves SensorInput (the forwarding path a future camera service -
        // or any other external perception service - would use) reaches
        // Predict/Observe/Compare and Working Memory exactly like a human
        // turn does, just tagged with a different SourceChannel.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles
            .sensor_input_tx
            .send(SensorInput {
                text: "someone entered the kitchen".to_string(),
                channel: SourceChannel::Environment,
                source: "camera",
                entity_label: None,
            })
            .await
            .unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut saw_compare_event = false;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.phase == CyclePhase::Compare {
                saw_compare_event = true;
                assert_eq!(event.payload.get("text").and_then(|v| v.as_str()), Some("someone entered the kitchen"));
            }
        }
        assert!(saw_compare_event, "a sensor observation should reach Compare exactly like any other input source");

        let snapshot = handles.snapshot_rx.borrow();
        assert!(
            snapshot.working_memory.iter().any(|m| m.text == "someone entered the kitchen"),
            "the sensor observation should compete for and win Working Memory admission like any other fresh observation"
        );
    }

    #[tokio::test]
    async fn a_camera_recognized_entity_label_resolves_to_the_same_interlocutor_a_voice_enrollment_already_built() {
        // The concrete proof behind video-interaction-plan.md section 4:
        // reusing the same enrolled name across voice (`RoomInput::speaker_label`)
        // and video (`SensorInput::entity_label`) must land both observations
        // on one persistent interlocutor node, not two - because both keys
        // fold into the same `data_tag["speaker_label"]`, which is all
        // `steps::interlocutor::find_or_create` ever keys on.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        handles
            .room_input_tx
            .send(RoomInput { text: "hello Omega".to_string(), speaker_label: Some("derek".to_string()), stream_id: "track-derek".to_string() })
            .await
            .unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        clock.advance(1_000);
        handles
            .sensor_input_tx
            .send(SensorInput { text: "Derek entered the camera view.".to_string(), channel: SourceChannel::Environment, source: "camera", entity_label: Some("derek".to_string()) })
            .await
            .unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let interlocutor_nodes: Vec<MentalObjectId> = actor
            .memory.graph
            .iter()
            .filter(|object| object.data.get("interlocutor_hint").and_then(|v| v.as_str()) == Some("derek"))
            .map(|object| object.id)
            .collect();
        assert_eq!(interlocutor_nodes.len(), 1, "voice and video observations of the same enrolled name must share exactly one interlocutor node, not mint a second one");

        let anchors = crate::steps::interlocutor::social_cloud_anchors(&actor.memory.graph, interlocutor_nodes[0]);
        assert_eq!(anchors.len(), 2, "both the voice utterance and the camera sighting should be linked to derek's one interlocutor node");
    }

    #[tokio::test]
    async fn snapshot_reports_memory_counts_goal_stack_and_tier_status() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 2, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("hello Omega".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        actor.tick().await;

        let snapshot = handles.snapshot_rx.borrow().clone();
        assert!(snapshot.memory_counts.working > 0, "the broadcast object should be tagged Working");
        assert_eq!(snapshot.tier_status.len(), 2, "both wired tiers (T3 and the escalation-only T4) should be reported");
        assert_eq!(snapshot.tier_status[0].tier, Tier::T3);
        assert_eq!(snapshot.tier_status[0].capacity, 2);
        assert!(snapshot.tier_status[0].available_permits <= 2);
        assert_eq!(snapshot.tier_status[1].tier, Tier::T4);
        assert_eq!(snapshot.tier_status[1].capacity, 1);
        assert!(snapshot.tier_status[1].available_permits <= 1);
        // No impasse forced in this scenario, so the goal stack should be empty -
        // this is as important to prove as the non-empty cases: an idle/settled
        // system should not fabricate goals that don't exist.
        assert!(snapshot.goal_stack.is_empty());
        // Nothing was ever synthesized in this scenario either - same
        // "don't fabricate what doesn't exist" discipline extended to the
        // provisional abstraction.
        assert!(snapshot.provisional_abstraction.is_none());
    }

    #[tokio::test]
    async fn snapshot_reports_a_synthesized_pattern_as_the_provisional_abstraction() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 2, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        let mut episode_a = aca_types::MentalObject::new_observation("went for a run in the rain", aca_util::EpochMillis(0), 0.5);
        episode_a.memory_roles.push(aca_types::MemoryRole::Episodic);
        let episode_a_id = episode_a.id;
        let mut episode_b = aca_types::MentalObject::new_observation("skipped a run because of rain", aca_util::EpochMillis(0), 0.5);
        episode_b.memory_roles.push(aca_types::MemoryRole::Episodic);
        let episode_b_id = episode_b.id;
        actor.memory.graph.insert(episode_a);
        actor.memory.graph.insert(episode_b);

        // Mirrors exactly what `steps::synthesize::synthesize` itself
        // produces (kind, roles, source_object_ids) - this test is
        // deliberately about `publish_snapshot`'s own reading of that shape,
        // not about `synthesize` itself (already covered in
        // `steps::synthesize`'s own tests).
        let mut pattern = aca_types::MentalObject::new_observation("weather strongly influences running habits", aca_util::EpochMillis(5_000), 0.5);
        pattern.kind = aca_types::MentalObjectKind::Memory;
        pattern.memory_roles.push(aca_types::MemoryRole::Semantic);
        pattern.confidence = 0.82;
        pattern.source_object_ids = vec![episode_a_id, episode_b_id];
        actor.memory.graph.insert(pattern);

        actor.publish_snapshot().await;

        let snapshot = handles.snapshot_rx.borrow().clone();
        let abstraction = snapshot.provisional_abstraction.expect("a synthesized pattern should be reported as the provisional abstraction");
        assert_eq!(abstraction.text, "weather strongly influences running habits");
        assert_eq!(abstraction.confidence, 0.82);
        assert_eq!(abstraction.source_count, 2);
        assert_eq!(abstraction.reinforced_count, 0, "a freshly-formed pattern has only its creation reference, no reconfirmations yet");
        assert_eq!(abstraction.supporting_episodes.len(), 2);
        assert!(abstraction.supporting_episodes.contains(&"went for a run in the rain".to_string()));
        assert!(abstraction.supporting_episodes.contains(&"skipped a run because of rain".to_string()));
    }

    #[tokio::test]
    async fn snapshot_prefers_the_most_recently_formed_abstraction_when_several_exist() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 2, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        let source_id = |actor: &mut CognitiveLoopActor, text: &str| {
            let mut episode = aca_types::MentalObject::new_observation(text, aca_util::EpochMillis(0), 0.5);
            episode.memory_roles.push(aca_types::MemoryRole::Episodic);
            let id = episode.id;
            actor.memory.graph.insert(episode);
            id
        };
        let a = source_id(&mut actor, "episode one");
        let b = source_id(&mut actor, "episode two");
        let c = source_id(&mut actor, "episode three");
        let d = source_id(&mut actor, "episode four");

        let mut older = aca_types::MentalObject::new_observation("an older pattern", aca_util::EpochMillis(1_000), 0.5);
        older.kind = aca_types::MentalObjectKind::Memory;
        older.memory_roles.push(aca_types::MemoryRole::Semantic);
        older.source_object_ids = vec![a, b];
        actor.memory.graph.insert(older);

        let mut newer = aca_types::MentalObject::new_observation("a more recent pattern", aca_util::EpochMillis(2_000), 0.5);
        newer.kind = aca_types::MentalObjectKind::Memory;
        newer.memory_roles.push(aca_types::MemoryRole::Semantic);
        newer.source_object_ids = vec![c, d];
        actor.memory.graph.insert(newer);

        actor.publish_snapshot().await;

        let snapshot = handles.snapshot_rx.borrow().clone();
        let abstraction = snapshot.provisional_abstraction.expect("the more recent pattern should be reported");
        assert_eq!(abstraction.text, "a more recent pattern", "the most recently formed abstraction should win, not the first one seen");
    }

    #[tokio::test]
    async fn snapshot_does_not_mistake_a_single_source_semantic_reclassification_for_a_synthesized_abstraction() {
        // A Reflection reclassified as Semantic via `memory_formation`'s
        // `SemanticUpdate` path always carries exactly one source id (the
        // single object it reflected on - see `cognitive_core::reflect`),
        // never two: `synthesize` itself refuses to run on fewer than two
        // usable episodic texts. This must never be picked up as a
        // provisional abstraction just because it happens to carry the
        // Semantic role too.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 2, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        let mut reflection = aca_types::MentalObject::new_observation("a single-source semantic reclassification", aca_util::EpochMillis(1_000), 0.5);
        reflection.kind = aca_types::MentalObjectKind::Reflection;
        reflection.memory_roles.push(aca_types::MemoryRole::Semantic);
        reflection.source_object_ids = vec![aca_types::MentalObjectId::new()];
        actor.memory.graph.insert(reflection);

        actor.publish_snapshot().await;

        assert!(handles.snapshot_rx.borrow().provisional_abstraction.is_none());
    }

    #[tokio::test]
    async fn an_idle_tick_with_no_input_does_not_panic_and_advances_cycle_seq() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, _handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );
        actor.tick().await;
        actor.tick().await;
        // No panic, no input needed - a fully idle loop is safe to run.
    }

    #[tokio::test]
    async fn a_forced_tie_produces_a_live_impasse_and_escalation_event() {
        // A wide tie epsilon guarantees propose_operators' two proposals
        // for any fresh object count as tied, so this proves the loop's
        // actual Impasse/Escalation wiring end to end - not just the
        // isolated select_operator logic already covered in executive.rs's
        // own tests.
        let mut config = LoopConfig::default();
        config.executive_config.preference_tie_epsilon = 1.0;
        // Phase 3's automatic memory formation (`steps::memory_formation::
        // maybe_automatic_remember`) would otherwise remove `Operator::
        // Remember` from propose_operators' output before this test's
        // forced tie ever gets a chance to form - a fresh/surprising
        // observation is exactly the case it's designed to catch. This test
        // is about impasse/escalation wiring, not automatic memory
        // formation, so it opts out explicitly rather than relying on an
        // incidental default.
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("ambiguous input".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut saw_impasse = false;
        let mut saw_escalation = false;
        while let Ok(event) = handles.events_rx.try_recv() {
            match event.event_type {
                aca_store::CycleEventKind::Impasse => saw_impasse = true,
                aca_store::CycleEventKind::Escalation => saw_escalation = true,
                _ => {}
            }
        }

        assert!(saw_impasse, "a forced tie should produce a live Impasse event");
        assert!(saw_escalation, "resolving that impasse via Tier 3 should produce a live Escalation event");
    }

    #[tokio::test]
    async fn an_escalated_tier3_answer_that_names_a_candidate_operator_is_actually_used() {
        // Regression test: the chunked resolution used to ignore what the
        // escalated tier actually said and just re-pick among the original
        // tied candidates by their own (usually indistinguishable)
        // self-reported confidence. When the tier's answer parses into one
        // of the operators actually in contention, that real answer - not a
        // fallback guess - must be what gets chunked.
        let mut config = LoopConfig::default();
        config.executive_config.preference_tie_epsilon = 1.0;
        // Phase 3's automatic memory formation (`steps::memory_formation::
        // maybe_automatic_remember`) would otherwise remove `Operator::
        // Remember` from propose_operators' output before this test's
        // forced tie ever gets a chance to form - a fresh/surprising
        // observation is exactly the case it's designed to catch. This test
        // is about impasse/escalation wiring, not automatic memory
        // formation, so it opts out explicitly rather than relying on an
        // incidental default.
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let tier3_client = ConfiguredChatClient { text: "remember", confidence: 0.9, tier: Tier::T3 };
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(tier3_client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("ambiguous input".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut chunked_operator: Option<String> = None;
        while let Ok(event) = handles.events_rx.try_recv() {
            if let Some(op) = event.payload.get("chunked_operator").and_then(|v| v.as_str()) {
                chunked_operator = Some(op.to_string());
            }
        }

        assert_eq!(
            chunked_operator.as_deref(),
            Some("Remember"),
            "the escalated tier's actual parsed answer should be used, not a fallback max-by-confidence guess"
        );
    }

    #[tokio::test]
    async fn an_escalated_impasse_resolution_is_also_applied_this_same_tick() {
        // Regression test: chunking a resolved impasse into a learned bias
        // for *future* occurrences of the same tie must not be the only
        // effect - the object the impasse was actually about should get
        // acted on this tick too. Without this, resolving "should I speak,
        // remember, or ask about this?" only ever paid off the next time
        // the same operator set happened to tie again, never for the
        // content that actually triggered the deliberation.
        let mut config = LoopConfig::default();
        config.executive_config.preference_tie_epsilon = 1.0;
        // Phase 3's automatic memory formation (`steps::memory_formation::
        // maybe_automatic_remember`) would otherwise remove `Operator::
        // Remember` from propose_operators' output before this test's
        // forced tie ever gets a chance to form - a fresh/surprising
        // observation is exactly the case it's designed to catch. This test
        // is about impasse/escalation wiring, not automatic memory
        // formation, so it opts out explicitly rather than relying on an
        // incidental default.
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let tier3_client = ConfiguredChatClient { text: "remember", confidence: 0.9, tier: Tier::T3 };
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(tier3_client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("ambiguous input".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut saw_remember_act = false;
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.payload.get("operator").and_then(|v| v.as_str()) == Some("remember") {
                saw_remember_act = true;
            }
        }

        assert!(
            saw_remember_act,
            "the escalated resolution should be applied to the impasse's own object this same tick, not only chunked for the future"
        );
    }

    /// A single client whose successive calls cycle through distinct canned
    /// responses - needed for pattern-synthesis tests because a constant
    /// reply (`FixedChatClient`) would make every Reflection embed
    /// identically, hitting `form_memory`'s near-duplicate `Reinforced`
    /// path instead of producing genuinely distinct `NewEpisodic` memories
    /// for the synthesis buffer to accumulate. Same shape as
    /// `cognitive_core.rs`'s own test-local `SequenceClient`.
    struct SequenceClient {
        responses: Vec<&'static str>,
        next: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl ChatClient for SequenceClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            let i = self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % self.responses.len();
            Ok(TierResponse { raw_text: self.responses[i].to_string(), confidence: 0.9, tier: Tier::T3 })
        }
    }

    fn sequence_client(responses: Vec<&'static str>) -> SequenceClient {
        SequenceClient { responses, next: std::sync::atomic::AtomicUsize::new(0) }
    }

    #[tokio::test]
    async fn a_same_channel_arrival_inside_the_refractory_window_is_deferred_not_lost() {
        // Regression test for a real bug caught while implementing the
        // attentional-refractory mechanism itself: deferring a refractory-
        // blocked raw Observation through the ordinary `pending_admission`
        // path (built for already-created Reflections, which never had a
        // surprise term) silently dropped its `new_surprise` term on its one
        // delayed shot at Coalition - without it, a genuinely surprising
        // second same-channel arrival could lose its own Coalition bid on a
        // low bare-activation score and simply vanish, never reaching
        // Reflection/Speak at all. Two inputs on the same channel, ticked
        // immediately back to back (well inside `attentional_refractory_ms`
        // in real wall-clock terms), must both eventually be spoken about.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let client = sequence_client(vec!["a reflection about topic A", "a reflection about topic B"]);
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("topic A".to_string()).await.unwrap();
        actor.tick().await;
        // Sent and ticked immediately after - in real wall-clock terms this
        // is almost certainly still inside the default 400ms refractory
        // window, which is exactly the scenario this test needs to exercise.
        handles.input_tx.send("topic B".to_string()).await.unwrap();

        let mut spoke_about_a = false;
        let mut spoke_about_b = false;
        let mut timed_spoken_turns = 0;
        for _ in 0..50 {
            actor.tick().await;
            while let Ok(event) = handles.events_rx.try_recv() {
                if event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                    if event.payload.get("foreground_turn_elapsed_us").and_then(|v| v.as_u64()).is_some() {
                        timed_spoken_turns += 1;
                    }
                    match event.payload.get("text").and_then(|v| v.as_str()) {
                        Some(text) if text.contains("topic A") => spoke_about_a = true,
                        Some(text) if text.contains("topic B") => spoke_about_b = true,
                        _ => {}
                    }
                }
            }
        }

        assert!(spoke_about_a, "the first, non-deferred arrival should be spoken about");
        assert!(spoke_about_b, "the second arrival, deferred by the refractory window, must still be spoken about, not silently lost");
        assert!(timed_spoken_turns >= 2, "both slow-path spoken responses should carry provenance-linked turn timing");
    }

    #[tokio::test]
    async fn pattern_synthesis_fires_once_and_the_resulting_object_can_win_broadcast_and_speak() {
        let mut config = LoopConfig::default();
        config.synthesis_config = crate::steps::synthesize::SynthesisConfig {
            min_new_episodic: 3,
            max_cluster_size: 8,
            min_interval_ms: 0,
            dedup_similarity_threshold: 0.93,
            promotion_gate_enabled: true,
        };
        // This test is about synthesis -> broadcast -> speak, driven under
        // real `SystemClock` timing across 150 real ticks - not about
        // Phase 3's automatic memory formation, which otherwise changes
        // which objects compete for Speak on which tick by removing
        // `Operator::Remember` from that competition for surprising
        // content. Opted out explicitly rather than re-tuning this test's
        // already-precise timing around an orthogonal new mechanism.
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let client = sequence_client(vec![
            "a reflection about the weather being unusually warm today",
            "a reflection about the market closing higher today",
            "a reflection about the new coffee shop opening downtown",
            "the pattern connecting several unrelated daily observations",
        ]);
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        let mut synthesize_events = 0u32;
        let mut synthesis_seen = false;
        let mut speak_or_ask_after_synthesis = 0u32;
        let mut saw_semantic_memory = false;

        for i in 0..150 {
            if i == 0 {
                handles.input_tx.send("the weather today".to_string()).await.unwrap();
            } else if i == 25 {
                handles.input_tx.send("the stock market today".to_string()).await.unwrap();
            } else if i == 50 {
                handles.input_tx.send("the new coffee shop".to_string()).await.unwrap();
            }
            actor.tick().await;

            while let Ok(event) = handles.events_rx.try_recv() {
                if event.phase == aca_store::CyclePhase::Synthesize {
                    synthesize_events += 1;
                    synthesis_seen = true;
                }
                if synthesis_seen {
                    if let Some(op) = event.payload.get("operator").and_then(|v| v.as_str()) {
                        if op == "speak" || op == "ask" {
                            speak_or_ask_after_synthesis += 1;
                        }
                    }
                }
            }
            if handles.snapshot_rx.borrow().memory_counts.semantic >= 1 {
                saw_semantic_memory = true;
            }
        }

        assert_eq!(synthesize_events, 1, "exactly one synthesis event should fire once 3 distinct episodic memories accumulate");
        assert!(saw_semantic_memory, "a new Semantic memory should exist in the graph after synthesis");
        // Not asserting an exact count here: under Working Memory
        // contention across 3 overlapping conversational turns, an
        // unrelated turn's own Speak can genuinely land after the
        // synthesis event by coincidence of tick timing - that's expected
        // architecture behavior, not a synthesis bug. "Never re-speaks the
        // *same* synthesized object" is already covered independently by
        // `propose_operators`'s existing dedup tests (unchanged by this
        // feature) plus `steps::synthesize`'s own unit tests. What this
        // integration test needs to prove is the new claim: a
        // self-synthesized memory can actually enter the Speak/Ask pipeline
        // unprompted, not that it's the only thing that ever speaks again.
        assert!(speak_or_ask_after_synthesis >= 1, "something should have been proposed to speak/ask after the pattern synthesized - proves the new memory can compete for broadcast like any other candidate");
    }

    #[tokio::test]
    async fn foreground_input_defers_due_synthesis_but_keeps_it_queued() {
        let mut config = LoopConfig::default();
        config.synthesis_config = crate::steps::synthesize::SynthesisConfig {
            min_new_episodic: 3,
            max_cluster_size: 2,
            min_interval_ms: 0,
            dedup_similarity_threshold: 0.93,
            promotion_gate_enabled: true,
        };

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        for i in 0..4 {
            let mut episode = MentalObject::new_observation(format!("episode {i}"), EpochMillis(i), 0.5);
            episode.memory_roles.push(MemoryRole::Episodic);
            episode.embedding = Some(vec![i as f32, 1.0]);
            let id = episode.id;
            actor.memory.graph.insert(episode);
            actor.memory.episodic_since_last_synthesis.push(id);
        }

        handles.input_tx.send("foreground turn".to_string()).await.unwrap();
        actor.tick().await;

        let mut saw_synthesis = false;
        let mut saw_deferred_telemetry = false;
        while let Ok(event) = handles.events_rx.try_recv() {
            saw_synthesis |= event.phase == aca_store::CyclePhase::Synthesize;
            saw_deferred_telemetry |= event.phase == aca_store::CyclePhase::Telemetry
                && event.payload.pointer("/deferred/synthesis").and_then(|v| v.as_bool()) == Some(true);
        }
        assert!(!saw_synthesis, "foreground input should keep due synthesis off the user-facing tick");
        assert!(saw_deferred_telemetry, "Telemetry should make the deferral visible");
        assert_eq!(actor.memory.episodic_since_last_synthesis.len(), 4, "deferred synthesis must remain queued");

        actor.tick().await;
        let saw_synthesis_after_idle = std::iter::from_fn(|| handles.events_rx.try_recv().ok())
            .any(|event| event.phase == aca_store::CyclePhase::Synthesize);
        assert!(saw_synthesis_after_idle, "the queued synthesis should still run once foreground pressure clears");
    }

    /// Builds `A(in Working Memory) -> M -> D`, none of them sharing a
    /// direct edge with `A` except the first hop, and gives `D` a stale
    /// single reference (own `base_level` alone stays below
    /// `attention_threshold`) so recall, not raw base-level activation, is
    /// the only thing that can pull it back into competition. Returns
    /// `(a_id, d_id)`.
    fn insert_two_hop_chain(actor: &mut CognitiveLoopActor) -> (MentalObjectId, MentalObjectId) {
        use aca_types::{AssociativeEdge, EdgeKind};

        let mut dormant = MentalObject::new_observation("a long-forgotten detail", EpochMillis(0), 0.5);
        let dormant_id = dormant.id;
        // Freeze the reference log to a single, stale entry so nothing but
        // spreading activation can lift this object's total above threshold.
        dormant.activation.reference_log = aca_util::RingBuffer::new(64);
        dormant.activation.reference_log.push(EpochMillis(0));
        actor.memory.graph.insert(dormant);

        let mut intermediate = MentalObject::new_observation("a bridging thought", EpochMillis(0), 0.5);
        intermediate.edges.push(AssociativeEdge {
            target_id: dormant_id,
            kind: EdgeKind::Associative,
            strength: 1.0,
            last_coactivated_at: EpochMillis(0),
        });
        let intermediate_id = intermediate.id;
        actor.memory.graph.insert(intermediate);

        let mut source = MentalObject::new_observation("the current topic", EpochMillis(0), 0.5);
        source.edges.push(AssociativeEdge {
            target_id: intermediate_id,
            kind: EdgeKind::Associative,
            strength: 1.0,
            last_coactivated_at: EpochMillis(0),
        });
        let source_id = source.id;
        actor.memory.graph.insert(source);
        actor.memory.working_memory.insert(source_id);

        (source_id, dormant_id)
    }

    #[tokio::test]
    async fn dormant_memory_re_enters_working_memory_via_multi_hop_recall() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.recall_config = crate::steps::recall::RecallConfig { max_hops: 2 };
        let (mut actor, _handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        let (_source_id, dormant_id) = insert_two_hop_chain(&mut actor);

        // 100 seconds since the dormant object's only reference: with
        // decay_d=0.5, base_level alone is ~-2.3 - below the default
        // attention_threshold of -2.0 - so it cannot win admission on
        // base-level activation alone.
        clock.advance(100_000);
        actor.tick().await;

        assert!(
            actor.memory.working_memory.contains(&dormant_id),
            "a dormant object two associative hops from Working Memory should be recalled and win broadcast, even though its own base-level activation alone cannot clear threshold"
        );
    }

    #[tokio::test]
    async fn recall_max_hops_one_does_not_reach_a_two_hop_dormant_memory() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.recall_config = crate::steps::recall::RecallConfig { max_hops: 1 };
        let (mut actor, _handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        let (_source_id, dormant_id) = insert_two_hop_chain(&mut actor);

        clock.advance(100_000);
        actor.tick().await;

        assert!(
            !actor.memory.working_memory.contains(&dormant_id),
            "max_hops=1 must not be able to reach a two-hop-away dormant object - isolates hop count as what makes recall work, not some coincidental other path"
        );
    }

    #[tokio::test]
    async fn recall_ablation_suppresses_dormant_memory_reactivation() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let clock = Arc::new(ManualClock::new(EpochMillis(0)));
        let mut config = LoopConfig::default();
        config.recall_config = crate::steps::recall::RecallConfig { max_hops: 2 };
        config.ablation_config.disable_recall = true;
        let (mut actor, _handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            Vec::new(),
        );

        let (_source_id, dormant_id) = insert_two_hop_chain(&mut actor);

        clock.advance(100_000);
        actor.tick().await;

        assert!(
            !actor.memory.working_memory.contains(&dormant_id),
            "disable_recall should fully suppress dormant-memory reactivation, for A/B benchmarking against the pre-Recall baseline"
        );
    }

    #[tokio::test]
    async fn a_drive_pressure_spawns_an_intention_that_surfaces_gets_planned_and_is_revised() {
        // End-to-end proof that commitment actually competes for attention,
        // not just bookkeeping in isolation: sustained curiosity pressure
        // spawns a real `Intention` object (`steps::agenda::revise_agenda`),
        // which a *later* tick's `surface_active_intentions` must explicitly
        // re-nominate into `pending_admission` for it to ever be reconsidered
        // by Coalition/Broadcast at all (see `steps::agenda`'s own doc
        // comment on why `record_reference` alone is not sufficient), which
        // then actually wins Broadcast, gets a real `Operator::Plan`
        // proposal from the Executive, produces a plan-tagged Reflection via
        // `steps::act::act`, and has that Reflection's content folded back
        // into the parent intention's own `data`/commitment - not left as a
        // disconnected side-chain.
        let mut config = LoopConfig::default();
        config.agenda_config = crate::steps::agenda::AgendaConfig {
            top_k_surfaced: 1,
            spawn_threshold: 0.05,
            replan_interval_ms: 0,
            min_spawn_interval_ms: 0,
            ..crate::steps::agenda::AgendaConfig::default()
        };

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        // Four unresolved, unconsulted low-confidence Reflections saturate
        // `steps::drives::curiosity_pressure` at 1.0 - enough for even one
        // EMA-smoothed tick to clear `spawn_threshold: 0.05`
        // (`0.1 * 1.0 = 0.1 >= 0.05`). Inserted directly into the graph
        // (never into `working_memory`/`pending_admission`), so they never
        // themselves compete for Coalition - they exist purely as the
        // unresolved-discrepancy signal `curiosity_pressure` scans for.
        for i in 0..4 {
            let mut reflection = aca_types::MentalObject::new_observation(format!("an unresolved thought {i}"), aca_util::EpochMillis(0), 0.5);
            reflection.kind = aca_types::MentalObjectKind::Reflection;
            reflection.confidence = 0.1;
            actor.memory.graph.insert(reflection);
        }

        let mut spawn_events = 0u32;
        let mut plan_operator_events = 0u32;
        for _ in 0..2 {
            actor.tick().await;
            while let Ok(event) = handles.events_rx.try_recv() {
                if event.phase == aca_store::CyclePhase::Agenda && event.payload.get("spawned").is_some() {
                    spawn_events += 1;
                }
                if event.phase == aca_store::CyclePhase::Executive {
                    // `Debug`-formatted (see `loop_actor`'s own Executive
                    // `emit_event` call: `format!("{:?}", proposal.operator)`),
                    // so "Plan" (capitalized), not `Operator::operator_keyword`'s
                    // lowercase prompt-facing spelling.
                    if event.payload.get("operator").and_then(|v| v.as_str()) == Some("Plan") {
                        plan_operator_events += 1;
                    }
                }
            }
        }

        assert_eq!(spawn_events, 1, "sustained curiosity pressure should spawn exactly one intention");
        assert!(plan_operator_events >= 1, "the surfaced intention should have won Broadcast and been proposed Operator::Plan");

        let intention = actor
            .memory.graph
            .iter()
            .find(|object| object.kind == aca_types::MentalObjectKind::Intention)
            .expect("the spawned intention should still be present in the graph");
        let intention_id = intention.id;
        let commitment_after = intention.goal.as_ref().expect("a spawned intention always carries GoalStackMembership").priority;
        assert!(commitment_after > 0.0, "a genuinely spawned intention should carry real, non-zero commitment");

        let next_action = intention.data.get("next_action").and_then(|v| v.as_str()).unwrap_or_default();
        assert_eq!(next_action, "a reply", "the plan-derived Reflection's text should have been folded back into the intention's own next_action, not left disconnected");

        let consumed_reflection = actor
            .memory.graph
            .iter()
            .find(|object| object.kind == aca_types::MentalObjectKind::Reflection && object.data.get("for_intention").and_then(|v| v.as_str()) == Some(&intention_id.to_string()));
        let consumed_reflection = consumed_reflection.expect("a plan-tagged reflection should have been created for the intention");
        assert_eq!(consumed_reflection.produced_by_operator.as_deref(), Some("Plan"), "the plan reflection should be marked consumed once folded back, so it's never folded twice");
    }

    #[tokio::test]
    async fn pattern_synthesis_does_not_fire_before_min_new_episodic_is_reached() {
        let mut config = LoopConfig::default();
        config.synthesis_config = crate::steps::synthesize::SynthesisConfig {
            min_new_episodic: 5,
            max_cluster_size: 8,
            min_interval_ms: 0,
            dedup_similarity_threshold: 0.93,
            promotion_gate_enabled: true,
        };

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let client = sequence_client(vec!["a reflection about topic one", "a reflection about topic two"]);
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        let mut synthesize_events = 0u32;
        for i in 0..80 {
            if i == 0 {
                handles.input_tx.send("topic one".to_string()).await.unwrap();
            } else if i == 25 {
                handles.input_tx.send("topic two".to_string()).await.unwrap();
            }
            actor.tick().await;
            while let Ok(event) = handles.events_rx.try_recv() {
                if event.phase == aca_store::CyclePhase::Synthesize {
                    synthesize_events += 1;
                }
            }
        }

        assert_eq!(synthesize_events, 0, "only 2 of the required 5 new episodic memories formed - synthesis should not fire");
    }

    #[tokio::test]
    async fn pattern_synthesis_respects_min_interval_ms_as_a_cost_control_valve() {
        let mut config = LoopConfig::default();
        config.synthesis_config = crate::steps::synthesize::SynthesisConfig {
            min_new_episodic: 2,
            max_cluster_size: 8,
            // Large enough that the real wall-clock time this test takes to
            // run can never cross it - proves a second synthesis-eligible
            // window forming shortly after the first does not retrigger.
            min_interval_ms: 10_000_000_000,
            dedup_similarity_threshold: 0.93,
            promotion_gate_enabled: true,
        };

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let client = sequence_client(vec![
            "a reflection about topic one",
            "a reflection about topic two",
            "a reflection about topic three",
            "a reflection about topic four",
            "a pattern across the first window",
        ]);
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        let mut synthesize_events = 0u32;
        for i in 0..150 {
            if i == 0 {
                handles.input_tx.send("topic one".to_string()).await.unwrap();
            } else if i == 25 {
                handles.input_tx.send("topic two".to_string()).await.unwrap();
            } else if i == 50 {
                handles.input_tx.send("topic three".to_string()).await.unwrap();
            } else if i == 75 {
                handles.input_tx.send("topic four".to_string()).await.unwrap();
            }
            actor.tick().await;
            while let Ok(event) = handles.events_rx.try_recv() {
                if event.phase == aca_store::CyclePhase::Synthesize {
                    synthesize_events += 1;
                }
            }
        }

        assert_eq!(synthesize_events, 1, "a second synthesis-eligible window forming shortly after the first must not retrigger before min_interval_ms elapses");
    }

    #[tokio::test]
    async fn a_dormant_self_memory_object_outlasts_an_ordinary_one_in_working_memory() {
        // specs.md's Memory Competition: "relevance to self -> elevated
        // baseline activation for Self Memory-linked nodes... rarely lose
        // the competition for relevance even when dormant." Both objects
        // start with identical activation (one reference, same instant);
        // only `MemoryRole::SelfMemory` distinguishes them. `Belief` kind
        // (not `Observation`) sidesteps `propose_operators`' "superseded by
        // its own Reflection" path, which is a different mechanism this
        // test isn't about.
        let clock = Arc::new(aca_util::ManualClock::new(aca_util::EpochMillis(0)));

        let mut self_memory_object = MentalObject::new_observation("I am Omega", aca_util::EpochMillis(0), 0.5);
        self_memory_object.kind = aca_types::MentalObjectKind::Belief;
        self_memory_object.memory_roles = vec![MemoryRole::SelfMemory];
        let mut ordinary_object = MentalObject::new_observation("just a regular note", aca_util::EpochMillis(0), 0.5);
        ordinary_object.kind = aca_types::MentalObjectKind::Belief;
        let (self_memory_id, ordinary_id) = (self_memory_object.id, ordinary_object.id);

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            clock.clone(),
            vec![self_memory_object, ordinary_object],
        );
        actor.memory.pending_admission.insert(self_memory_id);
        actor.memory.pending_admission.insert(ordinary_id);

        actor.tick().await;
        let ids_at_zero: Vec<_> = handles.snapshot_rx.borrow().working_memory.iter().map(|m| m.id).collect();
        assert!(ids_at_zero.contains(&self_memory_id), "both objects should win their first admission shot");
        assert!(ids_at_zero.contains(&ordinary_id), "both objects should win their first admission shot");

        // Five simulated days of dormancy, no further input at all - long
        // enough for an un-boosted object's ACT-R activation to decay well
        // below the default attention threshold.
        clock.advance(5 * 24 * 60 * 60 * 1000);
        actor.tick().await;

        let ids_after: Vec<_> = handles.snapshot_rx.borrow().working_memory.iter().map(|m| m.id).collect();
        assert!(ids_after.contains(&self_memory_id), "a dormant Self Memory object should still be admitted after 5 days");
        assert!(!ids_after.contains(&ordinary_id), "an ordinary dormant object should have decayed out of Working Memory by then");
    }

    #[tokio::test]
    async fn a_low_confidence_tier3_resolution_escalates_once_more_to_tier4() {
        // specs.md's Model Tiering: Tier 4 is "reached only when a Tier 3
        // impasse fails to resolve with adequate confidence" - a low-
        // confidence Tier 3 answer here should trigger exactly one further
        // escalation, and the *final* live event should reflect Tier 4's
        // (more confident) answer, not Tier 3's.
        let mut config = LoopConfig::default();
        config.executive_config.preference_tie_epsilon = 1.0;
        // Phase 3's automatic memory formation (`steps::memory_formation::
        // maybe_automatic_remember`) would otherwise remove `Operator::
        // Remember` from propose_operators' output before this test's
        // forced tie ever gets a chance to form - a fresh/surprising
        // observation is exactly the case it's designed to catch. This test
        // is about impasse/escalation wiring, not automatic memory
        // formation, so it opts out explicitly rather than relying on an
        // incidental default.
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let tier3_client = ConfiguredChatClient { text: "unsure", confidence: 0.2, tier: Tier::T3 };
        let tier4_client = ConfiguredChatClient { text: "a confident tier 4 answer", confidence: 0.95, tier: Tier::T4 };
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(tier3_client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(tier4_client),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("ambiguous input".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut escalation_tiers: Vec<Tier> = Vec::new();
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.event_type == aca_store::CycleEventKind::Escalation
                && let Some(tier) = event.tier_used
            {
                escalation_tiers.push(tier);
            }
        }

        assert_eq!(escalation_tiers, vec![Tier::T4], "a low-confidence Tier 3 answer should escalate to Tier 4, and only the final (Tier 4) resolution should be reported live");
    }

    #[tokio::test]
    async fn tier4_being_unavailable_falls_back_to_the_tier3_answer_already_in_hand() {
        // A busy/failing Tier 4 must not discard the real Tier 3 answer
        // already obtained - falling back to it is strictly better than
        // treating the whole impasse as unresolved.
        let mut config = LoopConfig::default();
        config.executive_config.preference_tie_epsilon = 1.0;
        // Phase 3's automatic memory formation (`steps::memory_formation::
        // maybe_automatic_remember`) would otherwise remove `Operator::
        // Remember` from propose_operators' output before this test's
        // forced tie ever gets a chance to form - a fresh/surprising
        // observation is exactly the case it's designed to catch. This test
        // is about impasse/escalation wiring, not automatic memory
        // formation, so it opts out explicitly rather than relying on an
        // incidental default.
        config.automatic_memory_formation_surprise_threshold = f32::INFINITY;

        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let tier3_client = ConfiguredChatClient { text: "a real but low-confidence answer", confidence: 0.2, tier: Tier::T3 };
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            config,
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(tier3_client),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(AlwaysFailsChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("ambiguous input".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut escalation_tiers: Vec<Tier> = Vec::new();
        while let Ok(event) = handles.events_rx.try_recv() {
            if event.event_type == aca_store::CycleEventKind::Escalation
                && let Some(tier) = event.tier_used
            {
                escalation_tiers.push(tier);
            }
        }

        assert_eq!(escalation_tiers, vec![Tier::T3], "a failing Tier 4 should fall back to reporting the Tier 3 answer, not silently drop it");
    }

    #[tokio::test]
    async fn every_pipeline_phase_can_produce_a_live_event() {
        // Regression test: Predict/Coalition/Broadcast/Learn previously
        // never emitted anything, so their pipeline stations in the
        // visualization could never pulse no matter how long it ran.
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, mut handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        // Two distinct inputs so Working Memory ends up with 2+ members,
        // which is what makes Learn's co-activation reinforcement fire.
        handles.input_tx.send("are you there?".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        handles.input_tx.send("the sky is blue".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;

        let mut seen_phases: std::collections::HashSet<CyclePhase> = std::collections::HashSet::new();
        while let Ok(event) = handles.events_rx.try_recv() {
            seen_phases.insert(event.phase);
        }

        for phase in [
            CyclePhase::Predict,
            CyclePhase::Compare,
            CyclePhase::Coalition,
            CyclePhase::Broadcast,
            CyclePhase::Executive,
            CyclePhase::Act,
            CyclePhase::Learn,
        ] {
            assert!(seen_phases.contains(&phase), "expected at least one {phase:?} event, got {seen_phases:?}");
        }
    }

    #[tokio::test]
    async fn associative_edges_appear_once_two_objects_co_occur_in_working_memory() {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        let (mut actor, handles) = CognitiveLoopActor::new(
            LoopConfig::default(),
            Arc::new(FakeEmbeddingClient::default()),
            Arc::new(FixedChatClient),
            empty_tier1(),
            empty_tier2(),
            TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
            TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
            Arc::new(FixedChatClient),
            store.clone(),
            store,
            ToolRegistry::empty(),
            Arc::new(SystemClock),
            Vec::new(),
        );

        handles.input_tx.send("are you there?".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        let snapshot_after_first = handles.snapshot_rx.borrow().clone();
        assert!(
            snapshot_after_first.associative_edges.is_empty(),
            "a single Working Memory member has no co-occurring partner to reinforce an edge with"
        );

        handles.input_tx.send("the sky is blue".to_string()).await.unwrap();
        tick_through_embedding_resolution(&mut actor).await;
        let snapshot_after_second = handles.snapshot_rx.borrow().clone();
        assert!(
            !snapshot_after_second.associative_edges.is_empty(),
            "two co-occurring Working Memory members should have a reinforced associative edge between them"
        );
        for edge in &snapshot_after_second.associative_edges {
            assert!(snapshot_after_second.working_memory.iter().any(|m| m.id == edge.source_id));
            assert!(snapshot_after_second.working_memory.iter().any(|m| m.id == edge.target_id));
        }
    }

    #[test]
    fn classify_provenance_tags_semantic_and_self_memory_by_promotion_status() {
        let mut durable = MentalObject::new_observation("a confirmed belief", EpochMillis(0), 0.5);
        durable.memory_roles.push(MemoryRole::SelfMemory);
        durable.promotion = aca_types::PromotionState::confirmed(EpochMillis(0));
        assert_eq!(classify_provenance(&durable), crate::prompt_templates::ProvenanceTier::Durable);

        let mut staged = MentalObject::new_observation("an unconfirmed pattern", EpochMillis(0), 0.5);
        staged.memory_roles.push(MemoryRole::Semantic);
        staged.promotion = aca_types::PromotionState::candidate(EpochMillis(0));
        assert_eq!(classify_provenance(&staged), crate::prompt_templates::ProvenanceTier::StagedCandidate);

        let narrative = MentalObject::new_observation("ordinary conversation", EpochMillis(0), 0.5);
        assert_eq!(classify_provenance(&narrative), crate::prompt_templates::ProvenanceTier::Narrative, "an object with no Semantic/SelfMemory role carries no special trust question");
    }
