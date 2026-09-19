use std::sync::Arc;
use std::time::{Duration, Instant};

use aca_api::{build_router, ApiState};
use aca_engine::{CognitiveLoopActor, CycleEvent, CyclePhase, LoopConfig, ToolRegistry};
use aca_store::SqliteStore;
use aca_tiers::{build_http_client, ChatClient, DivergentPool, EmbeddingClient, GenerateRequest, OllamaClient, TierError, TierPool, TierResponse};
use aca_types::{MentalObject, MentalObjectKind, Tier};
use aca_util::{EpochMillis, ManualClock};
use async_trait::async_trait;
use futures_util::StreamExt;
use tokio_tungstenite::connect_async;

struct ForbiddenModels;

struct FixedEmbedding;

#[async_trait]
impl EmbeddingClient for FixedEmbedding {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, TierError> {
        Ok(vec![1.0, 0.0, 0.0])
    }
}

#[async_trait]
impl ChatClient for ForbiddenModels {
    async fn generate(&self, _request: GenerateRequest) -> Result<TierResponse, TierError> {
        panic!("verified loopback fast path must not call chat inference")
    }
}

#[async_trait]
impl EmbeddingClient for ForbiddenModels {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, TierError> {
        panic!("verified loopback fast path must not call embedding inference")
    }
}

/// Run explicitly with `cargo test -p aca-api loopback_verified_turn_latency_probe
/// -- --ignored --nocapture`. This includes real local HTTP submission,
/// actor scheduling, and WebSocket event delivery, but not a real model,
/// TTS, Godot rendering, or a deployed LAN conversation.
#[tokio::test]
#[ignore = "local loopback performance probe; timings vary with host load"]
async fn loopback_verified_turn_latency_probe() {
    let mut verified = MentalObject::new_observation("verified sequence: hello omega", EpochMillis(0), 0.5);
    verified.kind = MentalObjectKind::Memory;
    verified.embedding = Some(vec![1.0, 0.0]);
    verified.data = serde_json::json!({
        "schema": "omega-verified-operator-sequence/v2",
        "stimulus": "hello omega",
        "steps": ["ContinueReflecting", "Speak"],
        "conditions": {"source_kind": "observation", "stimulus_policy": "routine_non_command", "match": "exact_normalized"},
        "expected_spoken_response": "Hello Derek.",
        "expected_consequence": {"kind": "spoke", "text": "Hello Derek."},
        "verified_successes": 3,
        "credited_observation_ids": [
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333"
        ]
    });
    let mut verified_ask = MentalObject::new_observation("verified sequence: status update", EpochMillis(0), 0.5);
    verified_ask.kind = MentalObjectKind::Memory;
    verified_ask.embedding = Some(vec![0.0, 1.0]);
    verified_ask.data = serde_json::json!({
        "schema": "omega-verified-operator-sequence/v2",
        "stimulus": "status update",
        "steps": ["ContinueReflecting", "Ask"],
        "conditions": {"source_kind": "observation", "stimulus_policy": "routine_non_command", "match": "exact_normalized"},
        "expected_asked_question": "Which project do you mean?",
        "expected_consequence": {"kind": "asked", "text": "Which project do you mean?"},
        "verified_successes": 3,
        "credited_observation_ids": [
            "44444444-4444-4444-8444-444444444444",
            "55555555-5555-4555-8555-555555555555",
            "66666666-6666-4666-8666-666666666666"
        ]
    });
    let clock = Arc::new(ManualClock::new(EpochMillis(100_000)));
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let models = Arc::new(ForbiddenModels);
    let mut config = LoopConfig::default();
    config.ablation_config.disable_agenda = true;
    config.ablation_config.disable_boredom = true;
    config.ablation_config.disable_synthesis = true;
    config.automatic_memory_formation_surprise_threshold = f32::INFINITY;
    let (actor, handles) = CognitiveLoopActor::new(
        config, models.clone(), models.clone(),
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![]),
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![]),
        TierPool::new(Tier::T3, 1, Duration::from_secs(5)),
        TierPool::new(Tier::T4, 1, Duration::from_secs(5)),
        models, store.clone(), store, ToolRegistry::empty(), clock.clone(), vec![verified, verified_ask],
    );
    let state = ApiState::new(handles.input_tx.clone(), handles.events_rx.resubscribe(),
        handles.snapshot_rx.clone(), handles.live_models.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(axum::serve(listener, build_router(state)).into_future());
    let actor_task = tokio::spawn(actor.run());
    let (mut socket, _) = connect_async(format!("ws://127.0.0.1:{port}/events")).await.unwrap();
    let client = reqwest::Client::new();
    let mut speech_us = Vec::with_capacity(50);
    let mut ask_us = Vec::with_capacity(50);
    for turn in 0..110 {
        clock.advance(1_000);
        let is_ask = turn % 2 == 1;
        let input = if is_ask { "status update" } else { "Hello Omega!" };
        let expected_operator = if is_ask { "Ask" } else { "Speak" };
        let started = Instant::now();
        let accepted = client.post(format!("http://127.0.0.1:{port}/input"))
            .json(&serde_json::json!({"text": input}))
            .send().await.unwrap();
        assert_eq!(accepted.status(), reqwest::StatusCode::ACCEPTED);
        loop {
            let message = tokio::time::timeout(Duration::from_secs(3), socket.next())
                .await.expect("loopback speech delivery timed out")
                .expect("event socket closed").expect("event socket error");
            if !message.is_text() { continue; }
            let event: CycleEvent = serde_json::from_str(message.to_text().unwrap()).unwrap();
            if event.phase == CyclePhase::Act
                && event.payload["operator"] == "speak"
                && event.payload["render_path"] == "CompiledProcedure"
                && event.payload["attempted_operator"] == expected_operator
            {
                if turn >= 10 {
                    let elapsed = started.elapsed().as_micros() as u64;
                    if is_ask { ask_us.push(elapsed); } else { speech_us.push(elapsed); }
                }
                break;
            }
        }
    }
    speech_us.sort_unstable();
    ask_us.sort_unstable();
    let speech_under_10ms = speech_us.iter().filter(|&&us| us < 10_000).count();
    let ask_under_10ms = ask_us.iter().filter(|&&us| us < 10_000).count();
    println!("loopback HTTP-to-WebSocket compiled actions: speak_n=50, speak_p50_us={}, speak_p95_us={}, speak_under_10ms={speech_under_10ms}/50, ask_n=50, ask_p50_us={}, ask_p95_us={}, ask_under_10ms={ask_under_10ms}/50",
        speech_us[24], speech_us[47], ask_us[24], ask_us[47]);
    assert_eq!(speech_us.len(), 50);
    assert_eq!(ask_us.len(), 50);
    actor_task.abort();
    server.abort();
}

/// A privacy-safe real-inference probe: prompts go only to a loopback Ollama
/// endpoint, while HTTP ingress, actor scheduling, Tier-3 reflection, and
/// WebSocket delivery are real. It deliberately uses a fixed local embedding
/// so an unavailable embedding model cannot disguise chat-model turnaround.
#[tokio::test]
#[ignore = "requires an explicitly available loopback Ollama model"]
async fn loopback_model_backed_turn_latency_probe() {
    let base_url = std::env::var("OMEGA_PROBE_LOOPBACK_MODEL_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    assert!(base_url.starts_with("http://127.0.0.1:") || base_url.starts_with("http://localhost:"),
        "model-backed probe refuses non-loopback endpoints");
    let model_name = std::env::var("OMEGA_PROBE_LOOPBACK_MODEL")
        .unwrap_or_else(|_| "qwen2.5:1.5b".to_string());
    let chat = Arc::new(OllamaClient::new(build_http_client(), base_url, model_name, Tier::T3));
    let clock = Arc::new(ManualClock::new(EpochMillis(100_000)));
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let mut config = LoopConfig::default();
    config.ablation_config.disable_agenda = true;
    config.ablation_config.disable_boredom = true;
    config.ablation_config.disable_synthesis = true;
    config.automatic_memory_formation_surprise_threshold = f32::INFINITY;
    let (actor, handles) = CognitiveLoopActor::new(
        config, Arc::new(FixedEmbedding), chat.clone(),
        DivergentPool::new(Tier::T1, Duration::from_secs(5), vec![]),
        DivergentPool::new(Tier::T2, Duration::from_secs(5), vec![]),
        TierPool::new(Tier::T3, 1, Duration::from_secs(30)),
        TierPool::new(Tier::T4, 1, Duration::from_secs(30)),
        chat, store.clone(), store, ToolRegistry::empty(), clock.clone(), Vec::new(),
    );
    let state = ApiState::new(handles.input_tx.clone(), handles.events_rx.resubscribe(),
        handles.snapshot_rx.clone(), handles.live_models.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(axum::serve(listener, build_router(state)).into_future());
    let actor_task = tokio::spawn(actor.run());
    let (mut socket, _) = connect_async(format!("ws://127.0.0.1:{port}/events")).await.unwrap();
    let client = reqwest::Client::new();
    let prompts = ["Acknowledge receipt of latency probe alpha."];
    let mut samples_ms = Vec::new();
    let mut terminal_operators = Vec::new();
    for prompt in prompts {
        clock.advance(1_000);
        let started = Instant::now();
        let accepted = client.post(format!("http://127.0.0.1:{port}/input"))
            .json(&serde_json::json!({"text": prompt})).send().await.unwrap();
        assert_eq!(accepted.status(), reqwest::StatusCode::ACCEPTED);
        let deadline = Instant::now() + Duration::from_secs(45);
        let mut recent_events = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "loopback model-backed turn produced no terminal action within 45 seconds; recent={recent_events:?}");
            let message = tokio::time::timeout(remaining, socket.next())
                .await.unwrap_or_else(|_| panic!("loopback model-backed turn timed out; recent={recent_events:?}"))
                .expect("event socket closed").expect("event socket error");
            if !message.is_text() { continue; }
            let event: CycleEvent = serde_json::from_str(message.to_text().unwrap()).unwrap();
            recent_events.push(format!("{:?}:{}", event.phase, event.payload));
            if recent_events.len() > 100 { recent_events.remove(0); }
            let attempted = event.payload["attempted_operator"].as_str();
            if event.phase == CyclePhase::Act
                && attempted.is_some_and(|operator| ["Speak", "Ask", "Ignore"].contains(&operator)) {
                samples_ms.push(started.elapsed().as_millis() as u64);
                terminal_operators.push(attempted.unwrap().to_string());
                break;
            }
            if event.phase == CyclePhase::Act && event.event_type == aca_engine::CycleEventKind::Error {
                panic!("loopback model-backed turn failed before speech: {}", event.payload);
            }
        }
    }
    samples_ms.sort_unstable();
    println!("loopback real-model foreground turn: n=1, elapsed_ms={}, terminal_operator={}", samples_ms[0], terminal_operators[0]);
    assert_eq!(samples_ms.len(), 1);
    actor_task.abort();
    server.abort();
}
