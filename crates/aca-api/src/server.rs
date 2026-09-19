use std::sync::Arc;

use aca_engine::{CycleEvent, EngineSnapshot, LiveModelHandles};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc, watch};

/// Everything the local API needs, handed to it by whoever wires up the
/// `CognitiveLoopActor` (`omega-acad`). Kept deliberately thin: the API is
/// a pure consumer of the actor's channels, never a second path into the
/// graph.
#[derive(Clone)]
pub struct ApiState {
    pub input_tx: mpsc::Sender<String>,
    /// Held only to hand out fresh `.resubscribe()`d receivers per
    /// WebSocket connection - the API itself never reads from this
    /// directly.
    events_rx_template: Arc<broadcast::Receiver<CycleEvent>>,
    pub snapshot_rx: watch::Receiver<EngineSnapshot>,
    /// Read directly at request time (never through `snapshot_rx`) so a
    /// poller can actually observe a Tier 1-4 candidate mid-call - see
    /// `LiveModelHandles`'s own doc comment for why `snapshot_rx` alone
    /// can't do this.
    live_models: LiveModelHandles,
}

impl ApiState {
    pub fn new(
        input_tx: mpsc::Sender<String>,
        events_rx: broadcast::Receiver<CycleEvent>,
        snapshot_rx: watch::Receiver<EngineSnapshot>,
        live_models: LiveModelHandles,
    ) -> Self {
        Self {
            input_tx,
            events_rx_template: Arc::new(events_rx),
            snapshot_rx,
            live_models,
        }
    }
}

/// Builds the router: `GET /health`, `GET /snapshot/overview` (the whole
/// engine - Working Memory, memory-role counts, goal stack, tier status),
/// `POST /input`, `GET /events` (WebSocket upgrade, live `cycle_events`
/// feed). Godot 4.x's built-in `HTTPRequest`/`WebSocketPeer` nodes can
/// consume every one of these with zero plugins.
///
/// `GET /tests/status` (runs `cargo test --workspace` on demand) only
/// exists behind the `dev-tools` feature - it shells out to `cargo` and
/// needs the source tree present, so it has no business in a binary built
/// for distribution without the source (see `test_status`'s module doc
/// comment for the rest of the reasoning).
pub fn build_router(state: ApiState) -> Router {
    let router = Router::new()
        .route("/health", get(health))
        .route("/snapshot/overview", get(get_overview_snapshot))
        .route("/input", post(post_input))
        .route("/events", get(ws_events));

    #[cfg(feature = "dev-tools")]
    let router = router.route("/tests/status", get(crate::test_status::get_test_status));

    // Axum's own default (2MB, applied automatically to any body-reading
    // extractor - `Json` included - unless overridden) rejects a real,
    // unremarkable use of `/input`: pasting a large block of text (a
    // document, a log excerpt) from the viz app's text box - confirmed
    // live, a 3MB paste got a bare 413 with the Godot client surfacing
    // nothing at all (see `viz/scripts/api_client.gd`'s own fix for that
    // half). This API is loopback-only (`main.rs` binds `127.0.0.1`, unlike
    // the LAN-reachable MCP server), so there's no untrusted-network actor
    // this limit is defending against - raised generously rather than
    // disabled outright, so a genuinely runaway body still can't wedge the
    // process indefinitely.
    let router = router.layer(DefaultBodyLimit::max(32 * 1024 * 1024));

    router.with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn get_overview_snapshot(State(state): State<ApiState>) -> impl IntoResponse {
    // `working_memory`/`goal_stack`/etc. only need tick-boundary freshness,
    // so they're still taken from the last-published snapshot - but
    // `active_models`/`tier_status`/`embedding_in_flight`/`attention_in_flight` are overwritten
    // from `live_models` instead of trusting that same stale-by-cadence
    // copy, otherwise a Tier 1-4 candidate's in-flight window is invisible
    // to every poller (see `LiveModelHandles`'s doc comment for why).
    let mut snapshot = state.snapshot_rx.borrow().clone();
    snapshot.active_models = state.live_models.active_models();
    snapshot.tier_status = state.live_models.tier_status();
    snapshot.embedding_in_flight = state.live_models.embedding_in_flight();
    snapshot.attention_in_flight = state.live_models.attention_in_flight();
    Json(snapshot)
}

#[derive(Debug, Deserialize)]
struct InputRequest {
    text: String,
}

async fn post_input(State(state): State<ApiState>, Json(payload): Json<InputRequest>) -> impl IntoResponse {
    match state.input_tx.send(payload.text).await {
        Ok(()) => (StatusCode::ACCEPTED, Json(serde_json::json!({ "accepted": true }))),
        Err(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "accepted": false, "reason": "cognitive loop is not accepting input" })),
        ),
    }
}

async fn ws_events(ws: WebSocketUpgrade, State(state): State<ApiState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_events_socket(socket, state))
}

async fn handle_events_socket(mut socket: WebSocket, state: ApiState) {
    let mut receiver = state.events_rx_template.resubscribe();
    loop {
        match receiver.recv().await {
            Ok(event) => {
                let Ok(json) = serde_json::to_string(&event) else { continue };
                if socket.send(Message::Text(json)).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(dropped)) => {
                let notice = serde_json::json!({ "lagged": dropped });
                if socket.send(Message::Text(notice.to_string())).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_store::{CycleEventKind, CyclePhase};
    use aca_util::EpochMillis;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> (ApiState, mpsc::Receiver<String>, broadcast::Sender<CycleEvent>) {
        let (input_tx, input_rx) = mpsc::channel(8);
        let (events_tx, events_rx) = broadcast::channel(8);
        let (_snapshot_tx, snapshot_rx) = watch::channel(EngineSnapshot::default());
        (ApiState::new(input_tx, events_rx, snapshot_rx, LiveModelHandles::empty()), input_rx, events_tx)
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let (state, _input_rx, _events_tx) = test_state();
        let app = build_router(state);
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn snapshot_returns_the_current_watch_value() {
        let (state, _input_rx, _events_tx) = test_state();
        let app = build_router(state);
        let response = app
            .oneshot(Request::builder().uri("/snapshot/overview").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn input_is_forwarded_to_the_actor_channel() {
        let (state, mut input_rx, _events_tx) = test_state();
        let app = build_router(state);
        let body = serde_json::json!({ "text": "hello" }).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/input")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let received = input_rx.recv().await.unwrap();
        assert_eq!(received, "hello");
    }

    #[tokio::test]
    async fn input_accepts_a_paste_well_over_axum_default_body_limit() {
        // Regression guard: confirmed live, axum's own default (2MB,
        // applied automatically unless overridden) rejected a real,
        // unremarkable paste (a large document) with a bare 413 and no
        // feedback surfaced anywhere in the viz app. 3MB comfortably clears
        // the old default while staying under `build_router`'s new limit.
        let (state, mut input_rx, _events_tx) = test_state();
        let app = build_router(state);
        let large_text = "a".repeat(3 * 1024 * 1024);
        let body = serde_json::json!({ "text": large_text }).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/input")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED, "a large paste must not be rejected by the default body limit");
        let received = input_rx.recv().await.unwrap();
        assert_eq!(received.len(), 3 * 1024 * 1024);
    }

    #[tokio::test]
    async fn input_reports_unavailable_when_the_actor_is_gone() {
        let (input_tx, input_rx) = mpsc::channel::<String>(8);
        drop(input_rx); // simulate the actor having shut down
        let (_events_tx, events_rx) = broadcast::channel(8);
        let (_snapshot_tx, snapshot_rx) = watch::channel(EngineSnapshot::default());
        let state = ApiState::new(input_tx, events_rx, snapshot_rx, LiveModelHandles::empty());
        let app = build_router(state);

        let body = serde_json::json!({ "text": "hello" }).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/input")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn a_broadcast_event_is_delivered_over_a_real_websocket_connection() {
        let (state, _input_rx, events_tx) = test_state();
        let app = build_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let (mut ws_stream, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/events"))
            .await
            .expect("should connect to the events websocket");

        let event = CycleEvent::new(1, EpochMillis(1_000), CyclePhase::Act, CycleEventKind::Normal, None, serde_json::json!({"operator": "speak"}));
        events_tx.send(event.clone()).unwrap();

        use futures_util::StreamExt;
        let received = tokio::time::timeout(std::time::Duration::from_secs(2), ws_stream.next())
            .await
            .expect("should receive a message before timing out")
            .expect("stream should not end")
            .expect("message should not be an error");

        let text = received.into_text().expect("expected a text frame");
        let parsed: CycleEvent = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed.cycle_seq, 1);
    }
}
