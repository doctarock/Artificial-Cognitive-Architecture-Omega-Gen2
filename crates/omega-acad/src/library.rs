//! Two independent readers over the same PDF library share, kept separate
//! because they serve different purposes and want very different triggers:
//!
//! - [`run_kl_ingest_loop`] walks every file once, embeds and stores each
//!   chunk in the Knowledge Library, and is only rate-limited enough to be
//!   polite to the embedding server - the goal is a searchable corpus in
//!   hours, not years. It never touches `SensorInput`; `consult` (see
//!   `aca-engine::steps::knowledge_library`) is the only thing that reads
//!   this content back. Still fully automatic - background indexing is
//!   harmless and self-terminates once the corpus is covered.
//! - [`DripState`]/[`deliver_next_chunk`] is Omega's "currently reading"
//!   cursor: one chunk forwarded as a `SensorInput` per delivery, giving
//!   Predict/Observe/Compare and Memory Formation something to chew on
//!   between real room-feed events. `SensorInput` is the lowest-priority
//!   input source in `tick()` (see `loop_actor.rs`), so this never displaces
//!   a real conversational turn or sensor observation, only fills the gaps
//!   between them.
//!
//!   This used to run on its own timer (a chunk every `drip_interval`,
//!   unattended); in practice that meant one slow book (a several-hundred-
//!   chunk cookbook, once) could sit in front of Omega for days with nobody
//!   having chosen it. Delivery is now a deliberate, human-triggered action
//!   instead - see `build_deliver_router`, wired to the viz's "Deliver a
//!   Book" button - one press advances the same persisted cursor by exactly
//!   one chunk.
//!
//! Both walk the same file list independently, at their own pace, and
//! persist their own progress sidecar so a restart resumes each exactly
//! where it left off rather than re-reading from the start. A file that
//! fails to extract (corrupt PDF, scanned-image-only, etc.) is marked
//! completed and skipped rather than retried forever - this is best-effort
//! content, not something that should be able to wedge either loop.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aca_engine::{SensorInput, SourceChannel};
use aca_store::KnowledgeLibraryStore;
use aca_tiers::EmbeddingClient;
use aca_util::{chunk_text, Clock};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex};

pub struct LibraryConfig {
    pub root: PathBuf,
    pub chunk_chars: usize,
    pub kl_interval: Duration,
    pub kl_state_path: PathBuf,
    pub drip_state_path: PathBuf,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Progress {
    completed: HashSet<String>,
    current_file: Option<String>,
    next_chunk_index: usize,
}

fn load_progress(path: &Path) -> Progress {
    std::fs::read_to_string(path).ok().and_then(|contents| serde_json::from_str(&contents).ok()).unwrap_or_default()
}

fn save_progress(path: &Path, progress: &Progress) {
    match serde_json::to_string_pretty(progress) {
        Ok(json) => {
            if let Err(err) = std::fs::write(path, json) {
                tracing::warn!(error = %err, path = %path.display(), "library reader: failed to persist progress");
            }
        }
        Err(err) => tracing::warn!(error = %err, "library reader: failed to serialize progress"),
    }
}

fn list_pdf_files(root: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(root) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("pdf")))
            .collect(),
        Err(err) => {
            tracing::warn!(error = %err, root = %root.display(), "library reader: failed to read directory");
            Vec::new()
        }
    };
    files.sort();
    files
}

/// Distinguishes "this file is bad, never retry it" from "extraction didn't
/// get to run at all because the runtime is tearing down" - conflating the
/// two would let a shutdown that happens to land mid-extraction (a restart,
/// Ctrl-C, a `JoinHandle` cancelled by runtime teardown) permanently
/// blacklist whatever file was in flight, via `Progress::completed`, without
/// it ever actually having been read.
enum ExtractOutcome {
    Chunks(Vec<String>),
    Unreadable,
    ShuttingDown,
}

/// Extracts and chunks one PDF off the blocking-task pool - `pdf_extract` is
/// synchronous and CPU-bound, and has been known to choke on malformed
/// input, so a panic here becomes a `JoinError` (caught below) rather than
/// taking down the caller's loop, let alone the daemon.
async fn extract_and_chunk(path: PathBuf, chunk_chars: usize) -> ExtractOutcome {
    let display_path = path.display().to_string();
    let result = tokio::task::spawn_blocking(move || pdf_extract::extract_text(&path)).await;
    match result {
        Ok(Ok(text)) => {
            let chunks = chunk_text(&text, chunk_chars);
            if chunks.is_empty() {
                tracing::warn!(file = %display_path, "library reader: no extractable text - skipping");
                ExtractOutcome::Unreadable
            } else {
                ExtractOutcome::Chunks(chunks)
            }
        }
        Ok(Err(err)) => {
            tracing::warn!(error = %err, file = %display_path, "library reader: failed to extract text - skipping");
            ExtractOutcome::Unreadable
        }
        Err(err) if err.is_cancelled() => {
            tracing::info!(file = %display_path, "library reader: extraction cancelled (runtime shutting down) - will retry on next start");
            ExtractOutcome::ShuttingDown
        }
        Err(err) => {
            tracing::warn!(error = %err, file = %display_path, "library reader: extraction task panicked - skipping");
            ExtractOutcome::Unreadable
        }
    }
}

/// Advances `progress`/`active` to the next unfinished chunk, extracting a
/// new file (and marking unreadable ones completed) as needed. Returns
/// `None` once every file in `files` is in `progress.completed`, or once
/// extraction reports the runtime is shutting down - in the latter case
/// `progress` is left untouched so the next start resumes exactly here
/// rather than skipping whatever was in flight. Shared by both loops below -
/// they only differ in what they do with the chunk once they have it.
async fn next_chunk(files: &[PathBuf], progress: &mut Progress, active: &mut Option<(PathBuf, Vec<String>)>, chunk_chars: usize) -> Option<(PathBuf, String)> {
    loop {
        if active.is_none() {
            let next_path = progress
                .current_file
                .as_ref()
                .map(PathBuf::from)
                .filter(|resumed| files.contains(resumed))
                .or_else(|| files.iter().find(|candidate| !progress.completed.contains(&candidate.display().to_string())).cloned());

            let path = next_path?;

            let chunks = match extract_and_chunk(path.clone(), chunk_chars).await {
                ExtractOutcome::Chunks(chunks) => chunks,
                ExtractOutcome::Unreadable => {
                    progress.completed.insert(path.display().to_string());
                    progress.current_file = None;
                    progress.next_chunk_index = 0;
                    continue;
                }
                ExtractOutcome::ShuttingDown => return None,
            };

            let path_key = path.display().to_string();
            if progress.current_file.as_deref() != Some(path_key.as_str()) {
                progress.current_file = Some(path_key);
                progress.next_chunk_index = 0;
            }
            *active = Some((path, chunks));
        }

        let (path, chunks) = active.as_ref().expect("just populated above");
        let idx = progress.next_chunk_index;

        if idx >= chunks.len() {
            progress.completed.insert(path.display().to_string());
            progress.current_file = None;
            progress.next_chunk_index = 0;
            *active = None;
            continue;
        }

        let path = path.clone();
        let chunk = chunks[idx].clone();
        progress.next_chunk_index += 1;
        return Some((path, chunk));
    }
}

/// Indexes every PDF under `config.root` into the Knowledge Library, once,
/// at `config.kl_interval` pace. Intended to be `tokio::spawn`ed and left
/// alone; returns once the corpus is fully indexed.
pub async fn run_kl_ingest_loop(config: Arc<LibraryConfig>, kl_store: Arc<dyn KnowledgeLibraryStore>, embedding_client: Arc<dyn EmbeddingClient>, clock: Arc<dyn Clock>) {
    let files = list_pdf_files(&config.root);
    if files.is_empty() {
        tracing::warn!(root = %config.root.display(), "library KL ingest: no PDF files found - nothing to do");
        return;
    }
    tracing::info!(root = %config.root.display(), count = files.len(), "library KL ingest: starting");

    let mut progress = load_progress(&config.kl_state_path);
    let mut active: Option<(PathBuf, Vec<String>)> = None;

    loop {
        let Some((path, chunk)) = next_chunk(&files, &mut progress, &mut active, config.chunk_chars).await else {
            tracing::info!("library KL ingest: every file has been indexed - stopping");
            return;
        };
        let source_uri = format!("file://{}", path.display());

        let embedding = match embedding_client.embed(&chunk).await {
            Ok(embedding) => embedding,
            Err(err) => {
                tracing::warn!(error = %err, file = %source_uri, "library KL ingest: failed to embed chunk - will retry");
                progress.next_chunk_index -= 1; // undo next_chunk's advance so the same chunk is retried
                tokio::time::sleep(config.kl_interval).await;
                continue;
            }
        };
        if let Err(err) = kl_store.insert_document(&source_uri, &chunk, embedding, clock.now()).await {
            tracing::warn!(error = %err, file = %source_uri, "library KL ingest: failed to store chunk - will retry");
            progress.next_chunk_index -= 1;
            tokio::time::sleep(config.kl_interval).await;
            continue;
        }

        save_progress(&config.kl_state_path, &progress);
        tokio::time::sleep(config.kl_interval).await;
    }
}

struct DripInner {
    progress: Progress,
    active: Option<(PathBuf, Vec<String>)>,
}

/// The drip cursor's persisted state, held for the life of the daemon and
/// advanced one chunk at a time by [`deliver_next_chunk`] - never on a
/// timer. `files` is computed once at startup like the KL loop's own list;
/// a `Mutex` (not `RwLock`) because every access here either reads-then-
/// writes `progress`/`active` together or does neither, so there's no
/// read-only path worth splitting out.
pub struct DripState {
    config: Arc<LibraryConfig>,
    files: Vec<PathBuf>,
    inner: Mutex<DripInner>,
}

impl DripState {
    pub fn new(config: Arc<LibraryConfig>) -> Self {
        let files = list_pdf_files(&config.root);
        let progress = load_progress(&config.drip_state_path);
        Self { config, files, inner: Mutex::new(DripInner { progress, active: None }) }
    }
}

pub enum DeliverOutcome {
    Delivered { file: String },
    Exhausted,
    Disconnected,
}

/// Advances the drip cursor by exactly one chunk, forwarding it as a
/// `SensorInput` - the manual, on-demand replacement for the old timer
/// loop's per-iteration body. Persists progress immediately on success so a
/// restart never redelivers (or skips) a chunk a press already resolved.
pub async fn deliver_next_chunk(state: &DripState, sensor_tx: &mpsc::Sender<SensorInput>) -> DeliverOutcome {
    let mut inner = state.inner.lock().await;
    let DripInner { progress, active } = &mut *inner;

    let Some((path, chunk)) = next_chunk(&state.files, progress, active, state.config.chunk_chars).await else {
        tracing::info!("library drip: every file has been read");
        return DeliverOutcome::Exhausted;
    };
    let file = path.display().to_string();

    if sensor_tx.send(SensorInput { text: chunk, channel: SourceChannel::ExternalKnowledge, source: "library", entity_label: None }).await.is_err() {
        tracing::warn!("library drip: cognitive loop is no longer accepting input");
        return DeliverOutcome::Disconnected;
    }

    save_progress(&state.config.drip_state_path, progress);
    tracing::info!(file = %file, "library drip: delivered one chunk");
    DeliverOutcome::Delivered { file }
}

#[derive(Clone)]
struct DeliverState {
    drip: Arc<DripState>,
    sensor_tx: mpsc::Sender<SensorInput>,
}

#[derive(Serialize)]
struct DeliverResponse {
    status: &'static str,
    file: Option<String>,
}

async fn handle_deliver(State(state): State<DeliverState>) -> impl IntoResponse {
    match deliver_next_chunk(&state.drip, &state.sensor_tx).await {
        DeliverOutcome::Delivered { file } => Json(DeliverResponse { status: "delivered", file: Some(file) }).into_response(),
        DeliverOutcome::Exhausted => Json(DeliverResponse { status: "exhausted", file: None }).into_response(),
        DeliverOutcome::Disconnected => (StatusCode::SERVICE_UNAVAILABLE, Json(DeliverResponse { status: "disconnected", file: None })).into_response(),
    }
}

/// `POST /library/deliver` - the whole manual-delivery surface, one route.
/// Built here rather than folded into `aca-api::build_router` since that
/// crate is a generic engine-facing API with no notion of the library
/// feature; `omega-acad::main` merges this into the same router instead.
pub fn build_deliver_router(drip: Arc<DripState>, sensor_tx: mpsc::Sender<SensorInput>) -> Router {
    Router::new().route("/library/deliver", post(handle_deliver)).with_state(DeliverState { drip, sensor_tx })
}
