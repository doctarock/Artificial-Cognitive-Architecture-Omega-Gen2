use std::sync::Arc;

use aca_engine::{EngineSnapshot, ExternalAgentInput};
use aca_store::{KnowledgeLibraryStore, ScoredDocument};
use aca_tiers::EmbeddingClient;
use aca_util::{chunk_text, Clock, SystemClock};
use axum::Router;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

/// Everything the MCP surface needs, handed to it by whoever wires up the
/// `CognitiveLoopActor` (`omega-acad`) - structured like `aca-api`'s
/// `ApiState`: a pure consumer of already-constructed handles/stores, never
/// a second path into the graph. `knowledge_library_write`/`_search`/
/// `_ingest_url` only ever touch `kl_store`, never `mental_objects` - the
/// Knowledge Library stays external, per specs.md.
///
/// `knowledge_library_ingest_url` fetches whatever URL it's given, from
/// wherever `omega-acad` runs - the same trust model already established by
/// every other tool here (no auth on this MCP server at all; it's a
/// household-LAN surface, not internet-facing). Worth knowing before
/// exposing this server beyond the LAN it was designed for.
#[derive(Clone)]
pub struct McpState {
    external_agent_input_tx: mpsc::Sender<ExternalAgentInput>,
    snapshot_rx: watch::Receiver<EngineSnapshot>,
    kl_store: Arc<dyn KnowledgeLibraryStore>,
    embedding_client: Arc<dyn EmbeddingClient>,
    http_client: reqwest::Client,
    tool_router: ToolRouter<Self>,
}

impl McpState {
    pub fn new(
        external_agent_input_tx: mpsc::Sender<ExternalAgentInput>,
        snapshot_rx: watch::Receiver<EngineSnapshot>,
        kl_store: Arc<dyn KnowledgeLibraryStore>,
        embedding_client: Arc<dyn EmbeddingClient>,
    ) -> Self {
        Self {
            external_agent_input_tx,
            snapshot_rx,
            kl_store,
            embedding_client,
            http_client: aca_tiers::build_http_client(),
            tool_router: Self::tool_router(),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct KnowledgeLibraryWriteRequest {
    /// Where this document came from - a URI, file path, or any other
    /// caller-meaningful label. Not interpreted or fetched by Omega.
    source_uri: String,
    text: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct KnowledgeLibraryWriteResponse {
    id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct KnowledgeLibrarySearchRequest {
    query: String,
    /// Defaults to 5 when omitted.
    top_k: Option<usize>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct KnowledgeLibraryMatch {
    id: String,
    source_uri: String,
    text: String,
    score: f32,
}

#[derive(Debug, Serialize, JsonSchema)]
struct KnowledgeLibrarySearchResponse {
    matches: Vec<KnowledgeLibraryMatch>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SendInputRequest {
    text: String,
    /// Identifies the calling agent for provenance - tagged onto the
    /// resulting Observation's `data` field, not treated as a privilege.
    agent_id: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SendInputResponse {
    accepted: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct KnowledgeLibraryIngestUrlRequest {
    url: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct KnowledgeLibraryIngestUrlResponse {
    ids: Vec<String>,
    chunks_ingested: usize,
}

const INGEST_FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Paragraph-grouped chunk target size. A document embedded whole as one
/// vector loses retrieval precision (a query matching one paragraph would
/// have to compete with the whole document's average) - not chunking at all
/// would make `knowledge_library_ingest_url` markedly worse than
/// `knowledge_library_write`'s already-precise, caller-sized documents.
const INGEST_CHUNK_TARGET_CHARS: usize = 1500;

/// Minimal, dependency-free HTML-to-text: strips everything between `<` and
/// `>`. Doesn't decode entities (`&amp;` etc.) and doesn't special-case
/// `<script>`/`<style>` bodies (their contents pass through as text) - a
/// named simplification chosen over pulling in a real HTML parser crate for
/// a first pass at ingestion; real usage should motivate swapping this out
/// before it motivates a rewrite of anything around it.
fn strip_html_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Fetches `url`, extracts text (stripping HTML tags when the response
/// looks like HTML), chunks it, embeds and inserts each chunk into the
/// Knowledge Library under `url` as the shared `source_uri`. Every failure
/// mode (unreachable host, non-2xx status, no extractable text, an embed or
/// store error partway through) degrades to `Err(String)`, matching every
/// other tool handler here - never a panic.
async fn ingest_url(http_client: &reqwest::Client, kl_store: &dyn KnowledgeLibraryStore, embedding_client: &dyn EmbeddingClient, url: &str) -> Result<Vec<String>, String> {
    let response = tokio::time::timeout(INGEST_FETCH_TIMEOUT, http_client.get(url).send())
        .await
        .map_err(|_| format!("fetching {url} timed out after {INGEST_FETCH_TIMEOUT:?}"))?
        .map_err(|err| err.to_string())?;
    if !response.status().is_success() {
        return Err(format!("fetching {url} returned HTTP {}", response.status()));
    }
    let is_html = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|content_type| content_type.contains("html"));
    let body = response.text().await.map_err(|err| err.to_string())?;
    let text = if is_html { strip_html_tags(&body) } else { body };

    let chunks = chunk_text(&text, INGEST_CHUNK_TARGET_CHARS);
    if chunks.is_empty() {
        return Err(format!("{url} had no extractable text content"));
    }

    let mut ids = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let embedding = embedding_client.embed(&chunk).await.map_err(|err| err.to_string())?;
        let id = kl_store.insert_document(url, &chunk, embedding, SystemClock.now()).await.map_err(|err| err.to_string())?;
        ids.push(id.to_string());
    }
    Ok(ids)
}

/// Embeds `text` and inserts it into the Knowledge Library. A plain
/// function (not a `#[tool]` method body) so it's independently testable
/// without going through rmcp's macro dispatch.
async fn write_document(
    kl_store: &dyn KnowledgeLibraryStore,
    embedding_client: &dyn EmbeddingClient,
    source_uri: &str,
    text: &str,
) -> Result<String, String> {
    let embedding = embedding_client.embed(text).await.map_err(|err| err.to_string())?;
    let id = kl_store
        .insert_document(source_uri, text, embedding, SystemClock.now())
        .await
        .map_err(|err| err.to_string())?;
    Ok(id.to_string())
}

/// Embeds `query` and searches the Knowledge Library. Plain function, same
/// reasoning as `write_document`.
async fn search_documents(
    kl_store: &dyn KnowledgeLibraryStore,
    embedding_client: &dyn EmbeddingClient,
    query: &str,
    top_k: usize,
) -> Result<Vec<ScoredDocument>, String> {
    let embedding = embedding_client.embed(query).await.map_err(|err| err.to_string())?;
    kl_store.search(&embedding, top_k).await.map_err(|err| err.to_string())
}

/// Forwards a turn of input from another agent into the actor's dedicated
/// `external_agent_input` channel - the same path `send_input` exposes, and
/// the same accepted/unavailable semantics as `aca-api`'s `POST /input`
/// (`Err` only means the actor has shut down, never a validation failure).
async fn submit_external_input(tx: &mpsc::Sender<ExternalAgentInput>, text: String, agent_id: String) -> bool {
    tx.send(ExternalAgentInput { text, agent_id }).await.is_ok()
}

/// Renders the current `EngineSnapshot` as JSON text rather than `Json<T>`
/// structured content - `EngineSnapshot` and its nested types derive only
/// `Serialize` (they're shared with `aca-api`'s plain-JSON HTTP response),
/// not `schemars::JsonSchema`, and pulling `schemars` into `aca-engine`
/// just for this one MCP-only path isn't worth it. The full snapshot is
/// still delivered - just as text content instead of `structured_content`.
fn render_snapshot_json(snapshot_rx: &watch::Receiver<EngineSnapshot>) -> String {
    let snapshot = snapshot_rx.borrow().clone();
    serde_json::to_string(&snapshot).unwrap_or_else(|_| "{}".to_string())
}

#[tool_router]
impl McpState {
    #[tool(description = "Write a document into Omega's Knowledge Library - external knowledge Omega can later consult, never injected directly into memory or Working Memory.")]
    async fn knowledge_library_write(&self, Parameters(req): Parameters<KnowledgeLibraryWriteRequest>) -> Result<Json<KnowledgeLibraryWriteResponse>, String> {
        let id = write_document(self.kl_store.as_ref(), self.embedding_client.as_ref(), &req.source_uri, &req.text).await?;
        Ok(Json(KnowledgeLibraryWriteResponse { id }))
    }

    #[tool(description = "Search Omega's Knowledge Library by semantic similarity, returning the closest matching documents.")]
    async fn knowledge_library_search(&self, Parameters(req): Parameters<KnowledgeLibrarySearchRequest>) -> Result<Json<KnowledgeLibrarySearchResponse>, String> {
        let top_k = req.top_k.unwrap_or(5);
        let matches = search_documents(self.kl_store.as_ref(), self.embedding_client.as_ref(), &req.query, top_k).await?;
        Ok(Json(KnowledgeLibrarySearchResponse {
            matches: matches
                .into_iter()
                .map(|m| KnowledgeLibraryMatch { id: m.id.to_string(), source_uri: m.source_uri, text: m.text, score: m.score })
                .collect(),
        }))
    }

    #[tool(description = "Send a turn of input to Omega, exactly as a human chat message would - full competition pipeline, no privilege.")]
    async fn send_input(&self, Parameters(req): Parameters<SendInputRequest>) -> Json<SendInputResponse> {
        let accepted = submit_external_input(&self.external_agent_input_tx, req.text, req.agent_id).await;
        Json(SendInputResponse { accepted })
    }

    #[tool(description = "Get Omega's current cognitive snapshot: Working Memory, associative edges, memory-role counts, and goal stack.")]
    async fn get_snapshot(&self) -> String {
        render_snapshot_json(&self.snapshot_rx)
    }

    #[tool(description = "Fetch a URL (documentation, an article, a README) and ingest its text into Omega's Knowledge Library, chunked and embedded for later semantic search.")]
    async fn knowledge_library_ingest_url(&self, Parameters(req): Parameters<KnowledgeLibraryIngestUrlRequest>) -> Result<Json<KnowledgeLibraryIngestUrlResponse>, String> {
        let ids = ingest_url(&self.http_client, self.kl_store.as_ref(), self.embedding_client.as_ref(), &req.url).await?;
        Ok(Json(KnowledgeLibraryIngestUrlResponse { chunks_ingested: ids.len(), ids }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for McpState {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Omega's household-facing surface: read from and write into its Knowledge Library, send input as a full conversational turn, and observe its current cognitive state.",
        )
    }
}

/// Mounts the MCP tool surface at `/mcp` over the streamable-HTTP
/// transport - HTTP, not stdio, since this needs to be reachable by other
/// devices/agents on the LAN, matching everything else in this stack
/// (local Ollama/llama.cpp instances by IP, no cloud dependency by
/// default).
pub fn build_router(state: McpState) -> Router {
    let service = StreamableHttpService::new(move || Ok(state.clone()), Arc::new(LocalSessionManager::default()), StreamableHttpServerConfig::default());
    Router::new().nest_service("/mcp", service)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_store::SqliteStore;
    use aca_tiers::testing::FakeEmbeddingClient;

    fn test_state() -> (McpState, mpsc::Receiver<ExternalAgentInput>, watch::Sender<EngineSnapshot>) {
        let (tx, rx) = mpsc::channel(8);
        let (snapshot_tx, snapshot_rx) = watch::channel(EngineSnapshot::default());
        let kl_store: Arc<dyn KnowledgeLibraryStore> = Arc::new(SqliteStore::open_in_memory().unwrap());
        let embedding_client: Arc<dyn EmbeddingClient> = Arc::new(FakeEmbeddingClient::default());
        (McpState::new(tx, snapshot_rx, kl_store, embedding_client), rx, snapshot_tx)
    }

    #[tokio::test]
    async fn write_then_search_round_trips_a_document() {
        let (state, _rx, _snapshot_tx) = test_state();
        write_document(state.kl_store.as_ref(), state.embedding_client.as_ref(), "household://note", "the wifi password is hunter2")
            .await
            .unwrap();

        let matches = search_documents(state.kl_store.as_ref(), state.embedding_client.as_ref(), "the wifi password is hunter2", 5)
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "the wifi password is hunter2");
    }

    #[test]
    fn strip_html_tags_removes_tags_but_keeps_text() {
        let html = "<html><body><h1>Title</h1><p>Some <b>bold</b> text.</p></body></html>";
        assert_eq!(strip_html_tags(html), "TitleSome bold text.");
    }

    #[tokio::test]
    async fn ingest_url_fetches_chunks_and_stores_plain_text() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/doc.txt"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("first paragraph\n\nsecond paragraph").insert_header("content-type", "text/plain"))
            .mount(&server)
            .await;

        let (state, _rx, _snapshot_tx) = test_state();
        let url = format!("{}/doc.txt", server.uri());
        let ids = ingest_url(&aca_tiers::build_http_client(), state.kl_store.as_ref(), state.embedding_client.as_ref(), &url).await.unwrap();
        // Both paragraphs together are well under INGEST_CHUNK_TARGET_CHARS,
        // so chunk_text merges them into a single chunk/document.
        assert_eq!(ids.len(), 1);

        let matches = search_documents(state.kl_store.as_ref(), state.embedding_client.as_ref(), "first paragraph", 5).await.unwrap();
        assert!(!matches.is_empty());
        assert_eq!(matches[0].source_uri, url);
    }

    #[tokio::test]
    async fn ingest_url_strips_html_before_storing() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/page.html"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw("<html><body><p>real content</p></body></html>", "text/html; charset=utf-8"))
            .mount(&server)
            .await;

        let (state, _rx, _snapshot_tx) = test_state();
        let url = format!("{}/page.html", server.uri());
        ingest_url(&aca_tiers::build_http_client(), state.kl_store.as_ref(), state.embedding_client.as_ref(), &url).await.unwrap();

        let matches = search_documents(state.kl_store.as_ref(), state.embedding_client.as_ref(), "real content", 5).await.unwrap();
        assert_eq!(matches[0].text, "real content");
        assert!(!matches[0].text.contains('<'), "HTML tags should have been stripped before storage");
    }

    #[tokio::test]
    async fn ingest_url_reports_a_non_2xx_status_as_an_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET")).and(wiremock::matchers::path("/missing")).respond_with(wiremock::ResponseTemplate::new(404)).mount(&server).await;

        let (state, _rx, _snapshot_tx) = test_state();
        let url = format!("{}/missing", server.uri());
        let result = ingest_url(&aca_tiers::build_http_client(), state.kl_store.as_ref(), state.embedding_client.as_ref(), &url).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn submit_external_input_forwards_to_the_channel() {
        let (state, mut rx, _snapshot_tx) = test_state();
        let accepted = submit_external_input(&state.external_agent_input_tx, "hello".to_string(), "agent-1".to_string()).await;
        assert!(accepted);
        let received = rx.recv().await.unwrap();
        assert_eq!(received.text, "hello");
        assert_eq!(received.agent_id, "agent-1");
    }

    #[tokio::test]
    async fn submit_external_input_reports_not_accepted_when_the_actor_is_gone() {
        let (state, rx, _snapshot_tx) = test_state();
        drop(rx);
        let accepted = submit_external_input(&state.external_agent_input_tx, "hello".to_string(), "agent-1".to_string()).await;
        assert!(!accepted);
    }

    #[test]
    fn render_snapshot_json_serializes_the_current_value() {
        let (state, _rx, snapshot_tx) = test_state();
        let snapshot = EngineSnapshot { cycle_seq: 42, ..Default::default() };
        snapshot_tx.send(snapshot).unwrap();

        let json = render_snapshot_json(&state.snapshot_rx);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["cycle_seq"], 42);
    }

    #[tokio::test]
    async fn the_router_mounts_the_mcp_service_at_slash_mcp() {
        // Smoke-level only: proves the streamable-HTTP service is actually
        // wired to the /mcp path (a request reaches the tower::Service and
        // gets a real HTTP response, not axum's 404 fallback) without
        // performing a full MCP initialize handshake.
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let (state, _rx, _snapshot_tx) = test_state();
        let app = build_router(state);
        let response = app
            .oneshot(Request::builder().method("POST").uri("/mcp").header("content-type", "application/json").body(Body::from("{}")).unwrap())
            .await
            .unwrap();
        assert_ne!(response.status(), StatusCode::NOT_FOUND, "the MCP service should be mounted at /mcp");
    }
}
