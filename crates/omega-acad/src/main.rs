use std::env;
use std::io::{self, BufRead};
use std::sync::Arc;
use std::time::Duration;

use aca_engine::{is_self_memory_seeded, seed_self_memory_objects, CognitiveLoopActor, CurrentTimeTool, KnowledgeLibraryStore, LoopConfig, MemoryStore, OutcomeFeedbackCommand, ProcedureFeedbackCommand, Tool, ToolRegistry, ToolRiskTier};
use aca_store::{CycleEventRetention, SqliteStore};
use aca_tiers::{
    build_http_client, AttentionClient, ChatClient, DivergentPool, EmbeddingClient, GenerateRequest, OllamaAttentionClient, OllamaClient,
    OpenAiCompatClient, TierError, TierPool, TierResponse, TimeoutEmbeddingClient,
};
use aca_types::{MentalObjectId, Tier};
use aca_util::{Clock, SystemClock};
use async_trait::async_trait;
use tracing_subscriber::EnvFilter;

mod library;
mod video;
mod voice;
use video::VideoClient;
use voice::VoiceClient;

/// This command is accepted only from the daemon's local stdin, never from
/// the ordinary conversation/API/agent input channels. The host must judge
/// the real outcome independently before submitting a label.
fn parse_local_feedback(line: &str) -> Option<Result<ProcedureFeedbackCommand, &'static str>> {
    let mut words = line.split_whitespace();
    if words.next()? != "/feedback" { return None; }
    let successful = match words.next() {
        Some("success") => true,
        Some("failure") => false,
        _ => return Some(Err("usage: /feedback success|failure <terminal-observation-id>")),
    };
    let Some(id) = words.next() else {
        return Some(Err("usage: /feedback success|failure <terminal-observation-id>"));
    };
    if words.next().is_some() {
        return Some(Err("usage: /feedback success|failure <terminal-observation-id>"));
    }
    let Ok(observation_id) = id.parse::<MentalObjectId>() else {
        return Some(Err("invalid terminal observation id"));
    };
    Some(Ok(ProcedureFeedbackCommand { observation_id, successful }))
}

fn parse_local_outcome(line: &str) -> Option<Result<OutcomeFeedbackCommand, &'static str>> {
    let mut words = line.split_whitespace();
    if words.next()? != "/outcome" { return None; }
    let successful = match words.next() {
        Some("success") => true,
        Some("failure") => false,
        _ => return Some(Err("usage: /outcome success|failure <compared-observation-id>")),
    };
    let Some(id) = words.next() else {
        return Some(Err("usage: /outcome success|failure <compared-observation-id>"));
    };
    if words.next().is_some() {
        return Some(Err("usage: /outcome success|failure <compared-observation-id>"));
    }
    let Ok(observation_id) = id.parse::<MentalObjectId>() else {
        return Some(Err("invalid compared observation id"));
    };
    Some(Ok(OutcomeFeedbackCommand { observation_id, successful }))
}

#[cfg(test)]
mod local_feedback_tests {
    use super::*;

    #[test]
    fn feedback_is_local_command_not_conversation_input() {
        let id = MentalObjectId::new();
        assert!(parse_local_feedback("nice to meet you").is_none());
        let success = parse_local_feedback(&format!("/feedback success {id}")).unwrap().unwrap();
        assert_eq!(success.observation_id, id);
        assert!(success.successful);
        let failure = parse_local_feedback(&format!("/feedback failure {id}")).unwrap().unwrap();
        assert_eq!(failure.observation_id, id);
        assert!(!failure.successful);
        assert!(parse_local_feedback("/feedback success not-an-id").unwrap().is_err());
        assert!(parse_local_feedback(&format!("/feedback success {id} extra")).unwrap().is_err());
        let outcome = parse_local_outcome(&format!("/outcome failure {id}")).unwrap().unwrap();
        assert_eq!(outcome.observation_id, id);
        assert!(!outcome.successful);
        assert!(parse_local_outcome("/outcome success unknown").unwrap().is_err());
    }
}

/// Falls back to a placeholder chat response when no Tier 3 endpoint is
/// configured, so the daemon can run (and the acceptance bar can be
/// demonstrated) without a live model server — the same graceful-
/// degradation spirit as `aca-tiers`' confidence-clamping contract, applied
/// one level up.
struct NullChatClient;

#[async_trait]
impl ChatClient for NullChatClient {
    async fn generate(&self, req: GenerateRequest) -> Result<TierResponse, TierError> {
        let preview: String = req.prompt.chars().take(120).collect();
        Ok(TierResponse {
            raw_text: format!("(no Tier 3 model configured — set OMEGA_TIER3_BASE_URL) heard: {preview}"),
            confidence: 0.5,
            tier: Tier::T3,
        })
    }
}

fn env_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// A minimal, readable narration of the cognitive pipeline's meaningful
/// beats (Compare -> Broadcast -> Executive -> Act) - the terminal
/// console's counterpart to `viz/scripts/event_log.gd`'s own
/// `_is_worth_logging`/`_summarize_payload` allowlist, so someone running
/// headless (no Godot client attached) can watch "heard -> broadcast ->
/// deciding -> reflected/spoke" unfold live instead of seeing only the
/// final spoken line with nothing in between. Same bar as that allowlist,
/// deliberately kept in sync with it rather than reinvented: routine
/// per-tick churn (an `Ignore` that keeps winning, an empty Broadcast tick)
/// stays silent, real transitions are shown. `None` for anything that
/// doesn't clear that bar, or that's already printed elsewhere - a "speak"
/// Act outcome is handled by the `Omega: ...` line at this loop's own call
/// site (with a `continue` right after it), so this function is never even
/// reached for that case and never duplicates it.
fn console_pipeline_line(event: &aca_engine::CycleEvent) -> Option<String> {
    use aca_engine::CyclePhase;
    match event.phase {
        CyclePhase::Compare => {
            let text = event.payload.get("text").and_then(|v| v.as_str())?;
            Some(format!("heard: {text}"))
        }
        CyclePhase::Broadcast => {
            // Already gated at emission (see `loop_actor::tick`'s own
            // guard) to fire only when something was actually admitted or
            // released - never an empty-tick flood.
            let admitted = event.payload.get("admitted").and_then(|v| v.as_u64()).unwrap_or(0);
            let released = event.payload.get("released").and_then(|v| v.as_u64()).unwrap_or(0);
            Some(format!("broadcast: admitted {admitted}, released {released}"))
        }
        CyclePhase::Executive => {
            let operator = event.payload.get("operator").and_then(|v| v.as_str())?;
            (operator != "Ignore").then(|| format!("deciding: {operator}"))
        }
        CyclePhase::Act => {
            let operator = event.payload.get("operator").and_then(|v| v.as_str())?;
            match operator {
                // "silent" is routine noise (the same bar `_is_worth_logging`
                // applies); "speak" is already printed as `Omega: ...` above
                // this function's call site.
                "silent" | "speak" => None,
                "reflect" => Some("reflected".to_string()),
                "remember" => Some(format!("remembered: {}", event.payload.get("outcome").and_then(|v| v.as_str()).unwrap_or("?"))),
                "act" => {
                    let tool = event.payload.get("tool").and_then(|v| v.as_str()).unwrap_or("?");
                    Some(format!("used tool '{tool}'"))
                }
                "consult-knowledge-library" => {
                    let found = event.payload.get("found").and_then(|v| v.as_bool()).unwrap_or(false);
                    Some(format!("consulted knowledge library: {}", if found { "found a match" } else { "no match found" }))
                }
                other => Some(other.to_string()),
            }
        }
        _ => None,
    }
}

/// Every raw client this returns is wrapped in `TimeoutEmbeddingClient` -
/// see that type's doc comment for why: unlike every reasoning tier, Tier 0
/// used to be called with no timeout anywhere, and a single hung response
/// (confirmed live, not theoretical) froze the entire cognitive loop
/// indefinitely since Observe awaits it inline mid-tick.
fn build_embedding_client() -> (String, Arc<dyn EmbeddingClient>) {
    let http = build_http_client();
    let timeout_secs: u64 = env_var("OMEGA_EMBEDDING_TIMEOUT_SECS").and_then(|v| v.parse().ok()).unwrap_or(30);
    let (label, raw_client): (String, Arc<dyn EmbeddingClient>) =
        match (env_var("OMEGA_EMBEDDING_BASE_URL"), env_var("OMEGA_EMBEDDING_MODEL")) {
            (Some(base_url), Some(model)) => {
                let provider = env_var("OMEGA_EMBEDDING_PROVIDER").unwrap_or_else(|| "ollama".to_string());
                tracing::info!(provider = %provider, base_url = %base_url, model = %model, timeout_secs, "configured embedding tier");
                let client: Arc<dyn EmbeddingClient> = if provider.eq_ignore_ascii_case("openai") {
                    Arc::new(OpenAiCompatClient::new(http, base_url, model.clone(), env_var("OMEGA_EMBEDDING_API_KEY"), Tier::T0))
                } else {
                    Arc::new(OllamaClient::new(http, base_url, model.clone(), Tier::T0))
                };
                (model, client)
            }
            _ => {
                tracing::warn!("OMEGA_EMBEDDING_BASE_URL/OMEGA_EMBEDDING_MODEL not set — using the deterministic FakeEmbeddingClient");
                ("(unconfigured)".to_string(), Arc::new(aca_tiers::testing::FakeEmbeddingClient::default()))
            }
        };
    (label, Arc::new(TimeoutEmbeddingClient::new(raw_client, Duration::from_secs(timeout_secs))))
}

/// Builds a Tier 1/2 "divergent parallel takes" pool: `OMEGA_{PREFIX}_{i}_
/// BASE_URL` / `OMEGA_{PREFIX}_{i}` pairs for `i` in `1..=max_slots`, each
/// slot independent (a missing/blank slot is just skipped, not a reason to
/// stop scanning later ones). An empty result is a valid, supported
/// "this tier isn't configured yet" state - `cognitive_core::reflect`'s
/// ladder treats an empty `DivergentPool` as "skip straight to the next
/// tier," not an error, so the daemon runs fine with only Tier 3 set up.
fn build_divergent_pool(prefix: &str, max_slots: usize, tier: Tier, timeout: Duration) -> DivergentPool {
    let http = build_http_client();
    let mut clients: Vec<Arc<dyn ChatClient>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for i in 1..=max_slots {
        let base_url = env_var(&format!("OMEGA_{prefix}_{i}_BASE_URL"));
        let model = env_var(&format!("OMEGA_{prefix}_{i}"));
        let (Some(base_url), Some(model)) = (base_url, model) else {
            continue;
        };
        let provider = env_var(&format!("OMEGA_{prefix}_{i}_PROVIDER")).unwrap_or_else(|| "ollama".to_string());
        tracing::info!(index = i, provider = %provider, base_url = %base_url, model = %model, tier = ?tier, "configured tier model slot");
        let client: Arc<dyn ChatClient> = if provider.eq_ignore_ascii_case("openai") {
            Arc::new(OpenAiCompatClient::new(http.clone(), base_url, model.clone(), env_var(&format!("OMEGA_{prefix}_{i}_API_KEY")), tier))
        } else {
            Arc::new(OllamaClient::new(http.clone(), base_url, model.clone(), tier))
        };
        clients.push(client);
        labels.push(model);
    }
    if clients.is_empty() {
        tracing::warn!(tier = ?tier, "no models configured for this tier — it will be skipped by the escalation ladder");
    }
    DivergentPool::new(tier, timeout, clients).with_labels(labels)
}

/// The Voice Interaction service (`E:\AI\Voice Interaction`) is a separate
/// process this daemon never spawns - same reasoning as the Godot
/// visualization being its own process. `None` means "not configured,"
/// treated as a no-op by the caller, not an error - voice output is
/// optional, same as every other external integration here.
fn build_voice_client() -> Option<Arc<VoiceClient>> {
    let base_url = env_var("OMEGA_VOICE_BASE_URL")?;
    let speaker = env_var("OMEGA_VOICE_SPEAKER").unwrap_or_else(|| "Omega".to_string());
    let conversation_id = env_var("OMEGA_VOICE_CONVERSATION_ID").unwrap_or_else(|| "omega".to_string());
    tracing::info!(base_url = %base_url, speaker = %speaker, "configured voice output");
    Some(Arc::new(VoiceClient::new(base_url, speaker, conversation_id)))
}

/// The Video Interaction service (`E:\AI\Video Interaction`) is a separate
/// process this daemon never spawns - same reasoning as `build_voice_client`
/// and the Godot visualization being their own processes. `None` means "not
/// configured," treated as a no-op by the caller - camera input is optional,
/// same as every other external integration here.
fn build_video_client() -> Option<Arc<VideoClient>> {
    let base_url = env_var("OMEGA_VIDEO_BASE_URL")?;
    // Also this client's poll interval - see `VideoClient::min_interval`'s
    // doc comment for why polling cadence doubles as the rate limit that
    // keeps a bursty camera feed from crowding conversation out of Working
    // Memory competition. 1500ms matches the service's own README example,
    // not a value derived from anything - reasonable to retune once a real
    // session's event log has been reviewed.
    let min_interval_ms = env_var("OMEGA_VIDEO_MIN_INTERVAL_MS").and_then(|v| v.parse::<u64>().ok()).unwrap_or(1500);
    tracing::info!(base_url = %base_url, min_interval_ms, "configured video input");
    Some(Arc::new(VideoClient::new(base_url, Duration::from_millis(min_interval_ms))))
}

/// Reads a PDF library share in the background: one loop indexes it into
/// the Knowledge Library at a brisk, embedding-server-limited pace; a
/// second, manually-triggered cursor drips it in as `SensorInput` one chunk
/// per `POST /library/deliver` (see `library.rs`). `None` means "not
/// configured" - same optional-integration pattern as `build_voice_client`.
fn build_library_config(db_path: &str) -> Option<library::LibraryConfig> {
    let root = env_var("OMEGA_LIBRARY_PATH")?;
    let chunk_chars: usize = env_var("OMEGA_LIBRARY_CHUNK_CHARS").and_then(|v| v.parse().ok()).unwrap_or(1500);
    // Indexing wants to finish in hours, not years - this only needs to be
    // polite to the embedding server, not paced for human attention.
    let kl_interval_secs: u64 = env_var("OMEGA_LIBRARY_KL_INTERVAL_SECS").and_then(|v| v.parse().ok()).unwrap_or(1);
    let kl_state_path = env_var("OMEGA_LIBRARY_KL_STATE_PATH").unwrap_or_else(|| format!("{db_path}.library_kl_state.json"));
    let drip_state_path = env_var("OMEGA_LIBRARY_DRIP_STATE_PATH").unwrap_or_else(|| format!("{db_path}.library_drip_state.json"));
    tracing::info!(root = %root, chunk_chars, kl_interval_secs, "configured library reader");
    Some(library::LibraryConfig {
        root: root.into(),
        chunk_chars,
        kl_interval: Duration::from_secs(kl_interval_secs),
        kl_state_path: kl_state_path.into(),
        drip_state_path: drip_state_path.into(),
    })
}

/// Overrides `ExecutiveConfig`'s impasse thresholds from the environment,
/// starting from `LoopConfig::default()` for everything else. Exists mainly
/// as a testing lever: `propose_operators`' current rule-based proposals
/// all hardcode confidence in the 0.7-0.9 range, so with the default 0.4
/// threshold a genuine `select_operator` impasse (as opposed to the
/// separate, content-triggerable low-confidence-Reflection -> Consult-
/// Knowledge-Library path) essentially never fires on real input - raising
/// `OMEGA_EXECUTIVE_CONFIDENCE_THRESHOLD` above ~0.9 makes nearly every
/// proposal an impasse on purpose, so `spawn_subgoal`/`EscalateTier` are
/// actually reachable to watch (look for `CycleEventKind::Impasse`/
/// `Escalation` - the console printer below already calls those out).
fn build_loop_config() -> LoopConfig {
    let mut config = LoopConfig::default();
    if let Some(threshold) = env_var("OMEGA_EXECUTIVE_CONFIDENCE_THRESHOLD").and_then(|v| v.parse().ok()) {
        config.executive_config.confidence_threshold = threshold;
    }
    if let Some(epsilon) = env_var("OMEGA_EXECUTIVE_TIE_EPSILON").and_then(|v| v.parse().ok()) {
        config.executive_config.preference_tie_epsilon = epsilon;
    }
    tracing::info!(
        confidence_threshold = config.executive_config.confidence_threshold,
        preference_tie_epsilon = config.executive_config.preference_tie_epsilon,
        "configured executive impasse thresholds"
    );
    if let Some(ms) = env_var("OMEGA_BOREDOM_IDLE_MS").and_then(|v| v.parse().ok()) {
        config.boredom_config.idle_threshold_ms = ms;
    }
    if let Some(ms) = env_var("OMEGA_BOREDOM_MIN_INTERVAL_MS").and_then(|v| v.parse().ok()) {
        config.boredom_config.min_interval_ms = ms;
    }
    if let Some(ms) = env_var("OMEGA_BOREDOM_SELF_STATUS_INTERVAL_MS").and_then(|v| v.parse().ok()) {
        config.boredom_config.self_status_interval_ms = ms;
    }
    tracing::info!(
        idle_threshold_ms = config.boredom_config.idle_threshold_ms,
        min_interval_ms = config.boredom_config.min_interval_ms,
        self_status_interval_ms = config.boredom_config.self_status_interval_ms,
        "configured idle/boredom self-stimulus"
    );
    if let Some(ms) = env_var("OMEGA_ATTENTION_TIMEOUT_MS").and_then(|v| v.parse().ok()) {
        config.attention_timeout = Duration::from_millis(ms);
    }
    if let Some(threshold) = env_var("OMEGA_ATTENTION_MIN_CONFIDENCE").and_then(|v| v.parse().ok()) {
        config.attention_min_confidence = threshold;
    }
    if let Some(interval) = env_var("OMEGA_ATTENTION_RECONCILIATION_INTERVAL").and_then(|v| v.parse().ok()) {
        config.attention_reconciliation_interval = interval;
    }
    tracing::info!(
        attention_timeout_ms = config.attention_timeout.as_millis() as u64,
        attention_min_confidence = config.attention_min_confidence,
        attention_reconciliation_interval = config.attention_reconciliation_interval,
        "configured attention-model Broadcast thresholds"
    );
    if let Some(ms) = env_var("OMEGA_PRESENCE_HALF_LIFE_MS").and_then(|v| v.parse().ok()) {
        config.presence_half_life_ms = ms;
    }
    tracing::info!(
        presence_half_life_ms = config.presence_half_life_ms,
        "configured conversational-presence decay (see LoopConfig::presence_half_life_ms)"
    );
    config
}

/// The `Operator::Act` trust dial (see `steps::tools::ToolRegistry`'s doc
/// comment): starts at `Harmless` - only tools with zero external side
/// effects - and can be turned up as trust is established, without any
/// code change, once tools above that tier actually exist to turn it up
/// *for*. `self_status` (needs the actor's own snapshot channel) isn't
/// built here at all - see `ToolRegistry::with_self_status`.
fn build_tool_registry(clock: Arc<dyn Clock>) -> ToolRegistry {
    let max_risk_tier = match env_var("OMEGA_MAX_TOOL_RISK_TIER").as_deref() {
        Some("reversible") => ToolRiskTier::Reversible,
        Some("consequential") => ToolRiskTier::Consequential,
        Some("harmless") | None => ToolRiskTier::Harmless,
        Some(other) => {
            tracing::warn!(value = %other, "unrecognized OMEGA_MAX_TOOL_RISK_TIER — defaulting to harmless");
            ToolRiskTier::Harmless
        }
    };
    tracing::info!(max_risk_tier = ?max_risk_tier, "configured Operator::Act tool registry");
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(CurrentTimeTool::new(clock))];
    ToolRegistry::new(tools, max_risk_tier)
}

fn build_tier3_client() -> (String, Arc<dyn ChatClient>) {
    let http = build_http_client();
    match (env_var("OMEGA_TIER3_BASE_URL"), env_var("OMEGA_TIER3_MODEL")) {
        (Some(base_url), Some(model)) => {
            let provider = env_var("OMEGA_TIER3_PROVIDER").unwrap_or_else(|| "ollama".to_string());
            tracing::info!(provider = %provider, base_url = %base_url, model = %model, "configured Tier 3 model");
            let client: Arc<dyn ChatClient> = if provider.eq_ignore_ascii_case("openai") {
                Arc::new(OpenAiCompatClient::new(http, base_url, model.clone(), env_var("OMEGA_TIER3_API_KEY"), Tier::T3))
            } else {
                Arc::new(OllamaClient::new(http, base_url, model.clone(), Tier::T3))
            };
            (model, client)
        }
        _ => {
            tracing::warn!("OMEGA_TIER3_BASE_URL/OMEGA_TIER3_MODEL not set — using NullChatClient (no real reflection/escalation will occur)");
            ("(unconfigured)".to_string(), Arc::new(NullChatClient))
        }
    }
}

/// Stands in for an unconfigured Tier 4: always *fails* rather than
/// returning a placeholder success. Deliberately not `NullChatClient` here -
/// by the time Tier 4 is ever called, a real (if low-confidence) Tier 3
/// answer is already in hand (see `tick()`'s `EscalateTier` arm), and a fake
/// "not configured" placeholder succeeding would silently replace that real
/// answer instead of falling back to it. `NullChatClient` is correct for
/// Tier 3 specifically because Tier 3 is the guaranteed backstop the whole
/// ladder depends on always producing *something* - Tier 4 has no such
/// requirement; specs.md calls it escalation-only and rare by construction.
struct UnconfiguredChatClient(Tier);

#[async_trait]
impl ChatClient for UnconfiguredChatClient {
    async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
        Err(TierError::TierNotConfigured(self.0))
    }
}

fn build_tier4_client() -> (String, Arc<dyn ChatClient>) {
    let http = build_http_client();
    match (env_var("OMEGA_TIER4_BASE_URL"), env_var("OMEGA_TIER4_MODEL")) {
        (Some(base_url), Some(model)) => {
            let provider = env_var("OMEGA_TIER4_PROVIDER").unwrap_or_else(|| "ollama".to_string());
            tracing::info!(provider = %provider, base_url = %base_url, model = %model, "configured Tier 4 model");
            let client: Arc<dyn ChatClient> = if provider.eq_ignore_ascii_case("openai") {
                Arc::new(OpenAiCompatClient::new(http, base_url, model.clone(), env_var("OMEGA_TIER4_API_KEY"), Tier::T4))
            } else {
                Arc::new(OllamaClient::new(http, base_url, model.clone(), Tier::T4))
            };
            (model, client)
        }
        _ => {
            tracing::warn!("OMEGA_TIER4_BASE_URL/OMEGA_TIER4_MODEL not set — Tier 4 escalation will always fall back to the Tier 3 answer");
            ("(unconfigured)".to_string(), Arc::new(UnconfiguredChatClient(Tier::T4)))
        }
    }
}

/// The Omega Attention model's endpoint - genuinely optional and not part
/// of the tier ladder (see `.env`'s own comment on these two vars, and
/// `steps::broadcast`'s doc comments for what it's used for). Returns
/// `None`, not a placeholder client, when unconfigured: unlike Tier 3/4,
/// there is no "always succeeds with a stub answer" fallback that would
/// make sense here - `CognitiveLoopActor` already treats a `None`
/// `attention_client` as "run Step 6 exactly as the deterministic
/// algorithm always has," which is the correct unconfigured behavior on
/// its own, not something a placeholder client needs to simulate.
fn build_attention_client() -> Option<Arc<dyn AttentionClient>> {
    match (env_var("OMEGA_ATTENTION_MODEL_BASE_URL"), env_var("OMEGA_ATTENTION_MODEL")) {
        (Some(base_url), Some(model)) => {
            tracing::info!(base_url = %base_url, model = %model, "configured Omega Attention model; mode determines whether votes are shadowed or active");
            Some(Arc::new(OllamaAttentionClient::new(build_http_client(), base_url, model)))
        }
        _ => {
            tracing::info!("OMEGA_ATTENTION_MODEL_BASE_URL/OMEGA_ATTENTION_MODEL not set — Step 6 Broadcast runs the deterministic algorithm only");
            None
        }
    }
}

fn build_orient_outcome_specialist() -> Option<aca_engine::OrientOutcomeSpecialist> {
    let path = env_var("OMEGA_ORIENT_OUTCOME_MODEL_PATH")?;
    match std::fs::read_to_string(&path)
        .map_err(|error| error.to_string())
        .and_then(|json| aca_engine::OrientOutcomeSpecialist::from_json(&json).map_err(|error| error.to_string()))
    {
        Ok(model) => {
            tracing::info!(path = %path, "loaded shadow-only ORIENT observed-outcome specialist");
            Some(model)
        }
        Err(error) => {
            tracing::warn!(path = %path, error = %error, "rejected ORIENT outcome specialist artifact");
            None
        }
    }
}

fn build_communicative_intent_specialist() -> Option<aca_engine::CommunicativeIntentSpecialist> {
    let path = env_var("OMEGA_COMMUNICATIVE_INTENT_MODEL_PATH")?;
    match std::fs::read_to_string(&path)
        .map_err(|error| error.to_string())
        .and_then(|json| aca_engine::CommunicativeIntentSpecialist::from_json(&json).map_err(|error| error.to_string()))
    {
        Ok(model) => {
            tracing::info!(path = %path, "loaded shadow-only communicative-intent specialist");
            Some(model)
        }
        Err(error) => {
            tracing::warn!(path = %path, error = %error, "rejected communicative-intent specialist artifact");
            None
        }
    }
}

/// `aca_store::prune_cycle_events` and `CycleEventRetention` have existed
/// since the idle-tick telemetry incident (see the attention-model journal)
/// but were only ever reachable by hand, via the `prune_cycle_events`
/// example binary - nothing in the running daemon ever called them. Over a
/// multi-day unattended run that gap is the whole story: `cycle_events` gets
/// one `Normal`-kind row per tick with no cost anyone pays at write time, so
/// it grows in direct proportion to wall-clock uptime with nothing to bound
/// it. Confirmed live: a 67.6-hour run with a tiny (575-object) graph still
/// grew this table to 122.4 million rows / 79.8 GB, because nothing ever
/// pruned it.
///
/// Runs on its own timer rather than from inside `tick()` - pruning is
/// maintenance, not cognition, and has no business sharing a failure mode or
/// a latency budget with the cognitive loop. Reads the live `cycle_seq` off
/// the same snapshot channel `self_status`/the API already read from, so it
/// always prunes relative to where the actor actually is, not a stale
/// number captured at startup.
fn spawn_cycle_event_pruner(store: Arc<SqliteStore>, snapshot_rx: tokio::sync::watch::Receiver<aca_engine::EngineSnapshot>) {
    let interval_ms: u64 = env_var("OMEGA_CYCLE_EVENT_PRUNE_INTERVAL_MS").and_then(|v| v.parse().ok()).unwrap_or(300_000);
    let retention = CycleEventRetention {
        keep_recent_cycles: env_var("OMEGA_KEEP_RECENT_CYCLES").and_then(|v| v.parse().ok()).unwrap_or(CycleEventRetention::default().keep_recent_cycles),
        keep_abnormal_cycles: env_var("OMEGA_KEEP_ABNORMAL_CYCLES").and_then(|v| v.parse().ok()).unwrap_or(CycleEventRetention::default().keep_abnormal_cycles),
    };
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms));
        loop {
            ticker.tick().await;
            let current_cycle_seq = snapshot_rx.borrow().cycle_seq;
            match store.prune_cycle_events(current_cycle_seq, retention).await {
                Ok(report) => {
                    if report.deleted_total() > 0 {
                        tracing::info!(
                            current_cycle_seq,
                            deleted_normal = report.deleted_normal_events,
                            deleted_abnormal = report.deleted_abnormal_events,
                            "pruned cycle_events"
                        );
                    }
                }
                Err(err) => tracing::warn!(error = %err, "cycle_events prune failed"),
            }
        }
    });
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Loads .env from the current working directory into the process
    // environment, if present - .ok() because running without a .env file
    // at all (real env vars set some other way, or intentionally using
    // every default) is a normal, supported case, not an error.
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let db_path = env_var("OMEGA_DB_PATH").unwrap_or_else(|| "omega.sqlite3".to_string());
    let api_port: u16 = env_var("OMEGA_API_PORT").and_then(|v| v.parse().ok()).unwrap_or(8787);
    let mcp_port: u16 = env_var("OMEGA_MCP_PORT").and_then(|v| v.parse().ok()).unwrap_or(8788);
    // Deliberately not loopback-only, unlike the local API above - this
    // server's entire purpose is being reachable by other agents/devices on
    // the household LAN.
    let mcp_host = env_var("OMEGA_MCP_HOST").unwrap_or_else(|| "0.0.0.0".to_string());
    let tier3_timeout_secs: u64 = env_var("OMEGA_TIER3_TIMEOUT_SECS").and_then(|v| v.parse().ok()).unwrap_or(300);
    // Escalation-only and rarer still than Tier 3, but when it does run
    // it's the largest model in the ladder - the same generous, dead-
    // connection-only backstop timeout as Tier 3, not a tighter one.
    let tier4_timeout_secs: u64 = env_var("OMEGA_TIER4_TIMEOUT_SECS").and_then(|v| v.parse().ok()).unwrap_or(300);
    let tier2_timeout_secs: u64 = env_var("OMEGA_TIER2_TIMEOUT_SECS").and_then(|v| v.parse().ok()).unwrap_or(60);
    let tier1_timeout_secs: u64 = env_var("OMEGA_TIER1_TIMEOUT_SECS").and_then(|v| v.parse().ok()).unwrap_or(20);

    tracing::info!(db_path = %db_path, api_port, "starting omega-acad");

    let store = Arc::new(SqliteStore::open(&db_path)?);
    let mut snapshot = store.load_all().await?;
    tracing::info!(object_count = snapshot.objects.len(), "loaded graph from durable store");

    let (embedding_label, embedding_client) = build_embedding_client();
    // The MCP server embeds Knowledge Library documents/queries with the
    // same client the cognitive loop uses - one embedding space, not two.
    let mcp_embedding_client = embedding_client.clone();
    // The library ingest loop is a third, independent consumer of the same
    // embedding space - see `library.rs`.
    let library_embedding_client = embedding_client.clone();

    // Self Memory (specs.md: "identity continuity") is seeded exactly once,
    // on the very first boot against a fresh database - never again, so a
    // restart doesn't duplicate it. Embeddings are resolved here (this is
    // the async, I/O-capable context `self_memory::seed_self_memory_objects`
    // itself deliberately isn't) and any object whose embedding fails to
    // resolve is dropped rather than inserted un-embedded and unfindable by
    // recall - the same discretion `steps::observe` already applies to
    // ordinary Observations.
    if !is_self_memory_seeded(&snapshot.objects) {
        tracing::info!("seeding Self Memory for the first time");
        for mut object in seed_self_memory_objects(SystemClock.now(), LoopConfig::default().decay_d) {
            match embedding_client.embed(&object.text).await {
                Ok(embedding) => {
                    object.embedding = Some(embedding);
                    snapshot.objects.push(object);
                }
                Err(err) => {
                    tracing::warn!(error = %err, text = %object.text, "failed to embed a Self Memory seed object - skipping it");
                }
            }
        }
    }
    let (tier3_label, chat_client) = build_tier3_client();
    let tier3_pool = TierPool::new(Tier::T3, 1, Duration::from_secs(tier3_timeout_secs)).with_label(tier3_label);
    let (tier4_label, tier4_client) = build_tier4_client();
    let tier4_pool = TierPool::new(Tier::T4, 1, Duration::from_secs(tier4_timeout_secs)).with_label(tier4_label);
    // Tier 2 is capped at the spec's own "up to three ~9B models" policy.
    let tier2_pool = build_divergent_pool("TIER2_MODEL", 3, Tier::T2, Duration::from_secs(tier2_timeout_secs));
    // Tier 1 (sub-4B models) is back in the reflection ladder as a real
    // first-pass proposer - live testing showed those models self-report
    // high confidence on flatly wrong/off-topic answers, which a threshold
    // on that self-report can't defend against, so the ladder no longer
    // trusts it. Instead it trusts cross-candidate embedding *agreement*
    // (see `cognitive_core::reflect`'s doc comment and `steps::arbitrate`) -
    // a scan bound of 5 slots leaves room for several concurrent cheap
    // models without being a hard architectural cap.
    let tier1_pool = build_divergent_pool("TIER1_MODEL", 5, Tier::T1, Duration::from_secs(tier1_timeout_secs));

    // `SqliteStore` implements both `MemoryStore` and `KnowledgeLibraryStore`
    // over the same connection/file (see aca-store::knowledge_library's doc
    // comment) - three `Arc` clones of one store, not three databases.
    let kl_store_for_actor: Arc<dyn KnowledgeLibraryStore> = store.clone();
    let kl_store_for_mcp: Arc<dyn KnowledgeLibraryStore> = store.clone();
    let kl_store_for_library: Arc<dyn KnowledgeLibraryStore> = store.clone();
    let store_for_pruner = store.clone();
    let tool_registry = build_tool_registry(Arc::new(SystemClock));

    let attention_mode = env_var("OMEGA_ATTENTION_MODE").unwrap_or_else(|| "off".to_string());
    let mut loop_config = build_loop_config();
    loop_config.attention_shadow_only = attention_mode != "active";
    let attach_attention = matches!(attention_mode.as_str(), "shadow" | "active");
    if !matches!(attention_mode.as_str(), "off" | "shadow" | "active") {
        tracing::warn!(mode = %attention_mode, "unknown attention mode; specialist is disabled");
    }
    tracing::info!(mode = %attention_mode, "attention specialist mode: off, shadow, or explicitly active");

    let (actor, handles) = CognitiveLoopActor::new(
        loop_config,
        embedding_client,
        chat_client,
        tier1_pool,
        tier2_pool,
        tier3_pool,
        tier4_pool,
        tier4_client,
        store,
        kl_store_for_actor,
        tool_registry,
        Arc::new(SystemClock),
        snapshot.objects,
    );
    let mut actor = actor.with_embedding_label(embedding_label);
    if attach_attention {
        if let Some(attention_client) = build_attention_client() {
            actor = actor.with_attention_client(attention_client);
        }
    }
    if let Some(specialist) = build_orient_outcome_specialist() {
        actor = actor.with_orient_outcome_specialist(specialist);
    }
    if let Some(specialist) = build_communicative_intent_specialist() {
        actor = actor.with_communicative_intent_specialist(specialist);
    }

    // The continuous cognitive cycle - runs independently of whether
    // anyone is watching the API or typing at the CLI.
    tokio::spawn(actor.run());

    spawn_cycle_event_pruner(store_for_pruner, handles.snapshot_rx.clone());

    // The KL ingest loop stays automatic - background indexing is harmless
    // and self-terminates once the corpus is covered. The drip ("currently
    // reading") side no longer runs on its own timer - see `library.rs`'s
    // module doc comment for why - so `library_router` is only `Some` when
    // the library is configured at all, giving the viz's "Deliver a Book"
    // button somewhere to POST.
    let mut library_router = None;
    if let Some(library_config) = build_library_config(&db_path) {
        let library_config = Arc::new(library_config);
        tokio::spawn(library::run_kl_ingest_loop(library_config.clone(), kl_store_for_library, library_embedding_client, Arc::new(SystemClock)));
        let drip_state = Arc::new(library::DripState::new(library_config));
        library_router = Some(library::build_deliver_router(drip_state, handles.sensor_input_tx.clone()));
    }

    let api_state = aca_api::ApiState::new(
        handles.input_tx.clone(),
        handles.events_rx.resubscribe(),
        handles.snapshot_rx.clone(),
        handles.live_models.clone(),
    );
    let router = aca_api::build_router(api_state);
    let router = match library_router {
        Some(library_router) => router.merge(library_router),
        None => router,
    };
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", api_port)).await?;
    tracing::info!(addr = %listener.local_addr()?, "local API listening");
    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, router).await {
            tracing::error!(error = %err, "API server exited");
        }
    });

    // The household-facing MCP surface: other agents read from and write
    // into the Knowledge Library, send input as a full conversational turn
    // (via `handles.external_agent_input_tx`, never the plain `input_tx`
    // human/voice/API channel), and observe Omega's current snapshot.
    let mcp_state = aca_mcp::McpState::new(handles.external_agent_input_tx.clone(), handles.snapshot_rx.clone(), kl_store_for_mcp, mcp_embedding_client);
    let mcp_router = aca_mcp::build_router(mcp_state);
    let mcp_listener = tokio::net::TcpListener::bind((mcp_host.as_str(), mcp_port)).await?;
    tracing::info!(addr = %mcp_listener.local_addr()?, "MCP server listening");
    tokio::spawn(async move {
        if let Err(err) = axum::serve(mcp_listener, mcp_router).await {
            tracing::error!(error = %err, "MCP server exited");
        }
    });

    // Print any spoken/asked replies (and impasses/escalations) to the console as
    // they happen, live, independent of the API. Also where voice output
    // hooks in: speaking is awaited sequentially in this same loop (not
    // spawned per-utterance) so Omega never talks over itself, and a voice
    // failure only ever logs a warning here - it can't affect the
    // cognitive loop, which knows nothing about this task's existence.
    let voice_client = build_voice_client();
    if let Some(voice_client) = &voice_client {
        // The voice service captures/transcribes on its own now; forward
        // anything it overhears (that isn't Omega's own voice) into the
        // same input channel stdin and the HTTP API already use.
        voice_client.clone().spawn_listener(handles.room_input_tx.clone());
    }
    let video_client = build_video_client();
    if let Some(video_client) = &video_client {
        // The video service captures/tracks/captions on its own now;
        // forward scene/entity events into the same generic sensor channel
        // `SensorInput` already exists for (see `video.rs`'s own doc
        // comment - no engine change was needed for this).
        video_client.clone().spawn_listener(handles.sensor_input_tx.clone());
    }
    let mut console_events = handles.events_rx.resubscribe();
    tokio::spawn(async move {
        loop {
            match console_events.recv().await {
                Ok(event) => {
                    if event.phase == aca_engine::CyclePhase::Compare
                        && let Some(id) = event.payload.get("observation_id").and_then(|v| v.as_str())
                    {
                        println!("[compared observation {id}; local outcome: /outcome success|failure {id}]");
                    }
                    if event.phase == aca_engine::CyclePhase::Act
                        && event.payload.get("reason").and_then(|v| v.as_str()) == Some("ignored")
                        && let Some(id) = event.payload.get("foreground_observation_id").and_then(|v| v.as_str())
                    {
                        println!("[ignored observation {id}; local procedure feedback: /feedback success|failure {id}]");
                    }
                    if let Some(text) = event.payload.get("text").and_then(|v| v.as_str()) {
                        if event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak") {
                            println!("Omega: {text}");
                            if let Some(id) = event.payload.get("foreground_observation_id").and_then(|v| v.as_str()) {
                                let terminal = if event.payload.get("attempted_operator").and_then(|v| v.as_str()) == Some("Ask") {
                                    "asked"
                                } else {
                                    "spoken"
                                };
                                println!("[{terminal} observation {id}; local feedback: /feedback success|failure {id}]");
                            }
                            if let Some(voice_client) = &voice_client {
                                if let Err(err) = voice_client.speak(text).await {
                                    tracing::warn!(error = %err, "voice output failed");
                                }
                            }
                            continue;
                        }
                    }
                    if matches!(event.event_type, aca_engine::CycleEventKind::Impasse | aca_engine::CycleEventKind::Escalation) {
                        println!("[cycle {}] {:?}: {}", event.cycle_seq, event.event_type, event.payload);
                        continue;
                    }
                    if let Some(line) = console_pipeline_line(&event) {
                        println!("[cycle {}] {line}", event.cycle_seq);
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    println!("Omega ACA is running. Type a line and press Enter (Ctrl-C to quit). Use /feedback success|failure <terminal-id> for verified procedures, or /outcome success|failure <compared-id> for independent specialist labels; judge the real consequence first.");
    let input_tx = handles.input_tx;
    let procedure_feedback_tx = handles.procedure_feedback_tx;
    let outcome_feedback_tx = handles.outcome_feedback_tx;
    // Blocking stdin reads have no place directly in an async fn - park
    // them on a dedicated blocking-pool thread via spawn_blocking, feeding
    // lines back to the actor through the same channel the API uses.
    tokio::task::spawn_blocking(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            if let Some(command) = parse_local_feedback(&line) {
                match command {
                    Ok(command) => {
                        if procedure_feedback_tx.blocking_send(command).is_err() {
                            tracing::error!("cognitive loop is no longer accepting procedure feedback");
                            break;
                        }
                    }
                    Err(message) => eprintln!("{message}"),
                }
                continue;
            }
            if let Some(command) = parse_local_outcome(&line) {
                match command {
                    Ok(command) => {
                        if outcome_feedback_tx.blocking_send(command).is_err() {
                            tracing::error!("cognitive loop is no longer accepting observation outcomes");
                            break;
                        }
                    }
                    Err(message) => eprintln!("{message}"),
                }
                continue;
            }
            if input_tx.blocking_send(line).is_err() {
                tracing::error!("cognitive loop is no longer accepting input");
                break;
            }
        }
    })
    .await?;

    Ok(())
}
