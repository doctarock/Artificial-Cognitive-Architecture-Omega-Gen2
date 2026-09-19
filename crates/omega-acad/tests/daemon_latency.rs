use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aca_engine::{CycleEvent, CyclePhase};
use futures_util::StreamExt;
use tokio::process::Command;
use tokio_tungstenite::connect_async;

fn free_loopback_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Run explicitly with `cargo test -p omega-acad daemon_loopback_latency_probe
/// -- --ignored --nocapture`. This launches the real daemon executable and
/// crosses its HTTP/WebSocket process boundary. Model/media endpoints are
/// explicitly blanked. `OMEGA_PROBE_ATTENTION_SHADOW=1` optionally permits
/// only a separately supplied loopback attention URL; non-loopback is refused.
#[tokio::test]
#[ignore = "launches a local daemon process; performance varies with host load"]
async fn daemon_loopback_latency_probe() {
    let api_port = free_loopback_port();
    let mcp_port = free_loopback_port();
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut db_path = PathBuf::from(std::env::temp_dir());
    db_path.push(format!("omega-daemon-latency-{}-{unique}.sqlite3", std::process::id()));

    let mut command = Command::new(env!("CARGO_BIN_EXE_omega-acad"));
    command
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null())
        .env("OMEGA_DB_PATH", &db_path)
        .env("OMEGA_API_PORT", api_port.to_string())
        .env("OMEGA_MCP_PORT", mcp_port.to_string())
        .env("OMEGA_MCP_HOST", "127.0.0.1");
    let shadow_attention = std::env::var("OMEGA_PROBE_ATTENTION_SHADOW").ok().as_deref() == Some("1");
    if shadow_attention {
        let base_url = std::env::var("OMEGA_ATTENTION_MODEL_BASE_URL").expect("shadow probe requires attention base URL");
        let model = std::env::var("OMEGA_ATTENTION_MODEL").expect("shadow probe requires attention model");
        let parsed = reqwest::Url::parse(&base_url).expect("attention base URL must parse");
        assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
            "daemon latency probe refuses non-loopback attention traffic");
        command.env("OMEGA_ATTENTION_MODE", "shadow")
            .env("OMEGA_ATTENTION_MODEL_BASE_URL", base_url)
            .env("OMEGA_ATTENTION_MODEL", model);
    } else {
        command.env("OMEGA_ATTENTION_MODE", "off")
            .env("OMEGA_ATTENTION_MODEL_BASE_URL", "")
            .env("OMEGA_ATTENTION_MODEL", "");
    }
    for name in [
        "OMEGA_EMBEDDING_BASE_URL", "OMEGA_EMBEDDING_MODEL", "OMEGA_EMBEDDING_API_KEY",
        "OMEGA_TIER3_BASE_URL", "OMEGA_TIER3_MODEL", "OMEGA_TIER3_API_KEY",
        "OMEGA_TIER4_BASE_URL", "OMEGA_TIER4_MODEL", "OMEGA_TIER4_API_KEY",
        "OMEGA_VOICE_BASE_URL", "OMEGA_VIDEO_BASE_URL", "OMEGA_LIBRARY_PATH",
        "OMEGA_ORIENT_OUTCOME_MODEL_PATH",
    ] { command.env(name, ""); }
    for tier in ["TIER1", "TIER2"] {
        for slot in 1..=5 {
            command.env(format!("OMEGA_{tier}_MODEL_{slot}"), "");
            command.env(format!("OMEGA_{tier}_MODEL_{slot}_BASE_URL"), "");
            command.env(format!("OMEGA_{tier}_MODEL_{slot}_API_KEY"), "");
        }
    }
    let mut child = command.spawn().expect("real omega-acad daemon should launch");
    let _stdin_guard = child.stdin.take().expect("stdin stays open so the daemon remains alive");
    let client = reqwest::Client::new();
    let health_url = format!("http://127.0.0.1:{api_port}/health");
    let startup_deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if client.get(&health_url).send().await.is_ok() { break; }
        assert!(Instant::now() < startup_deadline, "daemon health endpoint did not start");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let (mut events, _) = connect_async(format!("ws://127.0.0.1:{api_port}/events")).await.unwrap();
    let mut samples_us = Vec::new();
    let mut shadow_decisions = 0usize;
    let mut shadow_timeouts = 0usize;
    for greeting in ["Hi", "Hello", "Hey", "Hi Omega", "Hello Omega", "Hey Omega"] {
        let started = Instant::now();
        let response = client.post(format!("http://127.0.0.1:{api_port}/input"))
            .json(&serde_json::json!({"text": greeting})).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::ACCEPTED);
        loop {
            let message = tokio::time::timeout(Duration::from_secs(5), events.next())
                .await.expect("daemon speech event timed out")
                .expect("daemon event socket closed").expect("daemon event socket failed");
            if !message.is_text() { continue; }
            let event: CycleEvent = serde_json::from_str(message.to_text().unwrap()).unwrap();
            if event.payload["attention_model_decision"] == true {
                assert_eq!(event.payload["attention_shadow_only"], true,
                    "shadow probe must never permit specialist control");
                shadow_decisions += 1;
            }
            if event.payload["attention_model_fallback"] == "timeout" { shadow_timeouts += 1; }
            if event.phase == CyclePhase::Act && event.payload["operator"] == "speak" {
                samples_us.push(started.elapsed().as_micros() as u64);
                break;
            }
        }
    }
    samples_us.sort_unstable();
    println!("real daemon loopback innate turns: mode={}, n=6, p50_us={}, max_us={}, accepted_shadow_decisions={shadow_decisions}, attention_timeouts={shadow_timeouts}",
        if shadow_attention { "attention-shadow" } else { "attention-off" }, samples_us[2], samples_us[5]);
    assert_eq!(samples_us.len(), 6);
    if shadow_attention { assert!(shadow_decisions + shadow_timeouts > 0, "shadow specialist should emit a decision or explicit timeout"); }
    child.kill().await.ok();
    child.wait().await.ok();
    for suffix in ["", "-wal", "-shm"] {
        let candidate = PathBuf::from(format!("{}{suffix}", db_path.display()));
        if candidate.exists() { std::fs::remove_file(candidate).ok(); }
    }
}
