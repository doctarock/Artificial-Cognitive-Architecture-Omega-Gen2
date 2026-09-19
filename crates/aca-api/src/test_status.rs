//! `GET /tests/status`: runs `cargo test --workspace` and reports pass/fail
//! per test binary. Deliberately not polled automatically the way
//! `/snapshot/overview` is - a full workspace test run takes real seconds,
//! not milliseconds, so this is a caller-triggered ("click a button"), not
//! background-polled, endpoint. Loopback-only by construction (this crate
//! is never bound beyond `127.0.0.1` - see `omega-acad::main`), unlike
//! `aca-mcp`'s LAN-facing surface: shelling out to `cargo test` on any
//! request reaching the process is a local-dev capability, not something to
//! expose to other household agents.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use tokio::process::Command;

use crate::server::ApiState;

/// A dead-connection-only backstop, same reasoning as every per-call
/// timeout elsewhere in this codebase - `cargo test` normally finishes in
/// single-digit seconds once already built, but a cold/full rebuild
/// shouldn't be allowed to hang the request forever.
const TEST_RUN_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TestBinaryResult {
    pub name: String,
    pub passed: u32,
    pub failed: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TestStatus {
    pub all_passing: bool,
    pub total_passed: u32,
    pub total_failed: u32,
    pub binaries: Vec<TestBinaryResult>,
    pub ran_at_epoch_ms: i64,
    pub duration_ms: u64,
}

pub async fn get_test_status(State(_state): State<ApiState>) -> impl IntoResponse {
    let started = Instant::now();
    let ran_at_epoch_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);

    // `cargo test` splits its output across streams: build/status lines
    // ("Compiling...", "Running...") go to stderr, the test harness's own
    // output ("test result: ...") goes to stdout. `Command::output()`
    // captures them as two separate buffers with no cross-stream ordering
    // preserved, so naively concatenating stdout-then-stderr put every
    // "Running" line *after* every "test result" line - every binary name
    // parsed as "unknown" (confirmed live). Routing through `cmd /C ...
    // 2>&1` merges the streams at the OS level, in real chronological
    // order, before Rust ever sees them - the same fix `2>&1` always is.
    let run = tokio::time::timeout(TEST_RUN_TIMEOUT, Command::new("cmd").args(["/C", "cargo test --workspace 2>&1"]).output()).await;

    let output = match run {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("failed to spawn cargo test: {err}")}))).into_response();
        }
        Err(_) => {
            return (StatusCode::GATEWAY_TIMEOUT, Json(serde_json::json!({"error": format!("cargo test did not finish within {TEST_RUN_TIMEOUT:?}")}))).into_response();
        }
    };

    let combined = String::from_utf8_lossy(&output.stdout).into_owned();
    let binaries = parse_cargo_test_output(&combined);
    let total_passed: u32 = binaries.iter().map(|b| b.passed).sum();
    let total_failed: u32 = binaries.iter().map(|b| b.failed).sum();

    let status = TestStatus {
        all_passing: output.status.success() && total_failed == 0 && !binaries.is_empty(),
        total_passed,
        total_failed,
        binaries,
        ran_at_epoch_ms,
        duration_ms: started.elapsed().as_millis() as u64,
    };
    Json(status).into_response()
}

/// Parses `cargo test`'s human-readable output (there is no stable
/// structured format on the stable toolchain this workspace targets - the
/// JSON test formatter is nightly-only) into one result per test binary.
/// Tracks "what binary are we currently inside" from `Running unittests
/// .../CRATE-HASH.exe`, `Running tests\FILE.rs (...)`, and `Doc-tests
/// CRATE` lines, then attaches the next `test result: ...` summary line to
/// whichever binary was most recently announced. A `test result:` line
/// with no preceding "Running"/"Doc-tests" line (shouldn't happen in
/// practice) is attributed to "unknown" rather than dropped, so a parser
/// mismatch shows up as a visibly odd binary name instead of a silently
/// undercounted total.
fn parse_cargo_test_output(output: &str) -> Vec<TestBinaryResult> {
    let mut results = Vec::new();
    let mut current_name: Option<String> = None;

    for raw_line in output.lines() {
        let line = raw_line.trim();
        if let Some(name) = extract_binary_name(line) {
            current_name = Some(name);
        } else if line.starts_with("test result:")
            && let Some((passed, failed)) = parse_pass_fail_counts(line)
        {
            results.push(TestBinaryResult { name: current_name.clone().unwrap_or_else(|| "unknown".to_string()), passed, failed });
        }
    }
    results
}

fn extract_binary_name(line: &str) -> Option<String> {
    if let Some(rest) = line.strip_prefix("Doc-tests ") {
        return Some(format!("{} (doctests)", rest.trim()));
    }
    if let Some(rest) = line.strip_prefix("Running unittests ") {
        // e.g. "src\lib.rs (target\debug\deps\aca_engine-fcb15cd6b95b7dd7.exe)"
        let inner = rest.split('(').nth(1)?.trim_end_matches(')');
        let filename = inner.rsplit(['\\', '/']).next()?;
        let stem = filename.rsplit_once('-').map(|(n, _)| n).unwrap_or(filename);
        return Some(stem.replace('_', "-"));
    }
    if let Some(rest) = line.strip_prefix("Running tests") {
        // e.g. "\full_cycle.rs (target\debug\deps\full_cycle-....exe)"
        let file_part = rest.trim_start_matches(['\\', '/']);
        let name = file_part.split_whitespace().next()?.trim_end_matches(".rs");
        return Some(name.to_string());
    }
    None
}

fn parse_pass_fail_counts(line: &str) -> Option<(u32, u32)> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut passed = None;
    let mut failed = None;
    for i in 1..tokens.len() {
        let normalized = tokens[i].trim_end_matches([';', '.']);
        if normalized == "passed" {
            passed = tokens[i - 1].trim_end_matches([';', '.']).parse::<u32>().ok();
        } else if normalized == "failed" {
            failed = tokens[i - 1].trim_end_matches([';', '.']).parse::<u32>().ok();
        }
    }
    Some((passed?, failed?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_OUTPUT: &str = r#"
   Compiling aca-engine v0.1.0 (E:\AI\omega-aca\crates\aca-engine)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 2.31s
     Running unittests src\lib.rs (target\debug\deps\aca_engine-fcb15cd6b95b7dd7.exe)

running 120 tests
test steps::act::tests::ignore_resolves_to_silence ... ok

test result: ok. 120 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s

     Running tests\full_cycle.rs (target\debug\deps\full_cycle-e7618a56c75d93e6.exe)

running 1 test
test one_full_cycle_speaks_forms_memory_and_reinforces_an_edge ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests aca_engine

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"#;

    const SAMPLE_WITH_FAILURE: &str = r#"
     Running unittests src\lib.rs (target\debug\deps\aca_store-8093915a7be18790.exe)

running 14 tests
test dao::tests::cycle_event_round_trips ... FAILED

failures:

---- dao::tests::cycle_event_round_trips stdout ----
thread panicked

test result: FAILED. 13 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
"#;

    #[test]
    fn parses_a_clean_multi_binary_run() {
        let binaries = parse_cargo_test_output(SAMPLE_OUTPUT);
        assert_eq!(
            binaries,
            vec![
                TestBinaryResult { name: "aca-engine".to_string(), passed: 120, failed: 0 },
                TestBinaryResult { name: "full_cycle".to_string(), passed: 1, failed: 0 },
                TestBinaryResult { name: "aca_engine (doctests)".to_string(), passed: 0, failed: 0 },
            ]
        );
    }

    #[test]
    fn parses_a_failing_run() {
        let binaries = parse_cargo_test_output(SAMPLE_WITH_FAILURE);
        assert_eq!(binaries, vec![TestBinaryResult { name: "aca-store".to_string(), passed: 13, failed: 1 }]);
    }

    #[test]
    fn empty_output_yields_no_binaries() {
        assert!(parse_cargo_test_output("").is_empty());
    }

    #[test]
    fn a_test_result_line_with_no_preceding_running_line_is_attributed_to_unknown() {
        let binaries = parse_cargo_test_output("test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s");
        assert_eq!(binaries, vec![TestBinaryResult { name: "unknown".to_string(), passed: 5, failed: 0 }]);
    }
}
