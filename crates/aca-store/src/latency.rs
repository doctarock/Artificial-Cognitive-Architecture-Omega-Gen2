use rusqlite::{params, Connection};
use serde::Serialize;
use serde_json::Value;

use crate::types::StoreError;

/// One tick's Telemetry event, kept for the "slowest non-idle cycles" list —
/// the full `payload` (including `phase_ms`, once populated) is retained so
/// an operator can see *which* phase dominated each specific slow cycle,
/// not just that it was slow.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SlowCycle {
    pub cycle_seq: u64,
    pub elapsed_ms: u64,
    pub elapsed_us: u64,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LatencySlice {
    pub sample_count: u64,
    pub p50_elapsed_us: u64,
    pub p95_elapsed_us: u64,
    pub p99_elapsed_us: u64,
    pub max_elapsed_us: u64,
    pub mean_elapsed_us: f64,
    pub under_10ms_percent: f64,
}

impl LatencySlice {
    fn from_samples(samples: &[u64]) -> Self {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let percentile = |p: f64| -> u64 {
            if sorted.is_empty() { return 0; }
            let rank = ((p * sorted.len() as f64).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
            sorted[rank]
        };
        let count = sorted.len() as f64;
        Self {
            sample_count: sorted.len() as u64,
            p50_elapsed_us: percentile(0.50),
            p95_elapsed_us: percentile(0.95),
            p99_elapsed_us: percentile(0.99),
            max_elapsed_us: sorted.last().copied().unwrap_or(0),
            mean_elapsed_us: if sorted.is_empty() { 0.0 } else { sorted.iter().map(|&value| value as f64).sum::<f64>() / count },
            under_10ms_percent: if sorted.is_empty() { 0.0 } else { 100.0 * sorted.iter().filter(|&&value| value < 10_000).count() as f64 / count },
        }
    }
}

/// `scale-strategy.md`'s "Immediate Order" step 2 ("use Telemetry events to
/// identify slowest non-idle cycles"), implemented: p50/p95/p99 of
/// `elapsed_ms` over a recent-cycle window, plus the slowest non-idle
/// cycles in that window with their full payload for inspection.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LatencyReport {
    pub sample_count: u64,
    pub microsecond_sample_count: u64,
    pub p50_elapsed_ms: u64,
    pub p95_elapsed_ms: u64,
    pub p99_elapsed_ms: u64,
    pub max_elapsed_ms: u64,
    pub mean_elapsed_ms: f64,
    pub all_cycles_us: LatencySlice,
    pub compiled_foreground_us: LatencySlice,
    pub other_foreground_us: LatencySlice,
    /// Actor dequeue to provenance-linked Speak, spanning asynchronous work
    /// and scheduler waits between cognitive cycles.
    pub spoken_turn_us: LatencySlice,
    pub compiled_spoken_turn_us: LatencySlice,
    pub curated_spoken_turn_us: LatencySlice,
    pub other_spoken_turn_us: LatencySlice,
    pub attention_decision_count: u64,
    pub attention_timeout_count: u64,
    pub attention_client_error_count: u64,
    pub attention_rejected_vote_count: u64,
    pub communicative_intent_shadow_count: u64,
    pub communicative_intent_agreement_count: u64,
    pub communicative_intent_disagreement_count: u64,
    pub slowest_non_idle_cycles: Vec<SlowCycle>,
}

struct Sample {
    cycle_seq: u64,
    elapsed_ms: u64,
    elapsed_us: u64,
    measured_us: bool,
    non_idle: bool,
    foreground: bool,
    compiled: bool,
    payload: Value,
}

impl LatencyReport {
    /// `recent_cycles` bounds the query to `cycle_seq > max(cycle_seq) -
    /// recent_cycles` — with `idx_cycle_events_phase_cycle_seq` in place,
    /// this is an index range seek, not a full-table scan, and keeps the
    /// row count pulled into memory bounded regardless of total table size.
    /// "Non-idle" mirrors `tick()`'s own `activity_this_tick` philosophy:
    /// a cycle counts if it left anything dirty or held a non-empty Working
    /// Memory, not merely if it ran.
    pub fn from_connection(conn: &Connection, recent_cycles: u64, top_n: usize) -> Result<Self, StoreError> {
        let max_cycle_seq: u64 = conn.query_row("SELECT COALESCE(MAX(cycle_seq), 0) FROM cycle_events", [], |row| row.get::<_, i64>(0))? as u64;
        let floor_cycle_seq = max_cycle_seq.saturating_sub(recent_cycles);

        let mut stmt = conn.prepare("SELECT cycle_seq, payload_json FROM cycle_events WHERE phase = 'telemetry' AND cycle_seq > ?1 ORDER BY cycle_seq")?;
        let rows = stmt.query_map(params![floor_cycle_seq as i64], |row| {
            let cycle_seq: i64 = row.get(0)?;
            let payload_json: String = row.get(1)?;
            Ok((cycle_seq as u64, payload_json))
        })?;

        let mut samples = Vec::new();
        for row in rows {
            let (cycle_seq, payload_json) = row?;
            let payload: Value = serde_json::from_str(&payload_json)?;
            let elapsed_ms = payload.get("elapsed_ms").and_then(Value::as_u64).unwrap_or(0);
            let measured_us = payload.get("elapsed_us").and_then(Value::as_u64);
            let elapsed_us = measured_us.unwrap_or(elapsed_ms.saturating_mul(1_000));
            let dirty_object_count = payload.get("dirty_object_count").and_then(Value::as_u64).unwrap_or(0);
            let working_memory_count = payload.get("working_memory_count").and_then(Value::as_u64).unwrap_or(0);
            let foreground = payload.get("foreground_input").and_then(Value::as_bool).unwrap_or(false);
            let compiled = payload.get("compiled_procedure_hit").and_then(Value::as_bool).unwrap_or(false);
            samples.push(Sample { cycle_seq, elapsed_ms, elapsed_us, measured_us: measured_us.is_some(), non_idle: dirty_object_count > 0 || working_memory_count > 0, foreground, compiled, payload });
        }

        let sample_count = samples.len() as u64;
        let microsecond_sample_count = samples.iter().filter(|sample| sample.measured_us).count() as u64;
        let mut elapsed_sorted: Vec<u64> = samples.iter().map(|s| s.elapsed_ms).collect();
        elapsed_sorted.sort_unstable();
        let percentile = |p: f64| -> u64 {
            if elapsed_sorted.is_empty() {
                return 0;
            }
            let rank = ((p * elapsed_sorted.len() as f64).ceil() as usize).saturating_sub(1).min(elapsed_sorted.len() - 1);
            elapsed_sorted[rank]
        };
        let mean_elapsed_ms = if elapsed_sorted.is_empty() { 0.0 } else { elapsed_sorted.iter().sum::<u64>() as f64 / elapsed_sorted.len() as f64 };
        let max_elapsed_ms = elapsed_sorted.last().copied().unwrap_or(0);
        let all_cycles_us = LatencySlice::from_samples(&samples.iter().filter(|sample| sample.measured_us).map(|sample| sample.elapsed_us).collect::<Vec<_>>());
        let compiled_foreground_us = LatencySlice::from_samples(&samples.iter().filter(|sample| sample.measured_us && sample.foreground && sample.compiled).map(|sample| sample.elapsed_us).collect::<Vec<_>>());
        let other_foreground_us = LatencySlice::from_samples(&samples.iter().filter(|sample| sample.measured_us && sample.foreground && !sample.compiled).map(|sample| sample.elapsed_us).collect::<Vec<_>>());

        let mut spoken_turn_samples = Vec::new();
        let mut compiled_spoken_turn_samples = Vec::new();
        let mut curated_spoken_turn_samples = Vec::new();
        let mut other_spoken_turn_samples = Vec::new();
        let mut act_stmt = conn.prepare("SELECT payload_json FROM cycle_events WHERE phase = 'act' AND cycle_seq > ?1 ORDER BY cycle_seq")?;
        let act_rows = act_stmt.query_map(params![floor_cycle_seq as i64], |row| row.get::<_, String>(0))?;
        for row in act_rows {
            let payload: Value = serde_json::from_str(&row?)?;
            if payload.get("operator").and_then(Value::as_str) != Some("speak") { continue; }
            let Some(elapsed_us) = payload.get("foreground_turn_elapsed_us").and_then(Value::as_u64) else { continue; };
            spoken_turn_samples.push(elapsed_us);
            if payload.get("render_path").and_then(Value::as_str) == Some("CompiledProcedure") {
                compiled_spoken_turn_samples.push(elapsed_us);
            } else if payload.get("render_path").and_then(Value::as_str) == Some("CuratedAnswer") {
                curated_spoken_turn_samples.push(elapsed_us);
            } else {
                other_spoken_turn_samples.push(elapsed_us);
            }
        }
        let spoken_turn_us = LatencySlice::from_samples(&spoken_turn_samples);
        let compiled_spoken_turn_us = LatencySlice::from_samples(&compiled_spoken_turn_samples);
        let curated_spoken_turn_us = LatencySlice::from_samples(&curated_spoken_turn_samples);
        let other_spoken_turn_us = LatencySlice::from_samples(&other_spoken_turn_samples);

        let mut attention_decision_count = 0;
        let mut attention_timeout_count = 0;
        let mut attention_client_error_count = 0;
        let mut attention_rejected_vote_count = 0;
        let mut attention_stmt = conn.prepare("SELECT payload_json FROM cycle_events WHERE phase = 'broadcast' AND cycle_seq > ?1 ORDER BY cycle_seq")?;
        let attention_rows = attention_stmt.query_map(params![floor_cycle_seq as i64], |row| row.get::<_, String>(0))?;
        for row in attention_rows {
            let payload: Value = serde_json::from_str(&row?)?;
            if payload.get("attention_model_decision").and_then(Value::as_bool) == Some(true) {
                attention_decision_count += 1;
            }
            match payload.get("attention_model_fallback").and_then(Value::as_str) {
                Some("timeout") => attention_timeout_count += 1,
                Some("client_error") => attention_client_error_count += 1,
                Some("low_confidence_or_ineligible_target") => attention_rejected_vote_count += 1,
                _ => {}
            }
        }

        let mut communicative_intent_shadow_count = 0;
        let mut communicative_intent_agreement_count = 0;
        let mut communicative_intent_disagreement_count = 0;
        let mut intent_stmt = conn.prepare("SELECT payload_json FROM cycle_events WHERE phase = 'learn' AND cycle_seq > ?1 ORDER BY cycle_seq")?;
        let intent_rows = intent_stmt.query_map(params![floor_cycle_seq as i64], |row| row.get::<_, String>(0))?;
        for row in intent_rows {
            let payload: Value = serde_json::from_str(&row?)?;
            if payload.get("communicative_intent_shadow").and_then(Value::as_bool) != Some(true) { continue; }
            communicative_intent_shadow_count += 1;
            match payload.get("agreement").and_then(Value::as_bool) {
                Some(true) => communicative_intent_agreement_count += 1,
                Some(false) => communicative_intent_disagreement_count += 1,
                None => {}
            }
        }

        let mut non_idle: Vec<&Sample> = samples.iter().filter(|s| s.non_idle).collect();
        non_idle.sort_by(|a, b| b.elapsed_ms.cmp(&a.elapsed_ms));
        let slowest_non_idle_cycles =
            non_idle.into_iter().take(top_n).map(|s| SlowCycle { cycle_seq: s.cycle_seq, elapsed_ms: s.elapsed_ms, elapsed_us: s.elapsed_us, payload: s.payload.clone() }).collect();

        Ok(Self {
            sample_count,
            microsecond_sample_count,
            p50_elapsed_ms: percentile(0.50),
            p95_elapsed_ms: percentile(0.95),
            p99_elapsed_ms: percentile(0.99),
            max_elapsed_ms,
            mean_elapsed_ms,
            all_cycles_us,
            compiled_foreground_us,
            other_foreground_us,
            spoken_turn_us,
            compiled_spoken_turn_us,
            curated_spoken_turn_us,
            other_spoken_turn_us,
            attention_decision_count,
            attention_timeout_count,
            attention_client_error_count,
            attention_rejected_vote_count,
            communicative_intent_shadow_count,
            communicative_intent_agreement_count,
            communicative_intent_disagreement_count,
            slowest_non_idle_cycles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::run_migrations;

    fn insert_telemetry_event(conn: &Connection, cycle_seq: u64, elapsed_ms: u64, dirty_object_count: u64, working_memory_count: u64) {
        let payload = serde_json::json!({
            "elapsed_ms": elapsed_ms,
            "dirty_object_count": dirty_object_count,
            "working_memory_count": working_memory_count,
        });
        conn.execute(
            "INSERT INTO cycle_events (id, cycle_seq, ts, phase, event_type, tier_used, payload_json)
             VALUES (?1, ?2, 0, 'telemetry', 'normal', NULL, ?3)",
            params![format!("evt-{cycle_seq}"), cycle_seq as i64, payload.to_string()],
        )
        .unwrap();
    }

    fn insert_act_event(conn: &Connection, cycle_seq: u64, payload: Value) {
        conn.execute(
            "INSERT INTO cycle_events (id, cycle_seq, ts, phase, event_type, tier_used, payload_json)
             VALUES (?1, ?2, 0, 'act', 'normal', NULL, ?3)",
            params![format!("act-{cycle_seq}"), cycle_seq as i64, payload.to_string()],
        ).unwrap();
    }

    fn insert_broadcast_event(conn: &Connection, cycle_seq: u64, payload: Value) {
        conn.execute(
            "INSERT INTO cycle_events (id, cycle_seq, ts, phase, event_type, tier_used, payload_json)
             VALUES (?1, ?2, 0, 'broadcast', 'normal', NULL, ?3)",
            params![format!("broadcast-{cycle_seq}"), cycle_seq as i64, payload.to_string()],
        ).unwrap();
    }

    fn insert_learn_event(conn: &Connection, cycle_seq: u64, payload: Value) {
        conn.execute(
            "INSERT INTO cycle_events (id, cycle_seq, ts, phase, event_type, tier_used, payload_json)
             VALUES (?1, ?2, 0, 'learn', 'normal', NULL, ?3)",
            params![format!("learn-{cycle_seq}"), cycle_seq as i64, payload.to_string()],
        ).unwrap();
    }

    #[test]
    fn report_counts_communicative_intent_shadow_agreement_separately() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        insert_telemetry_event(&conn, 1, 1, 1, 1);
        insert_learn_event(&conn, 1, serde_json::json!({"communicative_intent_shadow": true, "agreement": true}));
        insert_learn_event(&conn, 2, serde_json::json!({"communicative_intent_shadow": true, "agreement": false}));
        insert_learn_event(&conn, 3, serde_json::json!({"other": true}));
        let report = LatencyReport::from_connection(&conn, 10, 5).unwrap();
        assert_eq!(report.communicative_intent_shadow_count, 2);
        assert_eq!(report.communicative_intent_agreement_count, 1);
        assert_eq!(report.communicative_intent_disagreement_count, 1);
    }

    #[test]
    fn report_distinguishes_accepted_attention_votes_from_fallback_reasons() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        insert_telemetry_event(&conn, 1, 1, 1, 1);
        insert_broadcast_event(&conn, 1, serde_json::json!({"attention_model_decision": true}));
        insert_broadcast_event(&conn, 2, serde_json::json!({"attention_model_fallback": "timeout"}));
        insert_broadcast_event(&conn, 3, serde_json::json!({"attention_model_fallback": "client_error"}));
        insert_broadcast_event(&conn, 4, serde_json::json!({"attention_model_fallback": "low_confidence_or_ineligible_target"}));
        let report = LatencyReport::from_connection(&conn, 10, 5).unwrap();
        assert_eq!(report.attention_decision_count, 1);
        assert_eq!(report.attention_timeout_count, 1);
        assert_eq!(report.attention_client_error_count, 1);
        assert_eq!(report.attention_rejected_vote_count, 1);
    }

    #[test]
    fn spoken_turn_latency_spans_cycles_and_separates_compiled_curated_and_other_speech() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        insert_telemetry_event(&conn, 1, 3, 1, 0);
        insert_telemetry_event(&conn, 2, 4, 1, 0);
        insert_telemetry_event(&conn, 3, 5, 1, 0);
        insert_telemetry_event(&conn, 4, 6, 1, 0);
        insert_act_event(&conn, 1, serde_json::json!({"operator": "speak", "render_path": "CompiledProcedure", "foreground_turn_elapsed_us": 3_000}));
        insert_act_event(&conn, 2, serde_json::json!({"operator": "speak", "render_path": "SocialRendering", "foreground_turn_elapsed_us": 200_000}));
        insert_act_event(&conn, 3, serde_json::json!({"operator": "speak", "render_path": "CompiledProcedure"}));
        insert_act_event(&conn, 4, serde_json::json!({"operator": "speak", "render_path": "CuratedAnswer", "foreground_turn_elapsed_us": 4_000}));
        let report = LatencyReport::from_connection(&conn, 10, 5).unwrap();
        assert_eq!(report.spoken_turn_us.sample_count, 3);
        assert_eq!(report.compiled_spoken_turn_us.sample_count, 1);
        assert_eq!(report.compiled_spoken_turn_us.p50_elapsed_us, 3_000);
        assert_eq!(report.curated_spoken_turn_us.sample_count, 1);
        assert_eq!(report.curated_spoken_turn_us.p50_elapsed_us, 4_000);
        assert_eq!(report.other_spoken_turn_us.sample_count, 1);
        assert_eq!(report.other_spoken_turn_us.p50_elapsed_us, 200_000);
    }

    #[test]
    fn percentiles_and_slow_list_over_a_small_fixed_sample() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();

        // Idle ticks (no dirty ids, empty WM) mixed with non-idle ones -
        // the idle ones must count toward percentiles but never appear in
        // `slowest_non_idle_cycles`.
        insert_telemetry_event(&conn, 1, 10, 0, 0);
        insert_telemetry_event(&conn, 2, 500, 1, 1);
        insert_telemetry_event(&conn, 3, 20, 0, 0);
        insert_telemetry_event(&conn, 4, 100, 2, 0);
        insert_telemetry_event(&conn, 5, 5, 0, 0);

        let report = LatencyReport::from_connection(&conn, 25_000, 10).unwrap();
        assert_eq!(report.sample_count, 5);
        assert_eq!(report.microsecond_sample_count, 0, "legacy millisecond rows must not masquerade as precise measurements");
        assert_eq!(report.all_cycles_us.sample_count, 0);
        assert_eq!(report.max_elapsed_ms, 500);
        assert_eq!(report.mean_elapsed_ms, (10 + 500 + 20 + 100 + 5) as f64 / 5.0);

        assert_eq!(report.slowest_non_idle_cycles.len(), 2);
        assert_eq!(report.slowest_non_idle_cycles[0].cycle_seq, 2);
        assert_eq!(report.slowest_non_idle_cycles[0].elapsed_ms, 500);
        assert_eq!(report.slowest_non_idle_cycles[1].cycle_seq, 4);
    }

    #[test]
    fn recent_cycles_window_excludes_older_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();

        insert_telemetry_event(&conn, 1, 9_999, 1, 0); // outside the window once cycle 10 exists
        for seq in 2..=10 {
            insert_telemetry_event(&conn, seq, 10, 1, 0);
        }

        let report = LatencyReport::from_connection(&conn, 5, 10).unwrap();
        assert_eq!(report.sample_count, 5, "only cycle_seq > 10 - 5 = 5 should be included");
        assert_eq!(report.max_elapsed_ms, 10, "the far-older slow cycle 1 must fall outside the window");
    }

    #[test]
    fn empty_table_reports_zeroed_percentiles_not_an_error() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();

        let report = LatencyReport::from_connection(&conn, 25_000, 10).unwrap();
        assert_eq!(report.sample_count, 0);
        assert_eq!(report.p50_elapsed_ms, 0);
        assert_eq!(report.p99_elapsed_ms, 0);
        assert!(report.slowest_non_idle_cycles.is_empty());
    }

    #[test]
    fn microsecond_report_separates_compiled_and_other_foreground_turns() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        for (seq, elapsed_us, compiled) in [(1, 2_500, true), (2, 12_000, true), (3, 200_000, false)] {
            let payload = serde_json::json!({
                "elapsed_ms": elapsed_us / 1_000,
                "elapsed_us": elapsed_us,
                "foreground_input": true,
                "compiled_procedure_hit": compiled,
                "dirty_object_count": 1,
                "working_memory_count": 1,
            });
            conn.execute(
                "INSERT INTO cycle_events (id, cycle_seq, ts, phase, event_type, tier_used, payload_json)
                 VALUES (?1, ?2, 0, 'telemetry', 'normal', NULL, ?3)",
                params![format!("fast-{seq}"), seq, payload.to_string()],
            ).unwrap();
        }
        let report = LatencyReport::from_connection(&conn, 10, 3).unwrap();
        assert_eq!(report.compiled_foreground_us.sample_count, 2);
        assert_eq!(report.microsecond_sample_count, 3);
        assert_eq!(report.compiled_foreground_us.p50_elapsed_us, 2_500);
        assert_eq!(report.compiled_foreground_us.p95_elapsed_us, 12_000);
        assert_eq!(report.compiled_foreground_us.under_10ms_percent, 50.0);
        assert_eq!(report.other_foreground_us.sample_count, 1);
        assert_eq!(report.other_foreground_us.p50_elapsed_us, 200_000);
        assert_eq!(report.all_cycles_us.sample_count, 3);
        assert_eq!(report.slowest_non_idle_cycles[0].elapsed_us, 200_000);
    }
}
