use std::env;

use aca_store::{CycleEventRetention, SqliteStore};
use rusqlite::Connection;

fn arg_u64(name: &str, default: u64) -> u64 {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn current_cycle_seq(path: &str) -> Result<u64, Box<dyn std::error::Error>> {
    if let Ok(value) = env::var("OMEGA_CURRENT_CYCLE_SEQ") {
        return Ok(value.parse()?);
    }
    let conn = Connection::open(path)?;
    let max_seq: i64 = conn.query_row("SELECT COALESCE(MAX(cycle_seq), 0) FROM cycle_events", [], |row| row.get(0))?;
    Ok(max_seq as u64)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).unwrap_or_else(|| "omega.sqlite3".to_string());
    let current_cycle_seq = current_cycle_seq(&path)?;
    let retention = CycleEventRetention {
        keep_recent_cycles: arg_u64("OMEGA_KEEP_RECENT_CYCLES", CycleEventRetention::default().keep_recent_cycles),
        keep_abnormal_cycles: arg_u64("OMEGA_KEEP_ABNORMAL_CYCLES", CycleEventRetention::default().keep_abnormal_cycles),
    };

    let store = SqliteStore::open(&path)?;
    let report = store.prune_cycle_events(current_cycle_seq, retention).await?;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "database": path,
        "current_cycle_seq": current_cycle_seq,
        "keep_recent_cycles": retention.keep_recent_cycles,
        "keep_abnormal_cycles": retention.keep_abnormal_cycles,
        "deleted_normal_events": report.deleted_normal_events,
        "deleted_abnormal_events": report.deleted_abnormal_events,
        "deleted_total": report.deleted_total(),
    }))?);
    Ok(())
}
