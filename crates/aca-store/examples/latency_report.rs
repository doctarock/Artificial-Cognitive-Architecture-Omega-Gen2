use std::env;

use aca_store::LatencyReport;
use rusqlite::Connection;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "omega.sqlite3".to_string());
    let mut recent_cycles = 25_000u64;
    let mut top_n = 20usize;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--recent-cycles" => recent_cycles = args.next().ok_or("--recent-cycles requires a value")?.parse()?,
            "--top" => top_n = args.next().ok_or("--top requires a value")?.parse()?,
            other => return Err(format!("unrecognized flag: {other}").into()),
        }
    }

    let conn = Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI)?;
    let report = LatencyReport::from_connection(&conn, recent_cycles, top_n)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
