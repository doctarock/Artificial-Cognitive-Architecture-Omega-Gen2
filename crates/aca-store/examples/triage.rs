use std::env;

use aca_store::StoreTriageReport;
use rusqlite::Connection;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).unwrap_or_else(|| "omega.sqlite3".to_string());
    let conn = Connection::open_with_flags(
        &path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )?;
    let report = StoreTriageReport::from_connection(&conn)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
