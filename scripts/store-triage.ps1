param(
    [string]$DatabasePath = "omega.sqlite3"
)

$ErrorActionPreference = "Stop"

cargo run -q -p aca-store --example triage -- $DatabasePath
