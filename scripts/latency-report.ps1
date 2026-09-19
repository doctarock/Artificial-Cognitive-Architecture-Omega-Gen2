param(
    [string]$DatabasePath = "omega.sqlite3",
    [int]$RecentCycles = 25000,
    [int]$Top = 20
)

$ErrorActionPreference = "Stop"

cargo run -q -p aca-store --example latency_report -- $DatabasePath --recent-cycles $RecentCycles --top $Top
