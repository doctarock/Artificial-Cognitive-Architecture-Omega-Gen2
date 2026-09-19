param(
    [string]$DatabasePath = "omega.sqlite3",
    [UInt64]$CurrentCycleSeq = 0,
    [UInt64]$KeepRecentCycles = 25000,
    [UInt64]$KeepAbnormalCycles = 250000
)

$ErrorActionPreference = "Stop"

if ($CurrentCycleSeq -gt 0) {
    $env:OMEGA_CURRENT_CYCLE_SEQ = "$CurrentCycleSeq"
} else {
    Remove-Item Env:\OMEGA_CURRENT_CYCLE_SEQ -ErrorAction SilentlyContinue
}
$env:OMEGA_KEEP_RECENT_CYCLES = "$KeepRecentCycles"
$env:OMEGA_KEEP_ABNORMAL_CYCLES = "$KeepAbnormalCycles"

cargo run -q -p aca-store --example prune_cycle_events -- $DatabasePath
