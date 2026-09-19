# Restarts any Ollama instance whose CLOSE_WAIT/FIN_WAIT2 socket count has
# grown large enough to threaten Windows' ephemeral port range (16384 ports,
# 49152-65535) - Ollama's Windows build leaks a socket into one of those
# states per cancelled/abandoned request, and the Tier1 divergent pool
# (crates/omega-acad/src/main.rs) cancels losing candidates every cognitive
# cycle, so this grows steadily. Full exhaustion breaks every new local TCP
# connection on the machine, surfacing in the viz as `[input not sent] HTTP
# 0` (see api_client.gd) - this script keeps that from ever being reached by
# restarting an instance well before its leak count gets there.
#
# Both Ollama instances this stack uses are covered (see startup.ps1):
#   - default (port 11434, GPU 0 / 4070) - auto-started by its own
#     Startup-folder shortcut, so a plain `ollama.exe serve` restart matches
#     how it normally comes up.
#   - P4-pinned (port 11437, GPU 1) - needs OLLAMA_HOST/CUDA_* set the same
#     way startup.ps1 sets them, or the restarted instance would come back
#     on the wrong GPU/port.
#
# Safe to run unattended on a schedule: does nothing when nothing is leaking.
#
# Threshold/cadence are tuned against an observed leak rate of ~370
# sockets/min on the 11434 instance under normal cognitive-loop load (Tier2
# + Attention both point at it - see .env) - confirmed live 2026-08-20: a
# freshly-restarted instance re-accumulated 5571 leaked sockets in ~15
# minutes. At that rate, unmitigated exhaustion of the whole 16384-port
# dynamic range takes well under an hour, so this is registered (see
# register-ollama-leak-guard.ps1) to run every 5 minutes.

param(
    [int]$Threshold = 3000,
    [string]$LogPath = (Join-Path $PSScriptRoot "ollama-leak-guard.log")
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $Message
    Add-Content -Path $LogPath -Value $line
}

$ollamaProcs = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcs) {
    Write-Log "no ollama.exe process running - nothing to check"
    exit 0
}

$listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -in $ollamaProcs.Id }
$leakStates = Get-NetTCPConnection -State CloseWait, FinWait2 -ErrorAction SilentlyContinue |
    Group-Object OwningProcess -AsHashTable -AsString

foreach ($proc in $ollamaProcs) {
    $port = ($listeners | Where-Object { $_.OwningProcess -eq $proc.Id } | Select-Object -First 1).LocalPort
    if (-not $port) {
        continue
    }
    $leakCount = 0
    # $leakStates is $null (not an empty hashtable) when zero connections are
    # in CloseWait/FinWait2 anywhere on the system - the common case this
    # script is meant to be a safe no-op for. Calling .ContainsKey() on that
    # $null under $ErrorActionPreference = "Stop" would crash the script
    # instead of the intended silent no-op.
    if ($leakStates -and $leakStates.ContainsKey("$($proc.Id)")) {
        $leakCount = $leakStates["$($proc.Id)"].Count
    }
    if ($leakCount -lt $Threshold) {
        continue
    }

    Write-Log "port $port (pid $($proc.Id)) has $leakCount leaked sockets (>= $Threshold) - restarting"
    Stop-Process -Id $proc.Id -Force
    Start-Sleep -Milliseconds 500

    if ($port -eq 11437) {
        $env:OLLAMA_HOST = "127.0.0.1:11437"
        $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
        $env:CUDA_VISIBLE_DEVICES = "1"
        Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
        Remove-Item Env:\OLLAMA_HOST, Env:\CUDA_DEVICE_ORDER, Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    } else {
        Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
    }
    Write-Log "port $port restarted"
}
