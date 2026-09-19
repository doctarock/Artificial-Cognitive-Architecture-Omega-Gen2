# Registers clear-ollama-leak.ps1 as a Scheduled Task ("this user only") that
# checks every 5 minutes and restarts only the Ollama instance(s) that have
# actually crossed the leak threshold - see that script's own header for why
# 5 minutes. Re-running this is safe: -Force replaces the existing task
# definition rather than erroring if it's already registered.

$scriptPath = Join-Path $PSScriptRoot "clear-ollama-leak.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration ([TimeSpan]::MaxValue)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "OmegaACA-OllamaLeakGuard" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Restarts Ollama instances whose CLOSE_WAIT/FIN_WAIT2 socket count threatens Windows' ephemeral port range - see clear-ollama-leak.ps1." `
    -Force | Out-Null

Write-Output "Registered OmegaACA-OllamaLeakGuard (every 5 min)."
