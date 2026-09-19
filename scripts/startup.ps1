# omega-aca full-stack startup - registered as a Scheduled Task ("At log on",
# this user only) via scripts\register-startup.ps1. Brings up every process
# Omega's cognitive loop depends on that doesn't already have its own
# auto-start entry:
#   - the P4-pinned Ollama instance (Tier1 models, port 11437) - the default
#     Ollama instance (port 11434, the 4070) already auto-starts via its own
#     Startup-folder shortcut, so it's left alone here.
#   - the voice interaction service (Piper TTS, port 8790).
#   - omega-acad itself, started last, once everything it talks to is up.
#
# Every step checks first and skips if already up, so re-running this by
# hand (or a flaky double-trigger) is always safe - it only ever starts what
# isn't already running.

function Test-Url {
    param([string]$Url)
    try { Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop | Out-Null; return $true }
    catch {
        # Any real HTTP response (even a 404) means something is listening -
        # only a connection-level failure (nothing bound to the port) counts
        # as "not up." Checked structurally (a `Response` property that's
        # actually set) rather than by exception type: Invoke-WebRequest
        # throws System.Net.WebException under Windows PowerShell 5.1 but
        # Microsoft.PowerShell.Commands.HttpResponseException under pwsh 7+
        # for the identical non-2xx-response case, both with a `.Response` -
        # matching only the 5.1 type here used to fall through to "not up"
        # under pwsh, spawning a second, port-colliding process alongside an
        # already-running one that merely answered with a non-2xx status.
        if ($_.Exception.PSObject.Properties['Response'] -and $_.Exception.Response) { return $true }
        return $false
    }
}

# --- Tier1: P4-pinned Ollama (port 11437) ---
if (-not (Test-Url "http://127.0.0.1:11437/api/tags")) {
    $env:OLLAMA_HOST = "127.0.0.1:11437"
    $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    $env:CUDA_VISIBLE_DEVICES = "1"
    Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
    Remove-Item Env:\OLLAMA_HOST, Env:\CUDA_DEVICE_ORDER, Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
}

# --- Voice interaction service (port 8790) ---
if (-not (Test-Url "http://127.0.0.1:8790/")) {
    $voiceDir = "E:\AI\Voice Interaction"
    $env:PYTHONPATH = Join-Path $voiceDir "src"
    $env:VOICE_INTERACTION_PIPER_COMMAND = '"E:\AI\omega-observer\vendor\piper\windows-x64\piper.exe" --model "E:\AI\omega-observer\vendor\piper\voices\en_US-danny-low.onnx" --output_raw --quiet'
    $env:VOICE_INTERACTION_PIPER_SAMPLE_RATE = "16000"
    Start-Process -FilePath (Join-Path $voiceDir ".venv-f5\Scripts\python.exe") `
        -ArgumentList "-m voice_interaction.cli serve --host 127.0.0.1 --port 8790 --storage-dir data-api" `
        -WorkingDirectory $voiceDir -WindowStyle Hidden
    Remove-Item Env:\PYTHONPATH, Env:\VOICE_INTERACTION_PIPER_COMMAND, Env:\VOICE_INTERACTION_PIPER_SAMPLE_RATE -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
}

# --- omega-acad itself ---
# Deliberately NOT hidden and NOT redirected: its input loop does a blocking
# `stdin.lock().lines()` read (crates/omega-acad/src/main.rs), and a
# redirected/inherited-null stdin hits immediate EOF, ending that loop and
# exiting the whole process right after it finishes logging startup -
# confirmed live 2026-08-10 (it ran, logged every "configured X" line, then
# quit clean before the API ever answered a request). A real console avoids
# that and also matches how it's normally driven, typing lines into it
# directly.
if (-not (Get-Process -Name "omega-acad" -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath "E:\AI\omega-aca\target\debug\omega-acad.exe" -WorkingDirectory "E:\AI\omega-aca"
}
