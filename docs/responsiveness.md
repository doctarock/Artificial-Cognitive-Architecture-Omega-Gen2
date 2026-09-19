# Responsiveness Runbook

Omega keeps the cognitive architecture intact, but foreground turns now get priority over background cognition.

## What Changed

- `ResponsivenessConfig.defer_synthesis_on_foreground_input`
  - Default: `true`
  - Due pattern synthesis stays queued when a human/room/external-agent turn arrives.
  - The queued synthesis runs again on a later idle tick.

- `ResponsivenessConfig.defer_new_agenda_spawns_on_foreground_input`
  - Default: `true`
  - Existing agenda maintenance still runs, but new autonomous intention spawning waits while foreground input is being processed.

- `Telemetry` events now include:
  - `foreground_input`
  - `deferred.synthesis`
  - `deferred.agenda_spawn`
  - `elapsed_ms`
  - `input_token_estimate`
  - tier capacity/busy state

- Act speech events include `render_path`:
  - `Verbatim`
  - `Tier1`
  - `CompiledSkill`

## Event Cleanup

Inspect:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\store-triage.ps1 omega.sqlite3
```

Prune routine events while preserving recent and abnormal diagnostics:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prune-cycle-events.ps1 omega.sqlite3
```

Optional tuning:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prune-cycle-events.ps1 omega.sqlite3 -KeepRecentCycles 50000 -KeepAbnormalCycles 500000
```

## Read The Results

Healthy responsiveness should show:

- lower p95 `Telemetry.elapsed_ms` on foreground turns
- fewer foreground ticks with `deferred.* = false` while background work is due
- fewer routine `cycle_events` retained after pruning
- no regression in `behavior_regression.rs`
- `CompiledSkill` and `Verbatim` paths increasing for routine speech over time
