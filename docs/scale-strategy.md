# Scale Strategy

Omega should scale the processors around the cognitive architecture, not split the identity-bearing architecture itself.

## Keep Single-Owner

- `CognitiveLoopActor`
- Working Memory membership
- activation graph mutation
- Executive operator selection
- Tier 3/Tier 4 escalation seats

Reason: these are coherence boundaries. Scaling them by replication risks two minds racing over one memory.

## Scale Horizontally First

| Component | Scale shape | Why |
| --- | --- | --- |
| Tier 1 divergent clients | more independent small models/hosts | cheap agreement improves confidence and absorbs slow candidates |
| Tier 2 divergent clients | 2-3 mid models/hosts | keeps escalation latency bounded when Tier 1 disagrees |
| Embedding service | local batchable worker or replicated endpoint | Observe, arbitration, KL, and synthesis all depend on it |
| Knowledge Library ingestion/search | separate worker/process | external knowledge should not block the cognitive tick |
| Store maintenance | offline retention/rollup job | current live evidence shows cycle events dominate disk |
| Pattern synthesis | gated background worker later | useful but not turn-critical; keep result re-entry serialized through actor |

## Do Not Scale Blindly

- Do not raise Working Memory capacity to hide latency; that changes cognition.
- Do not make Tier 3 concurrent unless the Executive has an explicit merge policy.
- Do not shard the graph until memory retrieval and write ownership have a formal consistency model.
- Do not optimize away reflection for conversation globally; use measured fast paths and behavior regressions.

## Immediate Order

1. Add event retention/rollup for `cycle_events`.
2. Use `Telemetry` events to identify the slowest non-idle cycles.
3. Run ablation comparisons with `AblationConfig`.
4. Add embedding worker batching/caching if telemetry shows embedding waits are material.
5. Add Tier 1/Tier 2 model hosts only after telemetry shows LLM agreement latency dominates.

## Success Metrics

- p50/p95 `Telemetry.elapsed_us` for real-input cycles, stratified by
  compiled procedures versus other foreground paths; use phase and tier
  payloads to investigate the latter further
- p50/p95 provenance-linked actor-dequeue-to-Speak duration, separately for
  compiled, curated, and other responses; do not confuse this with a producer-
  to-UI delivery metric
- Tier 3/Tier 4 calls per conversation turn
- Act `render_path` distribution
- cycle event rows added per minute
- behavioral regression suite remains green
- live eval shows no increase in echoing, silence, identity drift, or repetition
