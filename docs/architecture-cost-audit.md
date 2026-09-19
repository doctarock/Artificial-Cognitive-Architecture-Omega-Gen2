# Architecture Cost Audit

Purpose: keep biological fidelity load-bearing. Every cognitive mechanism should have a behavioral claim, an operational cost signal, and an ablation that can falsify whether the mechanism is earning its runtime/storage cost.

## Current Evidence

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\store-triage.ps1 omega.sqlite3
```

Live result on 2026-08-17:

- Database size estimate: 59,048,304,640 bytes
- Mental objects: 4,825
- Knowledge docs: 36,485
- Embedding bytes: ~126.7 MB total
- Cycle events: 224,573,766

Interpretation: the first storage target is event retention/rollup, not memory objects or embeddings.

## Mechanism Ledger

| Mechanism | Behavioral claim | Cost signal | Ablation |
| --- | --- | --- | --- |
| Predict/Compare | Fresh input is scored against expectation before attention | `Telemetry.elapsed_ms`, embedding cache misses, Compare events | Not yet switchable; compare baseline against cached identical input |
| GWT coalition/broadcast | Working Memory remains bottlenecked and non-echoic | Coalition/Broadcast event count, WM count | Reduce `working_memory_capacity` and compare behavior |
| SOAR executive/impasse | Ambiguity becomes subgoal/reflection instead of random action | Impasse/Escalation event count, Tier 3/4 use | Force clear proposal thresholds; compare quality |
| Cognitive Core reflection | Conversation is mediated by cognition, not direct echo | Tier 3 latency, Reflection objects, behavior regression | Compare with a direct-speak fast path only in experiments |
| Memory formation | Experiences consolidate without re-remember loops | Act Remember events, dirty object count, references | Raise memory thresholds or disable Remember proposals |
| Pattern synthesis | Episodic clusters produce semantic abstractions | Synthesize events, Tier use, new semantic objects | `AblationConfig.disable_synthesis` |
| Boredom/self-stimulation | Idle mind can initiate useful self-directed activity | Boredom source events, Act/tool events | `AblationConfig.disable_boredom` |
| Agenda/drives | Persistent intentions surface without external prompting | Agenda events, active intention count | `AblationConfig.disable_agenda` |
| Social rendering | Spoken output is phrased without originating content | Act `render_path`, Tier 1 calls | Compiled-skill/verbatim fast path ratios |

## Decision Rule

Keep a mechanism if ablation makes behavior measurably worse on regression scenarios or live evaluation while its cost remains bounded. Tune or gate it if cost rises without visible behavioral contribution. Remove or isolate it if repeated ablations show no behavioral loss.

## Next Cost Target

Implement cycle-event retention: keep recent raw events, roll up older telemetry/normal repeats, and preserve Impasse/Escalation/Error events longer than routine Normal events.
