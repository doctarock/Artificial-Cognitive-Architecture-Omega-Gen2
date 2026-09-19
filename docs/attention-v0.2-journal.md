# Journal: Omega Attention Model, v0.1 → v0.2

*Covers one continuous working session, 2026-08-19 through 2026-08-20 (with a
status check-in on 2026-08-23). Written up after the fact from the session
transcript.*

## 1. The starting question: can we even use this?

Context coming in: a LoRA-tuned Qwen2.5-0.5B ("Omega Attention v0.1"),
trained on the other machine via Unsloth, already registered in local
Ollama as `omega-attention-v0:latest`. First question was simply whether
the live Rust engine (`aca-engine`, run by the `omega-acad` daemon) was
even equipped to use a LoRA trained this way.

Answer: yes, but indirectly. `aca-tiers` (the crate that talks to models)
never loads weights directly — it only speaks HTTP to Ollama-native or
OpenAI-compatible endpoints (`OllamaClient`, `OpenAiCompatClient`). A raw
PEFT adapter can't be dropped in; it has to be served first. Path chosen:
Unsloth's `save_pretrained_gguf` → `ollama create`.

First concrete finding: the raw Unsloth GGUF export's `Modelfile` was
generic (`SYSTEM "You are Qwen, created by Alibaba Cloud..."`,
`temperature 1.5`) — didn't match the trained task at all. Fixed to the
real training-time system prompt and `temperature 0`, re-created in
Ollama, smoke-tested via `/api/generate` — came back with clean,
schema-correct JSON.

## 2. Finding out where it actually plugs in

Went looking for the real call site for an "attention operation" decision
and found a red herring: `crates/omega-cluster` implements exactly this
concept (`OclOperation`, `MockProcessor::handle_attention`) — but it's a
self-contained, never-depended-on crate (nothing else in the workspace
references it), absent from `specs.md` entirely. A parked prototype for a
hypothetical distributed multi-processor architecture, not the live path.

The real live engine, `aca-engine`, has a completely different model of
attention: continuous ACT-R-style scoring (`activation_total + surprise`,
`steps::coalition::attention_score`) then rank-and-admit-top-N
(`steps::broadcast`) — no discrete ATTEND/MAINTAIN/SWITCH/SUPPRESS/IGNORE
concept anywhere in it.

Traced the real per-tick data available and compared it against what
v0.1 was trained on:

| training field (v0.1) | real equivalent in `aca-engine`? |
|---|---|
| `source` (7-value fictional enum) | `MentalObjectKind` exists, but different values |
| `activation`, `salience` | only `activation.total` is real; `salience` is coalition's own *output*, not an input |
| `goal_relevance` | doesn't exist as a per-candidate score |
| `novelty` | closest is `precision_weighted_surprise`, conflated with source reliability |
| `confidence` | real, direct match |
| `age` | not exposed as a scalar; implicit in `activation.base_level`'s decay term |

Verdict: v0.1's input schema was mostly fabricated. Asked the user
whether to run it anyway as harmless shadow-mode telemetry (fed proxy
values) or retrain properly first. **Decision: retrain on real fields.**
Then, mid-discussion, a further decision: **have the retrained model
replace Broadcast's decision entirely**, not just add a suppression veto
on top of the existing ranking.

## 3. Design

Went into plan mode given the scope (multi-crate Rust refactor + a new
training pipeline). Key design problem: the model only ever emits **one**
operation + one target per call — it was never trained to emit a ranked
admission set. Resolved by treating "replace Broadcast" as: each tick, ask
the model for one state-transition operation and apply it directly to the
persistent Working Memory set, instead of re-deriving the whole ranked set
from scratch. The old deterministic algorithm is kept as a **mandatory
fallback** — never fully retired — mirroring how every other tier in this
codebase already degrades gracefully when a model is unreachable.

Plan approved, covering:

- **Part A** (Python, `training/attention-v0/`): a new `generate_scenarios_v0_2.py`
  building a schema (`omega-attention-workspace/v0.2`) from only real
  engine fields — realistic ACT-R activation ranges, the key fidelity fix
  that **at most one candidate per tick ever carries a non-null
  `surprise`** (v0.1 wrongly gave every candidate one), and a label
  heuristic that faithfully reproduces real Step 6 ranking for
  ATTEND/MAINTAIN/SWITCH while adding the one thing the deterministic
  algorithm structurally can't do: actively `SUPPRESS` a high-score but
  low-confidence candidate.
- **Part B** (Rust): `aca-tiers::AttentionClient`/`OllamaAttentionClient`
  (a standalone client, deliberately not `ChatClient` — different response
  contract, not part of the tier ladder), `aca-engine::attention_workspace`
  (workspace JSON builder), `steps::broadcast` split into
  `decide_admission_deterministic` (today's algorithm, demoted to
  fallback) and `decide_admission_from_attention_model` (new, applies one
  state transition), wired into `loop_actor::tick()`'s Step 5/6 boundary
  with a confidence gate.

## 4. Building it

Implemented in order: `aca-tiers` (client + parser, reusing
`response::extract_json_candidate`'s `<think>`-stripping logic rather than
reimplementing it), `aca-engine` (workspace builder, the `broadcast.rs`
refactor — careful to preserve all 9 original tests via a mechanical
two-call split rather than rewriting them), `loop_actor.rs` wiring,
`omega-acad` env-var construction. Every step built and tested before
moving to the next; the `broadcast.rs` refactor in particular was the
highest-risk part (touching tested, load-bearing core logic) and came out
clean — all original tests passed unchanged after the split.

Ended with 318 `aca-engine` tests (up from 306 baseline), including new
regression tests proving: an unconfigured attention client changes zero
behavior, a high-confidence decision actually overrides the deterministic
algorithm, and low-confidence/error/timeout all fall back identically.

## 5. Training run

User trained v0.2 on their own rig (AMD Radeon 8060S, ROCm, via the new
`generate_scenarios_v0_2.py` → `train_unsloth.py` scripts). Result, 3
epochs / 3,000 steps:

| metric | v0.2 base run | v0.2 3-epoch |
|---|---|---|
| valid JSON rate | 100% | 100% |
| operation accuracy | 56.5% | **98.0%** |
| target accuracy | 66.5% | **95.75%** |

Slightly ahead of v0.1's own 3-epoch numbers (96.0%/92.2%), despite a
harder task (unbounded real activation values vs. v0.1's clean 0–1
floats). Good sign the label heuristic was actually learnable.

GGUF-exported, same Modelfile fix applied (`SYSTEM` = the v0.2 prompt,
`temperature 0`), registered as `omega-attention-v0.2:latest`, smoke-tested
against a hand-built example — correctly `SUPPRESS`ed the one low-confidence
high-activation candidate, exactly matching the label heuristic.

## 6. "Have we just bottled ACT-R?"

User asked directly whether swapping the deterministic rank-and-admit
policy for a learned one costs real dynamism. Worth recording the answer
precisely: the ACT-R activation computation itself
(`aca_graph::recompute_activation` — recency/frequency decay, spreading
activation, noise) is untouched and still recomputed fresh every tick from
live graph state. What got replaced is only the arbitration policy sitting
on top of it.

Two real costs identified, both then *compensated for in code*, not just
written down:

1. **Distribution shift** — the deterministic sort generalizes to any
   input by construction; the model only knows what the generator sampled
   (candidate count 1–9, `activation_total` in `[-10, 10]`). Compensation:
   `attention_workspace::workspace_in_distribution` — skip the model
   entirely, no extrapolation gamble, whenever a tick's real data falls
   outside that envelope.
2. **Incremental vs. batch-rerank dynamics** — the model only ever emits
   one state transition per tick (unlike the deterministic algorithm's
   full re-rank every tick), so a long run of confident `MAINTAIN`/`IGNORE`
   could in principle let live activation drift underneath unnoticed.
   Compensation: `LoopConfig::attention_reconciliation_interval` — every
   Nth consecutive model-trusted tick is forced back through the
   deterministic algorithm regardless of confidence, bounding worst-case
   staleness. Proved with a dedicated test (interval=1, a mock that would
   always suppress — confirmed the forced tick bypasses it).

Both compensations surface as their own telemetry events
(`attention_out_of_distribution`, `attention_forced_reconciliation`)
alongside `attention_model_decision`.

## 7. First real commit

Went to commit and discovered the repository had never actually been
committed beyond a 2-line README stub — the entire ~170-file codebase
(engine, viz, training pipeline) was untracked. Before doing a blanket
`git add -A`: found `models/research/` held **68 GB** of downloaded model
checkpoints, not excluded by `.gitignore`. Added `/models/research/` to
`.gitignore` (kept `models/README.md` and `models/research-artifacts.json`
tracked, per the existing README's own description of those as the
intended metadata files), scanned the rest for secrets (clean — only
placeholder `_API_KEY=` variable names), then committed 170 files as the
project's real first commit.

Also found and removed a stale Windows Scheduled Task ("Omega ACA
Startup") that was silently auto-launching an old build from a different
directory (`E:\AI\omega-aca`, not this checkout) on every login.

## 8. Production incident: the idle-tick spam bug

First live run "spammed [output] then died." Root cause: the new
out-of-distribution guard treated **zero candidates** as out-of-distribution
(0 is outside the trained floor of 1) — but zero candidates is the
*ordinary idle state* in a loop with no scheduler or backoff (confirmed
elsewhere in the codebase's own doc comments: idle ticks run at roughly
1000/sec). So every idle tick, which used to be completely silent, started
firing a telemetry event. Ran undetected for ~14 hours (the process's
whole lifetime), ballooning `omega.sqlite3` to **72.8 GB**.

Found this by checking the running process's binary path (confirmed it
was this checkout, started before the relevant source fix existed) and DB
file size. Stopped the process immediately. An interim bandaid had already
been applied somewhere along the way (widening the OOD range to `0..=9`),
which fixed the *symptom* but left a self-contradicting unit test and
still would have meant the model got HTTP-called on every single idle tick
forever. Reverted the range to the honest `1..=9` and applied the real
fix: `loop_actor::tick()` now bypasses the entire attention-arbitration
path — no model call, no OOD check, no telemetry, no counter changes —
whenever `raw_candidates` is empty, exactly matching how Coalition's own
telemetry was already silent on empty ticks.

Per the user's call ("we can wipe its data clean, it is fine"), deleted
the corrupted `omega.sqlite3` (leaving an unrelated, pre-existing Aug-16
backup file untouched — not something this bug touched, not mine to
delete unilaterally).

## 9. Reviewing work from a second agent

Subscription limits meant the user used a different, less capable coding
agent for some interim work while this session was unavailable: adding an
`attention_in_flight` live status indicator (parity with the existing
`embedding_in_flight`) across `LiveModelHandles`, `EngineSnapshot`, the API
server, `loop_actor.rs`, and a new Godot viz indicator in
`pipeline_view.gd`.

Reviewed all of it. The Rust side was correct throughout — single shared
`Arc<AtomicBool>` (not duplicated atomics), `store(true)`/`store(false)`
placed only around the actual model HTTP call (not the bypass branches),
the one full `EngineSnapshot` literal construction site updated, every
other construction site correctly defaulting via `..Default::default()`.

Found one real bug in the GDScript: `_pulse_attention_indicator()` called
`Tween.set_parallel(true)` *between* two `tween_property` calls. In Godot
4, `set_parallel` affects everything appended afterward, not just the next
call — so the down-tween and the recursive `tween_callback` both started
alongside the up-tween instead of after it, and with no guard checking
`_attention_in_flight` before recursing, the pulse would run forever once
triggered even a single time, continuously spawning tweens and fighting
the idle-reset branch. Fixed: removed the stray `set_parallel(true)`
(sequential is Godot's default and what a rhythmic pulse needs), added an
early-return guard so the recursion stops the moment the model call ends.

## 10. Is it actually any good? (inconclusive, twice)

Asked directly whether the model was performing well enough to move on or
worth another training iteration. Answer both times: **no real data yet**,
for two different reasons in sequence.

**First check:** the only run that had ever happened was the one
corrupted by the spam bug above — 14 hours of pure `attention_out_of_distribution`
noise, zero genuine decisions. That DB was wiped and nothing had been
restarted yet.

**Second check** (after the user started a clean run themselves): the fix
held up well — 250,420 lightweight heartbeat events, essentially zero CPU
time, 45 MB DB, no repeat of the spam. But *also* zero
`attention_model_decision` events anywhere: `working_memory_count` had
been `0` and `graph_object_count` stuck at `5` (just the Self Memory seed
objects) for the entire run. Nothing had actually been said to Omega, so
Working Memory never had contents, so the model was never once consulted.

Conclusion, as of this journal entry: the bug is genuinely fixed and the
compensation mechanisms are in place and tested, but there is still no
real signal on live decision quality — that requires an actual
conversational session to generate candidates worth arbitrating over.

## Current state (as of this entry)

- `omega-attention-v0.2:latest` is live in Ollama and wired via `.env`
  (`OMEGA_ATTENTION_MODEL_BASE_URL`, `OMEGA_ATTENTION_MODEL`,
  `OMEGA_ATTENTION_TIMEOUT_MS`, `OMEGA_ATTENTION_MIN_CONFIDENCE`,
  `OMEGA_ATTENTION_RECONCILIATION_INTERVAL`).
- All of the above (the idle-tick fix, the `attention_in_flight` feature,
  the Tween fix) is committed (`5f472eb`, "claude guard") on top of the
  project's first real commit (`8d17442`).
- Since then, substantial further engine work has landed on top
  (`crates/aca-engine/src/coherence.rs`, `embedding_worker.rs`, a
  conversational-presence feature threading through prompts and
  `docs/conversational-presence.md`, sampling/temperature config changes)
  — outside this thread's scope, not detailed here.
- **Open:** no real telemetry yet on the v0.2 model's live decision
  quality vs. the deterministic baseline. Needs an actual interactive
  session before the "iterate vs. move on" question can be answered with
  data instead of a guess.

## Later checkout note

The generator and schema cited above were not present in the checkout
inspected in September 2026. `generate_scenarios_v0_2.py` and
`SCHEMA_V0_2.md` were restored as a *new reproducible synthetic recipe*
matching the current engine workspace fields, not as a reconstruction of
the exact historical training run. Do not conflate its generated-label
accuracy with live quality of the installed model. The attention model's
incremental admission now releases candidates that disappear or fall below
Step 5's floor, and out-of-distribution gating also detects multiple or
non-finite surprise values.
# September 2026 loopback usability check

The configured loopback Ollama attention model is present and its Modelfile
mentions the v0.2 protocol. A five-case, synthetic but contract-faithful
probe (`training/attention-v0/probe_loopback_attention.ps1`) returned
well-formed decisions in all five cases but matched only two expected
operations. It incorrectly chose `ATTEND` for `IGNORE`, `SWITCH` for
`MAINTAIN`, and `SWITCH` for `SUPPRESS`, with self-reported confidence
between 0.64 and 0.75 on the incorrect decisions. Warm requests took
roughly 154–165 ms; a separate initial request timed out at 20 seconds
while the model was cold. These five cases are a diagnostic, not a broad
accuracy estimate or evidence of live cognitive quality.

Confidence >= 0.6 is therefore not a defensible promotion gate for this
configured model. `omega-acad` now defaults `OMEGA_ATTENTION_MODE=off` even
if endpoint variables are present, so neither incorrect votes nor
per-event model latency enter the live actor accidentally. `shadow`
consults the model and logs paired decisions without letting it alter
Working Memory; `active` explicitly allows control and is for a model that
has passed wider held-out and live checks. The engine itself also defaults
to shadow-only when an attention client is attached. This is a fail-closed
deployment gate, not an additional usable specialist.

The real-daemon loopback probe was subsequently run with this model in
non-controlling `shadow` mode. Across six innate greeting turns, HTTP input
to WebSocket Act p50 was 310 ms and maximum 626 ms, versus a sub-2-ms
maximum with attention off. No model decision cleared the runtime gate;
telemetry recorded seven explicit 300-ms attention timeouts (additional
cognitive ticks can consult attention beyond the six foreground turns).
All deterministic responses still arrived because shadow/fallback cannot
change admission. The configured model is currently both unusable and
harmful to response latency; defaulting it to `off` is required, not merely
conservative.
