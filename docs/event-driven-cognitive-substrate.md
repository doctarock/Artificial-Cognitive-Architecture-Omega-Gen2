# Event-driven cognitive substrate

This document records the implemented boundary between Omega's cheap local
cognition and its model-backed deliberation. It is an implementation map, not a
claim that the system is a biological simulation.

## Runtime flow

```text
input or due deadline
        |
        v
local prediction + orienting + sparse object dynamics
        |
        +-- familiar compiled procedure --> direct action
        |
        +-- weak/losing candidate --> bounded preconscious trace
        |
        v
coalition competition + workspace ignition
        |
        v
executive/model tiers only when local machinery cannot resolve the event
```

`CognitiveScheduler` builds a min-priority queue of current coarse cognitive
deadlines and sleeps until its earliest event or an input channel wake. Boredom,
synthesis, and preconscious re-evaluation/expiry have explicit events; a bounded
idle cadence still services tick-count-based maintenance and write-behind. The
queue is rebuilt after every tick so changed state cannot strand canceled
timers. A separate actor-local min-priority queue wakes the scheduler for
delayed per-edge spikes; it bounds fanout, pending events, and events drained
per tick rather than scanning every dormant object.

## Implemented local mechanisms

| Mechanism | Current implementation |
| --- | --- |
| Sparse object state | Each `MentalObject` has transient potential, threshold, adaptation, firing time, and refractory state. Potential and adaptation leak lazily when touched. |
| Sparse spike propagation | A local firing schedules delayed signed pulses only along a bounded active outgoing neighborhood. Due pulses accumulate by target; excitation can fire and nominate dormant objects, while inhibitory pulses reduce their potential. Firing may cascade on later deadlines. |
| Orienting | A Tier-0 weighted calculation combines novelty, local prediction error, goal relevance, affective salience, social relevance, and sensor-owned numerical threat/urgency. Threshold crossing produces an attention pulse. |
| Sensor habituation | Only familiar, well-predicted, low-orienting environment readings with no recognized identity or calibrated urgency stay in graph/prediction history without a fresh Coalition bid. New, surprising, urgent, identity-bearing, and conversational input retain the normal path. |
| Local prediction | An online predictor learns an empirical source-transition distribution plus interval and affect means/dispersion. Active objects predict likely successors from signed graph links and forecast goal impact from recent supportive/contradictory outcome links. These errors and expectations are numerical. |
| Competition/inhibition | Coalition crowding provides immediate divisive suppression. Fresh workspace winners also learn sparse signed inhibitory edges to the strongest losing competitors. Ignition hysteresis and refractory/adaptation reduce immediate repetition. |
| Preconscious persistence | Coalition losers are reconsidered on scheduled events with decaying surprise, then expire after a bounded interval. |
| Associative recall | Working-memory objects seed bounded, fan-normalized multi-hop spreading activation over the sparse graph. |
| Sparse indexing | Mature compiled procedures and goal-bearing objects have local indexes, avoiding full autobiographical-memory scans on familiar input and goal checks. |
| Delayed credit | Coactive associative edges receive transient eligibility traces. Later reward applies decayed, bounded positive or negative credit without retrospective model narration. |
| Skill compilation | Repeated outputs remain candidate memories, not executable shortcuts. Three distinct host-verified successful outcomes mature observed `ContinueReflecting -> Speak`, `ContinueReflecting -> Ask`, or `ContinueReflecting -> Ignore` into a typed semantic procedure. Question/command stimuli, digits, time-sensitive words, tools, and external actions are excluded; an exact previously verified clarifying question is an allowed consequence. The actor indexes verified programs once at load and refreshes on feedback; a hit reuses its stimulus embedding and directly speaks, asks, or ignores without embedding/chat-model I/O. |
| Curated static answers | Only explicitly seeded, exact-match factual questions with a supplied embedding enter this O(1) index. A host-only actor command can upsert or revoke entries; the ordinary HTTP/text/agent/sensor input paths do not have that authority. No generated Reflection auto-promotes itself. Common time-sensitive markers are excluded; the host remains responsible for validating that a fact is stable. A hit speaks without embedding or chat-model I/O. Same-tick revisions apply in command order, and revocation marks memories Discarded rather than deleting them. |
| Innate social reflex | Only isolated greetings such as “Hi Omega” resolve to direct Tier-0 speech on their first occurrence. This skips chat generation but still uses the ordinary embedding/observation path until a learned procedure can supply its own embedding. |

Each behavior-changing substrate addition has an ablation switch so its causal
contribution can be measured without disabling the scheduler itself.

## Deliberate boundaries

- Dynamics and eligibility traces are runtime state; durable memories and
  learned edge strengths remain the persisted substrate.
- Signed inhibitory edges are learned from bounded winner/loser competition.
  Learned `Contradicts` goal-outcome edges now also suppress their targets in
  spreading activation, multi-hop recall, and spike propagation. This is
  local associative suppression, not an explicit goal policy.
- The predictor covers source, timing, affect, strongest linked successor
  objects, and signed goal impact. Goal links are weak local associations
  learned when an active goal succeeds or is abandoned soon after an
  Executive decision; they are not proof that the decision caused the
  outcome. Calibrated consequence distributions remain future work.
- A typed `SensorSignal` now accepts calibrated threat and urgency from an
  external sensor, optionally with an already-computed embedding. It preempts
  ordinary queued input without dropping that input. The engine
  does not itself supply a fall detector or validate calibration; ordinary
  prose never receives a fabricated threat score.
- The attention specialist remains the model-backed attention example.
  This checkout had no v0.2 generator/schema despite references to them in
  the engine and journal. A new reproducible v0.2 synthetic generator and
  schema now match the runtime fields; they do not recreate the exact prior
  training run or establish live model quality. The model admission path now
  releases stale/below-Coalition-floor members, and the distribution guard
  rejects workspaces with multiple or non-finite surprise values.
  Actionable model votes must also name an eligible Coalition target;
  unparseable or absent UUIDs fall back to deterministic admission, even
  when the model says it is confident.
  Consulted attention decisions now log their target and both the specialist
  membership set and a same-tick deterministic shadow set (IDs and an
  agreement boolean, without object text). This can measure policy
  divergence in a future live session; agreement is not an outcome label.
  A loopback five-case v0.2 probe found only 2/5 correct operation/target
  decisions, including confident mistakes. The daemon therefore defaults
  its attention mode to `off`; `shadow` collects comparisons without
  changing admission, and only explicit `active` permits model control.
  The new Tier-0 components expose structured numeric results suitable as
  targets for future classifiers, MLPs, or SNNs, but this pass does not invent
  untrained specialist models.
- A separate dependency-free communicative-intent specialist is trained on
  generic authored Speak/Ask/Ignore examples and clears a distinct 30-case
  held-out fixture at 30/30 with 1.0 recall for every class. The checked
  artifact is schema/shape/metric gated independently by Rust and is
  `shadow_only`: it compares with an already-selected Executive terminal
  operator on a Reflection and cannot propose or execute one. A forced
  disagreement test proves non-control. Local 10,000-call inference measured
  88 microseconds p50 and 107 microseconds p95. This makes it usable for
  prospective shadow measurement, while real-reflection outcome validation
  remains required before any control role.
- Compilation executes three narrow outcome-verified two-step macros; legacy
  speech-only routines remain non-executable candidate records. Broader conditional operator sequences
  and real-world expected-consequence validation are not yet compiled; their
  execution would need explicit
  condition, outcome, and safety semantics rather than replaying arbitrary
  previously generated actions.
  The actor's loaded-skill index and refresh path revalidate stored response
  length and finite embeddings before direct speech; old or corrupted
  routine objects cannot bypass the normal path on load. Legacy speech-only
  records cannot bypass it even after repeated identical outputs.
  Success-gated macros cover exact observed `ContinueReflecting -> Speak`,
  `ContinueReflecting -> Ask`, and `ContinueReflecting -> Ignore` chains for
  narrow, non-command stimuli.
  The host-only `ProcedureFeedbackCommand` credits three *distinct* terminal
  observation IDs with explicit successful-outcome labels. Each macro stores
  its typed steps, exact-normalized source conditions, and a matching
  expected consequence, then collapses the two steps to direct speech, an
  exact clarifying question, or direct Ignore; negative feedback durably
  demotes every path.
  The v2 loader refuses missing/inconsistent condition or consequence
  fields; old uncontracted v1 records remain auditable but non-executable.
  Its in-memory executable is also typed: an exact-normalized routine
  Observation condition, bounded `ContinueReflecting` then `SpeakExact`,
  `AskExact`, or `Ignore` steps, and a matching `SpokeExact`, `AskedExact`,
  or `Ignored` consequence. The
  fast-path accessor rechecks that the input condition and steps/consequence
  agree. It is not a
  generic string-response cache or an interpreter for arbitrary operators.
  Tool actions, question/command stimuli, and time-sensitive content cannot
  enter this macro. Ask may only replay the exact verified clarifying question.
  The owning host must provide outcome labels; the engine does not
  infer success from its own repeated speech. Broader conditional operator
  programs and real-world consequence validation remain open. The local
  daemon prints each spoken, asked, or ignored foreground observation ID and accepts
  `/feedback success <id>` or `/feedback failure <id>` on its own stdin only;
  ordinary API/text/agent input cannot submit such a label. The operator
  must independently judge the real outcome before using this command.
- Curated-answer ingestion requires the host to provide a vetted answer and
  finite question embedding. The engine enforces an exact static-question
  admission filter but cannot independently verify factual truth.
  Valid upserts and revocations force the otherwise periodic write-behind
  flush at the end of their tick; a failed store flush retains dirty state
  for retry and is logged as an error.
- The scheduler's coarse deadline queue and bounded per-edge spike queue are
  implemented. Agenda commitment decay and floor grace now use elapsed wall
  time, as does drive smoothing. Outcome-credit windows still count ticks:
  a short fixed wall-clock expiry would discard feedback that arrives after
  slow inference or ordinary human turn-taking. Object
  leakage is lazy instead of scheduling decay
  events for every dormant object. Spike weights are existing graph-edge
  strengths, not a trained spiking-neuron model.
  Dormant intentions now earn a reference/admission pulse at most once per
  10-second wall-clock surface cadence; resident intentions receive none
  merely because maintenance woke the actor. The sparse goal index replaces
  a full graph scan for surfacing.
- The illustrative latency percentages in the design discussion are targets,
  not measured claims. Per-tick telemetry records microsecond elapsed time
  and whether a compiled procedure was recognized. The existing durable
  `LatencyReport` now separates compiled from other foreground turns and
  computes sub-10ms percentages. It also reports provenance-linked
  actor-dequeue-to-Speak turn latency, separately for compiled, curated, and other
  speech. This spans asynchronous embedding, deliberation, and scheduler
  waits between ticks; it does not include time waiting in an upstream
  input channel or delivery/rendering after the Act event. The compiled-path integration test
  establishes that no embedding or chat client is touched; representative
  deployment data is still required before publishing latency SLOs.

Run the read-only report on a local event database with
`cargo run -p aca-store --example latency_report -- omega.sqlite3`.
Its cycle slices are actor-tick durations; the spoken-turn slices are
actor-dequeue-to-Act durations across cycles. Neither is a full
producer-to-user-delivery measurement, and periodic flushes after telemetry
are excluded from cycle slices.

An explicitly invoked local actor probe (`cargo test -p aca-engine
local_compiled_turn_latency_probe -- --ignored --nocapture`) measured 100
warmed verified compiled turns on this machine: p95 actor-dequeue-to-Speak 762 µs,
p95 whole tick 8.7 ms, and 100% of those spoken turns under 10 ms. It uses
an in-memory store, no inference clients, and disabled agenda/boredom/
synthesis; it does **not** validate deployed producer-to-user latency or
the cost distribution of general conversational turns. The historical
`omega.sqlite3` report has 25,000 legacy telemetry samples, zero new
microsecond/spoken-turn samples, and multi-minute cycles dominated by Act
or Synthesize; attention costs were about 300 ms in those slow cycles.
Seven of ten configured endpoint settings are LAN rather than loopback, so
no controlled live model session was launched without specific permission
to send private ACA prompts to those hosts.

A second opt-in loopback probe (`cargo test -p aca-api
loopback_verified_turn_latency_probe -- --ignored --nocapture`) includes
real HTTP `/input` submission, the actor's event-driven scheduler, and
WebSocket `/events` delivery. For 100 warmed verified turns on this machine
(50 Speak and 50 Ask), Speak p50/p95 was 1.520/7.977 ms and Ask p50/p95 was
1.522/7.800 ms from HTTP submission to the Act event; all 100 were under
10 ms. It uses an in-memory store, preseeded verified macros, and panic
model clients. It excludes real embedding/chat inference, voice synthesis,
Godot rendering, and LAN deployment, so general response-time SLOs remain
unverified.

An additional process-boundary probe (`cargo test -p omega-acad
daemon_loopback_latency_probe -- --ignored --nocapture`) launches the real
`omega-acad` executable with a temporary SQLite database and explicitly
blank model, voice, video, and library endpoints. Six distinct innate
greeting turns crossed its loopback HTTP and WebSocket surfaces with p50
0.597 ms and a 1.788 ms maximum in the latest run (an earlier run measured
0.60/1.86 ms). This validates real daemon wiring and startup
configuration without sending prompts to LAN hosts. Six deterministic
reflex turns are not a percentile sample for general conversation and do
not include inference, TTS, or UI rendering.

A separate privacy-safe smoke probe used an actual loopback Ollama Tier-3
model with generic input while retaining the real HTTP, scheduler, Reflection,
Executive, Act, and WebSocket path. One foreground turn reached terminal Speak
in 300 ms. Embedding was fixed locally because the installed chat model does
not implement Ollama's embedding endpoint. One sample is not a percentile or
a general conversational SLO; it establishes that the real inference path is
sub-second in this controlled loopback configuration. Diagnostic runs also
revealed repeated Ignore execution on an already terminally handled
Reflection. Terminal Ignore now archives and releases that Reflection (not raw
Observations); spoken/asked Reflections remain available to associative
learning and synthesis.

With the configured loopback attention model enabled in non-controlling
shadow mode, the same real-daemon shape measured p50 310 ms and maximum
626 ms across six greetings. No decision cleared the runtime gate and seven
explicit 300-ms attention timeouts were observed across foreground and
follow-up ticks. All replies still followed deterministic admission. This
proves that even shadow consultation currently destroys the fast-path
latency target, supporting the daemon's `off` default.

For future numeric specialists, Compare events now key seven orienting
inputs by observation ID and Broadcast events name actual admitted IDs.
[`training/specialists/export_orient_ignition.py`](../training/specialists/export_orient_ignition.py)
joins those local events without exporting text and refuses fewer than
1,000 real examples with both classes present in chronological train/eval
partitions. The current legacy DB yielded zero keyed examples, so no new
numeric specialist was trained or declared usable from it.
The local daemon now also accepts `/outcome success|failure <compared-id>`
on its own stdin after a host judges the *observed* consequence. The actor
accepts a recent Compare ID once and durably records a keyed, text-free
independent Learn label. `export_orient_outcomes.py` joins those verdicts
with numeric inputs and actual ignition, drops conflicting labels, and
refuses sparse or one-sided chronological partitions. The legacy DB has
zero such independent verdicts. Even with labels, this is observational
data rather than proof that the opposite ignition choice would have helped;
prospective shadow/active evaluation is still required for usability.
The outcome trainer and Rust loader now share a versioned eight-feature
contract (seven ORIENT inputs plus actual ignition). The loader independently
enforces held-out gates, finite parameters, feature order, and `shadow_only`
status. A loaded artifact emits a keyed post-Broadcast success-probability
forecast but has no API capable of changing attention or action. No real
artifact exists yet because the evidence gate correctly remains empty.

The governing rule is simple: information moves upward only when the cheaper
layer cannot resolve it. The architecture should become faster with experience
because successful controlled behavior can become a compiled local procedure.
