# Coherence Arbitration Policy

**Status: permanent hard path. No model, now or later.** Every decision
below is closed-form arithmetic over fields the engine already computes
(score, confidence, timestamp, id) with no soft judgment call and no
outcome signal to learn from — the two conditions that would justify a
trained model (see "Why no model" below). This is not a placeholder
awaiting a future upgrade; it is the permanent implementation.

Prerequisite artifact for the two gates named in `scale-strategy.md`:
"Do not make Tier 3 concurrent unless the Executive has an explicit merge
policy" and "Do not shard the graph until memory retrieval and write
ownership have a formal consistency model." This document is that merge
policy. It defines, deterministically, what happens if two concurrent
threads of cognition ever propose conflicting mutations to the four
single-owner surfaces (`scale-strategy.md`'s "Keep Single-Owner" list):
Working Memory membership, activation graph mutation, Executive operator
selection, and Tier 3/Tier 4 escalation seats.

Nothing here turns on concurrency by itself. `CognitiveLoopActor` stays the
only mutator until each rule below has been run in shadow mode against
replayed real traces and produces no coherence violations
(`crates/aca-engine/src/coherence.rs` has the implementation and a first
pass of unit-test sanity checks).

## Format

Each surface gets:
- **Input schema** — drawn from the real fields the engine already has at
  that point (`aca_types::MentalObject`, `steps::executive::OperatorProposal`,
  etc.), not fabricated ones. `training/attention-v0` shipped once on made-up
  fields (v0.1) and had to be retrained on real ones (v0.2) — this policy
  starts from real fields to avoid repeating that.
- **Resolution rule** — a total, deterministic function from "the competing
  proposals" to "the one outcome," with an explicit tie-break chain (never a
  silent default).
- **Output contract** — a small fixed vocabulary plus a `reason_code`, kept
  for the same reason `AttentionDecision` uses one (`crates/aca-tiers/src/attention.rs`):
  compact, loggable, and unambiguous to test against. It is not a model
  output contract — see "Why no model."

## 1. Working Memory admission (Broadcast, Step 6)

Already a single serial gate today (`decide_admission_deterministic` /
`decide_admission_from_attention_model` in
`crates/aca-engine/src/steps/broadcast.rs`). This rule only matters if two
concurrent Coalition evaluations ever produced two competing admission
proposals for the same tick.

**Input**: for each proposal, `(candidate_id: MentalObjectId, score:
f32, source_thread: ThreadId)` — `score` is the existing
`activation_total + surprise.unwrap_or(0.0)` computation already used by
`decide_admission_from_attention_model`. Two proposals conflict when they
disagree on membership of the same `candidate_id`, or when their combined
admits would exceed `working_memory_capacity`.

**Resolution rule**: higher `score` wins. Tie-break: earlier
`MentalObject.created_at` wins (older evidence is more corroborated).
Second tie-break: lower `MentalObjectId` (a stable, arbitrary total order,
never a coin flip). If admitting both would exceed capacity, evict the
current weakest Working Memory member by the same `score` field — identical
to the existing eviction rule in `decide_admission_from_attention_model`,
just applied across threads instead of within one.

**Output**: `{operation: ADMIT_A | ADMIT_B | ADMIT_BOTH | EVICT_WEAKEST,
winner: MentalObjectId, reason_code}`.

## 2. Activation graph mutation

**Input**: for each proposed write, `(object_id: MentalObjectId,
field: "activation" | "status" | "edges" | "workspace", proposed_value,
proposer_confidence: f32, tick: u64)`.

**Resolution rule**: writes to *different* `object_id`s never conflict —
apply both. Writes to the same `object_id` and same `field`: higher
`proposer_confidence` wins. Tie-break: later `tick` wins for `activation`
and `workspace` (most recent evidence should win for decaying/refreshing
fields); earlier `tick` wins for `status` (a `Discarded` transition should
not be silently un-done by a slower concurrent writer that hasn't seen it
yet — discard is one-way per `Graph::discard`'s own doc comment). Writes to
the same `object_id`, *different* `field`s: apply both (fields are
independent columns, not a single atomic record).

**Output**: `{operation: APPLY_A | APPLY_B | APPLY_BOTH, reason_code}`.

## 3. Executive operator selection

**Input**: competing `OperatorProposal { operator, target_id, preference,
confidence }` values (the real struct in
`crates/aca-engine/src/steps/executive.rs`) for the same tick.

**Resolution rule**: this surface already has an ordering score
(`preference`) and a confidence gate — reuse them rather than inventing new
ones. Highest `preference` wins among proposals with `confidence` above the
existing impasse threshold. If the top two proposals are within the
engine's existing tie-margin, or the winner's `confidence` is below
threshold, do not silently pick one: emit `ESCALATE`, which is exactly
today's `ImpasseKind::Confidence` path — routed to Tier 3/4, not resolved
here. This surface should almost never need a genuinely new merge rule; it
should degrade to the existing impasse machinery.

**Output**: `{operation: SELECT | ESCALATE, winner: OperatorProposal,
reason_code}`.

## 4. Tier 3/Tier 4 escalation seats

This is contention over the single-flight semaphore itself
(`crates/aca-tiers/src/pool.rs`'s `TierPool`, concurrency fixed at 1), not
over cognitive state — so the resolution rule is about queueing, not
merging.

**Input**: competing escalation requests `(request_id, tick, ImpasseKind,
requester_confidence: f32)` arriving while the one permit is held.

**Resolution rule**: `ImpasseKind::MissingInformation` never contends for
this seat — it already takes the subgoal-respawn path, not Tier 3/4. Only
`ImpasseKind::Confidence` requests compete. FIFO by `tick` (oldest request
first) with one exception: `executive.rs`'s existing `try_acquire` /
`DeferredToHeuristic` fallback stays authoritative — a request that would
have to wait is better served by the existing heuristic fallback than by
a longer queue, so this rule only decides ordering among requests that
arrive close enough together to both plausibly `try_acquire` before the
next tick's fallback would trigger.

**Output**: `{operation: GRANT | DEFER_TO_HEURISTIC, reason_code}`.

## Non-goals

- This document does not make any surface concurrent. It only defines what
  *would* happen if they were.
- It does not cover Tier 1/2 (`DivergentPool`) — those are already
  concurrent by design across independent model instances and already have
  a resolution mechanism (`arbitrate.rs`'s agreement-based candidate
  selection), which is a different kind of arbitration (best-of-N over
  already-complete answers) than this document's (conflicting in-flight
  writes).
- It does not decide *where* in the engine concurrency should first be
  introduced. That is a separate decision, to be made after telemetry
  (per `scale-strategy.md`'s "Immediate Order") shows a real bottleneck
  these rules would need to resolve.

## Why no model

Training a model (attention-v0's pattern) pays off when a decision needs
either (a) soft signals that are expensive to hand-encode as a closed-form
rule, or (b) a channel to improve from real outcome data instead of manual
retuning. Neither holds here:

- Every rule above is already closed-form: a handful of numeric
  comparisons and a fixed tie-break chain, not an interaction of soft
  signals the way attention's ATTEND/MAINTAIN/SWITCH/SUPPRESS/IGNORE
  choice can weigh `goal_priority` against `surprise` against attentional
  stickiness. Section 3 (Executive selection) makes this concrete: routing
  both threads' proposals through the existing `select_operator` produces
  the exact same decision as the deterministic function, verified in
  `coherence.rs`'s test suite — there was no room for a model to do
  anything different.
- There is no outcome signal to train against. The engine is single-owner
  by construction, so no real conflict ever occurs to log; any training
  set would have to be synthetic labels generated *from* this document's
  own rules, meaning a model could at best approximate the code it was
  trained on, with added inference latency and a new failure mode
  (timeout, malformed output) for zero behavioral upside. Unlike Broadcast
  admission, there's also no `OutcomeRegistry`/`CalibrationTracker`-style
  downstream reward signal a future retrain could ever plug into — a
  resolved write conflict has no separate "was this the right call"
  measurement distinct from the rule itself.

If either condition ever changes — e.g. a future surface's conflicts need
to weigh the *semantic content* of competing Mental Objects rather than
their numeric fields — that would be a new, separate decision with its own
rationale, not a retrofit onto this policy.
