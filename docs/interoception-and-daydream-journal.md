# Journal: Interoceptive Self-Status and Memory-Grounded Daydreaming

*Covers one working session, 2026-08-23. Written up after the fact from the
session transcript.*

## 1. The starting question

User's framing: Omega's only standing idle behavior in practice is the
`self_status` duty, firing on a flat 15-minute clock
(`BoredomConfig::self_status_interval_ms`). Two questions followed - what
biological machinery would give Omega something closer to a real
self-check and a real daydream, and how to get there by *emulation* (real
signals the architecture already computes competing for attention through
its existing machinery) rather than by *hardcoding* (a scheduler bolting a
new special-cased behavior onto the tick loop).

## 2. What was already there

Before writing anything, worth recording what a read of `aca-engine`
turned up, because both changes below turned out to be extensions of
existing organs rather than new ones:

- `steps::drives::DriveState` - five continuously-tracked pressures
  (`uncertainty`, `curiosity`, `competence`, `social_connection`,
  `resource_pressure`), each grounded in a real signal the engine already
  computes elsewhere (Compare's prediction error, `CalibrationTracker`,
  `ExecutionTracker`, tier-pool saturation). Active-inference machinery,
  already built, just not yet consulted by `steps::boredom`.
- `steps::boredom::generate` - duty-first, invention-second: when Working
  Memory has been idle long enough, it either fires the standing
  `self_status` duty or asks Tier 1 to invent a thought from nothing via
  `idle_initiative_prompt`. The invention branch already existed; it just
  had no memory content to draw on.
- Whatever `boredom::generate` returns re-enters the cognitive cycle as an
  ordinary `Incoming::Boredom` → `SourceChannel::SelfGeneratedThought`
  observation, and goes through the *exact same* Observe → Compare →
  Predict → Coalition → Broadcast pipeline as a real conversational turn or
  sensor reading. This is what made the daydream change low-risk: anything
  produced here has to win Coalition honestly, same as everything else -
  no new insertion path, no `pending_admission` bypass, nothing bolted on.
- `steps::synthesize::synthesize` - idle-time abstraction of several recent
  Episodic memories into a new Semantic one. Not reused directly (its job
  is deliberate consolidation, on its own separate cadence), but its shape
  - sample real memory, ask a Tier for a pattern, let the caller give the
  result an honest shot at Coalition - is exactly the shape a
  memory-grounded daydream needed.

Given that, the two changes below are additions to `steps::boredom` and
`steps::drives`, not a new subsystem.

## 3. Self-status: from a clock to an interrupt

Interoceptive-inference framing (Seth & Craig's "beast machine" theory):
attention turns inward when homeostasis is actually disrupted, not on a
calendared schedule. `DriveState` already tracks the three pressures that
are genuinely *about* Omega's own functioning rather than the world or the
conversation - `uncertainty` (the predictive model doing poorly),
`competence` (miscalibrated self-reports or failing tool calls), and
`resource_pressure` (the constrained model tiers running hot).
`curiosity`/`social_connection` are about the world and the relationship,
so they're deliberately excluded from this reading.

Added `DriveState::self_monitoring_pressure()` (`max` of those three), and
changed `boredom::generate`'s self-status due condition from

```
last_self_status_at.is_none_or(|t| now.0 - t.0 >= self_status_interval_ms)
```

to that condition **or** `self_monitoring_pressure() >=
self_status_drive_threshold` (default `0.75` - deliberately high, same
"fail closed" precedent as `attention_min_confidence`, since this bypasses
the ordinary interval backstop entirely). The 15-minute interval stays as
an upper bound, not the only trigger. A sustained run of surprising input,
a miscalibrated tier, or saturated model pools now pulls a self-check
forward on its own, the same way real interoceptive disruption doesn't
wait for a routine checkup.

**Scope boundary, noted rather than solved:** this still only ever fires
inside `boredom::generate`'s existing idle-only branch (Working Memory
already required to be empty). A drive spike mid-conversation can't yet
preempt anything - self-status just gets its next idle window instead of
waiting out the full 15 minutes. Making self-status genuinely interrupt
capable outside idle windows would mean routing it through Coalition as a
real competing candidate, a larger change than this session's scope.

## 4. Daydreaming: from free invention to memory recombination

The existing invention branch asked Tier 1 to produce a thought with zero
grounding - `idle_initiative_prompt(self_summary, available_tools)`, no
memory content at all. Two theories point at the same fix:

- **Constructive episodic simulation** (Schacter & Addis): imagining and
  remembering share the same machinery - a daydream is memory fragments
  recombined, not generated from nothing.
- **Hippocampal replay** (Foster & Wilson): offline replay preferentially
  reactivates recently-encoded, salient experience, not a uniform draw
  over everything ever stored.

New function `steps::boredom::generate_daydream`, tried before the plain
invention fallback whenever `max(curiosity, uncertainty) >=
daydream_min_drive` (default `0.25`, deliberately low - default-mode-
network theory treats mind-wandering as idle *default* behavior, not
something needing strong justification) and `resource_pressure <
daydream_resource_pressure_ceiling` (default `0.7` - undirected
exploration is the first thing to give way under real tier scarcity).

`select_replay_sources` picks `daydream_sample_size` (default 3) dormant
memories: filters the graph to `MemoryRole::Episodic`/`Semantic` objects
(explicitly excluding `SelfMemory` - identity content is reinforced, not
recombined), sorts by `activation.total` - already computed every tick, no
new signal invented - and shuffles only within the top
`sample_size * pool_multiplier` (default 3x) before truncating. That last
step is what keeps this from being a stuck loop: sorting alone would
recombine the same one or two memories every idle tick.

The resulting fragments feed a new prompt, `daydream_prompt` (same
`{"confidence","response"}` JSON contract as `idle_initiative_prompt`, so
it reuses the identical sampling/selection path), asking Tier 1 to notice
"whatever connection, pattern, tension, or new possibility" occurs across
them. `BoredomStimulus` gained a `source_ids` field so the caller can tag
the resulting Observation's `data.source` as `"daydream"` rather than
`"boredom"` for anyone inspecting the event stream - purely observational
today, nothing downstream branches on it yet.

**Not done, flagged as the natural next step:** the daydream's source
memories aren't yet Hebbian-linked back to the thought it produced (the
way `synthesize::link_sources` or `interlocutor::reinforce_link` do for
their own outputs). Wiring that in would make a recombination event
actually restructure the graph's associative edges, not just produce a
one-off spoken thought - closer to what real memory replay is theorized to
do. Skipped this pass to keep the change contained to `steps::boredom` and
`steps::drives` without touching `apply_embedding_result`'s already-large
signature.

## 5. Why this shape, not a new subsystem

Both changes stayed inside `steps::boredom` deliberately. The alternative -
a new periodic trigger in `loop_actor.rs`, its own buffer of "material
since last daydream" (mirroring `episodic_since_last_synthesis`), its own
cadence config - would have worked, but would have meant a second path for
content to reach Working Memory, parallel to the one that already exists
and already competes fairly through Coalition. Reusing
`SourceChannel::SelfGeneratedThought` and the existing idle-eligibility
gate (`working_memory.is_empty()`, `idle_threshold_ms`, `min_interval_ms`)
meant zero new insertion mechanism, and it's the concrete answer to the
original "emulate, don't hardwire" question: daydream content doesn't get
special-cased into Working Memory, it has to win the same admission
process a surprising sensor reading does.

## 6. What shipped

- `steps::drives::DriveState::self_monitoring_pressure` - `uncertainty` /
  `competence` / `resource_pressure`, the max of the three.
- `steps::boredom::BoredomConfig` - six new fields
  (`self_status_drive_threshold`, `daydream_sample_size`,
  `daydream_min_sources`, `daydream_pool_multiplier`, `daydream_min_drive`,
  `daydream_resource_pressure_ceiling`), all with reasoned defaults, none
  requiring a call-site change anywhere that already used
  `BoredomConfig::default()`.
- `steps::boredom::generate_daydream` / `select_replay_sources` - new,
  private to the module except for what `generate` exposes.
- `steps::boredom::BoredomStimulus::source_ids` - new field, empty for the
  duty and for plain invention.
- `prompt_templates::daydream_prompt` - new, mirrors
  `idle_initiative_prompt`'s shape and self-context-block handling exactly.
- `loop_actor::tick()` - `boredom::generate`'s call site now threads
  `&self.graph`, `&self.drive_state`, and `&mut self.rng` through; the
  `Incoming::Boredom` arm tags `data.source`/`data.daydream_source_ids`
  when a daydream (not a duty or plain invention) fired.

12 new tests across `steps::boredom` and `steps::drives` (self-status
pulled forward by drive pressure; daydream firing, falling back on too few
sources, falling back on resource pressure, staying dormant below the
drive floor; replay-source selection excluding Self Memory and empty text,
and never selecting outside the salience-weighted pool). Full
`aca-engine` suite: 357 unit tests + 3 integration tests, all passing,
zero new clippy warnings.

## 7. Where this left it, mid-session

Both changes are additions to existing organs, not new machinery, and both
stayed inside the architecture's one honest path into Working Memory.
What's still open is exactly what got flagged along the way: self-status
can pull itself forward on drive pressure but still can't interrupt
mid-conversation, only get its next idle window sooner, and a daydream's
source memories aren't yet Hebbian-linked back to what it produced, so
recombination doesn't yet restructure the graph the way real replay is
theorized to. Both are natural next passes, deliberately left out to keep
this one contained.

*(Both were picked back up the same session - see below.)*

## 8. Closing both gaps

### Self-status becomes a genuine interrupt

`boredom_eligible`'s own gate (`working_memory.is_empty()`,
`wm_empty_since` past `idle_threshold_ms`) is still exactly what it was -
that's the *idle* self-check, and daydreaming/plain invention still only
ever run there, deliberately, since undirected activity genuinely should
wait for nothing else to be going on. What changed is that self-status
duty no longer has to wait for that gate at all. A new, independent check
in `tick()`, `self_status_interrupt_due`, fires whenever
`DriveState::self_monitoring_pressure()` clears a second, stricter
threshold (`BoredomConfig::self_status_interrupt_drive_threshold`, default
`0.9` vs. the idle pull-forward's `0.75`) - it does not require Working
Memory to be empty, only that no real input arrived this exact tick (input
still always wins the same tick, per the `Incoming` enum's existing
priority order) and the same shared `min_interval_ms` cost-control
backstop `boredom_eligible` already uses.

This gives a real three-rung ladder, each rung more disruptive than the
last: wait out the full interval (routine), pull the interval forward
while still idle (`self_status_drive_threshold`), or interrupt outright
regardless of Working Memory (`self_status_interrupt_drive_threshold`).
The interrupt-fired stimulus also reads differently -
`steps::boredom::self_status_stimulus(interrupt: bool)` is now the single
source of truth for the duty's text, shared by both the idle path and the
interrupt path, so the two can never drift on *what* it says while still
differing appropriately on wording ("nothing has needed my attention"
doesn't hold when something demonstrably has).

Whether the interrupt candidate actually *wins* Broadcast once it's up for
competition is untouched - Coalition's ordinary ranked-admission logic
decides that exactly as it would for any other candidate, capacity
allowing. That's the honest version of "interrupt capable" this
architecture's own competitive-admission model supports: the candidate
gets a real, immediate shot at attention instead of being deferred behind
an idle requirement, not a hard override of whatever's already there.

### Daydreams Hebbian-link back to their source memories

`generate_daydream`'s output already carried `source_ids`; what was
missing was actually writing `DerivedFrom` edges from each source back to
the thought once it resolves into a real graph object - the same edge
shape `steps::synthesize::link_sources`/`steps::interlocutor::
reinforce_link` already use for their own outputs.

The one real wrinkle: a daydream's text is freshly generated every time,
so it is essentially never an `embedding_cache` hit - it near-always takes
the asynchronous `embedding_worker` round trip, resolving on a *later*
tick than the one that generated it. A plain tick-scoped local for the
source ids (the first thing tried) silently failed to link anything for
exactly this reason - by the time the reentry landed, the tick that
carried the ids was long over. Fixed by threading `daydream_source_ids`
through `PendingObservation` itself (the struct that already exists
specifically to survive this same async boundary for everything else
about an in-flight observation), read back out once the reentry resolves,
right before the new object's id is known. The link-back site itself sits
once, right after `new_object_id` resolves from *either* path (cache hit
or reentry) - not duplicated per path.

### What shipped, second pass

- `steps::boredom::self_status_stimulus` - new, shared text source for
  both firing paths.
- `steps::boredom::BoredomConfig::self_status_interrupt_drive_threshold` -
  new field, default `0.9`.
- `loop_actor::tick()` - new `self_status_interrupt_due` gate, independent
  of `boredom_eligible`; new Hebbian link-back block using
  `reinforce_edge`/`EdgeKind::DerivedFrom`.
- `embedding_worker::PendingObservation::daydream_source_ids` - new field,
  empty for every non-daydream observation.

3 new tests: the interrupt firing with Working Memory genuinely non-empty
(proven via the interrupt-specific wording turning up in a `Compare`
event, not by predicting Coalition's admission outcome, which is a
separate and already-tested concern); a full daydream round trip proving
real `DerivedFrom` edges land on all three source memories, specifically
through the async reentry path (the common case, not the rare cache-hit
one); and `self_status_stimulus`'s two wordings sharing the same tool
request. Full `aca-engine` suite: 363 unit tests + 3 integration tests,
all passing.
