# Omega Capability Audit

Verification report on `crates/aca-engine`'s cognitive-mechanism test suites.
Prepared 2026-08-23. 16 tests written across 3 files; 16/16 passing as of the
last full run.

## Direct answer

**No test suite could settle that question.**

What follows is real, falsifiable engineering verification: specific, named
claims about how this codebase's mechanisms are wired together, checked
against the live system with negative controls that prove the checks aren't
tautological. One test also produced genuine model-authored content worth
reading on its own terms. None of it is evidence of cognition in the
philosophical or functional sense — that was never a claim a test file could
adjudicate.

## How the ask evolved

This audit went through three rounds, each triggered by a sharper version of
the same question.

1. **Round 1** asked for tests verifying Omega's cognitive capabilities. The
   result was seven tests of individual mechanisms — real ACT-R math, real
   embeddings — but every input was hand-picked specifically to make the
   mechanism succeed. Nothing in that suite could fail against a system where
   the pieces existed but weren't actually wired together.
2. **Round 2** pushed for a "bulletproof test to fail against." The fix was
   structural, not more tests of the same shape: drive the real, unmodified
   `CognitiveLoopActor` end to end, and pair every positive result with a
   negative control — an ablation flag or a structural variant that must
   actively diverge in the same run, or the positive result proves nothing.
3. **Round 3** asked the plainest question directly — *"so you think we've
   proved independent artificial cognition?"* — and the honest answer named
   six specific things nothing so far had touched: real models, organic
   (unseeded) emergence, contradiction handling, a non-cognitive baseline,
   model independence, and real wall-clock concurrency. This report covers
   all three rounds.

## What was built

### `cognitive_capabilities.rs` — 7/7 pass

`crates/aca-engine/tests/cognitive_capabilities.rs`

Mechanism-level tests using real ACT-R math and real (hashed, not semantic)
embeddings, but synthetic scenarios built to exercise each mechanism
directly: attention gating from real prediction error, decay and
reinforcement against the engine's own documented time constants, two-hop
spreading-activation rescue, GWT's hard capacity bottleneck, a SOAR impasse
spawning a real subgoal, a learning curve across repeated chunked
resolutions, and Self Memory outlasting — but not being exempt from —
ordinary decay.

### `cognitive_capabilities_end_to_end.rs` — 3/3 pass

`crates/aca-engine/tests/cognitive_capabilities_end_to_end.rs`

The same claims, but driven only through the actor's public `input_tx` /
`tick()` / snapshot surface — no private state, no hand-built graphs. Each
test is an adversarial pair: recall works live and is confirmed to fail
under `disable_recall`; two real turns form a real associative edge, and two
turns kept from ever overlapping don't; one real turn produces exactly one
Speak across 200 ticks, a bound that catches both silence and the
runaway-repetition bug this codebase has hit before.

### `cognitive_capabilities_gaps.rs` — 5/6 pass, 1 ignored*

`crates/aca-engine/tests/cognitive_capabilities_gaps.rs`

One test per gap named in Round 3. *The network-dependent real-model test is
marked `#[ignore]` so it doesn't block ordinary test runs — it was run
explicitly, twice, against live infrastructure (see the timeline below).

## Findings by gap

| Gap | Result | What it actually shows |
|---|---|---|
| 1. Real semantic content | failed first, then passed | Real Tier 3 model, not a scripted stub. Failed reproducibly (120/120 ticks) against the model then configured in production; passed against a swapped-in model with genuinely substantive output. See the timeline below — this is the most consequential finding in the audit. |
| 2. Organic emergence | pass | Two real, unseeded turns form a real edge; once both fully decay through genuine silence, an unrelated new topic does *not* spontaneously drag them back. Recall needs something currently live to spread from — a real boundary, not a defect. |
| 3. Content coherence | absence documented | A full source search turns up zero call sites that ever construct `EdgeKind::Contradicts`. Two directly contradictory statements are stored side by side, un-flagged, un-reconciled. Nothing to test because nothing exists yet. |
| 4. Non-cognitive baseline | pass | The identical scenario under the default config, under `disable_recall`, and under `disable_agenda`: one passes, the other two fail for two independently different structural reasons. The closest honest proxy for "compare against a dumber system" without inventing a separate one. |
| 5. Model independence | pass | Two runs, two `ChatClient`s returning unrelated text at identical confidence. Working Memory size, every memory-role count, and whether/how often Speak fires come out identical; only the spoken words differ. specs.md's Core Principle, operationalized and holding. |
| 6. Real concurrency | pass | The actual production entry point, `actor.run()`, under `SystemClock`, fed by two independent concurrent senders and real sleeps — the regime every other test in this codebase avoids. No tick errors; real cognitive activity observed afterward. |

## The real-model story

Gap 1 is worth walking through in full — it's the one test in this entire
audit that touched anything resembling actual reasoning, and it did not go
smoothly on the first attempt.

**Attempt 1 — `richardyoung/qwen3.6-27b-abliterated` on the LAN Tier 3 host.**
Real embedding round trip confirmed genuine (768-dimension `nomic-embed-text`
vectors, not the 8-dimension test stub). The Tier 3 reflect call then
failed — identically — on all 60 ticks, twice, across two independent runs
on two different network paths (120 failures total):

```
malformed response from tier T3: completion was empty (after one retry)
~2.7s per call, zero variance, zero successes
```

Deterministic, not a flake. `aca_tiers::OllamaClient` sends no `think` or
`num_predict` override; this specific "thinking" model plausibly spends its
whole completion budget on its reasoning block before `done` fires, leaving
`response` empty. **This is a real client/model integration bug, not a test
artifact** — and it would affect live `omega-acad` Tier 3 escalations the
same way.

**Between attempts:** the endpoint was swapped, live, to `qwen3.8:27b` —
same host, same family, also thinking-capable. Confirming the new tag
required correcting a background poll that had matched `qwen3:8b` (an
unrelated, already-present model) because an unescaped `.` in the match
pattern treated the model-name separator as a wildcard.

**Attempt 2 — `qwen3.8:27b` on the LAN Tier 3 host.** Passed. ~62 seconds of
real thinking latency, then a full, real cycle: Predict → Compare →
Coalition → Broadcast → reflect → next tick's Executive selects Speak →
renders verbatim. Prompted with *"name one thing that comes to mind about a
lighthouse standing in a desert,"* it produced:

> It's a warning about a danger that isn't there. The lamp turns and turns
> for no one — no fog, no rocks, no ships. Just a pulse of light aimed at an
> empty horizon of sand, still doing its job because the job was built into
> it, even though the context that made the job necessary is gone.
>
> — qwen3.8:27b, via Tier 3, rendered verbatim by the Social Interface

Substantive, on-topic, genuinely metaphorical — not an echo, not empty, not
a Self Memory recitation. The strongest single positive data point in this
entire audit.

**Reading this correctly matters:** it is one good answer to one distinctive
prompt, produced once. It is real evidence that the pipeline can carry
genuine model output end to end when a working model is behind it — it is
not a demonstrated track record, and the first model tried failed completely
and reproducibly. Whatever "thinking" happens here is entirely contingent on
which exact model is loaded on the other end of a LAN connection.

## What this audit does not establish

- **No claim about consciousness, understanding, or intentionality.** Every
  mechanism tested here is Tier 0 — deterministic, non-LLM bookkeeping by the
  architecture's own design. Passing tests show the bookkeeping is correctly
  wired; they say nothing about whether anything it's keeping books on
  amounts to cognition in any sense beyond the engineering one.
- **n = 1 on real content.** One real model, one real prompt, one real
  answer. That is evidence the wiring can carry genuine content — it is not
  evidence about quality, consistency, or reliability over any real span of
  use.
- **Belief contradiction handling doesn't exist.** This original audit found
  no producer for `EdgeKind::Contradicts`. The later goal-outcome learner can
  now produce a `Contradicts` link from a recent decision to an abandoned
  goal, and local propagation treats it as suppressive. It still does not
  detect or reconcile contradictory factual statements.
- **The empty-completion bug needs a real look.** If
  `richardyoung/qwen3.6-27b-abliterated` is still configured anywhere in
  production, live Tier 3 escalations against it are very likely failing
  silently the same way they did here, 120 times out of 120.
- **Nothing here ran for longer than a few minutes.** "Coherent long-term
  behaviour" is a specs.md design objective; the longest test in this audit
  covers 200 ticks in a fraction of a second against a scripted client. Real,
  multi-day operation under real models is untested by anything written
  here.

## Running it

```powershell
cargo test -p aca-engine --test cognitive_capabilities
cargo test -p aca-engine --test cognitive_capabilities_end_to_end
cargo test -p aca-engine --test cognitive_capabilities_gaps
```

The real-model test is excluded from the run above by design. It needs the
LAN inference hosts from `.env` reachable (via `OMEGA_EMBEDDING_BASE_URL`
and `OMEGA_TIER3_BASE_URL`), and `OMEGA_TIER3_MODEL` kept in sync with
whatever's actually loaded on that host (currently `qwen3.8:27b`):

```powershell
cargo test -p aca-engine --test cognitive_capabilities_gaps -- --ignored --nocapture
```

## Addendum: causal-intervention verification of one introspective claim shape

Prepared 2026-08-24, in response to a direct follow-up: instead of asking
Omega "are you conscious" or "what are you thinking about" (both dismissed
above as unanswerable by any test), verify a specific, checkable class of
self-referential claim Omega's language could make - **"X entered my
awareness because it displaced Y"** - against the real engine mechanism
behind it, with a real experimental intervention (disable the mechanism,
confirm the claim disappears), not just a positive-case test.

**What existed already:** Step 6 Broadcast (`steps::broadcast`) was already
a real, capacity-limited competition - GWT's admit-top-N rule, with
`BroadcastResult::released`/`newly_admitted` computed every tick from real
ACT-R activation and prediction-error scores. What did not exist: any
pairing of a specific released object with the specific admitted object
that caused its release, any distinction between a competitive loss and an
object simply no longer being renominated (a different, non-competitive
release path already present at `loop_actor`'s Step 5), or any surface
where that fact could reach a self-report.

**What was built** - `crates/aca-engine/src/steps/displacement.rs`:

- `verify_displacement(raw_candidates, capacity, entrant, evicted) -> bool`
  - the actual intervention: re-runs the real admission algorithm
  (`rank_candidates`/`admit_top_n`) with `entrant` removed, and checks
  whether `evicted` would now be admitted. This is the literal "disable the
  mechanism and see whether the behaviour disappears" step, callable on
  real tick data.
- `explain_release(...) -> ReleaseReason` - classifies a real release as
  `NotRenominated` (dropped out of competition, not beaten), `Outranked`
  (competed and lost, but to the combined admitted set, not any single
  identifiable entrant - a real shape of loss this module deliberately
  refuses to force into a one-cause story), or `Displaced(Displacement)`
  (competed, lost, and `verify_displacement` has confirmed exactly which
  real entrant is why).
- Wired into `loop_actor::tick()`'s real Step 6, not called from a test
  harness: every tick with a real release now computes this classification
  against that tick's real `raw_candidates`, emits a `displacement` event
  for any confirmed case, and republishes the most recent one as
  `EngineSnapshot::last_displacement` (`snapshot::DisplacementSummary`,
  including a `claim_text()` in the exact "X entered my awareness because
  it displaced Y" shape, built only from verified fields).

**How it's tested** - `crates/aca-engine/tests/displacement_causality.rs`,
driven only through the real actor's public `input_tx`/`tick()`/
`snapshot_rx`/`events_rx`, matching this file's own established bar for
"wired together" rather than "correct in isolation":

| Test | Result | What it shows |
|---|---|---|
| A real capacity-1 eviction between two real conversational turns | pass | `last_displacement` names the real winner and a real, verifiably-absent loser; independently corroborated on the real event log. |
| The same scenario with the capacity constraint relaxed (4 slots) | pass | No competitive `Displaced` verdict is ever produced - the actual mission's intervention: disabling the mechanism makes the claim disappear, not just its label. |
| An object absent from a tick's real candidates | pass | Never reported as "displaced" by anything - the `NotRenominated` guard against mislabeling ordinary turnover as a causal claim. |

Also caught live, during test construction: at capacity 1, Omega's *own*
generated reply is a real competing object that can itself win and then
lose the sole Working Memory slot before a second external turn even
arrives - confirmed by direct observation of the actor's real state, not
assumed. The end-to-end test's assertions were written to check the
verified claim against whatever the live actor's real state actually is,
rather than hardcoding which specific object a hand-picked scenario was
expected to evict.

**What this does not establish:** it says nothing about whether Omega
*means* it when language grounded in `last_displacement` gets spoken - only
that if it does say "X entered my awareness because it displaced Y," that
specific sentence now corresponds to a real, checkable, falsifiable fact
about the engine's own admission algorithm, rather than an LLM's plausible
guess. It also covers exactly one shape of introspective claim (competitive
Working Memory eviction); other claims about attention, memory, or affect
would each need their own mechanism-level grounding and their own
intervention test, not an extrapolation from this one.

```powershell
cargo test -p aca-engine --lib displacement
cargo test -p aca-engine --test displacement_causality
```

## Addendum 2: moving toward Global Workspace Theory parity

Prepared 2026-08-24, in response to a direct follow-up to the addendum
above: does the capacity-limited competitive admission Step 6 Broadcast
already implements amount to a biological Global Workspace? **No** - and
the honest target isn't neural-mechanistic parity anyway (this engine has no
operational analog for spiking populations, lateral inhibition, or
oscillatory binding, and inventing one to chase the label would be exactly
the kind of fabrication the rest of this audit exists to catch). The target
is *functional/computational* parity - reproducing GWT/GNW's behavioral
signatures - which specs.md's own framing ("emulate the functional
processes of cognition") already commits this architecture to. Four real
gaps were closed across three passes (heterogeneous neural-level dynamics
were scoped out entirely as not worth building) - this addendum now covers
the full roadmap as originally scoped.

### Specialist clouds (Phase 4) - `steps::interlocutor::social_cloud_anchors`

Real GWT specialists differ not just in *judgment* but in *access* - a
cortical specialist has its own local associative memory, not a shared
undifferentiated store. Step 4.5 Recall previously only ever spread from
current Working Memory. The first real specialist cloud spreads instead
from one specific interlocutor's own past utterances (a reverse scan over
real `DerivedFrom` edges `steps::interlocutor::reinforce_link` already
creates), independently ablatable via `AblationConfig::disable_social_recall`,
firing only on the tick a fresh utterance from that exact person resolves -
no "last known speaker" fallback.

**A real, pre-existing bug this caught:** the interlocutor anchor node is
documented as never a Coalition candidate, but nothing enforced it - any
ordinary conversational turn already links its own utterance straight to
its speaker, so *ordinary* Step 4.5 recall (unrelated to this work) could
already spread onto the interlocutor node itself and let it win Broadcast.
Confirmed live while building the social pass, which reaches the same node
just as easily. Fixed once, centrally, right before Coalition - covers
every candidate source, not only the new one.

**How it's tested** - `crates/aca-engine/tests/social_cloud_recall.rs`,
driven only through the real actor: a memory two hops from a returning
interlocutor's own old utterance resurfaces specifically because *that*
person spoke again, stays gone with `disable_social_recall` set, does not
leak to a *different* interlocutor speaking, and never fires on a turn with
no resolved speaker at all.

### Ignition hysteresis (Phase 1) - `steps::broadcast::decide_admission_with_hysteresis`

GNW's ignition asymmetry: sustaining an already-conscious percept is easier
than a subliminal one crossing into consciousness for the first time. Step
6 Broadcast previously re-ranked from scratch every tick with one flat
threshold, whether a candidate was already resident or brand new - no
hysteresis at all. `decide_admission_with_hysteresis` reuses the existing
rank-and-admit-top-N rule over a narrower pool: a previously-admitted
member stays eligible at the old `attention_threshold` floor; a fresh
candidate must clear a new, stricter `ignition_threshold`. Wired into all
seven of `tick()`'s deterministic-admission call sites (the trained
attention-model path is a separate mechanism, untouched here).

**Calibrating the default was the real work, not the mechanism.** The first
value tried, `0.0`, was reasoned the same way `attention_threshold`'s own
`-2.0` already is in this codebase (`BaseLevel(t) = -0.5·ln(t_seconds)`
crosses zero at t=1s) - and it broke two of this codebase's own established
recall capabilities: a dormant memory rescued by two real associative hops
lands around **-1.3** total activation on first re-entry, nowhere near
`0.0`. Measured directly (a temporary diagnostic, run once, removed), not
guessed twice. The shipped default, `-1.8`, keeps a real, modest gap over
the `-2.0` sustain floor (~37s vs. ~55s of base-level-alone dwell time)
while leaving comfortable headroom below what recall actually produces -
entering fresh is the harder case, but "harder" must not mean "recall can
never win admission," which `0.0` effectively did.

**How it's tested** - six unit tests in `steps::broadcast`'s own suite
(fresh admission at each bar, sustain-vs-fresh at the identical score, real
capacity pressure still evicts a sustained member, an equal-thresholds case
reduces exactly to the pre-existing plain function). No new end-to-end test
was added for the hysteresis *gap* itself - landing a live actor's real
ACT-R-computed activation inside a narrow 0.2-wide window on purpose was
judged more likely to produce a flaky test than a meaningful one. The
regression evidence instead: every existing end-to-end recall/displacement/
social-cloud test in this suite now runs *through* the new hysteresis path
in production and still passes, which is what caught the `0.0` regression
in the first place.

**What this does not establish:** no claim of equivalence to the neural
mechanism GNW actually describes - no heterogeneous parallel specialists
beyond this one cloud, no true fan-out broadcast (Executive/Act/Learn still
read the winner sequentially within one tick), no lateral-inhibition or
oscillatory dynamics, and only the deterministic admission path gained
hysteresis, not the trained attention model's own ATTEND/MAINTAIN/SWITCH/
SUPPRESS/IGNORE judgment.

```powershell
cargo test -p aca-engine --lib steps::interlocutor
cargo test -p aca-engine --lib steps::broadcast::tests::hysteresis
cargo test -p aca-engine --test social_cloud_recall
```

### Attention/ignition dissociation (Phase 2) - `EngineSnapshot::attended_not_ignited`

GNW's claim that attention (selection into the competition) and ignition
(becoming globally available) are separable mechanisms - something can be
attended without ever becoming conscious. Phase 1's hysteresis already
delivered this *mechanism* as an unavoidable byproduct (it needed two real
thresholds - `attention_threshold` for Coalition, `ignition_threshold` for
Broadcast - for hysteresis to mean anything at all). What was still missing
was proof the dissociation is real and observable, not just two numbers
that happen to differ: a new snapshot field, computed fresh every tick from
that tick's real `CoalitionCandidate` scores, naming every candidate that
cleared `attention_threshold` (attended) but didn't end up in the fresh
`working_memory` (never ignited). Wired into `SelfStatusTool` too, plainly
worded - no "I sense" phenomenological language this architecture has no
basis to claim, just the real fact of which candidates lost this tick's
ignition competition.

**Follow-up:** the one-shot limitation found during this phase has since been
closed. A losing candidate now leaves a bounded preconscious trace. The
cognitive scheduler reconsiders it at configured event deadlines while its
residual surprise decays, removes it immediately if it ignites, and expires it
after the configured trace window. This is still a deliberately small
computational analogue of subliminal persistence, not a claim of neural
fidelity.

**How it's tested** - `crates/aca-engine/tests/attention_ignition_dissociation.rs`,
driven only through the real actor: with `ignition_threshold` set
deliberately out of reach (sidestepping the same narrow-decay-window
precision problem noted above), a real conversational turn is confirmed
attended (via `attended_not_ignited`) on the tick it's evaluated while
`working_memory` stays empty across every subsequent tick - paired with a
negative control proving the identical turn ignites normally under
ordinary thresholds, so the positive result is really about the raised bar
and not some other reason the content could never win at all.

```powershell
cargo test -p aca-engine --test attention_ignition_dissociation
cargo test -p aca-engine --lib self_status_tool
```

### Parallel fan-out (Phase 3) - `steps::memory_formation::maybe_automatic_remember`

Real global broadcast makes ignited content available to multiple
independent consumers at once. Before this, `Operator::Remember` was one of
several proposals competing for the Executive's single winning slot each
tick alongside Speak/Ask/ContinueReflecting - a genuinely surprising
broadcast winner could be remembered *or* spoken about in a given tick,
never both, purely because they shared one arbitration. `maybe_
automatic_remember` runs memory formation automatically, outside
Executive's proposal/selection machinery entirely, whenever `precision_
weighted_surprise` clears a new threshold (default `1.0`, the midpoint of
the `[0, 2]` range that signal already clamps to elsewhere in this engine).
`Operator::Remember` itself is untouched - content that doesn't clear the
automatic bar still gets its ordinary shot at winning the competition,
exactly as before; the two paths compose cleanly because the automatic
pass runs first, so `already_remembered` naturally stops Executive from
redundantly re-proposing Remember the same tick it already fired
automatically.

**A real, disruptive interaction found and fixed, not routed around
silently:** the first full test run broke six pre-existing tests, not two.
Five were this codebase's own impasse/escalation machinery tests, which
turned out to rely on `Operator::Remember` always being available as one of
exactly two competing proposals for a fresh observation, specifically to
force deterministic ties for testing purposes - the automatic path
legitimately removes Remember from that competition for genuinely
surprising content, which is exactly its job, but it meant those tests'
tie-forcing mechanism no longer worked. The sixth was a real-`SystemClock`,
150-real-tick synthesis test whose finely-tuned timing shifted under the
same change. All six were fixed the same way, not by weakening the
mechanism: `automatic_memory_formation_surprise_threshold` set explicitly
to `f32::INFINITY` in each, opting a test that's about something else
entirely out of a new default-on mechanism it never asked for - the same
explicit-override discipline `ignition_threshold`/`attention_threshold`
already established for `permissive_config()`-style tests earlier in this
document.

**What this does not establish:** not literal wall-clock concurrency - the
automatic pass and Executive's own proposal/Act sequence still `.await`
sequentially within one tick(), one after the other in code. What actually
changed is that they no longer have to *win* against each other; true
concurrent execution would require splitting exclusive `&mut Graph` access
between them, a further, separate piece of work this pass doesn't attempt.

**How it's tested** - six unit tests on `maybe_automatic_remember` itself
(fires above threshold, refuses below it, refuses an already-remembered or
embedding-less candidate, graceful on a missing target) plus
`crates/aca-engine/tests/parallel_fan_out.rs`, driven only through the real
actor: a genuinely surprising turn produces both a real `automatic_remember`
event and a real Act-phase `speak` event on the *identical* `cycle_seq` -
paired with a negative control (threshold unreachable) proving ordinary
communicative decisions are entirely unaffected when the automatic path
never fires.

```powershell
cargo test -p aca-engine --lib memory_formation
cargo test -p aca-engine --test parallel_fan_out
```

This closes the GWT-parity roadmap as originally scoped in the discussion
that opened this addendum - Phases 1, 2, and 4 (one specialist) plus this
one. Phase 5 (heterogeneous neural-level dynamics) remains explicitly out
of scope, for the reasons given at the top of this addendum.

### Phase 5, revisited: what it would actually take

Prepared 2026-08-24, in response to a direct follow-up asking what Phase 5's
real prerequisites are, not just why it was skipped. Two separate pieces,
not one, and they are not the same size:

1. **Continuous-time competitive dynamics.** Naive mutual-inhibition
   relaxation applied before the existing `admit_top_n` sort was evaluated
   and rejected - for a strict top-N cutoff, it's a monotonic rescaling in
   the common case, changing no actual admission outcome unless
   deliberately built asymmetric specifically to flip orderings, which
   would be reverse-engineering a reason for it to matter rather than a
   real capability gap. The version that *is* real: divisive normalization
   / crowding - each candidate's effective activation suppressed in
   proportion to how many other strong candidates are active the same
   tick, so a busy cognitive moment measurably raises the real bar to
   ignite, independent of any one candidate's own score. Distinct from
   Phase 1's hysteresis (time, not crowding) and from the hard
   `working_memory_capacity` cutoff (caps the output, not the eligibility
   bar). Not yet built - a real, scoped, falsifiable target if pursued,
   not generic biological flavor.
2. **Feature-decomposed symbolic objects.** The actual prerequisite for
   Phase 5's original justification (the neural binding problem) to apply
   at all - today's `MentalObject` is atomic (one id, one embedding, one
   activation), so there is no "which feature belongs to which coalition"
   question to answer; the binding function that would need answering is
   already covered by real Hebbian associative edges. Making it a live
   question would mean decomposing every graph node into independently-
   competing sub-features (content, affect, source, modality) that then
   need reassembling - a representational rewrite touching every step that
   currently operates on whole `MentalObject`s (Predict, Compare,
   Coalition, Broadcast, Recall), not an incremental addition.

**Decision: (2) is explicitly shelved for a future ground-up rewrite
("Gen 3"), not abandoned.** It is too large and too central to retrofit
into the current architecture incrementally, and is a different-generation
design decision rather than a Gen 2 feature. Keep this in mind whenever a
Gen 3 rewrite is actually scoped - real oscillatory-binding/attention
research on this architecture stays symbolic-only without it. (1) was
pursued immediately afterward, in its crowding/normalization form - see
below.

### Crowding / divisive normalization (Phase 5's item 1) - `steps::coalition::apply_crowding_normalization`

Built the same session the "Phase 5, revisited" analysis above was written.
Each candidate's effective score is divided by `1 + crowding_strength *
(sum of every other candidate's positive score this tick)` - a quiet tick
leaves a candidate's score untouched; a busy one suppresses it, even though
nothing about that candidate's own ACT-R activation changed. Applied once,
centrally, right after `form_coalition` in `loop_actor::tick()`, so every
downstream consumer (the deterministic path, the hysteresis path, every
Broadcast fallback branch) sees the same crowding-adjusted field - and
deliberately *not* applied to the trained attention model's own JSON
workspace, which still sees real, un-normalized ACT-R scores, unchanged
from what it was calibrated against.

**Calibrating the default, again the real work.** `crowding_strength = 0.05`
was checked against the full test suite exactly as `ignition_threshold`'s
default was, and - unlike that one - it passed clean on the first try, no
regressions to fix. The live-actor test told a different calibration story:
an aggressive first attempt (`crowding_strength = 2.0`, a crowd of six)
didn't merely fail to demonstrate the effect, it overshot into a different
real phenomenon - mutual suppression strong enough that competitors choke
each other out *during their own establishment*, leaving only the single
earliest arrival standing (protected by hysteresis's lower sustain floor)
and nothing else, ever, breaking through. Confirmed live, not assumed: a
moderate value (`0.5`, a crowd of three) is what actually demonstrates "an
established crowd suppresses a new arrival" without that crowd first
suppressing *itself* out of existence.

**How it's tested** - six unit tests on `apply_crowding_normalization`
(lone candidate untouched, exact identity at `0.0`, mutual suppression
between two strong candidates, a negative-scoring bystander never
contributing to the suppressive pool, stronger settings suppress more)
plus `crates/aca-engine/tests/crowding.rs`, driven only through the real
actor: the identical target turn ignites when nothing else is competing
and fails to ignite once three other genuinely strong turns are already
resident - with `working_memory_capacity` set high enough that ordinary
capacity-limited eviction (already covered by `displacement_causality.rs`)
cannot be what explains the difference - paired with a negative control at
`crowding_strength = 0.0` proving the suppression is really attributable
to crowding.

```powershell
cargo test -p aca-engine --lib steps::coalition
cargo test -p aca-engine --test crowding
```

This closes Phase 5's item 1. Item 2 (feature-decomposed symbolic objects)
remains shelved for Gen 3, per the decision above.

---

One pre-existing, unrelated test
(`loop_actor::tests::a_daydream_hebbian_links_its_source_memories_to_the_thought_it_produced`)
was observed to fail under a full `cargo test -p aca-engine` run but pass in
isolation — confirmed as a pre-existing flake, not something introduced by
this audit.

## September 2026 continuation status

The event-driven substrate work is detailed in
[`event-driven-cognitive-substrate.md`](event-driven-cognitive-substrate.md).
It adds local object dynamics, sparse event/spike activation, signed
inhibition, eligibility traces, Tier-0 orienting and predictors, cautious
compiled speech, typed urgent sensor ingress, and actor/turn latency
telemetry. The full workspace suite currently passes, including over 450 engine
unit tests; one LAN-dependent real-model test remains intentionally ignored.

The v0.2 attention specialist is **not newly trained or validated on live cognitive outcomes** by
that work. Its historical model was reportedly trained, but the v0.2 data
recipe and schema cited by this repository were absent from this checkout.
They are now restored as a reproducible *new* synthetic recipe, with a
training preflight that rejects mixed v0.1/v0.2 rows. Admission cannot keep
stale or below-Coalition-floor members, and the distribution gate rejects
multi-surprise/non-finite workspaces. None of that supplies the missing live
decision outcomes or makes generated-label accuracy a performance claim.
An opt-in, isolated 20-step LoRA smoke run on the local RTX 4070 completed
after disabling incompatible local Triton compilation. It saved an adapter
under `training/attention-v0/outputs/` but **was not installed**. On 20
held-out synthetic v0.2 workspaces, it produced valid JSON on all 20 yet
matched only 25% of teacher operations and 20% of targets; it predicted
`SWITCH` for every case. That is a failed quality probe, not a usable
specialist. The evaluation now additionally measures whether an actionable
target resolves to a current workspace candidate. The runtime rejects
confident votes with invalid/absent targets and falls back to its
deterministic policy.
An isolated 100-step probe on a larger reconstructed v0.2 set also saved
cleanly but reached only 54% teacher-operation and 30% teacher-target
accuracy on 50 held-out cases. It predicted no `SUPPRESS` or `IGNORE`
cases and emitted fixed 0.9 confidence, reflecting the original
replacement generator's constant label. Neither adapter was installed.
The generator now supports separately named class-balanced datasets and
variable score-margin *heuristic* confidence for a subsequent iteration;
that is not empirical probability calibration or live success feedback.
For consulted votes, local telemetry records the actual specialist membership
and a same-tick deterministic shadow membership, plus their agreement.
That makes live divergence inspectable without treating agreement as proof
of better or worse behavior.
The configured loopback model was then tested against five simple v0.2
workspaces and returned well-formed but correct operations/targets in only
2/5. Its incorrect votes reported 0.64–0.75 confidence, above the engine's
old 0.6 trust gate. Warm calls were ~154–165 ms; one cold call timed out at
20 seconds. The local daemon now defaults the model to `off` even when an
endpoint is configured, supports non-controlling `shadow`, and requires
explicit `active` for model control. This is a deployment safety gate,
not specialist usability evidence.

At that point, remaining pieces of the original proposal were conditional, multi-step
operator compilation with outcome/safety semantics; additional genuinely
trained numeric specialists; and measured deployment response-time
percentiles. Local telemetry currently has no new spoken-turn samples, so
the response-time percentages remain targets rather than achieved SLOs.

## September 2026 goal continuation

The actor now learns three narrow executable two-step macros from actual
`ContinueReflecting -> Speak`, `ContinueReflecting -> Ask`, or
`ContinueReflecting -> Ignore` provenance
plus host-only successful-outcome feedback tied to three distinct terminal
observation IDs. It stores typed step names, exact-source conditions, and a
matching `SpokeExact`, `AskedExact`, or `Ignored` consequence, executes through
the direct speak/ask/ignore path, and immediately demotes any of them on
negative feedback.
Missing or inconsistent v2 contract fields
fail closed on reload; old uncontracted v1 records remain auditable but
cannot execute. An actor integration test proves no model call on the matured next
turn. Legacy repeated-speech records no longer execute without independent
success evidence. These are safe allowlisted programs,
not a general conditional operator-program compiler. The host has a
feedback sender but no automatic truth oracle; human/host labels remain
required for genuine success evidence.

A local in-process actor probe measured 100 warmed verified compiled turns at p95
762 µs actor-dequeue-to-Speak and p95 8.7 ms whole tick with model clients
absent and agenda/boredom/synthesis disabled. This validates the isolated
fast path only. Read-only analysis of legacy `omega.sqlite3` found
multi-minute cycles mostly in Act/Synthesize and no new turn samples.
Dormant intention surfacing has been cadence-bounded to remove a repeated
idle attention pulse, but its deployed latency effect remains unmeasured.

A separate loopback HTTP-to-WebSocket probe delivered 100 warmed, preseeded
verified turns (50 Speak and 50 Ask). Speak measured p50 1.520 ms and p95
7.977 ms; Ask measured p50 1.522 ms and p95 7.800 ms from submission to the
Act event, with 100/100 under 10 ms. This includes the real local API and actor
scheduler, but still uses an in-memory store and no inference/TTS/Godot/LAN
delivery. It cannot establish general deployed-turn percentiles.

A real `omega-acad` child-process probe, with all inference/media endpoints
explicitly blank and a temporary SQLite database, measured six distinct
innate greeting turns across HTTP submission and WebSocket delivery at p50
0.597 ms and 1.788 ms maximum in the latest run (an earlier run measured
0.60/1.86 ms). This confirms the actual daemon's local fast
path, not general inference-backed conversation or user-perceived audio/UI
latency.

A privacy-safe model-backed smoke probe used generic input and an actual
loopback Ollama Tier-3 model through the real HTTP, actor, Reflection,
Executive, Act, and WebSocket path. Its one foreground turn reached terminal
Speak in 300 ms. A fixed local embedding isolated chat turnaround because the
installed chat model does not expose Ollama embeddings. This is real
sub-second inference-path evidence, but one sample is not a percentile and
does not establish deployed multi-model/TTS/Godot/LAN performance. Diagnostic
runs exposed repeated Ignore on a previously handled Reflection. Ignore now
archives and releases that Reflection while preserving raw Observation history;
spoken/asked Reflections remain available to associative learning and synthesis.

The same child-process shape with the configured loopback attention model
in shadow mode measured p50 310 ms and 626 ms maximum for six greetings.
Zero decisions cleared the gate; seven explicit 300-ms attention timeouts
were recorded across foreground/follow-up ticks. Deterministic replies still
arrived, proving both fail-closed behavior and the present specialist's
unacceptable live latency cost.

The engine now records keyed numeric ORIENT inputs and admission IDs;
a local-only exporter refuses training unless at least 1,000 new-schema
examples have both classes in chronological train/eval. The legacy DB has
zero such keyed examples. No additional specialist can honestly be called
usable from that dataset yet.
The host can now label a recent Compare observation's independently judged
outcome from the local daemon console, with a separate command from the
verified-procedure feedback. The actor refuses duplicate/unknown IDs and
flushes keyed text-free Learn evidence. A second local-only exporter joins
that evidence to numeric ORIENT inputs and actual ignition while excluding
contradictory labels and sparse chronological/2×2 partitions. The legacy
database still has zero such labels. This improves the path toward a real
outcome-trained specialist but does not itself produce a usable one.
The accompanying dependency-free logistic trainer refuses fewer than 200
held-out rows and only writes a `shadow_only` artifact after balanced
accuracy and Brier-score gates. Its synthetic unit fixture validates the
gate mechanics only. With zero real labels, no artifact was trained or
promoted.
The Rust runtime can load that exact artifact format but independently
rechecks status, metrics, feature order, scales, and finiteness. Its only
capability is keyed post-Broadcast shadow forecasting; it cannot alter
admission or operators. This closes the train-to-runtime plumbing without
misrepresenting an untrained or observational model as usable.

A second specialist is now trained for the closed Speak/Ask/Ignore
communicative-intent judgment. It is a 1,024-feature hashed softmax model over
word, bigram, and character n-gram features, trained only on generic authored
intent statements and evaluated on a separate 30-case fixture. The checked
artifact reached 30/30 accuracy and 1.0 recall for every class. Its Rust loader
independently enforces exact feature/label schemas, finite parameter shape,
held-out sample/accuracy/per-class-recall gates, and `shadow_only` status.
The actor compares it with an already-selected Executive terminal operator;
a forced-disagreement test proves that a predicted Ask cannot replace the
selected Ignore. A 10,000-call local probe measured 88 microseconds p50 and
107 microseconds p95 per prediction. This establishes a usable prospective
shadow specialist, not authority over real ACA reflections; live agreement and
outcome evidence are still required before any control proposal.
