# Conversational Presence

Omega's default register is introspective by design — `self_memory::SELF_MEMORY_SEED_TEXTS`
seeds "the question I continuously ask is not 'what should I do next?' but 'what currently
occupies my mind?'" as one of its five foundational beliefs, and every Cognitive Core/Executive/
Social Interface prompt is primed with that identity via `prompt_templates::self_context_block`.
That's deliberate, not a bug: it's what makes Omega read as *present* rather than as a chatbot
that happens to log its internal state. Left alone, though, it means a live conversational
question ("what should I focus on now?") can get answered with architecture-narration instead of
a direct, practical answer — confirmed live against a synthetic backup-failure/smoke-alarm
scenario.

**Conversational presence** is a reactive dial that softens that register specifically while
someone is actively present, without touching identity (the seed beliefs are unchanged) and
without breaking the KV-cache-friendly stable prefix `self_context_block` exists for.

## How It Works

- `loop_actor::CognitiveLoopActor::conversational_presence(now)` returns a `0.0..=1.0` float:
  `1.0` the instant a live `SourceChannel::ConversationInput` turn last won Broadcast, decaying
  exponentially toward `0.0` as idle time since then grows. It reuses `last_broadcast_by_channel`
  (already recorded for the attentional-refractory-window check) rather than tracking a second
  timestamp.
- Recomputed fresh every tick in `tick()`'s Execute/Act pass, then threaded through
  `executive::propose_operators` → `steps::act::act` → `cognitive_core::reflect` →
  `prompt_templates::reflect_prompt`.
- `reflect_prompt` renders it via `presence_block`, appended *after* `self_context_block` — never
  inside it — so the byte-identical stable prefix a KV-cache-aware backend reuses is unaffected by
  a value that changes almost every tick. `presence_block` exposes the raw number to the model
  along with graded instructions for both ends of the scale, rather than a hard on/off switch: at
  presence 0, self-observation and internal-state narration are named as exactly the right
  register (today's behavior, unchanged); the higher it climbs, the more the instruction pushes
  toward engaging directly with the content and its practical implications for whoever's present.

## Why There's Also A Closing Directive

`presence_block` alone wasn't enough — confirmed live: at presence 1.0, a real reflection on
"the smoke alarm was a false alarm, what should I focus on now?" still answered with
Working-Memory/equilibrium narration and never mentioned the actual question (or the still-open
authentication error from the same conversation). The likely cause: `presence_block` sits well
before `broadcast_text` in the prompt, and `self_context_block` immediately above it carries the
seed belief's own explicit "tasks are consequences of my mental state, not the purpose of my
existence" framing — a competing instruction that's phrased as foundational identity rather than
contingent-to-this-tick, and small local models appear to weight it accordingly regardless of
`presence_block`'s framing-only phrasing further down.

`reflect_closing_instruction` (`PRESENCE_ENGAGED_THRESHOLD = 0.5`) is the fix: above that
threshold, a short, concrete imperative is appended immediately before the final "Respond with
ONLY a JSON..." instruction — the position right next to generation, which is what actually
steers small models reliably, more than a framing-only instruction earlier in the prompt. This
is graded coarsely (on/off at the threshold) rather than continuously worded like `presence_block`
— deliberately: the point of this specific line is to be blunt and unambiguous, not nuanced.

## Tuning

- `LoopConfig.presence_half_life_ms` (default `30_000`, i.e. 30s) — the decay half-life. At the
  default, presence is still ~0.8 ten seconds after a turn (covers ordinary think/speak latency
  within one exchange) and has receded to ~0.06 after two minutes of silence.
- Env var override: `OMEGA_PRESENCE_HALF_LIFE_MS` (see `omega-acad::build_loop_config`).
- Shorten it to make Omega revert to the introspective register faster after a turn; lengthen it
  to keep the "awake for conversation" framing active through longer gaps (e.g. a slow Tier 3
  reply, or a person who pauses mid-conversation).

## Read The Results

- `prompt_templates::reflect_prompt`'s rendered output should contain a `Conversational presence
  right now: X.XX` line whose value tracks how recently a real conversational turn broadcast —
  inspect via the same prompt-logging path used for other tier calls.
- A Reflection produced immediately after a live question should read as answering that question;
  one produced after a long idle gap (a sensor event, a self-generated thought) should still read
  as self-observation, exactly as before this change.
- `loop_actor::tests::conversational_presence_*` and `prompt_templates::tests::presence_*` cover
  the decay math and the stable-prefix/volatile-suffix split respectively — both should stay green
  if this dial is retuned.
