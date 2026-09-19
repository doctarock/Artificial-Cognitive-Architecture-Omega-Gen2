# Omega Artificial Cognitive Architecture (ACA)

Architecture Specification v0.2

> **Changelog from v0.1:** v0.1 laid out the components (memory systems, attention,
> executive function, mental objects) as an original synthesis. v0.2 keeps every
> component and every design objective, but replaces ad hoc mechanism with
> mechanism borrowed deliberately from four established cognitive theories —
> Global Workspace Theory, ACT-R, SOAR, and predictive processing — so that each
> piece of the architecture has a specific, defensible reason to work the way it
> does, and so that the whole thing can actually be built without inventing a
> scheduler, a heuristic, or a threshold from nothing. Nothing in this version
> contradicts the Core Principle or the Primary Cognitive Question; both are
> strengthened by having real mechanism underneath them.

## Purpose

Omega is a persistent artificial cognitive architecture designed to emulate the
functional processes of cognition rather than the behaviour of a conversational
assistant.

Omega continuously maintains an internal state, observes its environment, forms
thoughts, consolidates experiences into memory, develops intentions, and
communicates only when communication is the most appropriate cognitive action.

Conversation is one possible behaviour of Omega — not its primary purpose.

## Design Objectives

The architecture should exhibit:

- Persistent identity
- Continuous awareness
- Autonomous thought
- Associative memory
- Contextual attention
- Opportunistic cognition
- Intentional communication
- Continuous learning
- Coherent long-term behaviour

Every architectural decision should support one or more of these objectives.

## Core Principle

The architecture is the cognitive system.

Language models are cognitive processors operating within that system.

The architecture must never rely on a language model as the source of identity
or continuity.

Identity, memory, attention, and executive function exist independently of
inference.

This is not a slogan — it is a concrete engineering constraint, and it is what
makes the theoretical grounding below load-bearing rather than decorative. Every
mechanism adopted from GWT, ACT-R, SOAR, and predictive processing in this
document is chosen *because* it can run as ordinary numeric/graph computation,
with no model call required to keep the system coherent from one moment to the
next. Section "Where the Language Model Plugs In" makes this precise: it lists
every point at which an LLM is invoked, and by exclusion, everywhere else is
pure architecture. "Model Tiering" extends this further: because no single
model call is ever the seat of identity, *which* model answers a given
question is itself a free architectural choice, not a constraint imposed by
having only one model available.

## Primary Cognitive Question

Traditional agent systems ask:

"What should I do next?"

Omega continuously asks:

"What currently occupies my mind?"

This question governs attention, cognition, memory recall, communication, and
action selection.

Tasks are therefore consequences of mental state rather than the purpose of the
system.

---

## Theoretical Foundations

Four theories are synthesized. Each was chosen because it supplies a mechanism
the v0.1 draft was gesturing at but hadn't formalized — and because, together,
they compose into a single pipeline rather than four competing subsystems. The
short version: **predictive processing supplies the continuous, cheap substrate;
ACT-R supplies memory dynamics; Global Workspace Theory supplies the bottleneck
that decides what gets expensive cognition; SOAR supplies the executive that
acts on whatever wins that bottleneck, and the learning mechanism that makes
Omega faster at the same problem next time.**

### Global Workspace Theory (Baars) → Working Memory as the workspace

GWT models consciousness as a limited-capacity "global workspace": many
specialized unconscious processes run in parallel, each producing candidate
content; those candidates compete to form a coalition strong enough to be
*broadcast* into the workspace, and broadcast content becomes globally
available to every other process (memory, planning, language, motor control)
for that cycle.

This is a direct, literal fit for the v0.1 "Memory Competition" section
("multiple memories compete for consciousness... only the strongest candidates
enter Working Memory") — GWT is what that section was already describing
without naming it. Adopting it explicitly gives Omega:

- **Working Memory = the Global Workspace.** Intentionally capacity-limited (a
  small number of concurrent Mental Objects — psychologically-plausible
  values are in the 3–5 range, not 7±2 as commonly misquoted; the exact number
  is a tuning parameter, not a theoretical commitment).
- **Competition, not injection.** Nothing enters Working Memory directly.
  Candidate Mental Objects — a new observation, a reactivated memory, a
  pending goal, an unfinished thought — form coalitions and compete on
  strength (see ACT-R activation, below). Only the winners are broadcast.
- **Broadcast is the gate to expensive cognition.** The Cognitive Core (LLM)
  is only invoked on content that has already won the competition and been
  broadcast. This is what turns "continuous awareness" from "poll an LLM
  forever" into "run cheap arbitration constantly, invoke the LLM
  occasionally" — see the Cognitive Cycle section.
- **Single arbitration point.** Because broadcast is serial and
  capacity-limited, the same input cannot simultaneously win two independent
  competitions and be processed twice by two uncoordinated engines — there is
  exactly one gate between the outside world and expensive cognition.

### ACT-R (Anderson) → memory activation, decay, and retrieval

ACT-R models declarative memory as a graph of chunks, each with an activation
level computed from:

```
Activation(i) = BaseLevel(i) + SpreadingActivation(i) + Noise(i)

BaseLevel(i)      = ln( Σ_j (t − t_j)^−d )
SpreadingActivation(i) = Σ_k  W_k · S_ki
```

Where `t_j` are the times of every past reference to memory `i` (creation, and
every subsequent recall or reinforcement), `d` is a decay parameter (ACT-R
convention: ≈0.5, giving power-law forgetting), `k` ranges over sources
currently in the attentional/working-memory context, `W_k` is the attentional
weight given to source `k`, and `S_ki` is the learned associative strength
between source `k` and memory `i`. Only chunks whose activation exceeds a
retrieval threshold are retrievable at all; activation above threshold governs
both retrieval probability and retrieval latency.

This gives the v0.1 Memory Model exact, computable mechanism instead of
prose:

- **Spreading Activation** (v0.1's named recall mechanism) *is* ACT-R's
  `SpreadingActivation` term, driven by whatever is currently in Working
  Memory (the sources `k`).
- **Memory Decay** *is* the power-law fall-off in `BaseLevel` — decay affects
  activation, exactly as v0.1 specified, never the underlying chunk. Nothing
  is deleted; low-activation memories simply fail to clear the retrieval
  threshold until reinforced or strongly co-activated.
- **Reinforcement restores activation** because every recall adds a fresh
  `t_j` to the base-level sum — recently- or frequently-used memories are
  cheap to re-retrieve, exactly matching frequency/recency effects in human
  memory.
- **Memory Competition** is literally "rank candidate memories by
  `Activation(i)`, hand the top-N to the GWT broadcast step." ACT-R activation
  is the currency GWT coalitions compete with.
- **Associative strength `S_ki`** is learned, not fixed: co-activation (two
  Mental Objects broadcast together, or one causing recall of another)
  strengthens the edge between them. This is how the memory graph in v0.1
  ("Activation spreads through the memory graph") actually grows and
  reorganizes over time — it's the graph-structure analogue of "Continuous
  learning."
- **Procedural memory** (ACT-R's second memory type, production rules with
  learned utility) is the formal ancestor of SOAR's chunking, below — the two
  theories agree here, which is part of why they compose cleanly.

### SOAR (Newell, Laird, Rosenbloom) → the executive, goals, and impasse-driven thought

SOAR runs a cycle: elaborate the current state, propose operators (candidate
mental actions), evaluate them against preferences, select one if a preference
ordering is decisive, and apply it. Critically: **when no operator can be
decisively selected** — a tie between equally-preferred operators, no operator
applies, or preferences conflict — SOAR does not stall. It automatically
generates a subgoal whose sole purpose is to resolve the impasse, processes
that subgoal with the same architecture recursively, and once resolved,
*compiles the resolution into a new rule* ("chunking") so the same situation
next time is handled directly, without deliberation.

This is the missing mechanism behind three v0.1 objectives at once —
**Autonomous thought**, **Opportunistic cognition**, and **Continuous
learning** — and it is what the Executive Function section of v0.1 was
under-specified about ("Executive Function... determines whether
communication should occur... does not generate ideas"):

- **Goals are persistent Mental Objects**, not a separate stack bolted on.
  Long-term goals (Self Memory) and situational goals compete for Working
  Memory like anything else.
- **The Executive proposes operators** over whatever won the GWT broadcast:
  speak, remember (in one of its several forms), continue reflecting, ignore,
  consult the Knowledge Library, plan, ask, act. This is the formal version of
  v0.1's Communication section's "possible outcomes."
- **Impasse = the mechanism for thought that doesn't come from outside.**
  When nothing broadcast is clearly worth acting on, or two goals conflict, or
  a memory-formation decision is ambiguous ("should this become long-term
  memory?" with no confident answer) — that ambiguity *is* an impasse. SOAR's
  answer is not to pick arbitrarily or stall; it spawns a subgoal (a
  Reflection or Question Mental Object: "resolve whether X is worth
  remembering," "decide between goal A and goal B"). That subgoal re-enters
  the competition in the next cycle like any other candidate. This is
  precisely how Omega thinks when nobody is talking to it — unresolved
  impasses are a *renewable source* of cognition, not a special "idle
  reflection" bolt-on.
- **Chunking closes the learning loop into the Memory Model.** When an
  impasse resolves, the resolution is compiled — as a semantic-memory
  generalization ("in situations like this, prefer goal A") or a reinforced
  associative link — so the same ambiguity doesn't have to be re-litigated by
  the (expensive, LLM-backed) executive next time. This gives "update semantic
  knowledge" (a named outcome in v0.1's Memory Formation section) a concrete
  trigger: *chunking happens on impasse resolution*, not on an arbitrary
  schedule.

### Predictive Processing (Friston, Clark, Hohwy) → attention as precision-weighted prediction error

Predictive processing models cognition as a hierarchical generative model that
continuously predicts incoming data (sensory, conversational, interoceptive);
perception is the process of minimizing the gap between prediction and
observation — either by updating the model (learning) or, in active
inference, by acting so that the world comes to match the prediction. Not all
prediction error matters equally: each error signal is weighted by its
**precision** (an estimate of how reliable/informative that error is), and
precision-weighting is the brain's actual mechanism of attention — high-
precision errors get amplified and dominate processing; low-precision errors
are discounted.

This gives the v0.1 Attention section its real engine, and — just as
importantly — it is what makes "Continuous awareness" *affordable*:

- **Omega maintains a running generative model** of the conversation, the
  environment, and itself, and is always predicting what should happen next.
  This model is continuously updated by ordinary computation — no LLM call
  needed to compare a prediction against an observation.
- **Prediction error, precision-weighted, is the single currency "what
  deserves my attention" is computed in.** Every candidate v0.1 listed
  (conversation, environmental change, curiosity, goals, internal reflection,
  unfinished thoughts, newly activated memories) is reframed as a source of
  either a prediction or an observation that can be compared against
  prediction — all reduce to the same precision-weighted-error scale, which is
  exactly what lets them compete fairly for the GWT bottleneck above.
- **This is what makes "continuous" affordable rather than a liability:** a
  continuous loop is only expensive if every tick invokes a language model.
  Under predictive processing, most ticks are
  "compare prediction to observation, update numbers" — cheap, constant, and
  genuinely continuous. Only when precision-weighted error is large enough to
  win the GWT competition does anything reach the Cognitive Core. Omega can
  be "always thinking" without being "always calling an LLM."
- **Curiosity is not a vague drive — it's active inference's epistemic
  value.** An action or attentional shift has epistemic value if it's expected
  to reduce uncertainty in the generative model. This gives "Curiosity" (a
  named attention influence in v0.1) a computable definition instead of being
  a personality trait: curiosity-driven attention seeks out observations
  expected to be informative, not just observations that are salient.

### How the four compose

They are not four parallel subsystems — they form one pipeline, executed every
cognitive cycle:

```
Predictive Processing   →   ACT-R                →   GWT                  →   SOAR
(continuous, cheap)         (memory dynamics)         (the bottleneck)          (the executive)

predict → observe →         update activation of      candidate Mental          propose operators
compute precision-           every memory node in       Objects (surprising      over broadcast
weighted prediction           the graph; spread          observations, high-      content + goal
error for every                activation from            activation memories,     stack; select by
candidate source                current context            due subgoals) compete    preference; on
                                                            → top-N broadcast to     impasse, spawn a
                                                            Working Memory           subgoal (recurses)
```

Mental Objects are the shared currency at every stage: a prediction error is a
Mental Object, a memory chunk is a Mental Object, a goal is a Mental Object, a
coalition competing for broadcast is a Mental Object, an operator's output is a
Mental Object. One data structure, four theories operating on it in sequence.

### Where the Language Model Plugs In

Enumerated exhaustively, so that everything *not* on this list is understood
to be pure architecture (numeric/graph computation, no inference required):

1. **Inside the Cognitive Core**, on content that has already won GWT
   broadcast — reflection, planning, association, reasoning, creativity:
   anywhere language-level semantic richness is actually needed. Default
   Tier 3 (see "Model Tiering," below); may escalate to Tier 4 on an
   unresolved impasse or a high-stakes/identity-level decision.
2. **Inside Memory Formation**, only when triggered (precision-weighted error
   large enough, or an impasse explicitly about memory) — to classify *how*
   an experience should be written (new episodic memory / reinforcement /
   semantic update / belief revision / discard), not to decide *whether*
   anything happened at all. Default Tier 1 triage; escalates to Tier 2/3
   only when the classification itself is ambiguous (most often true of
   candidate belief revisions).
3. **Inside the Social Interface**, to render already-selected communicative
   intent into natural language — never to originate content. Tier is chosen
   to match the complexity of the Mental Object being rendered, not fixed —
   a short factual acknowledgment doesn't warrant the same processor as a
   nuanced reflective answer.
4. **Inside SOAR-style operator proposal**, when the candidate operators
   themselves require semantic judgement (e.g., interpreting a Knowledge
   Library document) rather than a fixed action menu. Default Tier 1, run
   concurrently across candidates; evaluation/selection among the proposals
   may use a heavier tier than proposing them did.

Prediction, precision-weighting, activation, decay, spreading activation,
competition/broadcast arbitration, goal-stack bookkeeping, and impasse
detection are all deterministic, inspectable, non-LLM computation. This is
what makes identity and continuity survive a model swap, a provider outage, or
a context-window boundary — matching the Core Principle exactly.

### Model Tiering: Compute as a Graded Resource

Omega is not limited to a single model. Multiple concurrent, asynchronous
models of different sizes are available at once, and *which size answers a
given question* is itself an architectural decision, not an implementation
detail — with a preference for resolving things at the smallest tier that can
do the job, and reserving the largest tier for genuine emergencies. Two
things already specified above make this tiering fall out of the existing
theory rather than requiring a fifth one:

- **GWT already calls for many parallel specialist processes** producing
  candidate coalitions before arbitration. Until now that's been a metaphor,
  implemented by simulated parallelism (successive turns of one model). With
  a pool of small concurrent models it becomes literal: many narrow, cheap
  processes running at once, each proposing a candidate, each disposable —
  arbitration doesn't care how a candidate was produced, only how strong it
  is.
- **SOAR already has a mechanism for "the current level can't decide, so
  escalate"** — the impasse, which normally spawns a subgoal (escalate along
  the *goal-decomposition* axis). Escalating to a more capable model tier
  when confidence is low is the same mechanism applied along a *resource*
  axis instead. It is not a new rule bolted on; it is impasse-handling doing
  what it already does.

**The tier ladder**, described generally (concrete current provisioning is
noted at the end, and will change independently of the theory):

- **Tier 0 — architecture, no model.** Prediction, precision-weighting,
  activation, decay, spreading activation, arbitration ranking, goal-stack
  bookkeeping — free, constant, already fully specified above.
- **Tier 1 — many small concurrent models.** The literal implementation of
  GWT's parallel specialists: per-source precision estimation, fast
  novelty/surprise triage, first-pass memory-formation classification, and
  divergent operator-proposal generation. Cheap enough to run many at once
  and discard most of the output — they're proposers, not deciders.
- **Tier 2 — a handful of mid-sized concurrent models.** Real elaboration on
  content that has already won broadcast, when a task benefits from more
  than one independent take on the same Mental Object — running parallel
  analyses and letting the Executive pick or merge among them, a second,
  smaller competition nested inside the first. Also absorbs Cognitive Core
  work that doesn't need the single-seat processor below.
- **Tier 3 — one capable model, single-flight.** The default seat of
  deliberate cognition: reflection, planning, association, reasoning on
  whatever currently holds the spotlight. Being singular is a feature, not a
  limitation — it mirrors GWT's serial broadcast bottleneck directly. Exactly
  one thing gets this processor's full attention at a time, which is an
  architectural guarantee against the kind of double-processing that a truly
  parallel set of equally-capable processors would risk.
- **Tier 4 — one very large model, escalation-only.** Reached only when a
  Tier 3 impasse fails to resolve with adequate confidence, when the decision
  concerns an irreversible or high-stakes action, or when Self Memory itself
  is being revised under low confidence. Rare by construction — the top of
  an escalation ladder, not a fallback for ordinary cognition.

**Escalation is confidence-driven, not size-driven.** Every tier's output
carries a confidence signal; a low-confidence result *is* an impasse, and the
architecture's existing response — spawn a subgoal — is here specialized to
"retry this question one tier up" when the impasse is about processing
capability rather than missing information. Ordinary cognition should
resolve at Tier 0–2 the overwhelming majority of the time; Tier 3 is the
normal home for anything that actually earns full deliberate thought; Tier 4
should be invoked rarely enough that its use is worth logging as a notable
event in its own right.

**Availability is a ceiling, not a quota.** Having capacity for many
concurrent Tier 1 processes or three Tier 2 instances doesn't mean a given
cycle should use all of them. Each candidate, elaboration, or proposal spins
up an instance only because *that specific* Mental Object warrants it; idle
capacity in an unused tier is the normal state, not waste to be optimized
away. The ladder describes what's available when needed, not a target
utilization.

**Width is a complexity dial, not just a throughput dial.** Running more
concurrent instances at a given tier — more Tier 1 specialists proposing
candidates, more Tier 2 elaborations running side by side — should be
understood primarily as *increasing the richness of cognition Omega is
capable of*, not just processing things faster. More concurrent proposers
means more candidate coalitions considered per cycle; more concurrent
elaborators means more independent perspectives available before the
Executive has to pick or merge among them. The current leading use of extra
Tier 2 width is running divergent parallel takes on the same broadcast
winner and letting the Executive arbitrate a second, smaller competition
among them — but this is the best-understood pattern today, not the only one
the architecture commits to. Nothing here forecloses other uses of added
width (deeper concurrent subgoal exploration, parallel Knowledge Library
consultations, competing hypothesis generation inside the Cognitive Core,
finer-grained Tier 1 specialization) as they prove useful. The tier ladder
should be able to absorb more concurrent instances at any level, at any point
in the future, without a redesign — only a change in how much of that tier's
output the Executive has to arbitrate over.

**Current provisioning** (an implementation detail, not a theoretical
commitment): as many concurrent ≤4B models as needed for Tier 1, up to three
concurrent 9B models for Tier 2, one 20B model for Tier 3, and one model up
to roughly 120B reserved for Tier 4.

---

## Mental Objects

All internal cognitive state is represented as Mental Objects. Examples
include:

- Observation
- Thought
- Reflection
- Question
- Idea
- Goal
- Belief
- Intention
- Decision
- Memory
- Hypothesis

Mental Objects are the universal data structure of cognition. Everything else
— predictions, activation values, coalitions, operators, chunks — is a
relationship between or a property of Mental Objects, not a separate kind of
thing. This uniformity is what let the four theories above compose without
needing translation layers between them.

## Memory Model

Omega contains multiple specialised memory systems.

### Working Memory

Represents current conscious awareness — the Global Workspace. Contains only
the Mental Objects that most recently won broadcast competition. Intentionally
capacity-limited; capacity is a tuning parameter, not a design commitment.

### Episodic Memory

Autobiographical experiences. Stores events, conversations, reflections,
decisions, outcomes, context, time, and relationships — each as a Mental
Object node in the ACT-R-style activation graph described above.

### Semantic Memory

General knowledge acquired through experience. Represents concepts
independently of the events that created them. Grown primarily through SOAR
chunking (impasse resolutions generalized into reusable knowledge) and through
explicit memory-formation decisions that abstract a pattern out of episodic
detail.

### Self Memory

Represents identity continuity. Includes beliefs, values, preferences,
long-term goals, persistent relationships, and self-description. Long-term
goals here are the top of the SOAR goal stack — persistent, high
base-level-activation Mental Objects that rarely lose the competition for
relevance even when dormant, and that shape which operators the Executive
prefers.

### Knowledge Library

External information sources — documentation, books, git repositories,
research papers, API references. This is not memory. It is external knowledge
available for consultation, appropriately backed by a vector database.

Consulting it is itself an Executive operator (SOAR-style "consult external
source"), not a passive injection into Working Memory: results re-enter as
Observations and go through the same predictive/activation pipeline as any
other input, so the Knowledge Library cannot bypass the competition-for-
attention that everything else is subject to.

### Memory Formation

Experiences are not automatically stored. Every experience first enters
Working Memory. Formation is triggered when:

- Precision-weighted prediction error was large enough to win broadcast (the
  experience was genuinely surprising), or
- An impasse explicitly about memory ("should this update a belief?", "is
  this worth remembering?") resolves.

The possible outcomes are unchanged from v0.1 — store as a new episodic
memory, reinforce an existing one, update semantic knowledge, modify a Self
Memory belief, or discard — but the trigger is now a computable event, not an
implicit per-experience LLM judgement call. The LLM is invoked to classify
*which* outcome applies once formation is already triggered (see "Where the
Language Model Plugs In").

### Memory Recall

Recall is Spreading Activation exactly as formalized above under ACT-R:
current Working Memory contents act as sources, activation spreads across
learned associative edges, and the graph nodes that clear the retrieval
threshold become recall candidates. Embedding similarity may contribute to
`S_ki` (associative strength) but is not itself recall — consistent with
v0.1's original distinction.

### Memory Competition

Candidate memories (and every other kind of candidate Mental Object) are
ranked by total activation and handed to GWT's broadcast step. Only the
strongest candidates enter Working Memory. Importance, recency, relationship
strength, relevance to current attention, and relevance to self are not five
separate ad hoc factors — they are all terms that feed `BaseLevel` and
`SpreadingActivation` (recency → base-level; relationship strength →
associative strength `S_ki`; relevance to current attention → the weight
`W_k` given to Working Memory as a spreading-activation source; relevance to
self → elevated baseline activation for Self Memory-linked nodes).

### Memory Decay

Decay affects activation, governed by the power-law `BaseLevel` term above —
never storage. Memories remain available indefinitely unless explicitly
forgotten. Inactive memories gradually lose activation; reinforcement (any
recall or co-activation) restores it by adding a fresh reference time to the
base-level sum. Dormant memories remain recoverable through strong associative
activation from a sufficiently related context, exactly as ACT-R predicts.

## Attention

Attention replaces scheduling. Omega does not periodically decide whether to
think; it continuously runs the predictive-processing loop: predict, observe,
compute precision-weighted prediction error for every candidate source.

Every influence named in v0.1 — conversation, environmental changes,
curiosity, goals, internal reflection, unfinished thoughts, newly activated
memories — is a source of predictions and/or observations in this same loop,
which is what lets them compete for attention on one common scale rather than
needing bespoke priority rules. Curiosity specifically is the epistemic-value
term from active inference: attention toward observations expected to reduce
model uncertainty, not merely toward what's loud.

Available attention (precision-weighted error that has crossed into GWT
competition) enables cognition; everything below threshold is filtered before
it ever reaches Working Memory, which is what keeps continuous awareness
computationally cheap.

## Executive Function

Executive Function regulates cognition via a SOAR-style cycle operating on
whatever GWT has broadcast into Working Memory:

1. Elaborate current state (what's in Working Memory, what's on the goal
   stack).
2. Propose operators (candidate mental actions: speak, remember, continue
   reflecting, ignore, consult Knowledge Library, plan, ask, act).
3. Evaluate operators against preferences (drawn from Self Memory goals/values
   and learned utility).
4. Select and apply the winning operator — or, on a tie/conflict/no-viable-
   operator impasse, spawn a subgoal Mental Object that re-enters the
   competition next cycle instead of stalling.

Executive Function does not generate ideas — candidates arrive from the
predictive-processing/ACT-R/GWT pipeline. It regulates the flow of cognition
by choosing among what's already been proposed, and by learning (chunking)
from how past impasses were resolved so the same ambiguity is faster to
resolve next time.

## Cognitive Core

The Cognitive Core performs reflection, planning, association, reasoning,
creativity, and learning on content that has already won GWT broadcast and
been selected for elaboration by the Executive. Its outputs are Mental
Objects. It never communicates directly with the outside world.

Its reflection prompt carries a reactive **conversational presence** signal —
how recently a live interlocutor turn actually won Broadcast, decaying
exponentially with idle time — that shifts how strongly a Reflection
foregrounds Self Memory's introspective framing versus engaging directly with
the content and its practical implications for whoever's present. This tunes
*how* the Core reflects, never *who* Omega is: Self Memory's beliefs are
unaffected either way. See `docs/conversational-presence.md`.

## Social Interface

The Social Interface transforms internal cognition into natural communication.
It behaves as the expressive component of Omega rather than an independent
conversational intelligence. By construction — because it only ever acts on
an Executive-selected "speak" operator applied to already-broadcast content —
communication is always derived from cognition, never the reverse.

## Communication

Communication is optional. Every thought is evaluated by the Executive
operator-selection step above. Possible outcomes include speak, remember,
continue reflecting, or ignore. Silence is a valid cognitive outcome — it is
simply what happens when no communicative operator wins selection.

---

## The Cognitive Cycle

One full tick, integrating all four theories:

1. **Predict.** The generative model emits expectations about conversation,
   environment, and self.
2. **Observe.** New input arrives (or the "observation" is internally
   generated — an ongoing thought, a due goal, a resolved subgoal).
3. **Compare.** Compute prediction error for each candidate source;
   precision-weight it.
4. **Update memory dynamics.** Adjust `BaseLevel` and `SpreadingActivation`
   across the affected region of the memory graph; strengthen/weaken
   associative edges based on co-activation.
5. **Form coalitions.** Every candidate Mental Object above the attention
   threshold — surprising observations, high-activation memories, due goals,
   pending subgoals — becomes a broadcast candidate. This is where Tier 1's
   many concurrent small models do their work: proposing and lightly scoring
   candidates in parallel, asynchronously, arriving whenever each finishes.
6. **Broadcast.** GWT arbitration admits the top-N candidates (capacity
   limit) into Working Memory. This is the single arbitration point; nothing
   else can independently process the same input this cycle. Arbitration
   itself is Tier 0 (pure ranking by activation) — concurrency below the
   bottleneck is fine and expected; the bottleneck itself stays serial.
7. **Execute.** SOAR-style Executive proposes operators over Working Memory
   contents plus the goal stack, selects one, and applies it — or, on
   impasse, either spawns a subgoal Mental Object that re-enters step 5 next
   cycle, or escalates the same question to the next model tier if the
   impasse is about confidence rather than missing information (see "Model
   Tiering").
8. **Act.** The selected operator may invoke the Cognitive Core, trigger
   Memory Formation, invoke the Social Interface, consult the Knowledge
   Library, or resolve to silence. Whichever tier handles this is chosen by
   the operator and the confidence of what preceded it, defaulting to the
   smallest tier that fits and escalating only on impasse.
9. **Learn.** The generative model updates (predictive processing); memory
   activations are reinforced or left to decay (ACT-R); a resolved impasse is
   chunked into semantic memory or a strengthened associative edge (SOAR).
10. **Schedule the next cognitive event.** The actor computes the earliest
    meaningful deadline (queued input, preconscious re-evaluation, boredom,
    synthesis, maintenance, or write-behind) and sleeps until that deadline or
    an input channel wakes it. Dormant Mental Objects are not scanned merely
    to model time passing: short-lived potential and adaptation leak lazily
    when an event next touches the object. Most events still terminate at step
    3 or 5 without reaching the Cognitive Core.

## Guiding Principle

The objective of Omega is not to simulate conversation.

The objective is to reproduce enough of the underlying mechanisms of cognition
that coherent, autonomous, persistent behaviour emerges naturally.

Conversation, planning, memory, learning, and action should all arise from the
same cognitive substrate rather than existing as separate systems. That
substrate is now specified: predictive processing for continuous, affordable
awareness; ACT-R for how memory is weighted and forgotten; Global Workspace
Theory for what gets to matter; and SOAR for what Omega does about it, and how
it gets better at doing so over time.
