# Local numeric specialist data

`export_orient_ignition.py` joins a resolved observation's seven numeric
orienting inputs to its same-cycle *actual* Working Memory admission. It reads
the SQLite DB in read-only mode and exports no text, prompts, endpoints, or
embeddings. The label is observed architecture behavior, not human success
feedback; a model trained on it would need separate held-out and live utility
checks before replacing a deterministic path.

```powershell
python training/specialists/export_orient_ignition.py `
  --database omega.sqlite3 `
  --train-out training/specialists/data/orient_train.jsonl `
  --eval-out training/specialists/data/orient_eval.jsonl
```

The export deliberately fails before writing files unless it finds at least
1,000 new-schema examples and at least 20 ignited and 20 non-ignited cases
in each chronological partition. The current legacy local DB has too few
keyed Compare events for that. Training a small network on two legacy events
would be fiction, not another usable specialist.

`export_orient_outcomes.py` instead joins those same seven numeric inputs
and actual ignition to independently judged *observed* consequences. The
local daemon prints a recent Compare observation ID; its own stdin accepts
`/outcome success <id>` or `/outcome failure <id>` after an operator judges
what actually happened. Ordinary conversation/API/agent input cannot supply
that label. The actor accepts a verdict once for a recently compared ID and
flushes a keyed, text-free Learn event. The exporter excludes contradictory
labels and reads the database in read-only mode.
An accepted `/feedback` verdict for a genuinely spoken routine turn also
produces this independent observation-outcome label automatically, even if
positive feedback cannot credit a two-step macro on an already compiled
turn. The operator need not enter both commands for the same observation.

```powershell
python training/specialists/export_orient_outcomes.py `
  --database omega.sqlite3 `
  --train-out training/specialists/data/outcome_train.jsonl `
  --eval-out training/specialists/data/outcome_eval.jsonl
```

It refuses to write unless at least 1,000 independent labels exist and all
four ignition/outcome cells have at least 20 examples in both chronological
partitions. The current local DB has zero. These are outcomes under the
*observed* action, not counterfactual evidence that the opposite attention
choice would have worked better. A trained specialist must stay shadow-only
until a prospective comparison establishes real utility; no model is declared
usable from the present data.

Once the exporter passes, train the deliberately tiny logistic forecaster:

```powershell
python training/specialists/train_orient_outcome.py `
  --train training/specialists/data/outcome_train.jsonl `
  --eval training/specialists/data/outcome_eval.jsonl `
  --output training/specialists/outputs/orient_outcome.json
```

The trainer standardizes from the training partition only, class-balances
the loss, and requires at least 200 held-out chronological rows, balanced
accuracy >= 0.65, and Brier score <= 0.22 before writing an artifact. Saved
artifacts are marked `shadow_only`; passing an observational prediction gate
does not justify controlling attention. Its unit test uses synthetic
separable/noisy fixtures solely to verify these mechanics—it is not evidence
that an ACA specialist has passed them on real data.

Set `OMEGA_ORIENT_OUTCOME_MODEL_PATH` to a passing local artifact to load it.
The Rust loader independently rechecks schema, exact feature order, finite
parameters, positive scales, held-out sample count, balanced accuracy, Brier
score, and `shadow_only` status. A loaded model emits keyed
`orient_outcome_shadow` forecasts after actual admission is known. Its API
has no control operation, and those forecasts do not change Working Memory,
operators, or actions. This makes a future artifact operational for live
validation without prematurely making it authoritative.

## Communicative-intent admission probe

`probe_communicative_intent.py` tests a loopback Ollama model against a fixed,
generic 30-case `speak`/`ask`/`ignore` benchmark. It refuses non-loopback URLs
and sends no ACA runtime content. Model loading is warmed outside the measured
distribution. Admission requires every response to parse, at least 90% overall
accuracy, at least 80% recall for every class, and p95 latency no higher than
1.5 seconds:

```powershell
python training/specialists/probe_communicative_intent.py --model qwen2.5:1.5b
```

Passing this small closed-task probe is only enough to justify prospective
shadow validation. It does not by itself authorize replacing Executive
selection or establish quality on ACA's real reflection distribution.

On 2026-09-17 the installed loopback candidates did not clear admission. The
1.5B candidate was fast (73.17 ms p95) but reached only 76.67% accuracy and
40% `speak` recall. The 1B candidate reached 33.33% accuracy and collapsed
almost entirely to `speak`. The larger candidate exceeded the 30-second
warm-up timeout. None is wired as a specialist; these negative results are
kept so speed alone cannot be mistaken for usability.

The rejected generative candidates are not the only implementation option.
`train_communicative_intent.py` trains a dependency-free 1,024-feature hashed
softmax model from generic authored intent statements, then evaluates it on
the separate 30-case admission fixture above. Word, bigram, and character
n-gram features are L2-normalized; hashing and inference are implemented
identically in Python and Rust.

```powershell
python training/specialists/train_communicative_intent.py `
  --output training/specialists/outputs/communicative_intent.json
```

The checked artifact reached 30/30 held-out accuracy with 1.0 recall for each
class. The Rust loader independently requires the exact schema, feature and
label order, finite 3x1,024 parameters, at least 30 held-out cases, at least
90% accuracy, at least 80% per-class recall, and `shadow_only` status. Set
`OMEGA_COMMUNICATIVE_INTENT_MODEL_PATH` to load it. It compares its prediction
with an already-selected Executive Speak/Ask/Ignore action on a Reflection and
emits `communicative_intent_shadow` telemetry. Its type and actor integration
have no route to propose or execute an operator. Generic held-out quality makes
the specialist usable for prospective shadow measurement, not authoritative
on ACA's live distribution.
