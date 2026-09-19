# Omega attention workspace v0.2

This is the input contract emitted by `aca-engine::attention_workspace` and
used by `generate_scenarios_v0_2.py`. It is not the v0.1 synthetic scenario
schema. The original v0.2 training recipe cited in the project journal was
not present in this checkout; this replacement provides a reproducible new
dataset, **not** the exact historical dataset or a newly trained model.

The user prompt is `Select the next attention operation for this cognitive
workspace:\n` followed by compact JSON. The workspace fields are:

| field | runtime meaning |
| --- | --- |
| `protocol` | `omega-attention-workspace/v0.2` |
| `attention_threshold` | Step 5 Coalition's score floor |
| `working_memory_capacity` | maximum active members |
| `current_focus` | previous Working Memory's highest-scoring ID or null |
| `candidates` | 1–9 candidate objects in the sampled training distribution |

Each candidate has `id` (UUID string), `kind` (serialized MentalObjectKind),
`activation_total` (ACT-R score, sampled in [-10, 10]), `surprise` (number or
null; at most one candidate has a number per generated tick), `confidence`
(0–1), `goal_priority` (0–1 or null; sampled for goals and intentions), `in_working_memory` (boolean),
`broadcast_count` (nonnegative integer), and `age_ms` (nonnegative integer).
The Rust runtime obtains these values from live graph objects; unlike the
generator, it does not clamp or manufacture them.

The output is compact JSON with exactly `operation`, `target`, `confidence`,
and `reason_code`. `operation` is one of `ATTEND`, `MAINTAIN`, `SWITCH`,
`SUPPRESS`, `IGNORE`; `target` is a candidate UUID or null. The synthetic
teacher ranks by `activation_total + surprise.unwrap_or(0)` above the
attention threshold. It adds a low-confidence distractor suppression rule.
The generator's confidence is a bounded score-margin heuristic, not
empirically calibrated correctness. `--balanced` rejection-samples equal
operation classes while retaining the same workspace shape; use it for a
new adapter rather than silently combining it with earlier data. These are
training labels, **not** observed human success labels; evaluation
against the generator alone cannot demonstrate live cognitive quality.

The model is consulted only when candidate count is 1–9 and every activation
is in [-10, 10]. Rust falls back to deterministic arbitration outside those
bounds. In-range alone is not evidence of good predictions.
