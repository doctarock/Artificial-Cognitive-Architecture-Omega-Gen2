# Omega Attention v0.1 Training

## v0.2 workspace dataset

The engine now emits `omega-attention-workspace/v0.2`, not the v0.1
`omega-attention-scenario` fields used below. Use the separate
[`SCHEMA_V0_2.md`](SCHEMA_V0_2.md) and generator to create *new* v0.2
synthetic training/evaluation data:

```powershell
python training/attention-v0/generate_scenarios_v0_2.py --train 5000 --eval 500 --seed 3407
```

The default outputs are `data/train_v0_2.jsonl` and `data/eval_v0_2.jsonl`.
Pass those paths to `train_unsloth.py`; never train the runtime v0.2 client
on the v0.1 files below. The historical v0.2 model mentioned in the
project journal was trained with a recipe absent from this checkout; this
generator is a reproducible replacement, not proof of identical labels or
live quality. Do not overwrite the installed historical model without
evaluating the replacement against real attention decisions.
`train_unsloth.py` now checks every train/eval row's protocol before loading
the GPU model and rejects a mixed v0.1/v0.2 dataset. For a v0.2 run, pass
`--protocol omega-attention-workspace/v0.2` explicitly and save into a new
output directory.

Run the no-GPU schema and preflight checks with
`python training/attention-v0/test_v0_2_schema.py`.
On the local Windows PyTorch/Unsloth installation, a Triton symbol mismatch
can prevent compiled training at step 0. `--disable-compile` selects the
installed Unsloth and PyTorch compile-disable settings for this process only;
it does not alter the venv or installed model.
`infer_unsloth.py` accepts both protocol versions and decodes only generated
tokens. Use `--eval-jsonl data/smoke_eval_v0_2.jsonl --predictions-out
data/smoke_predictions_v0_2.jsonl --limit 20` to assess a short adapter
against *held-out* synthetic labels; do not evaluate labels against
themselves as if that measured model accuracy.
The evaluator separately reports `resolvable_action_rate`: an actionable
operation's target must be a real candidate UUID. JSON validity alone is
not enough to trust the model in the engine.
The 100-step local probe on the original reconstructed set reached only 54%
operation and 30% target accuracy on 50 held-out cases, with no `SUPPRESS`
or `IGNORE` predictions and fixed 0.9 output confidence. Both its adapter
and the earlier 20-step smoke adapter remain experimental and uninstalled.
The generator now offers `--balanced` and margin-based *heuristic* confidence
for a subsequent, separately named training iteration; previous saved data
and adapters were not overwritten. Margin confidence is not measured
probability calibration.

First training target: a small attention policy.

Why this first:

- the operation set is tiny: `ATTEND`, `MAINTAIN`, `SWITCH`, `SUPPRESS`, `IGNORE`
- synthetic labels are easy to inspect
- outputs are compact JSON, matching the Omega cognitive protocol style
- benchmarking against the heuristic policy is straightforward

## 1. Generate Data

```powershell
python training/attention-v0/generate_scenarios.py --train 5000 --eval 500 --seed 3407
```

This writes:

```text
training/attention-v0/data/train.jsonl
training/attention-v0/data/eval.jsonl
```

Each row contains `messages`, `input`, `output`, and `metadata`.

## 2. Install Training Dependencies

Use a CUDA-capable Python environment. The exact PyTorch wheel depends on your
GPU/driver, so install PyTorch first, then:

```powershell
pip install -r training/attention-v0/requirements.txt
```

On Windows, create an isolated venv with:

```powershell
powershell -ExecutionPolicy Bypass -File training/attention-v0/setup_windows_venv.ps1
```

On Linux with NVIDIA CUDA:

```bash
bash training/attention-v0/setup_linux_cuda.sh
```

Unsloth install details change by platform. If this fails, use the current
instructions from:

```text
https://github.com/unslothai/unsloth
```

## 3. Smoke Test The Data

```powershell
python training/attention-v0/evaluate_attention.py --predictions training/attention-v0/data/eval.jsonl --gold-field output --prediction-field output
```

This should report 100% because it compares generated labels with themselves.

## 4. Train

Default starter run:

```powershell
python training/attention-v0/train_unsloth.py `
  --model unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit `
  --train training/attention-v0/data/train.jsonl `
  --eval training/attention-v0/data/eval.jsonl `
  --output training/attention-v0/outputs/omega-attention-v0-lora `
  --max-steps 200
```

If using the venv:

```powershell
training/attention-v0/.venv/Scripts/python.exe training/attention-v0/train_unsloth.py `
  --model unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit `
  --train training/attention-v0/data/train.jsonl `
  --eval training/attention-v0/data/eval.jsonl `
  --output training/attention-v0/outputs/omega-attention-v0-lora `
  --max-steps 200
```

On Linux, run the first smoke training attempt:

```bash
bash training/attention-v0/run_linux_smoke.sh
```

For a first attempt, keep `--max-steps` small. Once the loss moves and output
JSON validates, increase steps or examples.

## Output Contract

The trained model should emit only compact JSON:

```json
{"operation":"SWITCH","target":"observation_4","confidence":0.91,"reason_code":"HIGHER_PRIORITY_INTERRUPT"}
```

No chain-of-thought, no prose, no hidden memory IDs.
