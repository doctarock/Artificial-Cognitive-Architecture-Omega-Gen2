#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-training/attention-v0/.venv-linux/bin/python}"

export HF_HOME="${HF_HOME:-$PWD/models/.hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PWD/models/.hf-cache}"

"$PYTHON" training/attention-v0/generate_scenarios.py --train 5000 --eval 500 --seed 3407
"$PYTHON" training/attention-v0/evaluate_attention.py \
  --predictions training/attention-v0/data/eval.jsonl \
  --gold-field output \
  --prediction-field output

"$PYTHON" training/attention-v0/train_unsloth.py \
  --model unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit \
  --train training/attention-v0/data/train.jsonl \
  --eval training/attention-v0/data/eval.jsonl \
  --output training/attention-v0/outputs/smoke-omega-attention-v0-lora \
  --max-steps 20 \
  --batch-size 1 \
  --grad-accum 4
