#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-training/attention-v0/.venv-linux}"

python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

# CUDA 12.1 wheels are a stable default for recent NVIDIA drivers.
"$VENV_PATH/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"$VENV_PATH/bin/python" -m pip install -r training/attention-v0/requirements.txt

"$VENV_PATH/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import unsloth
print("unsloth", getattr(unsloth, "__version__", "unknown"))
PY
