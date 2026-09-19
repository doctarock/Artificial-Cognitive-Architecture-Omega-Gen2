param(
    [string]$VenvPath = "training/attention-v0/.venv"
)

$ErrorActionPreference = "Stop"

python -m venv $VenvPath
& "$VenvPath/Scripts/python.exe" -m pip install --upgrade pip setuptools wheel

# This project machine currently has a CUDA 13.0-capable driver. PyTorch cu121
# wheels run on newer drivers and are broadly compatible with Unsloth Core.
& "$VenvPath/Scripts/python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
& "$VenvPath/Scripts/python.exe" -m pip install -r training/attention-v0/requirements.txt --no-deps
& "$VenvPath/Scripts/python.exe" -m pip install aiohttp cut_cross_entropy diffusers hf_transfer huggingface-hub pandas pyarrow requests safetensors tokenizers tqdm triton-windows tyro

# Some Windows dependency resolutions try to replace the CUDA PyTorch wheel
# with an old CPU wheel. Re-assert CUDA PyTorch last.
& "$VenvPath/Scripts/python.exe" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

@'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import unsloth
print("unsloth", getattr(unsloth, "__version__", "unknown"))
'@ | & "$VenvPath/Scripts/python.exe" -
