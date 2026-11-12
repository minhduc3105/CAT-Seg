#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo '[setup] Preparing isolated CAT-Seg environment (Python 3.10, PyTorch CUDA 11.8).'
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.10 .venv-r2s100k
else
  PYTHON_BIN="${PYTHON_BIN:-python3.10}"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[setup] ERROR: $PYTHON_BIN not found. Install Python 3.10 or install uv." >&2
    exit 1
  }
  "$PYTHON_BIN" -m venv .venv-r2s100k
fi
.venv-r2s100k/bin/python -c 'import ensurepip; ensurepip.bootstrap()'
.venv-r2s100k/bin/python -m pip install --upgrade pip wheel ninja "setuptools==69.5.1"
.venv-r2s100k/bin/python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
.venv-r2s100k/bin/python -m pip install -r r2s100k/requirements.txt
.venv-r2s100k/bin/python -m pip install fvcore==0.1.5.post20221221 iopath==0.1.10 omegaconf==2.3.0 cloudpickle==3.0.0 pycocotools==2.0.7 timm==0.9.12 ftfy==6.1.3 regex==2023.12.25 einops==0.7.0
if [ ! -d .deps/detectron2 ]; then
  mkdir -p .deps
  git clone --depth 1 --branch v0.6 https://github.com/facebookresearch/detectron2.git .deps/detectron2
fi

# Detectron2's editable build imports torch. Disable pip's isolated build env,
# which otherwise hides the torch installation in this venv (ModuleNotFoundError).
MAX_JOBS="${MAX_JOBS:-4}" .venv-r2s100k/bin/python -m pip install --no-build-isolation -e .deps/detectron2

echo '[setup] Verifying installed packages and CUDA access...'
.venv-r2s100k/bin/python - <<'PY'
import cv2
import detectron2
import open_clip
import torch

print(f"Python environment: ready")
print(f"PyTorch: {torch.__version__} (CUDA runtime {torch.version.cuda})")
print(f"Detectron2: {detectron2.__version__}")
print(f"OpenCLIP: {open_clip.__version__ if hasattr(open_clip, '__version__') else 'import OK'}")
print(f"OpenCV: {cv2.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print('[setup] WARNING: CUDA is unavailable. Check nvidia-smi and the host driver before inference.')
PY

echo '[setup] Environment setup complete.'
echo '[setup] Next: run inference with .venv-r2s100k/bin/python r2s100k/pipeline.py infer ...'
