#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.10 .venv-r2s100k
else
  PYTHON_BIN="${PYTHON_BIN:-python3.10}"
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
.venv-r2s100k/bin/python -m pip install -e .deps/detectron2
