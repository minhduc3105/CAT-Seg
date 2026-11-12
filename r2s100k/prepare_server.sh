#!/usr/bin/env bash
# Run on the server after sync_to_server.sh.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash r2s100k/prepare_server.sh [--dataset-root /absolute/path] [--skip-smoke]

Sets up .venv-r2s100k, verifies/downloads the official CAT-Seg ViT-B/16
checkpoint, prepares server-local manifests, and infers one validation image.
Default dataset root: CAT-Seg/data/r2s100k
EOF
}

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="$repo/data/r2s100k"
smoke=1
while (( $# )); do
  case "$1" in
    --dataset-root)
      if (( $# < 2 )); then echo '[server] Missing --dataset-root value.' >&2; exit 2; fi
      dataset_root="$2"; shift 2 ;;
    --skip-smoke) smoke=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[server] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
cd "$repo"

for split in train val test train_labels val_labels test_labels; do
  if [[ ! -d "$dataset_root/$split" ]]; then
    echo "[server] Missing dataset directory: $dataset_root/$split" >&2; exit 1
  fi
done

command -v nvidia-smi >/dev/null || { echo '[server] NVIDIA driver/nvidia-smi is required for CUDA inference.' >&2; exit 1; }
nvidia-smi >/dev/null || { echo '[server] NVIDIA GPU is unavailable.' >&2; exit 1; }

if [[ -x .venv-r2s100k/bin/python ]] && \
   .venv-r2s100k/bin/python -c 'import cv2, detectron2, open_clip, torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo '[server] Existing Python environment and CUDA are ready.'
else
  echo '[server] Installing Python environment...'
  bash r2s100k/setup_env.sh
fi
.venv-r2s100k/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in PyTorch"; print("[server] GPU:", torch.cuda.get_device_name(0))'

checkpoint=checkpoints/r2s100k/catseg_vitb16.pth
expected_sha256=a201a50f94f4c04196e3083773832b736c2b8b81d27ea2c59cc13af3aff05760
if [[ ! -f "$checkpoint" ]]; then
  command -v curl >/dev/null || { echo '[server] curl is required to download the checkpoint.' >&2; exit 1; }
  mkdir -p "$(dirname "$checkpoint")"
  tmp="$(mktemp "${checkpoint}.tmp.XXXXXX")"
  trap 'rm -f "$tmp"' EXIT
  echo '[server] Downloading official CAT-Seg ViT-B/16 checkpoint...'
  curl --fail --location --retry 3 --output "$tmp" \
    'https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/de5f330f3fe6a28e19b2f0c7855a6e3b4928a76a/model_base.pth'
  printf '%s  %s\n' "$expected_sha256" "$tmp" | sha256sum --check --status
  mv "$tmp" "$checkpoint"
  trap - EXIT
fi
printf '%s  %s\n' "$expected_sha256" "$checkpoint" | sha256sum --check || {
  echo '[server] Checkpoint hash mismatch; remove the file and rerun setup to download it.' >&2
  exit 1
}

manifest_dir=r2s100k/runs/server_data
echo '[server] Creating manifests with this server’s dataset paths...'
.venv-r2s100k/bin/python r2s100k/pipeline.py prepare \
  --dataset-root "$dataset_root" --output "$manifest_dir"

if (( smoke )); then
  echo '[server] Running one-image CUDA smoke inference...'
  .venv-r2s100k/bin/python r2s100k/pipeline.py infer \
    --manifest "$manifest_dir/manifests/val_pilot_images.json" \
    --checkpoint "$checkpoint" --output r2s100k/runs/server_smoke_one \
    --device cuda --max-images 1
fi

echo '[server] Ready. To infer the test split:'
echo '  .venv-r2s100k/bin/python r2s100k/pipeline.py infer \'
echo "    --manifest $manifest_dir/manifests/test_images.json \\"
echo "    --checkpoint $checkpoint \\"
echo '    --output r2s100k/runs/server_test --device cuda'
