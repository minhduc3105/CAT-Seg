#!/usr/bin/env bash
# Run on the local machine: sync source, R2S100K splits, and an optional checkpoint.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash r2s100k/sync_to_server.sh USER@HOST /absolute/remote/CAT-Seg [LOCAL_DATASET_ROOT]

Defaults:
  LOCAL_DATASET_ROOT  /run/media/minhduc/D/Study/r2s100k
  SSH_PORT            22 (set as an environment variable to override)

The remote dataset will be placed in CAT-Seg/data/r2s100k.
The checkpoint is copied when checkpoints/r2s100k/catseg_vitb16.pth exists locally;
otherwise prepare_server.sh downloads it on the server.
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# < 2 || $# > 3 )); then usage >&2; exit 2; fi

remote="$1"
remote_repo="${2%/}"
local_dataset="${3:-/run/media/minhduc/D/Study/r2s100k}"
ssh_port="${SSH_PORT:-22}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# These values enter an SSH/rsync remote path. Keep them simple and unambiguous.
if [[ ! "$remote" =~ ^[A-Za-z0-9_.@-]+$ || "$remote" == -* ]]; then
  echo '[sync] Invalid USER@HOST.' >&2; exit 2
fi
if [[ ! "$remote_repo" =~ ^/[A-Za-z0-9_./-]+$ || "$remote_repo" == *..* ]]; then
  echo '[sync] Remote repository path must be absolute and contain no spaces or .. components.' >&2; exit 2
fi
if [[ ! "$ssh_port" =~ ^[0-9]+$ ]]; then
  echo '[sync] SSH_PORT must be a number.' >&2; exit 2
fi
for split in train val test train_labels val_labels test_labels; do
  if [[ ! -d "$local_dataset/$split" ]]; then
    echo "[sync] Missing dataset directory: $local_dataset/$split" >&2; exit 1
  fi
done
command -v ssh >/dev/null || { echo '[sync] ssh is required.' >&2; exit 1; }
command -v rsync >/dev/null || { echo '[sync] rsync is required locally and on the server.' >&2; exit 1; }

echo "[sync] Destination: $remote:$remote_repo"
ssh -p "$ssh_port" "$remote" "mkdir -p -- '$remote_repo/data/r2s100k' '$remote_repo/checkpoints/r2s100k'"

echo '[sync] Copying repository source...'
rsync -az --partial --info=progress2 -e "ssh -p $ssh_port" \
  --exclude='/.git/' --exclude='/.deps/' --exclude='/.venv-r2s100k/' \
  --exclude='/checkpoints/' --exclude='/r2s100k/runs/' --exclude='/data/r2s100k/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  "$repo/" "$remote:$remote_repo/"

echo '[sync] Copying R2S100K images and labels...'
for split in train val test train_labels val_labels test_labels; do
  rsync -az --partial --info=progress2 -e "ssh -p $ssh_port" \
    "$local_dataset/$split" "$remote:$remote_repo/data/r2s100k/"
done

checkpoint="$repo/checkpoints/r2s100k/catseg_vitb16.pth"
if [[ -f "$checkpoint" ]]; then
  echo '[sync] Copying checkpoint...'
  rsync -a --partial --info=progress2 -e "ssh -p $ssh_port" \
    "$checkpoint" "$remote:$remote_repo/checkpoints/r2s100k/"
else
  echo '[sync] No local checkpoint; server setup will download it.'
fi

echo '[sync] Done. On the server, run:'
echo "  cd '$remote_repo' && bash r2s100k/prepare_server.sh"
