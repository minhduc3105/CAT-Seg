# CAT-Seg on R2S100K

This pipeline evaluates frozen CAT-Seg ViT-B/16 on the 15 R2S100K semantic IDs (background plus 14 road-region classes). Inference reads image manifests only; ground-truth manifests are consumed by `evaluate` after predictions exist.

## Move to a GPU server

From the local CAT-Seg repository, sync the source, standard R2S100K splits (`train`, `val`, `test` and their label directories), and the local checkpoint if present. SSH and rsync must be installed on both machines. Give an absolute path for the remote repository; the dataset is placed under its `data/r2s100k` directory. Set `SSH_PORT` if the server does not use port 22.

```bash
bash r2s100k/sync_to_server.sh USER@HOST /home/USER/CAT-Seg \
  /run/media/minhduc/D/Study/r2s100k
```

Then SSH into the server and run:

```bash
cd /home/USER/CAT-Seg
bash r2s100k/prepare_server.sh
```

The server script installs the pinned environment when needed, downloads the official checkpoint only when it was not copied, verifies its SHA-256, creates manifests with server-local dataset paths, and runs one CUDA smoke inference. It prints a command for full test inference when finished. Set `MAX_JOBS=2` for the environment build if host RAM is limited. Use `--dataset-root /other/absolute/path` when the dataset already lives elsewhere, or `--skip-smoke` to omit the one-image run.

## Environment

Use Python 3.10 with CUDA 11.8 PyTorch 2.1.2 / torchvision 0.16.2, matching Detectron2 v0.6. The isolated environment lives at `CAT-Seg/.venv-r2s100k`. The setup script installs the pinned dependencies, builds Detectron2 without pip build isolation (so its build can import the installed PyTorch), and verifies CUDA/package imports. Run it once on a new server:

```bash
cd CAT-Seg
bash r2s100k/setup_env.sh
```

The server needs `git`, a C++ compiler (`gcc`/`g++`), and a working NVIDIA driver. Set `MAX_JOBS=2` before running the script if Detectron2 compilation exceeds available host RAM. The script does not download the model checkpoint or dataset. Reuse the environment for later runs; do not rerun setup unless rebuilding the environment.

## Checkpoint

Download the [official CAT-Seg ViT-B/16 checkpoint](https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/de5f330f3fe6a28e19b2f0c7855a6e3b4928a76a/model_base.pth) (`model_base.pth`) to `CAT-Seg/checkpoints/r2s100k/catseg_vitb16.pth`. Do not use SAN or SAM 3 weights; the pipeline checks for CAT-Seg aggregation weights and fails if they are absent.

The downloaded checkpoint SHA-256 in this workspace is `a201a50f94f4c04196e3083773832b736c2b8b81d27ea2c59cc13af3aff05760`.

## Prepare and pilot

```bash
cd CAT-Seg
.venv-r2s100k/bin/python r2s100k/pipeline.py prepare \
  --dataset-root /run/media/minhduc/D/Study/r2s100k \
  --output r2s100k/runs/r2s100k

.venv-r2s100k/bin/python r2s100k/pipeline.py infer \
  --manifest r2s100k/runs/r2s100k/manifests/val_pilot_images.json \
  --checkpoint checkpoints/r2s100k/catseg_vitb16.pth \
  --output r2s100k/runs/catseg_b16_val_pilot --device cuda

.venv-r2s100k/bin/python r2s100k/pipeline.py evaluate \
  --image-manifest r2s100k/runs/r2s100k/manifests/val_pilot_images.json \
  --label-manifest r2s100k/runs/r2s100k/manifests/val_labels.json \
  --predictions r2s100k/runs/catseg_b16_val_pilot/predictions \
  --output r2s100k/runs/catseg_b16_val_pilot/evaluation
```

Use `--max-images 1` for an initial smoke run. Without `--vocabulary`, inference generates `model_vocabulary.json` in the output directory from `taxonomy.py`: 14 road prompts followed by 8 background prompts. It maps all 8 background output channels to dataset ID 0 after argmax. CAT-Seg input resolution and preprocessing follow its upstream ViT-B/16 config. Inputs remain RGB. CPU mode is available for functional checks and will be slower. Do not change resolution and still report it as the same baseline.

For a one-image smoke test with the existing pilot manifest:

```bash
.venv-r2s100k/bin/python -m unittest r2s100k.test_pipeline -q
.venv-r2s100k/bin/python r2s100k/pipeline.py infer \
  --manifest r2s100k/runs/r2s100k/manifests/val_pilot_images.json \
  --checkpoint checkpoints/r2s100k/catseg_vitb16.pth \
  --output r2s100k/runs/taxonomy_smoke_one --device cuda --max-images 1
```

## Full test

After locking the vocabulary and completing the pilot, run inference over `test_images.json` and evaluate against `test_labels.json`. Generated manifests contain absolute paths, so regenerate them on the RTX 3090 host with `prepare --dataset-root` set to that machine's dataset. Preserve vocabulary, code revision, checkpoint hash and config. Keep outputs out of version control.

```bash
.venv-r2s100k/bin/python r2s100k/pipeline.py infer \
  --manifest r2s100k/runs/r2s100k/manifests/test_images.json \
  --checkpoint checkpoints/r2s100k/catseg_vitb16.pth \
  --output r2s100k/runs/catseg_b16_test --device cuda

.venv-r2s100k/bin/python r2s100k/pipeline.py evaluate \
  --image-manifest r2s100k/runs/r2s100k/manifests/test_images.json \
  --label-manifest r2s100k/runs/r2s100k/manifests/test_labels.json \
  --predictions r2s100k/runs/catseg_b16_test/predictions \
  --output r2s100k/runs/catseg_b16_test/evaluation
```

## Outputs and metrics

Each prediction has a lossless class-ID PNG in `predictions/`, a lossless RGB PNG using the taxonomy palette in `colored_masks/`, an overlay in `overlays/`, and per-image metadata. Use the class-ID PNG for evaluation; the RGB PNG is for viewing. `run_metadata.json` stores checkpoint, vocabulary and manifest hashes. Evaluation emits `metrics.json`, per-class CSV, count and GT-normalized confusion matrices, and up to 12 qualitative examples. Metrics include 15-class and foreground-only mIoU, per-class IoU/Dice/precision/recall, binary foreground IoU/Dice and per-class semantic Boundary F1. Boundary F1 is a supplementary project metric: boundaries use 3x3 erosion and symmetric matching within 3 original-resolution pixels.

The dataset audit records basename overlap. Existing audit notes indicate val shares images with train, so do not treat val as independent if train labels were used. This CAT-Seg pipeline does not use R2S100K GT for inference or model updates.
