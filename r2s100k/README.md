# CAT-Seg on R2S100K

This pipeline evaluates frozen CAT-Seg ViT-B/16 on the 15 R2S100K semantic IDs (background plus 14 road-region classes). Inference reads image manifests only; ground-truth manifests are consumed by `evaluate` after predictions exist.

## Environment

Use Python 3.10 with CUDA 11.8 PyTorch 2.1.2 / torchvision 0.16.2, matching Detectron2 v0.6. The isolated environment lives at `CAT-Seg/.venv-r2s100k`.

```bash
cd CAT-Seg
bash r2s100k/setup_env.sh
```

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

Use `--max-images 1` for an initial smoke run. CAT-Seg input resolution and preprocessing follow its upstream ViT-B/16 config. Inputs remain RGB. A one-image CPU run and a 20-image CUDA pilot both completed on the local RTX 3050 4 GB. CPU mode is available for functional checks and will be slower. Do not change resolution and still report it as the same baseline.

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

Each prediction is a lossless class-ID PNG, with an overlay and per-image metadata. `run_metadata.json` stores checkpoint, vocabulary and manifest hashes. Evaluation emits `metrics.json`, per-class CSV, count and GT-normalized confusion matrices, and up to 12 qualitative examples. Metrics include 15-class and foreground-only mIoU, per-class IoU/Dice/precision/recall, binary foreground IoU/Dice and per-class semantic Boundary F1. Boundary F1 is a supplementary project metric: boundaries use 3x3 erosion and symmetric matching within 3 original-resolution pixels.

The dataset audit records basename overlap. Existing audit notes indicate val shares images with train, so do not treat val as independent if train labels were used. This CAT-Seg pipeline does not use R2S100K GT for inference or model updates.
