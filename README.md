# Surgical scene understanding with structured inference

A practice project in surgical perception. The plan is in
[project_plan_v2.md](project_plan_v2.md).
Pre-specified acceptance criteria are in [docs/validation_plan.md](docs/validation_plan.md).

| Phase | What | Data | Status |
|---|---|---|---|
| 1 | Dense segmentation (32 classes) + evaluation harness | SISVSE (train/test), EndoVis18 (zero-shot) | done: [report](docs/phase1_test_report.md) |
| 2 | Instrument keypoint heatmaps | SurgPose | done (K1 fails: background shift) |
| 3 | Shape prior + robust MAP (continuous LM, discrete candidates) | SurgPose | done: [pre-specified](docs/phase23_test_report.md), [post hoc](docs/phase23_posthoc_decentered_report.md), [summary](docs/phase1-3_summary.md) |
| 4–6 | Temporal, stereo/3D, C++/TensorRT | | not started |

## Setup

```bash
uv sync -p 3.12
# datasets under data/ (gitignored); see the plan's Data section for sources
uv run python scripts/prepare_seg_data.py      # SISVSE + EndoVis18 caches, frozen patient split
uv run python scripts/prepare_surgpose.py      # SurgPose frame cache, frozen trajectory split
uv run --group dev pytest
```

## Phase 1

```bash
uv run python scripts/train_seg.py configs/seg_unet_r34.yaml
uv run python scripts/calibrate_seg.py runs/<run>/best.pt           # temperature on dev
uv run python scripts/eval_seg.py runs/<run>/best.pt --split dev    # tuning
uv run python scripts/eval_seg.py runs/<run>/best.pt --split test   # once, frozen
```

## Phases 2–3

```bash
uv run python scripts/train_kp.py configs/kp_unet_r34_decentered.yaml
uv run python scripts/eval_kp.py runs/<run>/best.pt --occluder-offset 0.6
```

## Layout

- `src/surgscene/taxonomy.py`: class lists, harmonized `{tip, wrist, shaft}` mapping
- `src/surgscene/evaluation.py`: per-frame sufficient statistics, patient-clustered bootstrap, image-condition proxies
- `src/surgscene/keypoints.py`: heatmap dataset/targets/focal loss/sub-pixel decode, occluder augmentation
- `src/surgscene/shape.py`: Umeyama, generalized Procrustes, PCA shape model, hand-written robust LM MAP
- `src/surgscene/structured.py`: heatmap → Laplace observations, top-N candidates, discrete candidate MAP
- `splits/`: frozen split manifests with hashes (committed; data is not)
- `docs/`: validation plan, test reports
