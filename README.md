# Surgical scene understanding with structured inference

A practice project in surgical perception. The plan is in
[project_plan_v2.md](project_plan_v2.md).
The next iteration (foundation backbone, learned stereo, kinematics-fused factor graph) is planned in
[project_plan_v3.md](project_plan_v3.md). v4 draft (SAM 2 masks as a measurement): [project_plan_v4.md](project_plan_v4.md), step 0 [report](docs/v4_step0_sam2_feasibility.md).
Pre-specified acceptance criteria are in [docs/validation_plan.md](docs/validation_plan.md).

| Phase | What | Data | Status |
|---|---|---|---|
| 1 | Dense segmentation (32 classes) + evaluation harness | SISVSE (train/test), EndoVis18 (zero-shot) | done: [report](docs/phase1_test_report.md) |
| 2 | Instrument keypoint heatmaps | SurgPose | done (K1 fails: background shift) |
| 3 | Shape prior + robust MAP (continuous LM, discrete candidates) | SurgPose | done: [pre-specified](docs/phase23_test_report.md), [post hoc](docs/phase23_posthoc_decentered_report.md), [summary](docs/phase1-3_summary.md) |
| 3b | Gated MAP (argmax when confident, MAP otherwise) | SurgPose, stage-2 split | done: gate keeps precision, gain not significant ([report](docs/stage2_test_report.md)) |
| 4 | Per-keypoint Kalman filter, fixed-lag smoother | SurgPose 30 fps | done: jitter −68%, occlusion −1.9 px |
| 5 | Stereo triangulation + covariance, SO(3), hand-eye from kinematics, SGM tissue proximity | SurgPose stereo + dVRK kinematics | done: 3D tip 10 mm (5 mm requirement fails: baseline physics); kinematics + registration 4.1 mm |
| 6 | Deploy graph → ONNX → TensorRT FP16, C++ runtime + parity tests | | done: FP16 non-inferior, p99 2.6 ms per stereo pair ([report](docs/phase6_deploy_report.md)) |

Summaries: [Phases 1–3](docs/phase1-3_summary.md), [Phases 3b–6](docs/phase3b-6_summary.md). Component-by-component comparison: [conventional vs modern](docs/conventional_vs_modern.md).

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

## Stage 2 (Phases 3b–5)

```bash
uv run python scripts/cache_stage2_obs.py runs/<kp_run>/best.pt   # full-rate observations, both eyes
uv run python scripts/tune_stage2.py                              # all parameters, tune split only
uv run python scripts/cache_tissue_planes.py                      # SGM tissue planes
uv run python scripts/eval_stage2.py --split tune                 # sanity
uv run python scripts/eval_stage2.py --split test2                # once, pre-specified
```

## Phase 6 (TensorRT + C++)

```bash
uv sync --group deploy && cpp/scripts/fetch_deps.sh               # TensorRT/CUDA from wheels, TRT headers from OSS
uv run --group deploy python scripts/export_trt.py runs/<kp_run>/best.pt
uv run --group deploy python scripts/eval_deploy.py               # D1, D2, goldens for C++
cmake -S cpp -B cpp/build && cmake --build cpp/build -j && cpp/build/tests
uv run python scripts/prepare_cpp_bench.py
cpp/build/bench runs/deploy/kp_stereo_fp16.engine runs/deploy/goldens runs/deploy/bench_frames.u8 100 2000 6.86 runs/deploy/bench.csv
```

## Layout

- `src/surgscene/taxonomy.py`: class lists, harmonized `{tip, wrist, shaft}` mapping
- `src/surgscene/evaluation.py`: per-frame sufficient statistics, patient-clustered bootstrap, image-condition proxies
- `src/surgscene/keypoints.py`: heatmap dataset/targets/focal loss/sub-pixel decode, occluder augmentation
- `src/surgscene/shape.py`: Umeyama, generalized Procrustes, PCA shape model, hand-written robust LM MAP
- `src/surgscene/structured.py`: heatmap → Laplace observations, top-N candidates, discrete candidate MAP
- `src/surgscene/temporal.py`: Kalman filter, fixed-lag smoother, jitter metric
- `src/surgscene/geometry.py`: distortion, triangulation with covariance, SO(3), rigid transforms, robust registration
- `src/surgscene/proximity.py`: rectification, SGM, disparity-space tissue plane, keypoint instrument mask
- `src/surgscene/stage2.py`, `frontend.py`: stage-2 split and pipeline glue
- `src/surgscene/deploy.py`, `trt_runner.py`: exportable graph, TensorRT build/run
- `cpp/`: C++ runtime (TensorRT engine, Eigen MAP solver, Kalman, stereo), GoogleTest parity tests, latency bench
- `splits/`: frozen split manifests with hashes (committed; data is not)
- `docs/`: validation plan, test reports
