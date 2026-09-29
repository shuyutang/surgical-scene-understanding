# Surgical scene understanding: 3D instrument tracking and tool–tissue proximity

A perception pipeline for robotic-surgery stereo video that locates surgical instrument tips in 3D
and estimates their distance to tissue, built and evaluated on public data. It compares modern
learned components (foundation-model backbones, learned stereo, SAM 2) with classical estimation
(shape priors, Kalman filtering, fusion of the robot's own kinematics) one component at a time.

**Every test is specified in writing and committed before it runs**
([validation plan](docs/validation_plan.md)), and failures are reported as failures.

## Headline results

Target use case: a tool-to-tissue proximity alert. Requirements: 3D tip error ≤ 5 mm (R1), end-to-end
p99 latency < 33 ms (R3), calibrated uncertainty (R7). Full requirement → test → result mapping:
[docs/traceability.md](docs/traceability.md).

| Stage | Method | Key result (95% CI) |
|---|---|---|
| v2 | U-Net keypoints, shape prior + robust MAP, Kalman filter, stereo triangulation | 3D tip error 10.0 [7.6, 12.8] mm: stereo depth fails with a 6 mm baseline |
| v3 C | + causal fusion of dVRK kinematics (online hand-eye, iterated EKF) | 5.27 [3.56, 7.43] mm (−4.8 mm vs v2); 3.9 mm on standard instruments |
| v3 A | DINOv2 ViT-S keypoints with a learned uncertainty head | No clean-frame gain; −5.1 px on occluded keypoints; post-hoc uncertainty inflation halved |
| v3 B | RAFT-Stereo vs SGM, tissue depth (SERV-CT, CT ground truth) | 17.1 → 1.6 mm (1.9 mm at 10.7 ms) |
| v4 | Jaw-length constraint for long-jaw instruments (SAM 2 masks tried, dropped) | 4.82 [3.63, 6.21] mm; primary endpoint failed (one misclassified arm) |
| v5 | Occlusion-aware tissue memory (development) | Hidden-tissue depth 2.54 → 1.28 mm vs a local plane; tip error still dominates distance |
| Deploy | ONNX → TensorRT FP16, C++ runtime | U-Net path p99 2.6 ms end to end (without stereo); DINOv2 engine 4.4 ms, +0.08 px from FP16 |

Summary of what worked, component by component: [docs/conventional_vs_modern.md](docs/conventional_vs_modern.md).
Learned models won on stereo matching, occlusion and uncertainty; the largest 3D gain came from
fusing the robot's kinematics. **R1 is not met overall**; the evaluation set has been used four times
(disclosed in each pre-registration), so further claims need fresh data.

## Pipeline

```
 stereo video ──► keypoint network (U-Net / DINOv2) ──► shape prior + MAP ──► Kalman filter (2D) ─┐
      │                                                                                           ├─► 3D tips + covariance
      │          robot kinematics ──► online hand-eye + tool geometry ──► iterated EKF ───────────┘        │
      │                                                                                                    ▼
      └────────► learned stereo (RAFT) ──► tissue memory (instrument masks: SAM 2) ──────────► tip-to-tissue distance, alert
```

## Setup

```bash
uv sync -p 3.12                       # Python 3.12, PyTorch 2.8 (CUDA 12.8 wheels)
uv run --group dev pytest             # 44 unit tests; no data needed
```

Optional dependency groups: `stereo` (RAFT-Stereo), `sam` (SAM 2 via transformers), `deploy`
(TensorRT, ONNX). RAFT-Stereo code and weights are fetched into `third_party/` (gitignored):

```bash
git clone https://github.com/princeton-vl/RAFT-Stereo third_party/RAFT-Stereo
(cd third_party/RAFT-Stereo && bash download_models.sh)
```

### Data

Nothing is redistributed here; datasets go under `data/` (gitignored). Check each dataset's own
terms before use.

| Dataset | Used for | Source | Terms |
|---|---|---|---|
| SurgPose | Keypoints, stereo, dVRK kinematics (main dataset) | Zenodo record 15278516 | CC BY 4.0 |
| SISVSE | Semantic segmentation (Phase 1) | MICCAI 2022 SISVSE release | see the dataset's terms |
| EndoVis 2018 Robotic Scene Segmentation | Zero-shot segmentation test | HF mirror `BeileiCui/EndoVis18` | challenge terms |
| SERV-CT | Stereo depth accuracy (CT ground truth) | SERV-CT release | CC BY-NC-SA 4.0 (non-commercial) |
| GraSP (1 fps) | Phase and step recognition (scene-semantics track) | github.com/BCV-Uniandes/GraSP (Google Drive) | no data license stated; research use |

Pretrained models: DINOv2 (Apache 2.0, via `timm`), SAM 2.1 (Apache 2.0, via `transformers`),
RAFT-Stereo (MIT). TensorRT is installed from NVIDIA's pip wheels under NVIDIA's license.

Frozen split manifests (with hashes) are committed in `splits/`.

## Reproducing

Scripts are indexed in [scripts/README.md](scripts/README.md) and documents in
[docs/README.md](docs/README.md). Test-set commands are marked *once*: they were run a single time,
after the matching pre-registration was committed.

```bash
# data caches
uv run python scripts/prepare_seg_data.py && uv run python scripts/prepare_surgpose.py

# v2: segmentation, keypoints, structured inference, temporal, stereo
uv run python scripts/train_seg.py configs/seg_unet_r34.yaml
uv run python scripts/train_kp.py configs/kp_unet_r34_decentered.yaml
uv run python scripts/cache_stage2_obs.py runs/<kp_run>/best.pt
uv run python scripts/tune_stage2.py && uv run python scripts/cache_tissue_planes.py
uv run python scripts/eval_stage2.py --split tune            # --split test2: once

# v3: kinematics fusion (C), learned stereo (B), DINOv2 keypoints (A)
uv run python scripts/fit_tool_geometry.py
uv run python scripts/eval_v3c.py --split tune               # --split test2: once
uv run --group stereo python scripts/eval_stereo_tune.py && uv run --group stereo python scripts/eval_servct.py
uv run python scripts/train_kp_vit.py configs/kp_vit_s.yaml
uv run python scripts/eval_kp_vit.py --split tune            # --split test2: once

# v4: SAM 2 masks, instrument-type library, jaw-length constraint
uv run --group sam python scripts/v4_make_masks.py && uv run python scripts/v4_mask_qa.py
uv run python scripts/v4_tool_library.py
uv run python scripts/eval_v4.py --split tune                # --split test2: once

# v5 (development, tune only): stereo on instruments, tissue memory
uv run --group stereo --group sam python scripts/v5_tissue_memory.py

# deployment: TensorRT FP16 engines, C++ runtime and latency benchmark
uv sync --group deploy && cpp/scripts/fetch_deps.sh
uv run --group deploy python scripts/export_trt.py runs/<kp_run>/best.pt
uv run --group deploy python scripts/eval_deploy.py
cmake -S cpp -B cpp/build && cmake --build cpp/build -j && cpp/build/tests
```

## Layout

- `src/surgscene/`: the library
  - perception: `models`, `keypoints`, `kp_vit`, `learned_stereo`, `sam2_track`
  - structured inference and estimation: `shape`, `structured`, `temporal`, `geometry`, `fusion`
  - tissue and proximity: `proximity`, `tissue_memory`
  - evaluated pipelines and evaluation: `pipeline`, `stage2`, `evaluation`, `kp_eval`
  - deployment: `deploy`, `trt_runner`
- `scripts/`: data preparation, training, evaluation, export ([index](scripts/README.md))
- `configs/`: training configs and frozen, hashed evaluation configs
- `cpp/`: C++ runtime (TensorRT engine, Eigen MAP solver, Kalman filter, stereo), GoogleTest parity tests, latency bench
- `docs/`: validation plan, pre-specified test reports, plans and summaries ([index](docs/README.md))
- `splits/`: frozen split manifests

## License

Code: MIT (see [LICENSE](LICENSE)). Datasets and pretrained weights keep their own licenses (above).
