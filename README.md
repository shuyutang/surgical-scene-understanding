# Surgical scene understanding

Perception for robot-assisted surgery on public data. Two questions:
- **Where are the instruments, and how close are they to tissue?** 3D tracking from stereo video and
  robot kinematics.
- **What is happening?** Surgical steps, instrument types and actions.

Each component is compared **conventional vs modern**, one swap at a time: classical estimation
and ImageNet CNNs against foundation models, learned stereo and vision-language models. Every
test was pre-registered, and failures are reported as failures.

## Highlights

- **The robot's own kinematics beat a bigger network for 3D.** An EKF fusing dVRK kinematics with
  the vision cut the 3D tip error from **10.0 to 5.3 mm**. With a 6 mm stereo baseline, almost all
  of the error was depth, which no 2D detector fixes.
- **Learned stereo wins on tissue, by an order of magnitude.** On CT ground truth: SGM 17.1 mm →
  RAFT-Stereo 1.6 mm. Fast-FoundationStereo is another **0.5 mm better and 2× faster**.
- **On thin instruments, stereo still loses to kinematics.** The jaw's disparity bleeds into the
  tissue behind it. Fast-FoundationStereo's monocular prior cuts this error 3× (median 12.9 →
  4.0 mm), but its failures are still gross.
- **For "what is happening", temporal context and DINOv2 features win.** Step recognition reaches
  **0.47 macro-F1**. A fine-tuned 8B vision-language model gets 0.21, and a surgical
  self-supervised backbone (EndoSSL) 0.24.
- **Not met:** the 5 mm 3D target overall. Long-jaw instruments remain the hard case, and the 3D
  test set is used up (fresh data is needed for further claims).

## Results

95% CIs in brackets; every row links back to a pre-registered endpoint in
[docs/validation_plan.md](docs/validation_plan.md) unless marked *development*.

### 3D instruments and tissue (SurgPose stereo + dVRK kinematics; SERV-CT for tissue depth)

| Component | Conventional | Modern | Result | Winner |
|---|---|---|---|---|
| 3D tip position | Stereo triangulation of keypoints | + causal EKF fusing robot kinematics (online hand-eye) | **10.05 → 5.27 mm**, −4.78 [−6.93, −2.71]; 3.9 mm on standard instruments | Kinematics fusion |
| Keypoint backbone | ResNet-34 U-Net | DINOv2 ViT-S + learned uncertainty | Clean frames: no gain. Occluded: −5.1 px. Uncertainty–error correlation −0.25 → +0.53 | Modern, where appearance is ambiguous |
| Tissue depth | SGM | RAFT-Stereo | **17.06 → 1.58 mm** on CT ground truth | Learned |
| Tissue depth, newer model | RAFT-Stereo (2021) | Fast-FoundationStereo (2026) | **1.86 → 1.37 mm**, −0.49 [−0.95, −0.09]; 104 → 47 ms | Newer |
| Long-jaw instruments | Rigid tool model | + jaw-length constraint | 5.27 → 4.82 mm; primary endpoint failed (one misclassified arm) | Neither shown |
| Instrument depth from stereo | Kinematics (4.8 mm median) | Stereo on jaw pixels | RAFT 12.9 mm; Fast-FoundationStereo 4.0 mm median but a heavy tail (*development*) | Kinematics |
| Tissue hidden by the instrument | Local plane, current frame | Temporal tissue memory with SAM 2 masks | 2.54 → 1.28 mm (*development*) | Memory |
| Deployment | — | ONNX → TensorRT FP16, C++ runtime | p99 2.6 ms end to end without stereo; DINOv2 engine 4.4 ms (+0.08 px from FP16) | — |

### Scene semantics (GraSP, robot-assisted prostatectomy, 5 test surgeries)

Metric: macro-F1, the per-class F1 averaged with every class weighted equally, computed per
surgery. Rare steps and actions count as much as common ones.

| Task | Best model | Score | Comparisons [95% CI] |
|---|---|---|---|
| Surgical step, online (21 classes, every second) | DINOv2 features + causal temporal CNN (MS-TCN) | **0.469** | vs ResNet-50: +0.067 [+0.039, +0.096]<br>vs per-frame: +0.137 [+0.079, +0.224]<br>EndoSSL: 0.238<br>fine-tuned Qwen3-VL-8B: 0.21 (zero-shot 0.03) |
| Instrument type (7 classes, given its box) | DINOv2 features + MLP | **0.818** | fine-tuned Qwen3-VL-8B 0.785 (−0.033 [−0.063, +0.012])<br>ResNet-50 0.694 |
| Instrument action (14 classes, multi-label) | DINOv2 features + MLP | **0.273** | fine-tuned Qwen3-VL-8B 0.178 (−0.095 [−0.117, −0.065]); it only uses 4 of the 14 actions |

Component-by-component discussion: [docs/conventional_vs_modern.md](docs/conventional_vs_modern.md).
Requirements → tests → results: [docs/traceability.md](docs/traceability.md).

## How it's evaluated

- **Pre-registration.** Each test's endpoints, criteria and frozen configuration are committed
  before it runs, and the test runs once. Anything decided afterwards is a dated, marked
  amendment.
- **Honest units.** Confidence intervals come from bootstrapping over trajectories, stereo pairs
  or surgeries, not frames. Each reuse of a test set is disclosed; the SurgPose test split has
  been used four times.
- **Development only on train/tune splits** (official folds for GraSP). Frozen split manifests
  with hashes are in `splits/`.

## Pipeline

```
3D:   stereo video ─► keypoints (U-Net / DINOv2) ─► shape prior + MAP ─► Kalman filter ─┐
      robot kinematics ─► online hand-eye + tool model ─► iterated EKF ─────────────────┴─► 3D tips + covariance
      stereo ─► learned stereo ─► tissue memory (SAM 2 instrument masks) ─────────────────► tip-to-tissue distance, alert

Semantics: frames ─► frozen backbone (ResNet-50 / DINOv2 / EndoSSL) ─► causal MS-TCN ─► phase + step
           instance box ─► crop + frame features ─► MLP ─► instrument type + actions   (vs Qwen3-VL-8B, QLoRA)
```

## Quick start

```bash
uv sync -p 3.12                  # Python 3.12, PyTorch 2.8 (CUDA 12.8 wheels)
uv run --group dev pytest        # 51 unit tests on synthetic data; no datasets needed
```

Optional groups:

| Group | For |
|---|---|
| `stereo` | RAFT-Stereo, Fast-FoundationStereo |
| `sam` | SAM 2 |
| `vlm` | Qwen3-VL, QLoRA |
| `deploy` | TensorRT, ONNX |

Third-party model code goes in `third_party/` (gitignored); the fetch commands are in
[src/surgscene/learned_stereo.py](src/surgscene/learned_stereo.py) and
[src/surgscene/backbones.py](src/surgscene/backbones.py). Every experiment's commands are in
[scripts/README.md](scripts/README.md#reproducing).

**Data** (not redistributed; place under `data/`, check each dataset's terms):

| Dataset | Used for | Terms |
|---|---|---|
| [SurgPose](https://zenodo.org/records/15278516) | Stereo video, keypoints, dVRK kinematics (main 3D dataset) | CC BY 4.0 |
| SERV-CT | Stereo depth vs CT ground truth | CC BY-NC-SA 4.0 |
| SISVSE, EndoVis 2018 | Segmentation (v2 Phase 1) | dataset / challenge terms |
| [GraSP](https://github.com/BCV-Uniandes/GraSP) (1 fps) | Phases, steps, instruments, actions | no license stated; research use |

**Pretrained models:**
- Apache 2.0: DINOv2 (`timm`), SAM 2.1 (`transformers`), Qwen3-VL-8B-Instruct.
- MIT: RAFT-Stereo.
- BSD: torchvision ResNet-50.
- Research-only: Fast-FoundationStereo (NVIDIA research license), EndoSSL (via SurgVISTA's
  conversion; no license stated).
- TensorRT comes from NVIDIA's pip wheels.

## Repository layout

| Path | Contents |
|---|---|
| `src/surgscene/` | The library: keypoints, stereo, fusion, tracking, tissue, scene semantics, deployment |
| `scripts/` | Data preparation, training, evaluation ([index and commands](scripts/README.md)) |
| `configs/`, `splits/` | Training configs; frozen, hashed evaluation configs and split manifests |
| `cpp/` | C++ runtime: TensorRT engine, MAP solver, Kalman filter; parity tests; latency benchmark |
| `docs/` | Validation plan, test reports, plans and summaries ([index](docs/README.md)) |

## License

Code: MIT ([LICENSE](LICENSE)). Datasets and pretrained weights keep their own licenses.
