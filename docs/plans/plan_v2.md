# Practice Project v2: Instrument-Aware Surgical Scene Understanding with Structured Inference

> **Status (2026-09-25):** Phases 1–6 implemented. Results and deviations from this plan are in
> [phase1-3_summary.md](../phase1-3_summary.md) and [phase3b-6_summary.md](../phase3b-6_summary.md).

## What changed from v1

| v1 | v2 | Why |
|---|---|---|
| Gallbladder tip/neck landmarks | da Vinci **instrument keypoints** | No public dataset labels anatomical landmarks; instruments are rigid/articulated, so a structured prior is physically justified |
| Pairwise pixel-distance Gaussian prior | **Similarity-invariant statistical shape model** + MAP | Pixel distances change with zoom/depth/rotation; the v1 prior is invalid under camera motion |
| Phases 3 and 4 separate | Merged into one MAP formulation | Shape model *is* the richer prior |
| Metrics lists | **Statistical validation plan** with pre-specified acceptance criteria | Medical-device ML needs regulatory-grade validation, not just metric lists |
| No clinical anchor | **Tool–tissue proximity awareness** use case | Keeps the work product-oriented; drives metrics in mm, not just pixels |
| 3D as optional toy | Stereo on real da Vinci data | da Vinci is stereo; the geometry phase feeds the clinical metric |
| 8 phases | 6 phases, explicit cut list | Depth beats breadth |

---

## Clinical use case (the product framing)

> **Instrument proximity awareness.** During a robotic procedure, estimate in real time the
> 3D distance from each instrument tip to (a) the nearest tissue surface and (b) the nearest
> labeled critical structure, and flag when it falls below a threshold.

Every design decision below is justified against this feature. The surgeon is the user;
the failure modes that matter are **missed proximity events** (false negatives) and
**nuisance alerts** (false positives that train the surgeon to ignore the system).

Derived requirements (write these down before building — they become the acceptance criteria):

| ID | Requirement | Measured by |
|---|---|---|
| R1 | Instrument tip localized within X px (2D) / Y mm (3D) | Keypoint error, triangulated error vs. GT depth |
| R2 | Critical-structure segmentation adequate for proximity (boundary quality, not just Dice) | Boundary F-score, HD95 |
| R3 | End-to-end latency p99 < 33 ms at 30 FPS | Per-stage timing on target GPU |
| R4 | Stable output: no alert flicker from jitter | Jitter metric, alert toggle rate |
| R5 | Graceful degradation: known behavior under smoke/blood/occlusion | Subgroup analysis, confidence gating |
| R6 | FP16 deployment does not degrade accuracy | Paired accuracy delta with CI |

---

## Data

All datasets here are research-only practice data. Don't redistribute them or publish weights
trained on them. Local copies are under `data/` (checked 2026-09-24).

| Dataset | Role | What it has | Gaps |
|---|---|---|---|
| **SISVSE** (robotic gastrectomy, MICCAI 2022), `data/sisvse/` | **Phase 1 primary**: training, validation stats, subgroups | Human, da Vinci, 1280×1024. **40 patients / 87 videos / 4,510 real labeled frames**. 31 classes: instrument **head/wrist/body** for 7 tool types; liver, stomach, pancreas, spleen, gallbladder; gauze; specimen bag. Official 3 folds are **patient-disjoint** (verified). Plus ~4.6k synthetic frames + SEAN/SPADE translations | Monocular, no calibration. Frames sparse (median 13 s apart), so no temporal use. "TheOther_Tissues" = 40% of pixels; Needle 0.05%. License not stated in the archive |
| **EndoVis 2018 Robotic Scene Segmentation** (HF mirror `BeileiCui/EndoVis18`), `data/endovis18/` | **Phase 1 external test on a second domain** (no training on it) | da Vinci Xi, porcine. 15 train seqs (2,235 frames) + 4 test seqs (997 frames), all labeled. Instrument **shaft/wrist/clasper**, kidney, intestine, thread, clamps, needle, US probe. Stereo calibration per sequence | **No right frames** in the mirror, so no stereo use. Only 19 sequences, so CIs are wide. Test set has no suction; needle nearly absent |
| **SurgPose** (Zenodo, CC-BY-4.0), `data/surgpose/` | **Phases 2–4** keypoints, shape prior, temporal; Phase 5 stereo triangulation + hand-eye | dVRK (da Vinci IS1200), 30 continuous stereo trajectories × 1001 steps, ~120k instances, 6 tool types, 7 UV-paint keypoints each; stereo calibration, SAM2 instance masks, **forward kinematics + joint states** | Benchtop ex vivo tissue background (overfits to it). Calibration needs distortion compensation |
| Hamlyn Dataset 2 (optional) | Phase 5 tissue depth | ~40k rectified da Vinci stereo pairs (partial nephrectomy) | No depth GT |
| SERV-CT (optional) | Phase 5 quantitative depth accuracy | 16 stereo pairs with CT-derived depth | Tiny |

**Harmonized instrument-part taxonomy.** SISVSE head/wrist/body and EndoVis18
clasper/wrist/shaft map onto a shared `{tip, wrist, shaft}` set. The main generalization
experiment is: train on SISVSE, evaluate the shared classes on EndoVis18 with no fine-tuning.

**Known gap: no dataset has anatomy labels and stereo in the same frames.** So the proximity
demo is **tool-to-tissue** distance (from depth). Tool-to-labeled-organ distance is out of reach
with public data. Stated openly: it's a data-strategy limitation.

**SISVSE split (frozen):**
- **Test:** the 10 patients of official `real_val_1`. Frozen, hashed, touched only for final reporting.
- **Dev:** 6 patients held out of `real_train_1`, used for all tuning.
- **Train:** the remaining 24 patients.
- Official folds 2–3 stay available for a cross-validation sanity check.

**Data rules (non-negotiable, set up in week 1):**

- Split by **video/procedure (patient-equivalent)**, never by frame. Adjacent frames are
  near-duplicates; frame-level splits inflate every metric.
- Freeze the test split and record its hash. Touch it only for final reporting.
- Keep a separate dev split for all tuning, including MAP prior weights and filter noise.
- Tag frames with condition labels (smoke, blood, specular, motion blur, occlusion, out-of-view
  tip). Hand-tag a few hundred test frames if needed. These define the subgroups.

---

## Architecture

```text
Stereo endoscopic video (left, right)
        │
        ▼
Shared backbone (left eye) ──────────────┐
   │                │                     │
   ▼                ▼                     ▼
Seg head       Keypoint heatmap head    (right eye: keypoints only,
(anatomy +     (per-instrument           or stereo depth net / SGM)
 instr. parts)  keypoints)
   │                │                     │
   └───────┬────────┘                     │
           ▼                              │
Structured inference (MAP)                │
 shape prior + mask constraints           │
           │                              │
           ▼                              │
Temporal filter (Kalman / fixed-lag)      │
           │                              │
           └──────────────┬───────────────┘
                          ▼
             Stereo triangulation → 3D tips
             Depth → tissue surface
                          │
                          ▼
             Proximity estimate + confidence → alert logic
                          │
                          ▼
        C++ runtime: TensorRT FP16, Eigen/Ceres MAP, CUDA streams
```

---

## Phase 1 — Segmentation baseline + validation harness (≈1.5 weeks)

**Build**
- A simple, deployable model: SegFormer-B0/B1 or U-Net with a ResNet-18/34 encoder. Pick
  for ONNX/TensorRT friendliness (avoid exotic ops) as much as for accuracy.
- Classes: SISVSE instrument parts (head/wrist/body per tool type) + anatomy (liver, stomach, pancreas, spleen, gallbladder). Also report the harmonized `{tip, wrist, shaft}` view, which is what EndoVis18 is evaluated on.
- The **evaluation harness is the main deliverable of this phase**, not the model:
  - Per-class Dice, IoU, **boundary F-score, HD95** (proximity depends on boundaries)
  - Patient-clustered bootstrap 95% CIs (resample videos, not frames)
  - Subgroup breakdown by condition tag
  - Temporal stability without labels: warp frame $t$'s prediction to $t{+}1$ with optical
    flow and measure disagreement
  - Calibration: ECE / reliability diagrams on per-pixel softmax

**Discussion points**
- Why Dice alone is the wrong metric for a proximity feature
- Why frame-level splits are leakage
- Class imbalance (thin structures, needles, thread) and loss choices

---

## Phase 2 — Instrument keypoint detection (≈1 week)

**Build**
- Add a heatmap head to the shared backbone: $H_k(u,v)$ for each keypoint $k$ (e.g. shaft
  points, wrist, jaw tips).
- Train jointly with segmentation, then ablate joint vs. separate training.
- Independent baseline: $\hat{x}_k = \arg\max_{u,v} H_k(u,v)$ with sub-pixel refinement
  (soft-argmax or quadratic fit around the peak).
- Per-keypoint confidence from the peak value; calibrate it (does 0.8 confidence mean
  80% within tolerance?).

**Metrics:** PCK@{5,10,20 px}, mean/median error, detection rate, confidence calibration,
error stratified by occlusion and by segmentation quality.

**Purpose:** produce *uncertain* observations with realistic failure modes, including
tips occluded by tissue, left/right jaw swaps, and a keypoint firing on the other instrument.
These failures are what Phase 3 has to fix.

---

## Phase 3 — Structured inference: shape prior + MAP (≈2 weeks, the core of the project)

### Shape model

Let an instrument's keypoints be $X = (x_1,\dots,x_K)$, $x_k \in \mathbb{R}^2$.

1. Align all training configurations to a mean shape with **similarity Procrustes
   (Umeyama)**. This removes translation, in-plane rotation, and scale, which absorbs zoom
   and depth.
2. PCA on the aligned residuals gives a linear shape model:
   $$X = s\,R(\theta)\,(\bar{X} + P b) + t, \qquad b \sim \mathcal{N}(0, \Lambda)$$
   with pose $(s,\theta,t)$ and shape coefficients $b$. The leading modes should capture
   jaw opening, wrist articulation, and out-of-plane foreshortening. Plot them and check.

Be honest about the approximation: a 2D linear model of a projected articulated 3D object
is not exact. That limitation is worth stating (the alternative is a
3D kinematic model + projection, which needs robot kinematics you don't have).

### MAP energy

$$
E(s,\theta,t,b) = -\sum_k w_k \log \tilde{H}_k\big(x_k(s,\theta,t,b)\big)
\;+\; \tfrac{1}{2}\, b^\top \Lambda^{-1} b
\;+\; \lambda_{\text{mask}} \sum_k \rho\big(d_{\text{mask}}(x_k)\big)
$$

- $\tilde{H}_k$: bilinearly interpolated, floored heatmap. The floor stops $\log 0$ and
  bounds the penalty for an occluded keypoint.
- $w_k$: confidence gating. An occluded keypoint contributes little, and the prior fills it in.
- $d_{\text{mask}}$: distance transform of the matching instrument-part mask (tip ∈ clasper
  mask, wrist ∈ wrist mask). $\rho$ is a robust (Huber) loss.
- Solve with **Gauss–Newton / Levenberg–Marquardt**, initialized from Procrustes-fitting the
  shape model to the argmax detections.

### Alternative solver (for the ablation)

A discrete one: take the top-$N$ peaks per heatmap as candidates and run **max-product
belief propagation on a tree** (pictorial structures) with pairwise terms on
similarity-normalized relative positions. Compare the continuous and discrete solvers on
accuracy, runtime, and robustness to multi-modal heatmaps.

### Ablation (all evaluated on the frozen test split with clustered-bootstrap CIs)

1. Argmax detections only
2. + shape prior (continuous MAP)
3. + mask constraints
4. Discrete tree BP alternative

Metrics: keypoint error overall **and on the occluded subgroup** (the prior should help most
there), rate of anatomically impossible configurations, solver iterations, runtime.

---

## Phase 4 — Temporal model (≈1 week)

State per instrument: $z_t = (s, \theta, t, b)_t$ and their velocities.

$$
P(z_{1:T} \mid I_{1:T}) \propto \prod_t P(I_t \mid z_t)\, P(z_t \mid z_{t-1})
$$

- Start with a constant-velocity **Kalman filter** on $z_t$, using the Phase 3 MAP estimate
  (and its Laplace-approximation covariance, i.e. the inverse Gauss–Newton Hessian) as the
  measurement. The observation noise is then input-dependent rather than a fixed tuning knob.
- Compare with a **fixed-lag smoother** (e.g. 3 frames). Quantify the accuracy gain against
  the latency added, which is a product tradeoff.
- Occlusion handling: when confidence drops, predict only and let the covariance grow. Once
  the covariance passes a threshold, report "unknown" instead of a stale estimate.

**Metrics:** jitter (frame-to-frame displacement on static segments), tracking error,
recovery time after occlusion, **alert toggle rate** (R4), and added latency.

---

## Phase 5 — Stereo geometry and the proximity metric (≈1.5 weeks)

**Build (from first principles, with unit tests)**
- Pinhole projection/backprojection, stereo rectification from the provided calibration
- Triangulation of left/right tip keypoints (DLT, then check reprojection error)
- Tissue depth: classical SGM (OpenCV) first, then optionally a small learned stereo network.
  Evaluate against SERV-CT ground truth (SCARED if institutional access is available).
- Frames: camera ← world transforms. Implement and test SO(3) utilities: rotation matrix ↔
  quaternion ↔ axis-angle, composition, inversion, and Umeyama rigid/similarity registration
  (reuse from Phase 3). Property tests: $R^\top R = I$, $\det R = 1$, round-trip conversions,
  inverse composition = identity.
- Proximity: $d_{\text{tip}\to\text{tissue}}$ = distance from the 3D tip to the local depth
  surface. $d_{\text{tip}\to\text{structure}}$ = distance to the 3D points of the critical-structure mask.

**Uncertainty propagation:** push keypoint covariance (from Phase 4) and disparity
uncertainty through triangulation (first-order Jacobian) to get a **distance ± σ in mm**.
The alert logic should use a lower confidence bound, which gives a principled tradeoff
between missed events and nuisance alerts.

**Metrics:** 3D tip error (mm), depth error vs. SERV-CT GT, proximity-alert
sensitivity/specificity at the chosen threshold, ROC over thresholds.

*Stretch only:* a robot-base frame. Without real kinematics, simulate a hand-eye transform
$T_{\text{robot}\leftarrow\text{cam}}$ and show a proper composition/inversion chain.

---

## Phase 6 — Real-time C++ deployment (≈2 weeks)

### Export path

```text
PyTorch FP32 → ONNX (opset pinned) → TensorRT FP32 → TensorRT FP16
```

- **Numerical parity tests** at each step: max abs diff on logits/heatmaps against golden
  outputs, and a *metric-level* check (the FP16 accuracy delta with a paired bootstrap CI must
  sit inside a pre-set non-inferiority margin, per R6).
- Watch for FP16 pitfalls: softmax/log on heatmaps, layernorm overflow. Keep sensitive
  layers in FP32 if needed and document why.

### C++ runtime

```text
cpp/
  include/  src/
    TrtEngine.{h,cpp}          # engine load, preallocated device buffers, CUDA stream
    Preprocess.{h,cu}          # resize/normalize on GPU, pinned host memory
    ShapeModel.{h,cpp}         # loads mean shape + PCA modes exported from Python
    MapSolver.{h,cpp}          # LM on the Phase 3 energy (Eigen, or Ceres)
    KalmanTracker.{h,cpp}
    Stereo.{h,cpp}             # rectification, triangulation, SO(3) utils
    Pipeline.{h,cpp}           # orchestration, per-stage timing
    main.cpp
  tests/                       # GoogleTest: geometry properties, solver on synthetic data,
                               # Python↔C++ parity on golden frames
  CMakeLists.txt
```

- **Python↔C++ parity:** export golden inputs/outputs from Python for every stage
  (preprocess, network, MAP, Kalman, triangulation) and assert equality within tolerance in C++ tests.
- Preallocate everything so the steady-state loop never allocates. Use pinned memory
  and async H2D/D2H on a dedicated stream. Overlap decode and preprocessing of frame $t{+}1$
  with inference on frame $t$.
- Failure handling: engine/version mismatch at load, NaN guard on outputs, solver
  non-convergence fallback (use the previous state and mark it low-confidence).

### Latency budget (report p50 **and p99**, not the mean)

| Stage | Budget (ms) | p50 | p99 |
|---|---|---|---|
| Decode (L+R) | 3 | | |
| Preprocess + H2D | 2 | | |
| TRT inference (left + right keypoints) | 12 | | |
| Seg/heatmap postproc | 2 | | |
| MAP solve (per instrument) | 3 | | |
| Kalman + triangulation | 1 | | |
| Depth (SGM or net) | 6 | | |
| Render/output | 2 | | |
| **Total** | **< 33** | | |

Run on the RTX 4090, but also report numbers with the GPU clock-locked and power-limited,
or with a smaller input resolution, as a proxy for an embedded target. Explain which stages
would dominate on a Jetson-class device.

---

## Statistical validation plan (write in week 1, report in the final week)

This section is modeled on a design-verification test plan. It is the part of the project
that most directly signals "can ship in a regulated environment".

1. **Pre-specified endpoints and acceptance criteria.** Before looking at test data, fix
   the endpoints and the thresholds, e.g.:
   - Primary: tip localization, *lower bound of the 95% CI of PCK@10px ≥ 0.90*
   - Secondary: critical-structure boundary F-score, proximity-alert sensitivity at the
     operating threshold, p99 latency
2. **Unit of analysis = procedure/video.** Use a cluster bootstrap over videos. Report how
   few independent videos you actually have, and what that implies for CI width.
3. **Subgroup analysis** on the condition tags (smoke, blood, specular, occlusion, motion blur),
   with CIs, plus a **failure-mode catalog**: annotated examples of each failure and what the
   system does (wrong vs. correctly abstaining).
4. **Non-inferiority testing** for deployment changes (FP32→FP16, PyTorch→TensorRT) using
   paired bootstrap on per-video differences with a pre-set margin.
5. **Calibration** of the confidence outputs that drive abstention and alerts.
6. **Traceability matrix** mapping each requirement (R1–R6) → test → result → pass/fail.
7. **Reproducibility:** every reported number is tied to a git SHA, a data-split hash,
   a config, and a seed. Rerun the final eval from a clean checkout.

Keep it short (2–3 pages), and mention that a real submission would sit under design
controls (IEC 62304 software lifecycle, ISO 13485 QMS). Don't pretend this project is compliant.

---

## Repository layout

```text
surgical-scene-understanding/
  configs/
  data/                 # split manifests + hashes only, not the data
  training/             # datasets, models, losses, train.py
  inference/            # shape model fit, MAP (Python reference), Kalman, stereo
  eval/                 # metrics, bootstrap, subgroup analysis, report generation
  export/               # ONNX export, TRT build, golden-output generation
  cpp/                  # runtime (see Phase 6)
  docs/
    design_doc.md       # architecture, alternatives considered, tradeoffs
    validation_plan.md  # pre-specified (committed before final eval — the git history proves it)
    validation_report.md
    latency_report.md
  tests/
```

Experiment tracking: MLflow or W&B, but the source of truth is the config + SHA + split hash.

---

## Schedule (≈9–10 weeks part-time) and cut list

| Week | Deliverable |
|---|---|
| 1 | Data access, video-level splits + hashes, condition tagging, **validation plan committed** |
| 2–3 | Phase 1 model + full eval harness |
| 3–4 | Phase 2 keypoints |
| 4–6 | Phase 3 shape model + MAP + ablation |
| 6–7 | Phase 4 temporal |
| 7–8 | Phase 5 stereo + proximity |
| 8–10 | Phase 6 C++/TensorRT, validation report, demo video, design doc |

**If time is short, protect this minimum story:** Phase 1 (harness) → Phase 2 → Phase 3 →
Phase 6 (TensorRT + C++ MAP) → validation report. Cut in this order: learned stereo network,
discrete BP alternative, fixed-lag smoother, robot-frame stretch goal, INT8 (don't start it).

---

## Final artifacts

1. **Demo video**: side-by-side raw vs. overlay (masks, keypoints with confidence, 3D
   distance readout, alert state), including a smoke/occlusion segment showing graceful
   degradation.
2. **Design doc**: problem → requirements → architecture → alternatives → failure modes →
   what I'd do with real kinematics / more data / an embedded target.
3. **Validation report** with the traceability matrix.
4. **Latency report** with the per-stage p50/p99 table.
5. Clean repo with passing Python and C++ test suites.

---

## Coverage map

| Topic | Where |
|---|---|
| Dense segmentation, structure detection | Phases 1–2 |
| Statistical modeling, graphical models, MAP, optimization | Phase 3 (shape model, LM, tree BP), Phase 4 (Kalman/Laplace) |
| Rotation representations, rigid/similarity registration, camera models | Phase 3 (Umeyama), Phase 5 |
| Statistical shape modeling (preferred) | Phase 3 |
| Depth estimation / 3D reconstruction (preferred) | Phase 5 |
| Real-time constraints, ONNX/TensorRT, mixed precision | Phase 6 |
| C++ integration | Phase 6 |
| Clinically meaningful metrics, regulatory-style validation | Use case, validation plan |
| Product orientation (surgeon as user) | Use case, alert logic, abstention |
| Reproducibility, testing, best practices | Data rules, parity tests, repo layout |
