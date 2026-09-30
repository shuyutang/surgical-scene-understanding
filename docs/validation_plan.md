# Validation Plan: Phase 1 Segmentation (pre-specified)

Committed **before** any evaluation on the frozen test split. The acceptance criteria below
may not be changed after the test split is first evaluated. Changes after that point go into
a dated amendment section with the reason, and are reported as post hoc.

This is a practice project, modeled on a design-verification test plan. It is not
IEC 62304 / ISO 13485 compliant and doesn't claim to be.

## Data under test

| Set | Source | Unit of analysis | Size | Frozen hash |
|---|---|---|---|---|
| Internal test | SISVSE, patients of official `real_val_1` | patient (10) | 1,135 frames | `edc5fb33ae75cc85` |
| External test | EndoVis 2018, all 19 labeled sequences (never trained on) | sequence (19) | 3,232 frames | n/a (full dataset) |
| Dev (tuning only) | SISVSE, 6 patients from `real_train_1` | patient | 650 frames | `5cd63fbfc667f8e9` |

Model selection (checkpoint, thresholds, subgroup cut-points) uses train/dev only.

## Statistical method

- Metrics are **pooled over pixels** within the sampled frames: Dice, IoU, and boundary F at a
  3 px tolerance (640×512 working resolution). HD95 is the median of per-frame values.
  Calibration is the pixel-level ECE with 15 bins.
- 95% CIs come from a **cluster bootstrap over patients (SISVSE) or sequences (EndoVis18)**,
  with 2,000 replicates and percentile intervals. Frames are never resampled independently.
- A criterion **passes if the lower 95% CI bound clears the threshold** (upper bound for
  "lower is better").
- Stated limitation: 10 test patients and 19 external sequences give wide intervals. The
  CI width is itself a reported result.

## Endpoints and acceptance criteria

Harmonized instrument classes: tip (SISVSE `*_Head` / EndoVis `clasper`), wrist, shaft
(`*_Body` / `shaft`). "Instrument" = the union of the three.

| ID | Endpoint | Set | Criterion |
|---|---|---|---|
| P1 (primary) | Instrument Dice | Internal | lower bound ≥ 0.85 |
| P2 (primary) | Tip Dice | Internal | lower bound ≥ 0.70 |
| S1 | Tip boundary F (3 px) | Internal | lower bound ≥ 0.60 |
| S2 | Liver, stomach Dice | Internal | lower bound ≥ 0.60 each |
| S3 | Pixel ECE, all 32 classes | Internal | upper bound ≤ 0.05 |
| S4 | Instrument Dice, zero-shot | External | lower bound ≥ 0.70 |
| E1 (exploratory) | Pancreas, gallbladder, spleen Dice; all 32-class Dice/IoU/BF/HD95 | Internal | report only |
| E2 (exploratory) | Tip/wrist/shaft Dice, zero-shot | External | report only |

## Robustness subgroups

Label-free image-condition proxies are computed per frame: sharpness (Laplacian variance),
specular fraction, haze (dark-channel mean), and brightness. The subgroup cut-points are the
**train-split quartiles** (fixed before test):

- blurry = sharpness < Q1(train)
- glare = specular > Q3(train)
- hazy = haze > Q3(train)
- dark = brightness < Q1(train)

**Criterion R1:** in every subgroup of the internal test set, the instrument Dice point
estimate is within 0.10 of the overall value. Subgroup CIs are reported. They are
expected to be wide, and R1 is judged on the point estimate for that reason.

These proxies are not validated clinical condition labels. Hand-tagging a sample to check
them is future work.

## Not evaluated in Phase 1

Temporal stability (SISVSE frames are a median of 13 s apart; covered in Phase 4 on
SurgPose), latency (Phase 6), and 3D accuracy (Phase 5).

## Methods note (2026-09-24, before any test evaluation)

The dev evaluation showed pixel ECE 0.068, which fails S3 on dev. A single softmax temperature
was fitted on dev pixels by NLL (`scripts/calibrate_seg.py`; T = 1.56) and is applied to all
confidences from then on. Argmax is unchanged, so no Dice/IoU/BF endpoint is affected.
Acceptance criteria are unchanged. Dev ECE after scaling (0.013) is in-sample and optimistic;
the test set is the real check.

Model under test: `runs/seg_unet_r34_20260924-230723/best.pt` (epoch 55, selected on dev mIoU).

---

# Validation Plan: Phases 2–3 Keypoints + Structured Inference (pre-specified 2026-09-25)

Committed before any evaluation on SurgPose test trajectories 20–33. Those trajectories use
different ex vivo backgrounds from train/dev (chicken thigh / beef vs. gizzard / liver), so the
test set is also a domain-shift test.

- **Model under test:** `runs/kp_unet_r34_20260925-001134/best.pt` (epoch 20, dev PCK@10 = 0.967).
- **Data:** test = 14 trajectories, every 5th frame, left eye (2,814 frames, hash `977ab824eb176fa9`).
  Unit of analysis = trajectory (14); trajectory-clustered bootstrap, 2,000 replicates.
- **Occlusion protocol:** in every test frame, one uniformly random keypoint per instrument is
  covered by a 36 px-radius (native) tissue patch from the same image (fixed seed). This is a
  controlled stand-in for real occlusion. Real occlusions (tissue folds, other instrument) are
  not labeled in SurgPose.
- **Solver hyperparameters** (confidence gate, prior weight β, Cauchy scale) are chosen by grid
  search on dev (clean + occluded), never on test.
- **Shape models** are fit on train ground truth only.

| ID | Hypothesis / endpoint | Criterion |
|---|---|---|
| K1 | Keypoint detection works under the background shift: argmax mean error, clean test | upper 95% bound ≤ 15 px |
| H1 (primary) | The shape prior recovers occluded keypoints: MAP − argmax error on occluded keypoints | upper 95% bound < 0 |
| H2 | The prior doesn't hurt when evidence is good: MAP − argmax mean error, clean test | upper 95% bound ≤ +1.0 px |
| H3 | MAP outputs are anatomically plausible: implausible rate (train-GT 99th-percentile threshold), occluded test | MAP < argmax (point estimate) |
| E1 (exploratory) | Discrete top-N candidate MAP vs continuous MAP; PCK@5/10/20; per-keypoint error; confidence reliability; runtime | report only |

## Amendment 1 (2026-09-25, POST HOC: after the Phase 2–3 test set was seen)

**Pre-specified outcome (stands as reported):** K1 FAIL, H1 FAIL, H2 PASS, H3 PASS
([report](phase23_test_report.md)).

**Why H1 failed: a protocol leak, found by diagnosis.** Training augmentation pasted occluders
*centered on keypoints*, and the test protocol did the same. The network learned "the center of a
pasted blob is a keypoint". Evidence: the same test keypoints were localized *better* when
occluded (13.9 → 10.9 px mean) with almost no drop in confidence.

**Diagnostic** (same model, test occluders offset up to 0.6× radius, still covering the keypoint):
occluded-keypoint error rose to 18.8 px, and MAP − argmax on occluded keypoints was
−1.43 [−2.22, −0.71] px ([report](phase23_posthoc_diagnostic_offcenter_report.md)).

**Fix:** retrained with off-center occluders plus random distractor patches
(`configs/kp_unet_r34_decentered.yaml`). Evaluated with off-center test occluders; solver
hyperparameters re-selected on dev ([report](phase23_posthoc_decentered_report.md)):

| ID | Post-hoc result | Would-be verdict |
|---|---|---|
| K1 | argmax clean 14.01 [10.37, 17.40] px | still FAIL (background domain shift; shaft keypoint dominates) |
| H1 | occluded MAP − argmax −1.47 [−2.09, −0.83] px | holds |
| H2 | clean MAP − argmax +0.01 [−0.84, 0.75] px | holds |
| H3 | implausible 17.2% → 1.4% (occluded) | holds |

These are post-hoc results on a test set that had already been seen. They generate hypotheses
and are not confirmatory. A confirmatory re-test needs fresh data, e.g. SurgPose's
green-channel videos or new trajectories.

---

# Validation Plan: Stage 2 (Phases 3b–6), pre-specified 2026-09-25

Committed before any stage-2 evaluation on the test2 trajectories. Everything below was selected
on the tune trajectories only (`scripts/tune_stage2.py`, output `configs/stage2_selected.json`,
sha256 prefix `35413a18cfe1efd5`).

## Why a new split

Tuning on dev (18–19) alone always concludes "the shape prior doesn't help": dev shares the
training backgrounds, so the argmax is already near-perfect there, and the gain only appears under
the background shift. Every stage-2 parameter needs to see that shift, so one beef (20) and one
chicken-thigh (23) trajectory move from test into tuning.

| Set | Trajectories | Frames per eye (30 fps) | Hash |
|---|---|---|---|
| tune | 18, 19, 20, 23 | 4,004 | `0ef34738c3c2c070` |
| test2 | 21, 22, 24–33 | 12,012 | `ee86807581bb2b50` |

**Prior exposure, stated plainly.** Every 5th left frame of the test2 trajectories was evaluated in
Phases 2–3 (argmax/MAP errors, the confidence reliability table). The gate idea (G1–G3) was
motivated by that reliability table, so G1–G3 on test2 are pre-specified but *not blind*. The
temporal, 3D, hand-eye, proximity and deployment endpoints (T, S, D) have never been computed on
these trajectories. Trajectories share tissue piles across tune and test2 (e.g. 20/21, 23/25/27),
so tuning transfers more easily than it would to a new setup.

Model under test: `runs/kp_unet_r34_decentered_20260925-004241/best.pt` (unchanged). Unit of
analysis = trajectory (12); trajectory-clustered bootstrap, 2,000 replicates, percentile CIs.

## Fix 2 (shaft line constraint) was not built

The premise was that shaft-keypoint error runs *along* the shaft, so a line from the Phase 1 mask
would constrain it. On the tune trajectories that have the shift (20, 23), the shaft error is
mostly *perpendicular* to the instrument axis: median 45.8 px perpendicular vs 12.6 px along it
(PSM1). Visual inspection shows the labeled shaft point sitting beside the visible shaft on these
trajectories, a labeling-convention difference rather than a detection failure. There is also no
video/label time offset: the best lag is 0 on every trajectory. Snapping to the mask would move
predictions *away* from the labels. In addition, the Phase 1 segmentation model does not transfer
to SurgPose: zero-shot it labels beef tissue as instrument and misses the dVRK jaws, so it isn't
used anywhere in stage 2. The instrument mask for depth comes from keypoints instead.

## Pipeline under test

Per eye and frame: heatmap observations → **gated MAP** (argmax where conf ≥ τ, shape-prior MAP
elsewhere). Per eye over time: **constant-velocity Kalman filter** per keypoint, measurement
noise from the heatmap's Laplace σ and confidence, χ² innovation gate. Across eyes:
**triangulation** (DLT + Gauss–Newton with distortion) with first-order covariance, inflated by
one scalar κ fitted on tune (the 3D analog of temperature scaling). Proximity: signed distance
from the 3D tip (midpoint of the two jaw tips) to a robust local tissue plane from SGM, fitted in
disparity space.

Occlusion protocol (T3/T4): every 60 frames, one random keypoint per instrument is covered for
15 frames (0.5 s) by a tissue patch that moves with it, center offset ≤ 0.6× radius.

## Endpoints and acceptance criteria

| ID | Endpoint (test2) | Criterion |
|---|---|---|
| G1 (primary, 3b) | Gated − argmax mean keypoint error, clean left, every 5th frame | upper 95% bound < 0 |
| G2 | Gated − argmax PCK@5 | lower bound ≥ −0.02 |
| G3 | Gated − argmax on occluded keypoints | upper bound < 0 |
| T1 (primary, 4) | Jitter ratio KF / gated (acceleration error vs GT, all frames) | upper bound ≤ 0.6 |
| T2 | KF − gated mean error, clean | upper bound ≤ +0.5 px |
| T3 | KF − gated on occluded keypoints during episodes | upper bound < 0 |
| T4 | KF − gated in the 15 frames after an episode ends | upper bound ≤ +1.0 px |
| S1 (R1-3D) | Mean 3D tip error, KF pipeline | upper bound ≤ 5 mm |
| S2 | 3D tip error, KF − frame-wise (paired) | upper bound < 0 |
| S3 | Coverage of the κ-inflated 95% ellipsoid | lower bound ≥ 0.85 and point ≤ 0.99 |
| S4a | Hand-eye (registered from vision on each trajectory's first half, evaluated on the second half): kinematics − vision 3D jaw-pivot error | upper bound < 0 |
| S4b | Kinematics + registration jaw-pivot error | upper bound ≤ 5 mm |
| S5 (R4) | Alert toggles/min, (KF + 3 mm hysteresis) / frame-wise, alert at 10 mm | upper bound ≤ 0.7 |
| E (exploratory) | Lag 0/3/6 tradeoff, "unknown" flag, NIS, depth vs lateral 3D error, distance error, alert sensitivity/specificity, GT stereo label consistency | report only |

**S1 is a requirement, not a tuned threshold.** On tune it is 15.2 [9.3, 22.5] mm and is expected
to fail. The cause is physical: tissue sits 150–230 mm from a 5.5 mm baseline, so 1 px of
disparity error is 2–5 mm of depth. The lateral 3D error on tune is about 1 mm, and nearly all of
the error is along the viewing ray. S4 tests the engineering answer: robot kinematics plus a
registration fitted from the same noisy vision estimates.

Exclusions: a GT stereo pair with reprojection RMS > 3 px is excluded from 3D endpoints (tune: 0.01%).
Triangulations outside 20–2000 mm depth count as missing.

## Phase 6 deployment criteria

Deploy graph = uint8 BGR stereo pair (2×986×1400×3) → avg-pool resize, pad, normalize → U-Net →
sigmoid → decode (argmax, sub-pixel, Laplace σ). Everything runs inside the engine. ONNX opset 17,
TensorRT 10.16, static batch 2.

| ID | Endpoint | Criterion |
|---|---|---|
| D1 | TensorRT FP32 vs PyTorch FP32 deploy graph, 200 tune frames: keypoint difference | 99th percentile ≤ 0.1 px |
| D2 (R6) | TensorRT FP16 − PyTorch FP32 mean argmax keypoint error, test2 left every 5th frame (paired, trajectory-clustered) | upper bound ≤ +0.25 px |
| D3 | C++ runtime reproduces the Python reference on goldens (MAP fit, Kalman filter, triangulation, engine outputs) | all parity tests pass at stated tolerances |
| D4 (R3) | C++ end-to-end latency per stereo pair (H2D, TensorRT FP16, D2H, gated MAP ×4, Kalman, triangulation), ≥ 1,000 frames, video decode excluded and reported separately | p99 < 33 ms |

---

# Validation Plan: v3 Phase C (kinematics-fused 3D tool state), pre-specified 2026-09-28

Committed before any v3 number is computed on the test2 trajectories. Plan:
[v3 plan](plans/plan_v3.md). Development record:
[v3_progress.md](v3_progress.md). All design choices were made on train (0–17: tool geometry
priors) and tune (18, 19, 20, 23: everything else).

## Prior exposure, stated plainly

- **test2 is being used a second time.** The v2 stage-2 evaluation ran on it once (`b0d7522`),
  and v3 exists because v2's S1 failed there, with the error almost entirely along the viewing ray.
  So v3's *direction* was motivated by a test2 result. No v3 component (kinematics, calibration,
  fusion, hybrid) has been computed on any test2 trajectory.
- **Trajectory 21 (test2) was looked at during method discussions:** heatmap peaks, a MAP trace,
  Kalman traces, an SGM example, and its rectification offset. None of these involved kinematics
  or the Phase C method.
- **Tool types:** long-jaw instruments were discovered on train (17) and tune (18, 19). Whether
  test2 contains them is unknown, which is why the subgroup below is pre-specified.

## Method under test

For each arm, causally (frame i uses frames 0..i only; the code is `src/surgscene/fusion.py`):

1. **Calibration**, refitted every 15 frames after a 30-frame warm-up:
   - hand-eye by robust rigid registration of the kinematic pivot (`api_cp`) to per-frame
     triangulated vision pivots;
   - then the tool geometry (tip-midpoint offset; wrist L, b, ax) by robust reprojection LM with
     the hand-eye fixed, using train-fitted priors (`configs/tool_geometry.json`, sha256
     `efff97598918ef94`).
2. **Per-frame state** (iterated EKF, Cauchy-robust): a kinematic correction δ, mean-reverting and
   constrained to be lateral (δ·ray = 0, sd 0.5 mm), plus the signed jaw half-opening h.
3. **Output ("hybrid"):** each jaw tip is triangulated from the v2 Kalman 2D detections of both
   eyes, with a Gaussian depth prior (sd 5 mm) along the viewing ray centred on the fused point.
   Covariance = inverse Hessian × κ.
4. Before the first calibration (the first second), the v2 estimate is used.

Configuration: `configs/v3_fusion_selected.json` (sha256 `f409b1a40ec23b43`), κ = 10.4649.
Inputs are unchanged from stage 2: same keypoint model, gated MAP and Kalman parameters
(`configs/stage2_selected.json`). Script: `scripts/eval_v3c.py --split test2`, run once.

Unit of analysis = trajectory (12). Trajectory-clustered bootstrap, 2,000 replicates, percentile
CIs. Frames and exclusions are as for S1: every 5th frame, GT stereo pairs with reprojection RMS
> 3 px excluded, and tip = the midpoint of the two jaw tips.

## Endpoints and acceptance criteria

| ID | Endpoint (test2) | Criterion | Tune (in-sample) |
|---|---|---|---|
| C1 (primary, R1) | Mean 3D tip error, hybrid (all frames; v2 fills the warm-up) | upper 95% bound ≤ 5.0 mm | 6.86 [3.96, 9.50] |
| C2 | 3D tip error, hybrid − v2 (paired) | upper bound < 0 | −8.38 [−14.50, −0.90] |
| C3 (R7) | Coverage of the κ-inflated 95% ellipsoid (post-warm-up) | lower bound ≥ 0.85 and point ≤ 0.99 | 0.950 [0.914, 0.974] (κ fitted here) |
| C4 | Fused jaw pivot 3D error (post-warm-up) | upper bound ≤ 5.0 mm | 3.11 [2.65, 3.60] |
| C6 | 2D left-image tip error, hybrid − v2 (non-inferiority, so a depth gain can't hide a lateral loss) | upper bound ≤ +0.5 px | −0.33 [−1.19, 0.23] |
| Subgroup (pre-specified, report) | C1 and C2 for long-jaw vs standard arm-trajectories (online-estimated tip offset > 15 mm at the last calibration; cluster = trajectory-arm) | report with CIs | long −4.5 [−16.0, 3.5]; standard −12.2 [−22.6, −4.3] |
| Reported | C5 tip-swap rate (> 3 mm) hybrid vs v2; depth/lateral split; 2D errors; post-warm-up only; per-trajectory calibration | report only | swaps 0.003 vs 0.043 |

**C1 is the R1 requirement, not a tuned threshold. It's expected to fail:** the upper bound on tune
is 9.5 mm. Part of every 3D number is reference noise: the GT tips are triangulated from hand labels
with the same 5.5 mm baseline, with a depth std of about 1.8 mm (median) on tune. C2 is the
engineering claim: kinematics fixes the depth failure that made S1 fail. C3 checks κ out of
sample. Trajectory 18 (long-jaw) is worse than v2 on tune, even with an oracle calibration. The
subgroup is there to test whether that generalizes.

The plan's C5 (tip swaps) is demoted to "reported" because its tune base rate is too low for a
useful criterion.

## Sensitivity check (tune, before freezing)

`scripts/eval_fusion.py --grid` (output `runs/v3_fusion/results_grid.json`), with one parameter varied
at a time around the frozen defaults. Numbers are the causal fusion tip-midpoint error after the warm-up, mean in mm:

| Parameter | Values tried | Range of mean error |
|---|---|---|
| sig_kin × τ | {1, 2, 4} mm × {10, 30, 100} frames | 6.72–6.73 |
| Warm-up | 15, 30, 60 frames | 6.63–6.74 (all frames: 6.82–7.17) |
| sig_ray | 0.2, 0.5, 2.0 mm | 6.65–6.72 (unconstrained: 7.30) |
| Refit interval | 5, 15, 45 frames | 6.61–6.80 |
| Hybrid depth-prior sd | 2, 5 mm | 6.69–6.75 |

The defaults aren't on a cliff, and they were kept as set; nothing was selected from this grid.
The per-frame parameters barely matter. The error is set by the calibration (hand-eye + tool
geometry) and the tool model, not by the per-frame filter. The one structural choice that
matters, constraining δ laterally, was made before the grid.

## Amendment 1 to v3 Phase C (2026-09-28, POST HOC: after the pre-specified test2 run)

The pre-specified run (`5eb5190`) showed no gain on long-jaw instruments. On train 17 and tune
18/19, the tip midpoint of that tool scatters 4–6 mm in the kinematic tool frame (standard tools:
about 1 mm). Kinematics is less precise at 24 mm from the pivot, so the rigid model's oracle floor
is 3.7–6.7 mm.

The fitted tip offset c was also estimated by 2D reprojection, the fit that proved
depth-ill-conditioned for the hand-eye. A variant estimating c as a robust mean of per-frame
triangulated tip midpoints in the tool frame (`tip_mode="3d"`,
`configs/v3_fusion_posthoc_tip3d.json`) was mixed on tune: 18: 10.4 → 6.1, 19: 6.5 → 10.4,
20: 7.1 → 4.4, 23: 2.7 → 3.2 mm. Its only justification is consistency with the hand-eye.

It was run on test2 **after** the pre-specified run, so everything below is exploratory:

| test2 | Pre-specified | Post hoc (c in 3D) |
|---|---|---|
| C1 mean 3D tip error | 5.27 [3.56, 7.43] | 4.17 [2.97, 5.84] |
| C2 vs v2 | −4.78 [−6.93, −2.71] | −5.88 [−7.92, −3.96] |
| C3 coverage (same κ) | 0.971 | 0.977 |
| Long-jaw subgroup (hybrid / v2) | 12.31 / 11.81 (4 arm-trajectories) | 8.42 / 15.24 (5) |
| Standard subgroup | 3.86 | 3.06 |

The variant improves 10 of 12 trajectories. Even so, C1's upper bound stays above 5 mm. The
hypothesis for a future confirmatory test is: `tip_mode="3d"` reduces 3D tip error, especially on
long-jaw instruments. It needs data neither tune nor test2 has seen. The frozen v3 config is unchanged.
A re-run of the frozen config reproduced the pre-specified numbers exactly (`runs/v3c_test2_repro/`).

---

# Validation Plan: v3 Phase B (learned stereo on SERV-CT), pre-specified 2026-09-28

Committed before any learned-stereo or SGM output is computed on SERV-CT. **SERV-CT is fresh
data**: it hasn't been used for any choice in this project. Its disparity range (1st–99th
percentile 34–126 px) was read from the GT files to set SGM's search range to 160 px; no method
output was compared with the GT.

**Method under test:** RAFT-Stereo (Princeton VL, MIT license), pretrained, zero-shot. The
checkpoint was selected on SurgPose tune (18, 19, 20, 23) by lowest photometric error on tissue
around the tips (`scripts/eval_stereo_tune.py`, `runs/v3_stereo_tune/`): **middlebury**. The tune
proxies for SGM / middlebury were photometric error 12.56 / 12.97 (not comparable: SGM is scored
only on its 82% valid pixels), plane scatter 5.42 / 1.54 mm, and 19 / 413 ms per pair.

**Comparator:** SGM as in v2 (`proximity.make_sgbm`, block 5, numDisparities 160). Both methods
use the B3 vertical-offset rule.

**Data:** SERV-CT `Reference_CT`, 16 pairs (8 per experiment, 2 porcine specimens), 720×576.
Valid pixels: GT disparity > 0 and not flagged in OcclusionL.

| ID | Endpoint | Criterion |
|---|---|---|
| B1 (primary) | Mean absolute depth error (mm), learned − SGM, on pixels valid for both (paired over the 16 pairs, bootstrap) | upper 95% bound < 0 |
| Reported | Depth MAE of each method; learned on all valid pixels; SGM coverage; bad-pixel rate (> 3 px) and disparity EPE differences; per-experiment numbers; the per-pair vertical offset | report only |

**Limits, stated up front:**
- 16 pairs from 2 specimens, so the pair-level CI overstates independence. Per-experiment numbers
  are reported, and a result driven by one experiment will be called out.
- Scoring only the pixels valid for both methods is conservative for the learned model: SGM
  drops the hardest pixels.
- Latency (B4) is reported, not tested. RAFT-Stereo at 32 iterations is far outside the 33 ms
  budget without optimization.

Script: `uv run --group stereo python scripts/eval_servct.py`, run once.

---

# Validation Plan: v3 Phase A (DINOv2 keypoint network), pre-specified 2026-09-28

Committed before the Phase A model is evaluated on any test2 trajectory. Training used train
(0–17) with dev (18–19) for checkpoint selection. Evaluation choices used tune (18, 19, 20, 23):
`scripts/eval_kp_vit.py --split tune`, `runs/v3_kp_vit_tune/`.

**Prior exposure:** test2's left frames (every 5th, clean and occluded) were used in the Phase 2–3
evaluation and in the stage-2 D2 check of the v2 network. The Phase A network has never been run
on them.

**Model under test:**
- DINOv2 ViT-S/14 (timm, fine-tuned), with a conv stem and a U-Net decoder;
- a learned per-keypoint log-variance head, trained by Gaussian NLL on the detached decoding error;
- focal loss with argmax positives (A0);
- the same data, augmentation and 20-epoch schedule as v2.

The seed-0 run (`kp_vit_s_20260928-175301`) is under test, against the v2 seed-0 U-Net
(`kp_unet_r34_decentered_20260925-004241`). The seed-1 runs of both are reported for variability.
Config: `configs/v3_kp_vit_selected.json`; learned-σ scale k = 6.086, fitted on tune.

**Data:** test2 (21, 22, 24–33), left eye, every 5th frame from the frame cache. Clean frames, plus the v2
de-centered occluder protocol (seed 7, offset 0.6). Unit of analysis = trajectory.
Trajectory-clustered bootstrap, 2,000 replicates.

| ID | Endpoint (test2) | Criterion | Tune (in-sample) |
|---|---|---|---|
| A1 (primary, as in the v3 plan) | PCK@10, DINOv2 − v2, clean | lower 95% bound > 0 | −0.038 [−0.081, 0.005]: expected to fail |
| A2 | Mean error, DINOv2 − v2, clean | upper bound < 0 | +0.17 [−0.94, 2.05] |
| A7 | Occluded keypoints: mean error, DINOv2 − v2 | upper bound < 0 | −3.21 [−4.72, −0.86] |
| A5 | Spearman(σ, error): learned σ vs v2 Laplace σ | learned > Laplace (point) | +0.66 vs −0.20 |
| A8 | Coverage of the 95% radius, 2.4477 · k · learned σ | lower bound ≥ 0.85 and point ≤ 0.99 | 0.950 (k fitted here) |
| Reported | Median, PCK@5, per-trajectory errors, seed-1 runs | report only | |

A1 stays primary because the plan fixed it before any Phase A result. On tune, the DINOv2 model is
sharper on the in-domain trajectories (18, 19) but not better on the shifted, label-convention
trajectories (20, 23). A1 is therefore expected to fail. The plan's A4 (mask head) wasn't built,
per its cut list. Downstream use of the new keypoints (Phase C with Phase A inputs) isn't part of
this test.

---

# Validation Plan: v3 step 5 (DINOv2 front end downstream), E1 (ViT FP16), B5 (fast stereo), pre-specified 2026-09-28

Committed before any of these endpoints is computed on test2 or on SERV-CT.

## Step 5: DINOv2 keypoints + learned σ through the Phase C pipeline (test2, 3rd use)

**Method.** Only the inputs to the frozen v3 Phase C pipeline change:
- DINOv2 argmax keypoints for both eyes (full rate: `data/cache/surgpose_obs_vit`);
- measurement std = k · learned σ (k = 6.086, from Phase A), with confidence used only as the
  predict-only gate;
- Kalman r0 = 0.5, selected on tune from {0.5, 1, 2} by mean 3D hybrid error;
- κ = 5.093, refitted on tune.

Config: `configs/v3_downstream_vit_selected.json` (sha256 `d4caf829dd71fe06`). Comparator: the
pre-registered v3 Phase C pipeline on v2 inputs, on the same frames. Script:
`scripts/eval_downstream_vit.py --split test2`.

| ID | Endpoint | Criterion | Tune |
|---|---|---|---|
| V1 (primary) | 3D tip error, DINOv2 inputs − v2 inputs (paired) | UB < 0 | −0.21 [−1.26, 1.00]: expected to fail |
| V2 (R1) | Mean 3D tip error, DINOv2 inputs | UB ≤ 5.0 mm | 6.65 [5.17, 8.03] |
| V3 | Occlusion episodes: Kalman 2D error on occluded keypoints, DINOv2 − v2 | UB < 0 | −2.35 [−6.17, 1.42] |
| V4 (R7) | Coverage of the κ-inflated 95% ellipsoid | LB ≥ 0.85, point ≤ 0.99 | 0.950 (κ fitted here) |

## E1 (R6): DINOv2 TensorRT FP16 engine (test2)

**Engine:** `runs/deploy_vit/kp_vit_stereo_fp16.engine` (`scripts/export_trt_vit.py`). The
preprocessing and decode are pinned to FP32 (without that, every keypoint was NaN from FP16
index overflow), as are LayerNorm, softmax and reductions.

**Development checks on tune** (`scripts/eval_deploy_vit.py`):
- FP32 engine vs PyTorch: p50 0.003 px.
- FP16 engine vs PyTorch FP32: p50 0.88 px. Unresolved: it's TensorRT-specific (PyTorch FP16
  autocast: 0.008 px) and unaffected by the precision pinning.
- Against GT on tune: +0.12 px mean.
- Latency per stereo pair: FP16 4.4 ms, FP32 17.6 ms.

| ID | Endpoint | Criterion |
|---|---|---|
| E1 | TRT FP16 − PyTorch FP32 mean keypoint error vs GT, test2 left every 5th frame (paired, trajectory-clustered) | UB ≤ +0.25 px (D2's margin) |

## B5: fast stereo configuration on SERV-CT (2nd use of SERV-CT)

B1 validated RAFT-Stereo `middlebury` at 32 iterations (98 ms per half-resolution pair in PyTorch
mixed precision). The latency check puts `realtime` at 4 iterations at 10.7 ms: with the 4.4 ms
keypoint engine, that's the only tested configuration that could fit the 33 ms budget. It was chosen
for latency alone: no configuration was compared on SERV-CT before this commit, except B1 itself.

| ID | Endpoint | Criterion |
|---|---|---|
| B5 | SERV-CT depth MAE, `realtime` @ 4 iterations − SGM (same protocol as B1) | UB < 0 |
| Reported | `realtime` @ 4 vs `middlebury` @ 32 depth MAE (from the two runs) | report only |

Script: `uv run --group stereo python scripts/eval_servct.py --checkpoint realtime --iters 4`.

---

# Validation Plan: v4 (jaw-length constraint for long-jaw instruments), pre-specified 2026-09-29

## Prior exposure, stated plainly

- **This is test2's fourth use.** It was used by v2 stage 2, by v3 Phase C (pre-specified), by
  v3 Phase C Amendment 1 (post hoc) and by v3 step 5. All 34 trajectories have been seen.
- **v4's direction was motivated partly by test2.** v3 Phase C showed no gain on long-jaw
  instruments (trajectories 28 and 29, 12.3 mm), and the post-hoc amendment explored the tip
  offset on test2. So it is known that test2 contains long-jaw instruments on 28 and 29, and that
  v3 does badly there. The diagnosis and the method were developed on train and tune only; no v4
  component has been computed on a test2 trajectory. SAM 2 masks were generated for test2
  trajectories, and only label-free statistics were looked at.
- **Consequence:** a pass here is *supportive*, not confirmatory. Confirmation needs data none of
  this has seen.

## Development (train and tune), what was tried and dropped

The v4 plan proposed SAM 2 masks as a measurement. In development, none of the three mask uses
earned a place, so **the method under test uses no masks**:
- Instrument type from the mask (jaw length / shaft width): overlapping classes on train + tune
  (standard up to 1.93, long from 1.86). A kinematic feature separates them perfectly.
- Depth from the apparent shaft width: 3–15% within-trajectory noise (6–10 mm at 200 mm), and
  the implied diameter varies 5.0–8.2 mm between trajectories. Worse than the kinematic pivot.
- Visibility gate (length constraint off when a tip detection is > 10 px outside its mask):
  neutral on tune (5.97 vs 5.97 mm).

Diagnosis on tune: the long-jaw error is almost entirely in the pivot-to-tip vector, along the
viewing ray (18 PSM3: fitted length 17.8 mm vs 29.2 GT). With the correct type prior it
persists (13.5 mm), because the tip offset in the kinematic tool frame isn't rigid on long jaws:
its deviation correlates 0.67–0.80 with the last wrist joint (cable-driven error), on train 17
and tune 18 and 19.

## Method under test

`scripts/eval_v4.py`. The v3 Phase C fusion and hybrid (`configs/v3_fusion_selected.json`,
sha256 `f409b1a40ec23b43`), unchanged except on arms classified as long-jaw:

1. **Type**, causal (`fusion.classify_type`): the running 90th percentile, over frames seen so
   far, of the lateral jaw length in mm (left-image pivot to tip-midpoint distance × fused pivot
   depth / f). It's decided from 30 valid frames on; above 13 mm = long. The threshold sits in the
   train gap (standard ≤ 11.1 mm, long ≥ 16.0).
2. **Jaw-length constraint** (`fusion.length_constrained_depth`), long-type arms only. The tip
   midpoint lies on its detected left viewing ray at the type's pivot-to-tip length L from the
   fused pivot; of the two intersections, the one nearer the kinematic prediction is kept. Its
   depth std is √((sd_L L)² + (d⊥ sd_lat)²)/√disc. It's combined by inverse variance with the
   kinematic tip depth (std = the type's train tip-offset scatter), and the result becomes each
   tip's depth prior in the v3 hybrid triangulation.
3. Standard-type arms, undecided frames and the warm-up are exactly v3.

**Frozen:**
- `configs/v4_selected.json` (sha256 `a85ecb4d244b594f`), κ = 12.7906, fitted on tune;
- `configs/v4_tool_library.json` (sha256 `7b9276761f1d719d`), train only; the long type rests
  on one train trajectory (17, both arms).

**Tune-informed choices (disclosed):**
- Applying the constraint to long-type arms only. Applied to all arms, it hurt one standard
  arm with buried tips (20/PSM1: 8.4 → 10.8 mm).
- Using the type scatter as the kinematic sd there, instead of v3's 5 mm.

## Endpoints (test2, 12 trajectories; trajectory-arm clusters for M1)

| ID | Endpoint | Criterion | Tune (in-sample) |
|---|---|---|---|
| M1 (primary) | Arm-trajectories the causal classifier calls long: 3D tip error, v4 − v3 (paired, trajectory-arm clusters) | UB < 0 | −2.27 [−5.32, −0.02] (4 arms) |
| M2 (R1) | Mean 3D tip error, v4, all arms | UB ≤ 5.0 mm | 5.73 [3.84, 7.11] |
| M3 | Causal type = GT type (GT pivot-to-tip distance > 15 mm, v3's subgroup rule), fraction of arm-trajectories | ≥ 0.90 | 1.00 (8) |
| M4 (R7) | Coverage of the κ-inflated 95% ellipsoid (post-warm-up) | LB ≥ 0.85 and point ≤ 0.99 | 0.950 (κ fitted here) |
| M5 | 2D left tip error, v4 − v2 (non-inferiority) | UB ≤ +0.5 px | −0.30 [−1.22, 0.37] |
| Reported | All-arm v4 − v3; long/standard subgroups; depth/lateral split; per arm-trajectory table | report only | −1.13 [−2.61, 0.00] |

**Why M1 is on long-classified arms:** the method changes nothing else, so an all-arm paired
difference is zero on most trajectories. With long-jaw tools on 2 of 12 trajectories, about 11%
of bootstrap resamples contain neither, so its upper bound would be 0 by construction.
**M1 has low power:** probably 4 arm-trajectories (28 and 29).
**M2 is expected to fail:** v3 was 5.27 [3.56, 7.43], and v4 can remove at most the long-jaw
excess.

Run once, after this section is committed: `uv run python scripts/eval_v4.py --split test2`.

---

# Validation Plan: scene semantics on GraSP (phase and step recognition), pre-specified 2026-09-29

Committed before any model output is computed on a GraSP test case. Plan and arms:
`docs/plans/plan_scene_semantics.md`.

## Prior exposure, stated plainly

**First use of the GraSP test split** (5 cases, CASE041, 047, 050, 051, 053; 42,897 frames; split
hash `6fb97152f3d58a5b`). Test frames have been read only to extract frozen backbone features
(`scripts/grasp_features.py`, no labels involved); no prediction has been made on them. Labels
are the official December 2024 revision (`splits/grasp_split.json`).

## Development (train cases only, official folds)

- **Temporal arms** (`scripts/grasp_tcn.py dev`, `runs/grasp_dev/report.md`): both fold
  directions × 3 seeds, held-out metric every 25 epochs up to 300. Per (backbone, model) the class
  weighting and epoch count with the highest mean held-out step macro-F1 were selected:

  | Arm | Config | Held-out step F1 | Phase F1 |
  |---|---|---|---|
  | A: ResNet-50 + causal MS-TCN | sqrt_inv, 100 epochs | 0.349 | 0.570 |
  | A0: ResNet-50 + linear probe | sqrt_inv, 75 epochs | 0.285 | 0.451 |
  | B: DINOv2 ViT-B/14 + causal MS-TCN | sqrt_inv, 200 epochs | 0.403 | 0.634 |
  | B0: DINOv2 ViT-B/14 + linear probe | sqrt_inv, 300 epochs | 0.345 | 0.527 |

- **VLM arm** (Qwen3-VL-8B-Instruct, `scripts/grasp_vlm.py`; comparison in
  `runs/grasp_dev_compare/report.md`, fold1 → fold2, every 10th frame):
  - zero-shot, bf16: step F1 0.006; it answers "Denonvilliers_Fascia, Denon_Dissection" for
    almost every frame. 4-bit zero-shot: 0.008.
  - QLoRA fine-tuned on fold1 (every 5th frame, 7,682 samples, 1 epoch): step F1 0.229, phase F1
    0.435 (B on the same frames: 0.403 / 0.620).
  - Causal majority smoothing over 30–300 s lowers step F1 for every VLM run, so the frozen
    window is **0 s** (raw answers) for both zero-shot and fine-tuned.

## Frozen configuration (`configs/grasp_selected.json`)

- A, A0, B, B0: selected configs above, trained on all 8 train cases, 3 seeds each
  (`runs/grasp_final/`); the prediction is the arg max of the mean of the 3 seeds' probabilities.
- C, fine-tuned: QLoRA adapter trained on all 8 train cases, every 10th frame (1 epoch, same
  hyperparameters as development; 7,366 samples, 460 steps): `runs/grasp_vlm/final_ft/adapter`
  (`adapter_model.safetensors` sha256 prefix `350271e9b8282fa8`).
- C, zero-shot: bf16, the prompt in `scripts/grasp_vlm.py` (`zero_shot_prompt`).
- VLM test frames: every 5th frame of each test case (8,582 frames). Unparsed answers count as
  wrong.

## Endpoints (5 test cases; unit = case; paired case-level bootstrap, 2,000 replicates)

| ID | Endpoint | Criterion |
|---|---|---|
| S1 (primary) | Step macro-F1, B − A (backbone effect, both causal MS-TCN), all frames | LB > 0 |
| S2 | Step macro-F1, A − A0 (temporal model effect), all frames | LB > 0 |
| S3 | Step macro-F1, fine-tuned VLM − B, on the VLM's frames | report with CI |
| S4 | Phase macro-F1, B − A, all frames | LB > 0 |
| Reported | Every arm's step / phase macro-F1 and accuracy; B − B0; zero-shot VLM; per-case values and the number of cases with a positive difference | report only |

Macro-F1 per case is over the classes present in the case's ground truth or prediction.

**Expectations, stated before the data:** development says S2 (+0.064) and S1 (+0.054) are
positive, and S3 is strongly negative (−0.17). With 5 cases, a true effect of ~0.05 may still
produce a lower bound below 0; a FAIL there means "not shown on 5 cases", not "no effect".

## Run order

1. `uv run --group vlm python scripts/grasp_vlm.py predict --tag test_zs_bf16 --cases <test> --stride 5 --batch 16`
2. `uv run --group vlm python scripts/grasp_vlm.py predict --tag test_ft --adapter runs/grasp_vlm/final_ft/adapter --cases <test> --stride 5 --batch 16`
3. `uv run python scripts/eval_grasp.py test`, once. Results are committed as they come out.

---

# Validation Plan: v5 step 2 (Fast-FoundationStereo vs RAFT-Stereo on SERV-CT), pre-specified 2026-09-29

**Third use of SERV-CT** (B1: RAFT-Stereo `middlebury`@32 vs SGM; B5: `realtime`@4 vs SGM).
Every model compared here was already scored there (RAFT) or has never seen SERV-CT (FFS), and no
Fast-FoundationStereo output has been computed on SERV-CT before this commit.

**Method under test:** Fast-FoundationStereo (NVIDIA, 2026; a distilled FoundationStereo;
research-only license), zero-shot, fp16, `max_disp` 192. The checkpoint was selected on SurgPose
tune by the same rule as B1 (lowest photometric error on tissue around the tips,
`scripts/eval_stereo_tune.py --select-prefix ffs:`, `runs/v5_ffs_tune/report.md`), among the
three checkpoints documented in its readme: **`23-36-37` at 8 iterations**.

| Model (SurgPose tune, 32 frames) | Photometric error | Plane scatter (mm) | ms / half-res pair* |
|---|---|---|---|
| SGM | 12.56 | 5.42 | 12 |
| RAFT `middlebury`@32 | 12.97 | 1.54 | 475 |
| RAFT `realtime`@4 | 13.11 | 1.66 | 55 |
| FFS `23-36-37`@8 | 12.95 | 1.29 | 93 |
| FFS `23-36-37`@4 | 12.96 | 1.30 | 79 |
| FFS `20-26-39`@8 | 12.98 | 1.15 | 79 |
| FFS `20-30-48`@4 | 12.96 | 1.30 | 57 |

\*Timed while another job was using the GPU; latency is re-measured in the test run on an idle
GPU. The photometric proxy separates the learned models by < 0.2 gray levels, so the selection is
weak. A fourth checkpoint in the release (`15-44-51`) is not in the readme and its configuration
lacks a key the code needs; it was dropped without being run.

**Comparator:** RAFT-Stereo `middlebury`@32, the model validated in B1. Both use the B3
vertical-offset rule.

| ID | Endpoint | Criterion |
|---|---|---|
| B6 (primary) | Mean absolute depth error (mm) on all valid pixels, FFS − RAFT `middlebury`@32 (paired over the 16 pairs, bootstrap) | upper 95% bound < 0 |
| Reported | Each model's depth MAE; bad-pixel (> 3 px) and disparity-EPE differences; FFS − SGM with the B1 protocol; per-experiment numbers; latency on one 720×576 pair | report only |

Limits, as in B1: 16 pairs from 2 specimens (per-experiment numbers reported), and the B1 RAFT
error is already low (1.86 mm on all valid pixels), so a real but small improvement may not be
detectable.

Script: `uv run --group stereo python scripts/eval_servct.py --method ffs:23-36-37@8 --reference middlebury@32`, run once.

The instrument-jaw check of v5 step 0 (tune, development only) is repeated with FFS:
`scripts/v5_dev_stereo_tip.py --models middlebury@32 ffs:23-36-37@8 --out runs/v5_ffs_tip`.

---

# Validation Plan: scene semantics round 2 on GraSP (S5 EndoSSL; ST1–ST4 short-term), pre-specified 2026-09-29

Committed before any EndoSSL prediction on a test case and before any short-term test output.

## Prior exposure, stated plainly

- **GraSP test cases, 2nd use.** The long-term labels (phases, steps) of the 5 test cases were used
  once, for S1–S4 (commit `31f51d7`). Nothing here was tuned on that result.
- **The short-term test labels (instances, instruments, actions) have not been used.** The test
  frames were seen before, only as inputs.
- **Surgical video foundation models:** SurgVISTA's backbone isn't released (its Hugging Face repo
  is empty), and SurgMotion's weights are gated (access requested, not granted). Neither is tested.
  EndoSSL, SurgVISTA's image teacher, is released and tested here.

## S5: in-domain self-supervised backbone (EndoSSL ViT-L/16) for steps

- **Backbone:** EndoSSL ViT-L/16 (MSN on private laparoscopy video; Hirsch et al., MICCAI 2023),
  via the PyTorch conversion released with SurgVISTA.
  - Its weights equal the official JAX checkpoint.
  - With raw 0–255 input it reproduces the official TF SavedModel: cosine 0.9998 on one frame, 0.98
    with the pipeline's own resize and bf16.
  - Features: [CLS, mean patch] after the final norm, 2048-d, frozen.
- **Arm E:** EndoSSL features + the same causal MS-TCN, selected on the official folds by the rule
  used for A and B (`runs/grasp_dev/report.md`): sqrt_inv class weights, 150 epochs.
  Development step F1 **0.271** (B: 0.403, A: 0.349), phase F1 0.448. Final model: 3 seeds on the
  8 train cases.

| ID | Endpoint | Criterion |
|---|---|---|
| S5 | Step macro-F1, E − B (EndoSSL vs DINOv2, both causal MS-TCN), all frames, 5 test cases | LB > 0 |
| Reported | Phase macro-F1 E − B; E − E0 (temporal model on EndoSSL) | report |

Expectation from development: S5 fails; EndoSSL is below both general-domain backbones on this
robotic prostatectomy data.

## ST1–ST4: short-term scene understanding (instrument type and atomic actions per instance)

**Task.** For each annotated instrument instance on the GraSP keyframes (every 35 s; train: 2,324
keyframes, 6,170 instances; test: 1,125 keyframes, 2,861 instances), with its **ground-truth box
given**, predict:
- its instrument type (7 classes);
- its atomic actions (14 classes, multi-label).

This is recognition only, not GraSP's official detection task. Code: `surgscene.grasp_st`.

**Arms:**
- **Frozen-feature heads** (`scripts/grasp_shortterm.py`):
  - Input: the backbone's feature of the padded box crop, the whole-frame feature, and the box
    geometry → MLP with an instrument head and an action head.
  - Variant "prev": adds the same crop one second earlier, a motion cue a single image lacks.
  - Backbones: ResNet-50, DINOv2 ViT-B/14, EndoSSL ViT-L/16.
  - Selection on the official folds (both directions, 3 seeds): epochs and the action threshold
    with the highest held-out action macro-F1. A first grid chose its edge values everywhere
    (lowest threshold, 10–30 of 150 epochs), so it was widened to thresholds 0.05–0.5 and epochs
    2–60. Both reports are kept (`runs/grasp_st_dev/report_grid1.md`, `report.md`).
- **VLM, SurgMLLM-style:** Qwen3-VL-8B-Instruct with the instance's box drawn in red on the frame.
  - The answer is "<instrument>; <action>, <action>" (`scripts/grasp_vlm.py --task instances`).
  - Zero-shot: the prompt lists the names. Fine-tuned: QLoRA as in S3, 1 epoch over all 6,170
    train instances (385 steps) → `runs/grasp_vlm/st_final_ft/adapter` (`adapter_model.safetensors`
    sha256 prefix `ca2723487eb17269`).
  - SurgMLLM itself (InternVL2.5-4B, trained on cholecystectomy) was not tested: no released
    weights were found.

**Development comparison** (train fold1 → evaluate fold2, 4 cases; `runs/grasp_st_dev_compare/report.md`):

| Arm | Instrument F1 | Action F1 | Action mAP |
|---|---|---|---|
| ResNet-50, single / prev | 0.703 / 0.740 | 0.248 / 0.268 | 0.258 / 0.275 |
| **DINOv2, single** / prev | **0.780** / 0.799 | **0.269** / 0.285 | 0.298 / 0.317 |
| EndoSSL, single / prev | 0.649 / 0.632 | 0.241 / 0.250 | 0.248 / 0.261 |
| Qwen3-VL-8B, zero-shot | 0.180 | 0.084 | — |
| Qwen3-VL-8B, QLoRA fine-tuned on fold1 | 0.670 | 0.171 | — |

**Frozen configuration** (`configs/grasp_st_selected.json`):
- reference arm = the best single-frame head on development, **DINOv2 single**;
- heads: `runs/grasp_st_dev/selected.json`, trained on all train instances, 3 seeds, mean
  probabilities;
- VLM: zero-shot tag `st_test_zs`, fine-tuned tag `st_test_ft`.

**Endpoints** (5 test cases; unit = case; paired case-level bootstrap, 2,000 replicates):

| ID | Endpoint | Criterion |
|---|---|---|
| ST1 (primary) | Action macro-F1, fine-tuned VLM − DINOv2 single | LB > 0 |
| ST2 | Instrument macro-F1, fine-tuned VLM − DINOv2 single | LB > 0 |
| ST3 | Action macro-F1, DINOv2 single − ResNet-50 single | LB > 0 |
| ST4 | Action macro-F1, DINOv2 prev − DINOv2 single (motion cue) | LB > 0 |
| Reported | Every arm's instrument F1, action F1 and action mAP; zero-shot VLM; EndoSSL heads; per-case values | report |

**Expectations from development:**
- ST1 and ST2 fail: the fine-tuned VLM is below the DINOv2 head on both tasks.
- ST3 and ST4 are positive but small (+0.02 each), likely inside the CI with 5 cases.

## Run order

1. `grasp_vlm.py predict --task instances --tag st_test_zs --split test --batch 16`
2. `grasp_vlm.py predict --task instances --tag st_test_ft --adapter runs/grasp_vlm/st_final_ft/adapter --split test --batch 16`
3. `eval_grasp_shortterm.py test` and `eval_grasp.py test-s5`, once each. Results are committed as they come out.
