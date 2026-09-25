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
