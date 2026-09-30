# v5 step 2: Fast-FoundationStereo vs RAFT-Stereo, 2026-09-29

**Question.** RAFT-Stereo (2021) is this project's learned stereo: 1.86 mm depth error on
SERV-CT tissue (B1), but 12–17 mm on thin instrument jaws (v5 step 0). Fast-FoundationStereo
(NVIDIA, 2026) distils FoundationStereo, which adds a monocular foundation-model prior (Depth
Anything V2 features) to the cost volume, into a real-time network. Is it better on tissue, and
does the monocular prior fix the thin-jaw failure?

**Answer: better on both, and 2× faster. On tissue it passes the pre-registered test; on jaws it
more than halves RAFT's error, but kinematic fusion is still the better tip-depth source.**

## Setup

- **Model:** Fast-FoundationStereo, research checkpoints (NVIDIA research-only license), zero-shot,
  fp16, `max_disp` 192. Wrapper: `surgscene.learned_stereo.FastFoundationStereo`; both models are
  built from one name by `make_stereo` (`"middlebury@32"`, `"ffs:23-36-37@8"`).
- **Checkpoint selection** on SurgPose tune, by the B1 rule (lowest photometric error on tissue
  around the tips; `runs/v5_ffs_tune/report.md`):

  | Model (32 tune frames, half resolution) | Photometric error | Plane scatter (mm) |
  |---|---|---|
  | SGM | 12.56 | 5.42 |
  | RAFT `middlebury`@32 | 12.97 | 1.54 |
  | RAFT `realtime`@4 | 13.11 | 1.66 |
  | **FFS `23-36-37`@8 (selected)** | **12.95** | 1.29 |
  | FFS `23-36-37`@4 | 12.96 | 1.30 |
  | FFS `20-26-39`@8 | 12.98 | 1.15 |
  | FFS `20-30-48`@4 | 12.96 | 1.30 |

  The proxy barely separates the learned models (< 0.2 gray levels). Plane scatter, the
  secondary proxy, favours every FFS checkpoint over RAFT.

## 1. Tissue depth on SERV-CT (B6, pre-registered, 3rd use of SERV-CT)

Pre-registration: `docs/validation_plan.md` (commit `d6fa43e`). 16 pairs, 2 porcine specimens, CT
reference, all valid pixels, paired bootstrap over pairs. Output: `runs/v5_servct_ffs_23-36-37_it8/`.

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| B6 | Depth MAE, FFS `23-36-37`@8 − RAFT `middlebury`@32, mm | **−0.49 [−0.95, −0.09]** | UB < 0 | **PASS** |

| Reported | FFS | RAFT `middlebury`@32 |
|---|---|---|
| Depth MAE, all valid pixels (mm) | **1.37** [1.06, 1.70] | 1.86 [1.25, 2.55] (reproduces B1 exactly) |
| Experiment 1 / Experiment 2 (mm) | 1.43 / 1.30 | 1.73 / 1.99 |
| Latency, one 720×576 pair, PyTorch fp16 (ms) | **47** | 104 |

- Bad-pixel rate (> 3 px) difference −0.026 [−0.053, 0.001]; disparity EPE difference
  −0.20 [−0.42, −0.00] px.
- FFS − SGM on SGM's valid pixels (the B1 protocol): −15.84 [−23.71, −9.43] mm.
- The gain holds on both specimens, so it is not one specimen's result.

## 2. Instrument jaws on SurgPose tune (development; repeats v5 step 0)

Same estimator as step 0 (SAM 2 jaw pixels, affine disparity fit along the jaw, evaluated at the
tip midpoint), every 10th frame, paired. Mean 3D tip-midpoint error, mm
(`runs/v5_ffs_tip/report.md`):

| Arm-trajectory | v3 kinematic | v4 hybrid | Stereo jaw, RAFT `middlebury`@32 | Stereo jaw, FFS |
|---|---|---|---|---|
| 18/PSM1 | 6.80 | 6.71 | 20.66 | 7.34 |
| 18/PSM3 | 14.06 | 6.69 | 19.96 | 6.96 |
| 19/PSM1 | 4.73 | 4.36 | 21.83 | 9.29 |
| 19/PSM3 | 9.18 | 6.94 | 19.19 | 7.21 |
| 20/PSM1 | 8.32 | 8.16 | 42.35 | 16.90 |
| 20/PSM3 | 5.27 | 5.76 | 87.55 | 43.67 |
| 23/PSM1 | 3.17 | 3.12 | 19.05 | 4.07 |
| 23/PSM3 | 2.23 | 2.23 | 1.99 | **1.53** |
| **Mean** | 6.72 | **5.50** | 29.07 | 12.12 |

- FFS removes most of RAFT's thin-structure bias: jaw error drops from ~20 mm to 7–9 mm on six
  of the eight arm-trajectories, and it beats v3 kinematics on 18/PSM3 and 19/PSM3 (the long-jaw
  arms kinematics gets wrong).
- Trajectory 20 still fails (17 and 44 mm); step 0 found the same trajectory hardest for RAFT.
- The v4 hybrid (kinematics + jaw-length constraint) is still better on average.

## 3. Upper bound: disparity read at the GT pixel locations

As in step 0: disparity at the projections of the GT pivot and GT tips (3 × 3 median), which
removes pixel selection and extrapolation. Absolute depth error, mm (median / mean;
`v5_dev_stereo_tip.py --oracle --models ffs:23-36-37@8 --out runs/v5_ffs_tip`):

| Source | Pivot | Tips |
|---|---|---|
| RAFT `middlebury`@32, half res (step 0) | 2.9 / 5.8 | 12.9 / 19.5 |
| FFS `23-36-37`@8, half res | 2.3 / 3.3 | **4.0** / 10.3 |
| FFS `23-36-37`@8, full res | 2.3 / 4.5 | 5.6 / 15.2 |
| v3 kinematic fusion (along the ray) | **2.0 / 2.7** | 4.8 / **6.3** |

- **The monocular prior largely fixes the thin-jaw bias.** The median tip error drops from 12.9 to
  4.0 mm, below kinematics (4.8 mm). The comparison isn't strictly paired: stereo is scored per
  tip (721), kinematics per tip midpoint (365), as in step 0.
- **The tail remains.** The mean (10.3 mm) is well above kinematics (6.3 mm): some frames still
  bleed tissue disparity into the jaw. Full resolution (1280 px wide) is worse, not better,
  consistent with the readme's note that the model works best below 1000 px width.

## Reading

- **For tissue, Fast-FoundationStereo replaces RAFT-Stereo:** more accurate on CT ground truth
  (pre-registered), 2× faster, and dense. Its license is research-only; the commercial checkpoint
  (NVIDIA Open Model Agreement) was not tested.
- **For instruments, stereo is now a competitive measurement but not a replacement.** Its median
  beats kinematics and its failures are gross, which is the case an innovation-gated fusion
  handles well. The next step would be to feed the jaw-stereo depth into the v3/v4 filter as a
  measurement, gated against the kinematic prediction, and test it on fresh data (test2 has been
  used four times).
