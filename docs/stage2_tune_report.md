# Stage 2 (Phases 3b-5), split `tune` (trajectories [18, 19, 20, 23])

## Pre-specified criteria

| ID | Endpoint | Value [95% CI] | Rule | Result |
|---|---|---|---|---|
| G1 | gated − argmax mean error, clean (px) | -1.068 [-3.108, 0.052] | UB < 0 | FAIL |
| G2 | gated − argmax PCK@5 | -0.003 [-0.005, -0.001] | LB ≥ −0.02 | PASS |
| G3 | gated − argmax, occluded keypoints (px) | -1.108 [-3.023, 0.696] | UB < 0 | FAIL |
| T1 | jitter ratio KF / gated | 0.359 [0.320, 0.475] | UB ≤ 0.6 | PASS |
| T2 | KF − gated mean error, clean (px) | -0.213 [-0.335, -0.091] | UB ≤ +0.5 | PASS |
| T3 | KF − gated, occlusion episodes (px) | -1.661 [-2.289, -1.181] | UB < 0 | PASS |
| T4 | KF − gated, 15 frames after episodes (px) | 0.013 [-0.364, 0.395] | UB ≤ +1.0 | PASS |
| S1 | 3D tip error, KF pipeline (mm) | 15.240 [9.322, 22.478] | UB ≤ 5.0 | FAIL |
| S2 | 3D tip error KF − frame-wise (mm) | -3.801 [-5.723, -2.014] | UB < 0 | PASS |
| S3 | 95%-ellipsoid coverage | 0.950 [0.903, 0.985] | LB ≥ 0.85 and point ≤ 0.99 | PASS |
| S4a | hand-eye pivot: kinematics − vision (mm) | -4.986 [-8.332, -2.122] | UB < 0 | PASS |
| S4b | hand-eye pivot error, kinematics (mm) | 3.328 [2.372, 4.250] | UB ≤ 5.0 | PASS |
| S5 | alert toggles ratio (KF + hysteresis) / frame-wise | 0.478 [0.365, 0.614] | UB ≤ 0.7 | PASS |

Selected on tune: gate `{'conf_min': 0.05, 'beta': 0.3, 'cauchy_c': 5.0, 'tau': 0.2}`, Kalman `{'q': 2.0, 'r0': 2.0, 'conf_min': 0.05, 'max_rejects': 3, 'unknown_px': 20.0}`, 3D covariance inflation kappa = 6.86. Errors in native pixels (1400x986) or mm; 95% CIs are trajectory-clustered bootstrap.

## Phase 3b: gated shape-prior MAP (left eye, every 5th frame)

| Metric | Value [95% CI] |
|---|---|
| Argmax mean error, clean | 11.52 [3.17, 19.86] |
| Gated mean error, clean | 10.45 [3.22, 17.67] |
| Gated − argmax, clean | -1.07 [-3.11, 0.05] |
| Gated − argmax PCK@5 | -0.003 [-0.005, -0.001] |
| Argmax / gated PCK@5 | 0.610 / 0.607 |
| Occluded keypoints: argmax / gated | 20.05 [13.98, 26.04] / 18.95 [14.25, 23.57] |
| Occluded keypoints: gated − argmax | -1.11 [-3.02, 0.70] |

## Phase 4: temporal filter (left eye, all frames, 30 fps)

| Metric | Value [95% CI] |
|---|---|
| Causal KF mean error, clean | 10.31 [3.13, 17.49] |
| KF − gated, clean | -0.21 [-0.34, -0.09] |
| Jitter (acceleration error, px/frame²): gated / KF | 6.32 [3.17, 9.48] / 2.27 [1.51, 3.03] |
| Jitter ratio KF / gated | 0.359 [0.320, 0.475] |
| Occlusion episodes, occluded keypoints: KF | 17.62 [12.64, 22.52] |
| Occlusion episodes: KF − gated | -1.66 [-2.29, -1.18] |
| 15 frames after an episode: gated / KF | 10.18 [2.97, 17.27] / 10.19 [3.37, 16.91] |
| Flagged 'unknown' (position std > 20 px) | 0.003 [0.000, 0.005] |
| Error when flagged / not flagged | 35.93 [0.00, 39.09] / 10.20 [3.12, 17.34] |
| Mean NIS of accepted updates (consistent filter: 2) | 0.16 |
| KF cost (Python, 10 keypoints) | 0.16 ms/frame |

Fixed-lag smoother (accuracy vs latency):

| Lag (frames / ms) | Mean error | Jitter |
|---|---|---|
| 0 / 0 | 10.31 [3.13, 17.49] | 2.27 [1.51, 3.03] |
| 3 / 99 | 10.01 [2.91, 17.12] | 1.26 [1.06, 1.46] |
| 6 / 198 | 9.98 [2.90, 17.07] | 1.10 [1.02, 1.17] |

## Phase 5: stereo 3D tips (every 5th frame)

| Metric | Value [95% CI] |
|---|---|
| 3D tip error, KF per eye then triangulate (mm) | 15.24 [9.32, 22.48] |
| 3D tip error, frame-wise gated (mm) | 18.79 [11.45, 27.35] |
| KF − frame-wise (paired) | -3.80 [-5.72, -2.01] |
| Depth-direction / lateral component (mm) | 15.11 [9.26, 22.22] / 0.95 [0.48, 1.53] |
| 95%-ellipsoid coverage (kappa-inflated) | 0.950 [0.903, 0.985] |
| GT left/right label consistency: reprojection RMS median / p95 | 0.34 / 1.04 px |
| GT pairs excluded (> 3 px) | 0.0001 |

Hand-eye (robot kinematics → camera), registered from vision estimates on the first half of each trajectory, evaluated on the second half; jaw-pivot keypoint:

| Metric | Value [95% CI] |
|---|---|
| Vision-only 3D pivot error (mm) | 8.31 [4.82, 12.43] |
| Kinematics + vision-fitted registration (mm) | 3.33 [2.37, 4.25] |
| Kinematics − vision | -4.99 [-8.33, -2.12] |

## Phase 5b: tip-to-tissue distance and the proximity alert (all frames; alert < 10 mm)

Tissue = robust local plane from SGM around the GT tip; d_ref uses the GT tip on the same plane, so these numbers isolate the error added by tip perception (SGM's own depth error is not measured: no depth GT).

| Metric | Value [95% CI] |
|---|---|
| Frames with reference alert on | 0.076 [0.028, 0.127] |
| abs(d − d_ref), KF pipeline (mm) | 11.81 [7.03, 18.06] |
| abs(d − d_ref), frame-wise (mm) | 13.92 [8.56, 20.94] |
| Alert sensitivity / specificity, KF | 0.885 [0.578, 0.967] / 0.912 [0.825, 0.982] |
| Alert sensitivity / specificity, frame-wise | 0.879 [0.591, 0.958] / 0.892 [0.804, 0.966] |

Alert toggles per minute (per instrument):

| Source | Toggles/min [95% CI] |
|---|---|
| reference (GT tip) | 91.0 [27.9, 154.2] |
| frame-wise | 228.4 [111.5, 345.3] |
| Kalman | 129.9 [54.4, 184.3] |
| Kalman + 3 mm hysteresis | 109.2 [49.0, 156.4] |

Toggle ratio (Kalman + hysteresis) / frame-wise: 0.478 [0.365, 0.614]

