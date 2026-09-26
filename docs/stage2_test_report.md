# Stage 2 (Phases 3b-5), split `test2` (trajectories [21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])

## Pre-specified criteria

| ID | Endpoint | Value [95% CI] | Rule | Result |
|---|---|---|---|---|
| G1 | gated − argmax mean error, clean (px) | -0.300 [-1.061, 0.199] | UB < 0 | FAIL |
| G2 | gated − argmax PCK@5 | -0.004 [-0.006, -0.003] | LB ≥ −0.02 | PASS |
| G3 | gated − argmax, occluded keypoints (px) | -0.088 [-0.435, 0.294] | UB < 0 | FAIL |
| T1 | jitter ratio KF / gated | 0.323 [0.304, 0.355] | UB ≤ 0.6 | PASS |
| T2 | KF − gated mean error, clean (px) | -0.562 [-0.840, -0.317] | UB ≤ +0.5 | PASS |
| T3 | KF − gated, occlusion episodes (px) | -1.948 [-2.296, -1.615] | UB < 0 | PASS |
| T4 | KF − gated, 15 frames after episodes (px) | -0.020 [-0.202, 0.118] | UB ≤ +1.0 | PASS |
| S1 | 3D tip error, KF pipeline (mm) | 10.046 [7.624, 12.847] | UB ≤ 5.0 | FAIL |
| S2 | 3D tip error KF − frame-wise (mm) | -1.683 [-2.556, -0.951] | UB < 0 | PASS |
| S3 | 95%-ellipsoid coverage | 0.973 [0.956, 0.987] | LB ≥ 0.85 and point ≤ 0.99 | PASS |
| S4a | hand-eye pivot: kinematics − vision (mm) | -7.341 [-11.625, -4.142] | UB < 0 | PASS |
| S4b | hand-eye pivot error, kinematics (mm) | 4.069 [3.377, 4.860] | UB ≤ 5.0 | PASS |
| S5 | alert toggles ratio (KF + hysteresis) / frame-wise | 0.535 [0.377, 0.648] | UB ≤ 0.7 | PASS |

Selected on tune: gate `{'conf_min': 0.05, 'beta': 0.3, 'cauchy_c': 5.0, 'tau': 0.2}`, Kalman `{'q': 2.0, 'r0': 2.0, 'conf_min': 0.05, 'max_rejects': 3, 'unknown_px': 20.0}`, 3D covariance inflation kappa = 6.86. Errors in native pixels (1400x986) or mm; 95% CIs are trajectory-clustered bootstrap.

## Phase 3b: gated shape-prior MAP (left eye, every 5th frame)

| Metric | Value [95% CI] |
|---|---|
| Argmax mean error, clean | 13.23 [9.19, 17.54] |
| Gated mean error, clean | 12.93 [9.02, 17.19] |
| Gated − argmax, clean | -0.30 [-1.06, 0.20] |
| Gated − argmax PCK@5 | -0.004 [-0.006, -0.003] |
| Argmax / gated PCK@5 | 0.507 / 0.503 |
| Occluded keypoints: argmax / gated | 21.07 [17.75, 24.37] / 20.99 [17.82, 24.24] |
| Occluded keypoints: gated − argmax | -0.09 [-0.44, 0.29] |

## Phase 4: temporal filter (left eye, all frames, 30 fps)

| Metric | Value [95% CI] |
|---|---|
| Causal KF mean error, clean | 12.27 [8.59, 16.35] |
| KF − gated, clean | -0.56 [-0.84, -0.32] |
| Jitter (acceleration error, px/frame²): gated / KF | 9.34 [6.77, 11.80] / 3.01 [2.38, 3.60] |
| Jitter ratio KF / gated | 0.323 [0.304, 0.355] |
| Occlusion episodes, occluded keypoints: KF | 19.10 [15.92, 22.49] |
| Occlusion episodes: KF − gated | -1.95 [-2.30, -1.62] |
| 15 frames after an episode: gated / KF | 11.58 [8.15, 15.50] / 11.56 [8.13, 15.46] |
| Flagged 'unknown' (position std > 20 px) | 0.002 [0.000, 0.003] |
| Error when flagged / not flagged | 58.06 [37.94, 97.46] / 12.19 [8.55, 16.26] |
| Mean NIS of accepted updates (consistent filter: 2) | 0.27 |
| KF cost (Python, 10 keypoints) | 0.16 ms/frame |

Fixed-lag smoother (accuracy vs latency):

| Lag (frames / ms) | Mean error | Jitter |
|---|---|---|
| 0 / 0 | 12.27 [8.59, 16.35] | 3.01 [2.38, 3.60] |
| 3 / 99 | 11.85 [8.23, 15.85] | 1.58 [1.32, 1.82] |
| 6 / 198 | 11.81 [8.20, 15.80] | 1.31 [1.16, 1.45] |

## Phase 5: stereo 3D tips (every 5th frame)

| Metric | Value [95% CI] |
|---|---|
| 3D tip error, KF per eye then triangulate (mm) | 10.05 [7.62, 12.85] |
| 3D tip error, frame-wise gated (mm) | 11.80 [8.84, 15.41] |
| KF − frame-wise (paired) | -1.68 [-2.56, -0.95] |
| Depth-direction / lateral component (mm) | 9.98 [7.58, 12.76] / 0.62 [0.45, 0.83] |
| 95%-ellipsoid coverage (kappa-inflated) | 0.973 [0.956, 0.987] |
| GT left/right label consistency: reprojection RMS median / p95 | 0.46 / 1.23 px |
| GT pairs excluded (> 3 px) | 0.0026 |

Hand-eye (robot kinematics → camera), registered from vision estimates on the first half of each trajectory, evaluated on the second half; jaw-pivot keypoint:

| Metric | Value [95% CI] |
|---|---|
| Vision-only 3D pivot error (mm) | 11.41 [7.96, 15.96] |
| Kinematics + vision-fitted registration (mm) | 4.07 [3.38, 4.86] |
| Kinematics − vision | -7.34 [-11.63, -4.14] |

## Phase 5b: tip-to-tissue distance and the proximity alert (all frames; alert < 10 mm)

Tissue = robust local plane from SGM around the GT tip; d_ref uses the GT tip on the same plane, so these numbers isolate the error added by tip perception (SGM's own depth error is not measured: no depth GT).

| Metric | Value [95% CI] |
|---|---|
| Frames with reference alert on | 0.062 [0.029, 0.098] |
| abs(d − d_ref), KF pipeline (mm) | 8.21 [6.16, 10.48] |
| abs(d − d_ref), frame-wise (mm) | 10.11 [7.58, 12.89] |
| Alert sensitivity / specificity, KF | 0.927 [0.839, 0.965] / 0.969 [0.949, 0.986] |
| Alert sensitivity / specificity, frame-wise | 0.911 [0.825, 0.955] / 0.963 [0.943, 0.980] |

Alert toggles per minute (per instrument):

| Source | Toggles/min [95% CI] |
|---|---|
| reference (GT tip) | 61.9 [22.2, 106.5] |
| frame-wise | 127.1 [88.3, 173.5] |
| Kalman | 78.0 [40.0, 122.7] |
| Kalman + 3 mm hysteresis | 68.0 [35.2, 106.4] |

Toggle ratio (Kalman + hysteresis) / frame-wise: 0.535 [0.377, 0.648]

