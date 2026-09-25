# Phase 2–3: keypoints and structured inference (SurgPose test, trajectories 20–33)

Hyperparameters selected on dev: `{'conf_min': 0.05, 'beta': 0.3, 'cauchy_c': 5.0}`. Errors in native pixels (1400×986). 95% CIs: trajectory-clustered bootstrap.

Shape models (train GT): PSM1: 5 modes, variance fractions [0.4818, 0.2109, 0.1901, 0.082, 0.0352]; PSM3: 4 modes, variance fractions [0.5043, 0.2816, 0.1633, 0.0508]

## Clean test frames

| Method | Mean err [CI] | Δ vs argmax [CI] | Occluded-kp err [CI] | Δ occluded vs argmax [CI] | Visible-kp err | PCK@5/10/20 | Implausible | ms/instr |
|---|---|---|---|---|---|---|---|---|
| argmax | 13.65 [10.30, 16.66] | n/a | n/a | n/a | n/a | 0.503/0.707/0.813 | 0.146 | 0.00 |
| map | 12.51 [9.88, 14.95] | -1.14 [-1.89, -0.34] | n/a | n/a | n/a | 0.431/0.674/0.809 | 0.015 | 0.49 |
| discrete | 14.41 [11.97, 16.68] | 0.76 [-0.31, 2.05] | n/a | n/a | n/a | 0.418/0.637/0.755 | 0.002 | 6.92 |

Argmax confidence reliability (fraction of keypoints within 10 px, per confidence bin):

| Confidence | n | Within 10 px |
|---|---|---|
| 0.0–0.1 | 1474 | 0.187 |
| 0.1–0.2 | 3548 | 0.345 |
| 0.2–0.3 | 3341 | 0.499 |
| 0.3–0.4 | 3256 | 0.655 |
| 0.4–0.5 | 2854 | 0.778 |
| 0.5–0.6 | 3254 | 0.853 |
| 0.6–0.7 | 3647 | 0.906 |
| 0.7–0.8 | 3782 | 0.916 |
| 0.8–0.9 | 2574 | 0.918 |
| 0.9–1.0 | 410 | 0.998 |

Per-keypoint median error (px): PSM1.shaft 18.6, PSM1.wrist 6.5, PSM1.jaw_pivot 10.4, PSM1.tip_a 3.4, PSM1.tip_b 2.8, PSM3.shaft 15.2, PSM3.wrist 4.2, PSM3.jaw_pivot 4.3, PSM3.tip_a 3.8, PSM3.tip_b 3.7

## Occluded test frames

| Method | Mean err [CI] | Δ vs argmax [CI] | Occluded-kp err [CI] | Δ occluded vs argmax [CI] | Visible-kp err | PCK@5/10/20 | Implausible | ms/instr |
|---|---|---|---|---|---|---|---|---|
| argmax | 13.91 [10.61, 17.05] | n/a | 10.88 [8.65, 13.31] | n/a | 14.66 [11.06, 17.97] | 0.513/0.716/0.815 | 0.180 | 0.00 |
| map | 12.79 [10.28, 15.15] | -1.12 [-1.89, -0.35] | 11.53 [9.54, 13.65] | 0.66 [0.20, 1.08] | 13.10 [10.41, 15.60] | 0.431/0.678/0.811 | 0.026 | 0.50 |
| discrete | 14.58 [12.48, 16.56] | 0.67 [-0.66, 2.10] | 13.85 [12.37, 15.28] | 2.97 [1.74, 4.21] | 14.76 [12.49, 16.87] | 0.420/0.639/0.754 | 0.005 | 6.92 |

Argmax confidence reliability (fraction of keypoints within 10 px, per confidence bin):

| Confidence | n | Within 10 px |
|---|---|---|
| 0.0–0.1 | 1302 | 0.195 |
| 0.1–0.2 | 3508 | 0.353 |
| 0.2–0.3 | 3655 | 0.519 |
| 0.3–0.4 | 3612 | 0.672 |
| 0.4–0.5 | 3061 | 0.806 |
| 0.5–0.6 | 3524 | 0.869 |
| 0.6–0.7 | 3595 | 0.919 |
| 0.7–0.8 | 3349 | 0.925 |
| 0.8–0.9 | 2235 | 0.920 |
| 0.9–1.0 | 299 | 0.997 |

Per-keypoint median error (px): PSM1.shaft 15.1, PSM1.wrist 5.2, PSM1.jaw_pivot 6.1, PSM1.tip_a 3.7, PSM1.tip_b 3.2, PSM3.shaft 12.2, PSM3.wrist 3.9, PSM3.jaw_pivot 4.3, PSM3.tip_a 4.1, PSM3.tip_b 4.0

