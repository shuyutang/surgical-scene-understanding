# Phase 2–3: keypoints and structured inference (SurgPose test, trajectories 20–33)

Checkpoint `runs/kp_unet_r34_20260925-001134/best.pt`; occluder center offset ≤ 0.6×radius. Hyperparameters selected on dev: `{'conf_min': 0.05, 'beta': 0.3, 'cauchy_c': 5.0}`. Errors in native pixels (1400×986). 95% CIs: trajectory-clustered bootstrap.

Shape models (train GT): PSM1: 5 modes, variance fractions [0.4818, 0.2109, 0.1901, 0.082, 0.0352]; PSM3: 4 modes, variance fractions [0.5043, 0.2816, 0.1633, 0.0508]

## Clean test frames

| Method | Mean err [CI] | Δ vs argmax [CI] | Occluded-kp err [CI] | Δ occluded vs argmax [CI] | Visible-kp err | PCK@5/10/20 | Implausible | ms/instr |
|---|---|---|---|---|---|---|---|---|
| argmax | 13.65 [10.30, 16.66] | n/a | n/a | n/a | n/a | 0.503/0.707/0.813 | 0.146 | 0.00 |
| map | 12.51 [9.88, 14.95] | -1.14 [-1.89, -0.34] | n/a | n/a | n/a | 0.431/0.674/0.809 | 0.015 | 0.50 |
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
| argmax | 15.44 [12.28, 18.37] | n/a | 18.82 [16.93, 20.98] | n/a | 14.60 [11.10, 17.85] | 0.396/0.600/0.792 | 0.207 | 0.00 |
| map | 14.25 [11.69, 16.62] | -1.19 [-1.89, -0.55] | 17.39 [15.57, 19.19] | -1.43 [-2.22, -0.71] | 13.47 [10.70, 16.02] | 0.344/0.585/0.789 | 0.033 | 0.53 |
| discrete | 16.22 [14.05, 18.25] | 0.78 [-0.41, 2.05] | 19.81 [18.01, 21.63] | 0.99 [0.12, 1.76] | 15.33 [13.04, 17.45] | 0.329/0.542/0.727 | 0.007 | 6.91 |

Argmax confidence reliability (fraction of keypoints within 10 px, per confidence bin):

| Confidence | n | Within 10 px |
|---|---|---|
| 0.0–0.1 | 1328 | 0.163 |
| 0.1–0.2 | 3712 | 0.298 |
| 0.2–0.3 | 3857 | 0.410 |
| 0.3–0.4 | 3726 | 0.535 |
| 0.4–0.5 | 3162 | 0.632 |
| 0.5–0.6 | 3301 | 0.721 |
| 0.6–0.7 | 3459 | 0.795 |
| 0.7–0.8 | 3253 | 0.848 |
| 0.8–0.9 | 2058 | 0.867 |
| 0.9–1.0 | 284 | 0.958 |

Per-keypoint median error (px): PSM1.shaft 17.6, PSM1.wrist 7.8, PSM1.jaw_pivot 12.5, PSM1.tip_a 4.5, PSM1.tip_b 3.6, PSM3.shaft 15.0, PSM3.wrist 5.5, PSM3.jaw_pivot 5.9, PSM3.tip_a 5.0, PSM3.tip_b 4.8

