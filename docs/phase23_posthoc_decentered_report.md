# Phase 2–3: keypoints and structured inference (SurgPose test, trajectories 20–33)

Checkpoint `runs/kp_unet_r34_decentered_20260925-004241/best.pt`; occluder center offset ≤ 0.6×radius. Hyperparameters selected on dev: `{'conf_min': 0.05, 'beta': 0.3, 'cauchy_c': 5.0}`. Errors in native pixels (1400×986). 95% CIs: trajectory-clustered bootstrap.

Shape models (train GT): PSM1: 5 modes, variance fractions [0.4818, 0.2109, 0.1901, 0.082, 0.0352]; PSM3: 4 modes, variance fractions [0.5043, 0.2816, 0.1633, 0.0508]

## Clean test frames

| Method | Mean err [CI] | Δ vs argmax [CI] | Occluded-kp err [CI] | Δ occluded vs argmax [CI] | Visible-kp err | PCK@5/10/20 | Implausible | ms/instr |
|---|---|---|---|---|---|---|---|---|
| argmax | 14.01 [10.37, 17.40] | n/a | n/a | n/a | n/a | 0.485/0.693/0.801 | 0.154 | 0.00 |
| map | 14.01 [10.91, 17.14] | 0.01 [-0.84, 0.75] | n/a | n/a | n/a | 0.418/0.653/0.793 | 0.011 | 0.46 |
| discrete | 15.77 [12.88, 18.50] | 1.76 [0.80, 2.76] | n/a | n/a | n/a | 0.400/0.609/0.734 | 0.005 | 6.84 |

Argmax confidence reliability (fraction of keypoints within 10 px, per confidence bin):

| Confidence | n | Within 10 px |
|---|---|---|
| 0.0–0.1 | 1255 | 0.129 |
| 0.1–0.2 | 3719 | 0.308 |
| 0.2–0.3 | 3554 | 0.529 |
| 0.3–0.4 | 3733 | 0.686 |
| 0.4–0.5 | 3233 | 0.764 |
| 0.5–0.6 | 3579 | 0.853 |
| 0.6–0.7 | 3854 | 0.909 |
| 0.7–0.8 | 3450 | 0.901 |
| 0.8–0.9 | 1631 | 0.887 |
| 0.9–1.0 | 132 | 0.985 |

Per-keypoint median error (px): PSM1.shaft 19.5, PSM1.wrist 6.9, PSM1.jaw_pivot 5.3, PSM1.tip_a 3.6, PSM1.tip_b 3.1, PSM3.shaft 15.4, PSM3.wrist 5.2, PSM3.jaw_pivot 4.2, PSM3.tip_a 3.8, PSM3.tip_b 3.9

## Occluded test frames

| Method | Mean err [CI] | Δ vs argmax [CI] | Occluded-kp err [CI] | Δ occluded vs argmax [CI] | Visible-kp err | PCK@5/10/20 | Implausible | ms/instr |
|---|---|---|---|---|---|---|---|---|
| argmax | 16.22 [12.77, 19.45] | n/a | 23.00 [20.10, 25.81] | n/a | 14.52 [10.87, 17.96] | 0.388/0.598/0.751 | 0.172 | 0.00 |
| map | 15.78 [12.72, 18.82] | -0.44 [-1.02, 0.07] | 21.53 [18.77, 24.23] | -1.47 [-2.09, -0.83] | 14.34 [11.19, 17.50] | 0.348/0.574/0.747 | 0.014 | 0.47 |
| discrete | 17.48 [14.80, 20.14] | 1.26 [0.41, 2.23] | 21.49 [19.30, 23.67] | -1.51 [-2.62, -0.11] | 16.48 [13.55, 19.27] | 0.326/0.526/0.690 | 0.003 | 6.79 |

Argmax confidence reliability (fraction of keypoints within 10 px, per confidence bin):

| Confidence | n | Within 10 px |
|---|---|---|
| 0.0–0.1 | 1344 | 0.121 |
| 0.1–0.2 | 5197 | 0.275 |
| 0.2–0.3 | 4727 | 0.447 |
| 0.3–0.4 | 3951 | 0.604 |
| 0.4–0.5 | 3013 | 0.717 |
| 0.5–0.6 | 3049 | 0.811 |
| 0.6–0.7 | 3014 | 0.887 |
| 0.7–0.8 | 2563 | 0.880 |
| 0.8–0.9 | 1191 | 0.875 |
| 0.9–1.0 | 91 | 0.989 |

Per-keypoint median error (px): PSM1.shaft 20.4, PSM1.wrist 8.4, PSM1.jaw_pivot 7.2, PSM1.tip_a 4.8, PSM1.tip_b 4.0, PSM3.shaft 15.8, PSM3.wrist 6.2, PSM3.jaw_pivot 5.7, PSM3.tip_a 5.0, PSM3.tip_b 5.3

