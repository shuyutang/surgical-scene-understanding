# v3: DINOv2 front end downstream (V1–V4), ViT FP16 engine (E1), fast stereo (B5), 2026-09-28

All three were pre-registered in `f22f390` and run once:
- `scripts/eval_downstream_vit.py --split test2` (test2, 3rd use);
- `scripts/eval_e1_vit.py` (test2);
- `scripts/eval_servct.py --checkpoint realtime --iters 4` (SERV-CT, 2nd use).

## Step 5: DINOv2 keypoints + learned σ through the Phase C pipeline

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| V1 (primary) | 3D tip error, DINOv2 inputs − v2 inputs (paired) | −0.24 [−1.30, 0.65] mm | UB < 0 | FAIL |
| V2 (R1) | Mean 3D tip error, DINOv2 inputs | 5.03 [3.95, 6.29] mm | UB ≤ 5.0 | FAIL |
| V3 | Occlusion episodes, Kalman 2D error on occluded keypoints, DINOv2 − v2 | −4.34 [−6.01, −2.57] px | UB < 0 | PASS |
| V4 (R7) | Coverage of the κ-inflated 95% ellipsoid (κ = 5.09) | 0.967 [0.939, 0.989] | LB ≥ 0.85, point ≤ 0.99 | PASS |

| Trajectory | 21 | 22 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2 inputs (mm) | 5.2 | 5.4 | 4.6 | 4.0 | 3.2 | 6.4 | 9.7 | 8.2 | 2.6 | 4.0 | 2.8 | 4.2 |
| v2 inputs (mm) | 3.6 | 4.2 | 3.8 | 3.7 | 6.3 | 5.7 | 13.0 | 11.6 | 2.3 | 2.7 | 2.5 | 4.0 |

**Reading**
- On average, the new front end doesn't change 3D accuracy. It trades between trajectories:
  - the long-jaw instrument (28, 29) improves by 3–3.5 mm, consistent with occlusion robustness
    and a σ that down-weights bad detections;
  - several standard-instrument trajectories get 1–1.6 mm worse.
- The occlusion gain from A7 survives the Kalman filter (V3).
- The learned σ roughly **halves κ** (10.5 → 5.1) at the same coverage. The uncertainty the
  pipeline reports now needs much less post-hoc inflation.
- R1 remains unmet with either front end: 5.03 vs 5.27 mm.

## E1 (R6): DINOv2 TensorRT FP16 engine

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| E1 | TRT FP16 − PyTorch FP32 mean keypoint error vs GT (test2 left, every 5th frame) | +0.076 [+0.057, +0.095] px | UB ≤ +0.25 | PASS |

- The FP16 engine runs at **4.4 ms per stereo pair**, against 17.6 ms in FP32 and 1.5 ms for the
  v2 U-Net.
- Two FP16 lessons, both caught by parity checks rather than mean metrics:
  1. The decode's argmax-index arithmetic overflowed FP16, making every keypoint NaN. It's fixed
     by pinning the decode to FP32.
  2. Even after pinning the decode, LayerNorm, softmax and reductions, FP16 keypoints sit 0.88 px
     (median) from FP32. The cause is TensorRT-specific (PyTorch FP16: 0.008 px) and unresolved.
     Against GT it costs only +0.08 px, which is why E1 passes.

## B5: fast stereo on SERV-CT

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| B5 | Depth MAE, RAFT-Stereo `realtime` @ 4 iterations − SGM (common pixels) | −15.18 [−23.02, −8.75] mm | UB < 0 | PASS |

(The run's own `report.md` labels this row "B1", because the script's report template is shared
with B1. It's the B5 endpoint.)

| Configuration | Latency (half res, PyTorch mixed precision) | Depth MAE common / all valid |
|---|---|---|
| `middlebury` @ 32 (B1) | 98 ms | 1.58 / 1.86 mm |
| `realtime` @ 4 (B5) | 10.7 ms | 1.88 / 2.33 mm |
| SGM | 19 ms (CPU) | 17.06 mm / 59% coverage |

## R3 latency budget (GPU stages measured separately; not an end-to-end C++ measurement)

| Stage | ms per stereo pair |
|---|---|
| DINOv2 keypoints, TRT FP16 (preprocess + net + decode) | 4.4 |
| RAFT-Stereo `realtime` @ 4, PyTorch (not yet TensorRT) | 10.7 |
| Kalman + fusion + hybrid (C++ estimate from the v2 runtime: MAP + Kalman + triangulation 0.03 ms; the fusion LM is larger but small) | < 1 |
| **Sum** | **about 16**, inside 33 ms |

The fusion isn't ported to C++ and the stereo network isn't in TensorRT. An end-to-end p99
measurement (the v2 D4 equivalent) is still open.
