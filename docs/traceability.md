# Traceability: requirements → tests → results (v2 + v3), 2026-09-28

Every row is a pre-specified endpoint from [validation_plan.md](validation_plan.md) unless it's
marked **post hoc**. Unit of analysis: patient (Phase 1), trajectory (SurgPose), or stereo pair
(SERV-CT). CIs are 95% clustered bootstrap. The "Data" column shows reuse: SurgPose test2 was
used by v2 stage 2 and again by v3 (disclosed in each pre-registration). SERV-CT was used once.

## Requirements (clinical use case: instrument-to-tissue proximity awareness)

| ID | Requirement | Status |
|---|---|---|
| R1 | Instrument tip localized in 3D within 5 mm | **Not met.** v2 10.0 → v3 5.27 mm (upper bound 7.4). Met on standard instruments (3.86 mm, pre-specified subgroup), not on the long-jaw tool |
| R2 | Critical-structure segmentation adequate for proximity | **Open.** Segmentation passes on SISVSE, but no dataset has anatomy and stereo together |
| R3 | End-to-end p99 < 33 ms at 30 fps | **Met without stereo** (v2 D4 2.59 ms). With the v3 front end and fast stereo, the measured GPU stages sum to about 16 ms (4.4 + 10.7), but there's no end-to-end C++ measurement yet |
| R4 | Stable output, no alert flicker | **Met** (S5: toggles −46%) |
| R5 | Known behaviour under occlusion | **Met for synthetic occluders** (T3, A7). Real smoke and blood aren't labelled in SurgPose |
| R6 | FP16 deployment doesn't degrade accuracy | **Met** for the v2 U-Net (D2) and the DINOv2 ViT (E1: +0.076 px) |
| R7 | Reported uncertainty is calibrated | **Met after one tune-fitted scalar**, every time (S3, C3, A8). Raw uncertainties under-cover |

## R1: 3D tip accuracy

| Test | Data | Result | Verdict |
|---|---|---|---|
| S1: v2 Kalman + triangulation | test2 (1st use) | 10.0 [7.6, 12.8] mm | FAIL |
| S2: Kalman − frame-wise | test2 (1st) | −1.68 mm | PASS |
| S4b: kinematics + hand-eye (offline), jaw pivot | test2 (1st) | 4.07 mm | PASS |
| **C1: v3 kinematics-fused hybrid** | test2 (2nd) | 5.27 [3.56, 7.43] mm | **FAIL (narrowly)** |
| C2: hybrid − v2 | test2 (2nd) | −4.78 [−6.93, −2.71] mm | PASS |
| C4: fused pivot, causal hand-eye | test2 (2nd) | 3.38 mm | PASS |
| C6: 2D non-inferiority | test2 (2nd) | −0.33 px | PASS |
| C subgroup: standard / long-jaw | test2 (2nd) | 3.86 / 12.31 mm | report |
| **Post hoc:** 3D tip offset (`tip_mode 3d`) | test2 (2nd) | 4.17 [2.97, 5.84] mm | hypothesis only |
| V1: DINOv2 inputs − v2 inputs through Phase C | test2 (3rd) | −0.24 [−1.30, 0.65] mm | FAIL |
| V2: mean 3D tip error, DINOv2 inputs | test2 (3rd) | 5.03 [3.95, 6.29] mm | FAIL |

## 2D keypoints and structured inference (support R1, R5)

| Test | Data | Result | Verdict |
|---|---|---|---|
| K1: argmax mean error ≤ 15 px (background shift) | Phase 2–3 test | 14.0 [10.4, 17.4] px | FAIL |
| H1: shape prior on occluded keypoints | Phase 2–3 test | failed as run (occluder shortcut); post-hoc fix −1.47 px | FAIL, post-hoc hold |
| H2, H3: prior harmless / plausible shapes | Phase 2–3 test | +0.01 px; implausible 17% → 1.4% | PASS |
| G1, G3: gated MAP gain | test2 (1st) | −0.30, −0.09 px (CIs cross 0) | FAIL |
| G2: gate keeps precision | test2 (1st) | PCK@5 −0.004 | PASS |
| A1, A2: DINOv2 accuracy gain | test2 | PCK@10 −0.049; mean +0.2 px | FAIL |
| A0: focal-loss positives fix | tune | within seed noise | not adopted |

## R5: occlusion

| Test | Data | Result | Verdict |
|---|---|---|---|
| T3: Kalman during occlusion episodes | test2 (1st) | −1.95 px | PASS |
| T4: recovery after episodes | test2 (1st) | −0.02 px | PASS |
| A7: DINOv2 on occluded keypoints | test2 | −5.09 [−6.77, −3.41] px | PASS |
| V3: DINOv2 occlusion episodes, through the Kalman filter | test2 (3rd) | −4.34 [−6.01, −2.57] px | PASS |
| C5: tip swaps (reported) | test2 (2nd) | 2.1% → 0.1% | report |

## R7: uncertainty

| Test | Data | Result | Verdict |
|---|---|---|---|
| Phase 1 S3: pixel ECE after temperature scaling | SISVSE test | 0.021 | PASS |
| S3: v2 triangulation, κ = 6.9 | test2 (1st) | 0.973 | PASS |
| C3: v3 hybrid, κ = 10.5 | test2 (2nd) | 0.971 | PASS |
| A5: learned σ ranks errors | test2 | ρ +0.53 vs −0.25 (Laplace σ) | PASS |
| A8: learned σ × k (tune) | test2 | 0.952 | PASS |
| V4: hybrid with learned σ, κ = 5.09 (half of C3's) | test2 (3rd) | 0.967 | PASS |

## Tissue depth and proximity (R1 → proximity, R4)

| Test | Data | Result | Verdict |
|---|---|---|---|
| B1: learned stereo − SGM, depth MAE | SERV-CT (fresh) | −15.5 [−23.5, −9.1] mm (17.1 → 1.6) | PASS |
| B5: fast stereo (`realtime` @ 4, 10.7 ms) − SGM | SERV-CT (2nd) | −15.2 [−23.0, −8.8] mm (→ 1.9) | PASS |
| B3: rectification row offset | train + tune | 18/22 trajectories off 1–2 px; shift rule | rule fixed |
| S5: alert toggles, Kalman + hysteresis / frame-wise | test2 (1st) | 0.54 | PASS |
| Distance error vs same-plane reference (reported) | test2 (1st) | 8.2 mm (Kalman) | report |

## R2: segmentation (Phase 1, SISVSE internal, EndoVis18 external)

| Test | Result | Verdict |
|---|---|---|
| P1: instrument Dice | 0.928 [0.909, 0.941] | PASS |
| P2: tip Dice | 0.829 [0.794, 0.856] | PASS |
| S1: tip boundary F (3 px) | 0.683 | PASS |
| S2: liver / stomach Dice | 0.850 / 0.814 | PASS |
| S4: zero-shot instrument Dice (EndoVis18) | 0.778 [0.748, 0.810] | PASS |
| Transfer to SurgPose (not pre-specified) | labels beef as instrument | not usable downstream |

## R3, R6: deployment

| Test | Result | Verdict |
|---|---|---|
| D1: TensorRT FP32 vs PyTorch | p99 0.003 px | PASS |
| D2: FP16 non-inferiority | +0.008 px | PASS |
| D3: C++ vs Python parity | 10/10 | PASS |
| D4: C++ end-to-end p99 (no stereo) | 2.59 ms | PASS |
| E1: DINOv2 TRT FP16 non-inferiority vs GT | +0.076 [+0.057, +0.095] px | PASS |
| DINOv2 FP16 engine latency (tune, development) | 4.4 ms per pair | report |
| E2–E4: C++ fusion parity, end-to-end with stereo, embedded proxy | — | not yet run |

## Open items, by what would change a verdict

1. **R1 on long-jaw instruments:** the kinematic tip precision at 24 mm and keypoint training
   coverage (one of 18 training trajectories). The post-hoc 3D tip offset needs fresh data to confirm.
2. **R3 end to end:** a TensorRT stereo engine, the C++ fusion port, and one measured p99 for the whole v3 pipeline.
3. **R2 end to end:** needs data with anatomy labels and stereo in the same frames.
4. **Fresh SurgPose-like data** for any further confirmatory claim: test2 has been used twice.
