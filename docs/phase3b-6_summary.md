# Phases 3b–6: results and lessons (2026-09-25)

Pre-specified in [validation_plan.md](validation_plan.md) (stage 2, commit `d7bfdb5`) before any
test2 or deployment number was computed. Reports: [stage-2 test](stage2_test_report.md),
[stage-2 tune](stage2_tune_report.md) (in-sample, for reference), [deployment](phase6_deploy_report.md).

## What was built

| Phase | Component | Code |
|---|---|---|
| 3b | Gated MAP: argmax where the heatmap is confident, shape-prior MAP elsewhere | `structured.fit_gated` |
| 4 | Constant-velocity Kalman filter per keypoint; noise from heatmap σ and confidence, χ² gate, re-init, "unknown" flag, fixed-lag RTS smoother | `temporal.py` |
| 5 | Camera model with distortion, DLT + Gauss–Newton triangulation with covariance, SO(3) utilities, robust rigid registration | `geometry.py` |
| 5 | Hand-eye: robot kinematics → camera, registered from the pipeline's own vision estimates | `eval_stage2.py` |
| 5b | SGM tissue depth, disparity-space robust tissue plane, tip-to-tissue distance, proximity alert with hysteresis | `proximity.py` |
| 6 | Deploy graph (preprocess + net + decode) → ONNX → TensorRT FP32/FP16; C++ runtime (TensorRT, Eigen MAP, Kalman, triangulation); GoogleTest parity tests; latency bench | `deploy.py`, `cpp/` |

## Pre-specified results (test2: 12 trajectories, never used for tuning)

| ID | Endpoint | Result [95% CI] | Verdict |
|---|---|---|---|
| G1 | Gated − argmax error, clean | −0.30 [−1.06, +0.20] px | FAIL |
| G2 | Gated − argmax PCK@5 | −0.004 [−0.006, −0.003] | PASS |
| G3 | Gated − argmax, occluded keypoints | −0.09 [−0.44, +0.29] px | FAIL |
| T1 | Jitter ratio, Kalman / gated | 0.32 [0.30, 0.36] | PASS |
| T2 | Kalman − gated error, clean | −0.56 [−0.84, −0.32] px | PASS |
| T3 | Kalman − gated, during occlusion episodes | −1.95 [−2.30, −1.62] px | PASS |
| T4 | Kalman − gated, 0.5 s after episodes | −0.02 [−0.20, +0.12] px | PASS |
| S1 | 3D tip error (R1 requirement ≤ 5 mm) | 10.0 [7.6, 12.8] mm | FAIL |
| S2 | 3D tip error, Kalman − frame-wise | −1.68 [−2.56, −0.95] mm | PASS |
| S3 | 95%-ellipsoid coverage | 0.973 [0.956, 0.987] | PASS |
| S4a | Jaw pivot, kinematics + registration − vision | −7.3 [−11.6, −4.1] mm | PASS |
| S4b | Jaw pivot, kinematics + registration | 4.07 [3.38, 4.86] mm | PASS |
| S5 | Alert toggles, (Kalman + hysteresis) / frame-wise | 0.54 [0.38, 0.65] | PASS |
| D1 | TensorRT FP32 vs PyTorch, p99 keypoint difference | 0.003 px | PASS |
| D2 | TensorRT FP16 − PyTorch FP32 error (R6) | +0.008 [−0.013, +0.036] px | PASS |
| D3 | C++ vs Python parity | 10/10 tests | PASS |
| D4 | C++ end-to-end latency per stereo pair (R3) | p99 2.59 ms | PASS |

14 of 17 pass. The three failures are reported as they came out.

## Lessons

1. **The tuning set has to contain the shift you deploy into.** Tuned on dev (training
   backgrounds), every structured-inference knob said "turn it off". Moving one beef and one
   chicken trajectory into the tuning set was the fix. This is the practical version of "your
   validation set must match deployment".
2. **Check the premise before building the fix.** The planned shaft-line constraint assumed
   along-axis error. The error was actually perpendicular (median 46 px), because the label sits
   beside the visible shaft on some trajectories. That's a labeling-convention difference, and no
   model change fixes it.
3. **The gate's gain didn't transfer.** On tune it saved 1.1 px; on test2 only 0.3 px, with a CI
   that crosses zero. Mean error is dominated by a few trajectories with label-convention offsets,
   which neither argmax nor MAP can fix. The gate does keep fine precision (G2), so it costs
   nothing, but the pre-specified claim of improvement fails.
4. **The temporal filter gives the most per line of code.** Jitter falls 68%, occluded keypoints
   improve by 1.9 px, 3D error by 1.7 mm, and alert flicker is roughly halved (with hysteresis).
   All of it at 0.16 ms/frame in Python and about 1 µs in C++. A 3-frame (100 ms) fixed-lag
   smoother halves jitter again: a latency-vs-stability product decision, quantified.
5. **Stereo physics sets the 3D ceiling.** Tissue sits 150–230 mm from a 5.5 mm baseline, so 1 px of
   disparity error means 2–5 mm of depth error. The 3D tip error is 10.0 mm along the viewing ray
   but only 0.6 mm laterally. No network improvement closes a 5 mm requirement at this geometry.
   Epipolar patch matching was tried on tune and didn't help robustly, so it wasn't adopted.
6. **Robot kinematics are the right 3D source; vision calibrates them.** The dVRK tool frame
   tracks the jaw pivot. A rigid kinematics→camera registration fitted from the pipeline's own
   noisy vision estimates (robust IRLS over half a trajectory) predicts the second half to
   4.1 mm, versus 11.4 mm from vision alone. Averaging over hundreds of frames cancels the depth
   noise that dominates any single frame.
7. **Uncertainty needed calibration, like the segmentation softmax did.** The raw triangulation
   covariance is about 2.6× too narrow in standard deviation (κ = 6.9 in variance, fitted on
   tune). After that one scalar, test coverage of the 95% ellipsoid is 97%.
8. **The references are noisy too.** GT stereo labels agree to 0.46 px (median), about 1–2 mm of
   depth. The reference alert itself toggles 62 times a minute at a 10 mm threshold. So S5 was
   written as a relative reduction, not an absolute toggle rate.
9. **Robustness bugs found by the evaluation, not by unit tests:** a near-zero-disparity
   triangulation landed 67 km away and dominated a mean. It's fixed with a depth-range guard
   (20–2000 mm) in both Python and C++. A plane fit on 3D points turned out ill-conditioned at this
   geometry, and fitting in disparity space fixed it (with a unit test).
10. **Deployment:** fusing preprocessing and decoding into the engine made the C++ side tiny and
    removed the need for custom kernels. FP16 is non-inferior on the mean (D2), but 0.4% of
    keypoints flip to a different near-tied peak, which per-frame parity tests catch and mean
    metrics hide.

## Deviations from the v2 plan

- The Kalman filter state is per-keypoint image position/velocity, not shape parameters. The
  gated output mixes argmax and MAP points, and the "unknown" flag and the alert are per keypoint.
- There is no quantitative tissue-depth validation (SERV-CT not used). SurgPose has no depth GT, so
  proximity numbers isolate the error that tip perception adds on a shared SGM surface.
- The Phase 1 segmentation model isn't used in stage 2: zero-shot on SurgPose it labels beef as
  instrument. The instrument mask for depth comes from keypoints.
- The robot-frame chain uses real dVRK kinematics instead of a simulated hand-eye transform.
- SGM is not in the C++ runtime (it runs about 18 ms on the CPU in Python). No clock-locked
  embedded-proxy run (needs root).

## Next steps

1. Kinematics-vision fusion for the jaw *tips*: kinematic pivot plus vision-estimated jaw
   direction/opening, instead of the constant tool-frame offset that failed on tune.
2. Target-domain fine-tuning or a stronger backbone for the label-convention trajectories (the
   remaining 2D error).
3. SGM or a small learned stereo network on GPU (or VPI) inside the C++ runtime, then a real
   latency budget with depth included.
4. A traceability matrix (R1–R6 → test → result) and a failure-mode catalog with example frames.
