# v3 progress (exploratory, train + tune data only)

Plan: [plans/plan_v3.md](plans/plan_v3.md).
Nothing here touches test2. These are development results, used to fix design rules before the
v3 endpoints are pre-registered.

## Step 2 (B3): residual vertical misalignment after rectification — 2026-09-27

Script: `scripts/check_rectification.py`. Output: `runs/rectification/{check.json, report.md}`.
Trajectories: 0–20 and 23 (train + tune).

**The question.** Stereo rectification should put each 3D point on the same image row in both
eyes; SGM searches along that row only. How far off is it with the SurgPose calibrations?

**Two independent measurements** of dy = y_left − y_right (rectified, half-res px):
- labels: the GT keypoint pairs, rectified with each eye's (R, P);
- images: SIFT matches between the rectified frames (8 frames per trajectory). This needs no
  labels, so it can run at deployment.

**Findings**

1. **Most calibrations leave a 1–2 px row offset.** Image-based dy is +1.0 to +1.9 px on 18 of
   22 trajectories. It's close to zero only on 17, 18, 19 and 23. Within a trajectory it's
   stable: the per-frame medians span about 0.3 px.
2. **On tissue it's a constant shift, not a rotation.** A robust fit of
   dy = a + b·x + c·y + e·disparity shows b and c ≤ 0.1 px per 100 px (trajectories 0, 10, 11, 20, 23).
3. **It shrinks for nearer points on some trajectories.** On 0 and 23, dy falls in the nearest
   disparity tercile (trajectory 0: +1.25 → +0.29). The instruments are nearer than the tissue.
   That explains why the label-based dy (instrument keypoints) reads 0.3–0.9 px lower than the
   image-based dy (mostly tissue) on trajectories 0–5 and 23. The pattern is consistent with a
   constant row offset plus a small vertical baseline error, whose effect scales with disparity.
   I haven't separated the two.
4. **Correcting it helps SGM.** Shift the right image by the image-based dy, then rerun SGM:

   | Trajectories with \|dy\| > 0.5 px (n = 18) | Change |
   |---|---|
   | Valid-pixel fraction | median +3.7 points; improved on 18/18 |
   | Per-pixel scatter about the local tissue plane (pixels valid in both versions) | median −0.40 mm; improved on 16/18 |

   Trajectory 10 is the clear exception (4.2 → 6.3 mm), and I haven't explained it. On the four
   trajectories with |dy| ≤ 0.5 px, the correction changes scatter by only +0.02 to +0.32 mm.

**Rule fixed for v3 (from train + tune only):** before any stereo matching, measure dy from SIFT
matches on the rectified pair (median over a few frames per trajectory, or a running median
online). Shift the right image by dy when |dy| > 0.5 half-res px. Apply the same rule to SGM and
to learned stereo.

**What this means for v2**
- The v2 SGM tissue planes were computed without the correction.
- v2 keypoint triangulation is affected less: Gauss–Newton absorbs a vertical inconsistency as
  a residual, and depth comes from horizontal disparity.
- A proper fix would re-estimate the rectification (principal-point row offset and vertical
  baseline) from the matches. That's out of scope for now.
- A sanity check was run once on trajectory 21 (test2), which had already been inspected by hand
  before this check existed. It isn't part of these results.

## Step 1 (A0): focal-loss positives — 2026-09-28

The v2 loss marks a pixel positive only if its target ≥ 0.99. The Gaussians are rendered at
sub-pixel positions, so only 33.6% of training keypoints ever get a positive pixel. The fix
(`focal_pos: argmax`, in `keypoints.focal_loss`) makes the peak pixel of every rendered channel
the positive.

**Setup.** Four runs with identical configs except `focal_pos` and the seed:
v2 (the original seed-0 run plus a new seed 1) and A0 (seeds 0 and 1).
Evaluation: `scripts/eval_a0.py`, output `runs/v3_a0/`. Tune trajectories 18, 19, 20 and 23,
left eye, every 5th frame (804 frames, 7,998 labeled keypoints), clean and with the v2 de-centered
occluder protocol. There are only 4 trajectories, so I report seed spread instead of CIs.

| | v2 (s0, s1) | A0 (s0, s1) |
|---|---|---|
| Clean mean error (px) | 11.70, 11.16 | 10.38, 11.24 |
| Clean PCK@5 / PCK@10 | 0.603 / 0.759, 0.591 / 0.755 | 0.608 / 0.766, 0.578 / 0.741 |
| Occluded-keypoint mean error (px) | 19.69, 22.33 | 17.88, 19.11 |
| Median peak confidence | 0.59, 0.56 | 0.70, 0.67 |
| Keypoints with conf ≥ 0.6 that are within 10 px | 0.956, 0.933 | 0.906, 0.918 |
| Spearman(conf, error) | −0.66, −0.60 | −0.63, −0.67 |
| Spearman(Laplace σ, error) | −0.20, −0.21 | −0.21, −0.25 |
| dev PCK@10 at epoch 2 / 4 (seed 0) | 0.10 / 0.53 | 0.43 / 0.87 |

**Reading**

1. **No accuracy gain beyond seed noise on clean frames.** The two v2 seeds differ by 0.5 px in
   mean error, and the A0 − v2 differences are the same size with mixed signs on PCK@5/10.
2. **Occluded keypoints may improve** (seed means 18.5 vs 21.0 px), but the v2 seeds alone differ
   by 2.6 px. Not established.
3. **The clear effects are training speed and confidence scale.** A0 learns much faster early.
   Its peaks are taller (median conf 0.69 vs 0.58), so more keypoints land in the high-confidence
   bin, and that bin is less reliable (91–92% vs 93–96% within 10 px). Downstream thresholds were
   tuned on the v2 confidence scale: the gate τ = 0.2, conf_min = 0.05, and the Kalman
   R = (r0 σ)²/conf. A0 can't be swapped in without re-tuning them.
4. **Side finding: the Laplace σ is *anti*-correlated with error on tune** (ρ ≈ −0.2 for all
   four models). Broader peaks are slightly *more* accurate, probably because gross errors are
   sharp, confident peaks on the wrong structure. On trajectory 21 it was +0.26. Either way,
   the heatmap width isn't a usable error bar. That supports a learned variance head (v3 A5),
   and explains why the covariance needed κ.

**Decision**
- Keep `focal_pos: argmax` for new training (Phase A): it's correct by construction and
  converges faster.
- Don't replace the v2 network with the A0 network. There's no accuracy gain, and it would force
  re-tuning.
- Phase C starts from the v2 keypoints, as planned.

## Step 3 (Phase C): kinematics-fused 3D tool state — 2026-09-28

Code: `src/surgscene/fusion.py`, `scripts/fit_tool_geometry.py` (train 0–17 → `configs/tool_geometry.json`),
`scripts/eval_fusion.py` (tune → `runs/v3_fusion/results.json`), tests in `tests/test_fusion.py`.
Inputs are the v2 per-frame gated keypoints of both eyes, v2 Kalman outputs, and the robot's
`api_cp` / `api_jp`. The metric is v2 S1: the jaw-tip midpoint against the GT-triangulated tips,
every 5th frame. Everything is causal: the calibration at frame i uses frames 0..i only.

### What the data showed along the way

1. **Kinematics tracks the jaw pivot to 1.0–1.8 mm** on train, given an oracle hand-eye. There's
   no meaningful video/kinematics lag (≤ 0.1 mm either way).
2. **Tool frame:** z is the tool axis and the jaws open along ±y. The wrist sits at
   (ax, L sin(q5 + b), −L cos(q5 + b)), with q5 the wrist-yaw joint (residual 1.5–1.8 mm). The jaw
   angle isn't in `api_jp`, so the half-opening h comes from vision. It's **signed**: tip_a and
   tip_b are swapped in about 5% of train labels.
3. **SurgPose mixes instrument types.** The tip-to-pivot distance ranges from 8–12 mm to
   24–29 mm (trajectory 17; tune 18 and 19 are the same long-jaw tool), and the wrist
   length from 5.8 to 9.4 mm. One global geometry gave 22.7 mm. **The fix: estimate the tip
   offset and wrist parameters online per trajectory** (train medians as weak priors). A real
   da Vinci knows the instrument type, so this is the conservative setting.
4. **Hand-eye by reprojection is ill-conditioned in depth.** Fitted jointly with the geometry, it
   moved the solution 5–46 mm along depth to explain small, view-dependent label offsets (9.3 mm).
   A causal 3D robust registration (kinematic pivot → per-frame triangulated pivot, all past
   frames) with the geometry then fitted by reprojection gave 7.3 mm.
5. **A free per-frame vision correction also leaks into depth.** Restricting the correction δ to
   be lateral (a pseudo-measurement δ·ray = 0, std 0.5 mm) brought it to 6.7 mm.
6. **Vision is still better laterally for the tips** (2D left-image error, median: 4.1 px for v2 vs
   9.3 px for fusion). So the output is a **hybrid**: each tip is triangulated from the v2 Kalman 2D
   detections with a Gaussian depth prior (sd 5 mm) along the viewing ray, centred on the fused point.

### Results (tune: 18, 19, 20, 23; after the 1 s warm-up; v2 fills the warm-up in "all frames")

| Variant | 3D tip, mean / median (mm) | Along the ray (mm) | Jaw pivot (mm) | 2D left tips, mean / median (px) |
|---|---|---|---|---|
| v2 Kalman + triangulation | 15.3 / 6.7 | 15.2 | 8.5 | 9.6 / 4.1 |
| Kinematics only (causal calibration) | 7.0 / 5.3 | 6.5 | 3.4 | 32.1 / 29.3 |
| Fusion, δ unconstrained | 7.3 / 5.4 | 7.1 | 4.5 | 11.5 / 9.0 |
| Fusion, δ lateral only | 6.7 / 5.0 | 6.5 | 3.1 | 11.7 / 9.3 |
| **Hybrid (sd 5 mm)** | **6.7 / 5.1** (all frames 6.9) | 6.4 | 3.1 | **9.2 / 4.2** |
| Oracle calibration (GT, whole trajectory), δ lateral | 5.7 / 3.6 | 5.5 | 1.8 | 10.7 / 8.2 |

Per trajectory (hybrid vs v2, mean): **18: 10.4 vs 7.7 (worse)**, 19: 6.5 vs 18.8, 20: 7.1 vs 24.4,
23: 2.7 vs 10.5. Trajectory 18 is worse even with the oracle calibration (9.1), so the rigid tip
model limits the long-jaw instrument. That's an open item.

**Reference noise:** from the GT label disagreement between eyes (0.46 px median), the
GT-triangulated tip midpoint has a depth std of about 1.8 mm (median). That is part of every
number above.

**Uncertainty:** the raw hybrid covariance covers only 60% of GT tips at the 95% level. κ = 10.5
(v2: 6.9) restores 0.89–0.97 per trajectory. That's in-sample on tune; test2 is the real check.

### Caveats
- All parameters are hand-set defaults, compared across a handful of variants on 4 tune
  trajectories (sig_kin 2 mm, τ = 30 frames, ray sd 0.5 mm, depth-prior sd 5 mm, 30-frame
  warm-up, refit every 15 frames). There's no grid search, deliberately, to limit overfitting
  4 trajectories. `--grid` exists for a later sensitivity check.
- The reprojection-based hand-eye is kept as `handeye_mode="reproj"` for the record. It isn't used.
- Python only, with numeric Jacobians; a full tune evaluation takes minutes. The C++ port (E2)
  comes after the design is frozen.

### Plan amendments (for the v3 pre-registration)
- The Phase C model gains per-trajectory online tool-geometry estimation, a 3D-registered hand-eye,
  a lateral-only kinematic correction, and the hybrid output. C1–C3 are evaluated on the hybrid
  tip midpoint.
- C4 (jaw pivot) uses the fused pivot.
- Add an endpoint: 2D left-image tip error, hybrid vs v2 (non-inferiority), so the depth gain
  can't hide a lateral loss.

## Step 3 on test2 (pre-registered, `9a7cdc3`) — 2026-09-28

See [v3c_test_report.md](v3c_test_report.md).
- C1 **FAIL**: 5.27 [3.56, 7.43] mm.
- C2 PASS: −4.78 mm against v2's 10.05.
- C3 PASS: coverage 0.971.
- C4 PASS: pivot 3.38 mm.
- C6 PASS: 2D non-inferior.
- Standard instruments: 3.86 [3.04, 4.84] mm. Long-jaw (28, 29): no gain over v2.

## Step 4: long-jaw instruments (post hoc) — 2026-09-28

The full account, with the test2 table, is Amendment 1 in `validation_plan.md`.
- **Where the error comes from.** On the long-jaw tool (train 17, tune 18/19) the tip midpoint
  scatters 4–6 mm in the kinematic tool frame, against about 1 mm for standard tools, and jaw
  opening or wrist joints don't explain it. So kinematics is less precise at 24 mm from the pivot:
  the rigid model's oracle floor is 3.7–6.7 mm.
- **Tip offset estimated in 3D** (`tip_mode="3d"`) instead of by reprojection: mixed on tune
  (two better, two worse). On test2 it's better (C1 5.27 → 4.17 mm; long-jaw 12.3 → 8.4), but that
  run is post hoc and left as a hypothesis for fresh data. The frozen config is unchanged and
  reproduces exactly.
- The deeper limit is that the keypoint network has seen this tool on 1 of 18 training trajectories.

## Step 5 (Phase B): learned stereo — 2026-09-28

- RAFT-Stereo (zero-shot), checkpoint chosen on SurgPose tune proxies (`middlebury`). Tune: plane
  scatter 5.4 → 1.5 mm; photometric error not separable.
- **B1 on SERV-CT (fresh, pre-registered in `0c011f7`): PASS.** Depth MAE 17.1 → 1.6 mm on common
  pixels, in both specimens. A post-hoc check shows SGM's typical error is only 1–2 px: the gap is
  its 3–11% gross-error tail plus 41% missing coverage. Report: `v3b_servct_report.md`.
- Latency is the open problem: 74–410 ms per pair in PyTorch.

## Step 6 (Phase A): DINOv2 keypoints — 2026-09-28

- DINOv2 ViT-S/14 plus a conv-stem decoder, with a learned log-variance head (Gaussian NLL on the
  detached decoding error). The first training run produced NaN from a masked-NaN gradient; it's
  fixed and has a regression test.
- **Test2 (pre-registered in `b1396db`):**
  - A1 FAIL (PCK@10 −0.049) and A2 FAIL (mean +0.2 px): no 2D accuracy gain under the background
    and label shift.
  - A7 PASS (occluded −5.1 px).
  - A5 PASS: the learned σ ranks errors (ρ +0.53 vs −0.25).
  - A8 PASS: 95.2% coverage with k fitted on tune.
- Report: `v3a_test_report.md`.

## Step 7: DINOv2 downstream, ViT FP16, fast stereo — 2026-09-28

Pre-registered in `c52302c`. Report: `v3_step5_e1_b5_report.md`.
- **V1 FAIL, V2 FAIL:** 3D is unchanged on average (5.03 vs 5.27 mm). The long-jaw tool improves
  by about 3 mm and some standard trajectories get worse.
- **V3 PASS:** occlusion −4.3 px through the Kalman filter.
- **V4 PASS:** κ halves (10.5 → 5.1).
- **E1 PASS:** ViT FP16 +0.08 px vs GT, 4.4 ms. Two FP16 traps: an index overflow (fixed) and a
  0.88 px TensorRT-specific shift (unresolved, harmless vs GT).
- **B5 PASS:** `realtime` @ 4 iterations gives 1.9 mm on SERV-CT at 10.7 ms. The GPU budget now
  sums to about 16 ms.
