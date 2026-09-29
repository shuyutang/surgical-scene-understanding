# v3 Phase C on test2: kinematics-fused 3D tool state (pre-specified), 2026-09-28

Pre-registration: `docs/validation_plan.md`, section "v3 Phase C", committed in `54aef7e` before
this run. It ran once: `uv run python scripts/eval_v3c.py --split test2`. Raw output is in
`runs/v3c_test2/` (report.md, results.json). **This is test2's second use**; the pre-registration
states the prior exposure.

## Pre-specified endpoints (12 trajectories, trajectory-clustered bootstrap)

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| C1 (primary, R1) | Mean 3D tip error, hybrid | 5.27 [3.56, 7.43] mm | UB ≤ 5.0 | **FAIL** |
| C2 | Hybrid − v2 (paired) | −4.78 [−6.93, −2.71] mm | UB < 0 | PASS |
| C3 (R7) | Coverage of the κ-inflated 95% ellipsoid | 0.971 [0.957, 0.983] | LB ≥ 0.85, point ≤ 0.99 | PASS |
| C4 | Fused jaw-pivot error | 3.38 [3.03, 3.80] mm | UB ≤ 5.0 | PASS |
| C6 | 2D left-tip error, hybrid − v2 | −0.33 [−1.02, 0.11] px | UB ≤ +0.5 | PASS |

**Pre-specified subgroup (cluster = trajectory-arm):**

| Instruments | n | Hybrid | v2 | Hybrid − v2 |
|---|---|---|---|---|
| Standard (online tip offset ≤ 15 mm) | 20 | 3.86 [3.04, 4.84] mm | 9.69 [6.81, 13.03] | −5.84 [−8.56, −3.41] |
| Long-jaw (> 15 mm; trajectories 28, 29) | 4 | 12.31 [7.03, 17.65] mm | 11.81 [8.87, 14.73] | +0.50 [−4.09, 4.83] |

**Reported:**
- Along the viewing ray, hybrid vs v2: 5.14 vs 9.98 mm.
- Lateral: 0.54 vs 0.62 mm.
- Tip swaps: 0.1% vs 2.1%.
- After the 1 s warm-up only: 5.18 mm.

| Trajectory | 21 | 22 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid (mm) | 3.6 | 4.2 | 3.8 | 3.7 | 6.3 | 5.7 | **13.0** | **11.6** | 2.3 | 2.7 | 2.5 | 4.0 |
| v2 (mm) | 13.7 | 6.8 | 14.8 | 8.8 | 9.8 | 15.7 | 14.7 | 8.9 | 4.5 | 8.7 | 7.1 | 7.2 |

## Reading

1. **The depth failure behind S1 is largely fixed.** The mean 3D tip error falls from 10.05 to
   5.27 mm (−48%, C2), all of it along the viewing ray (9.98 → 5.14 mm), without losing lateral or
   2D accuracy (C6). The hybrid is better than v2 on 11 of 12 trajectories.
2. **C1 fails narrowly.** The point estimate of 5.27 mm is just above the 5 mm requirement, and the
   upper bound is 7.43. It's reported as a fail.
3. **The failure is concentrated in the pre-specified weak spot.** On standard instruments (20 of
   24 arm-trajectories) the hybrid reaches 3.86 [3.04, 4.84] mm, inside the requirement. On the
   long-jaw instrument (trajectories 28 and 29, the same tool as tune 18 and 19) there's no gain
   over v2 (+0.5 mm). The rigid tip model doesn't fit that instrument, as tune trajectory 18
   suggested. This is a subgroup result, not a claim that R1 is met: the primary endpoint covers
   all instruments.
4. **Uncertainty transfers.** κ = 10.46, fitted on tune, gives 97.1% coverage on test2 (v2's
   S3: 97.3%).
5. **The causal hand-eye beats v2's offline one.** Pivot error is 3.38 mm here, against v2 S4b's
   4.07 mm, which fitted on the first half of each trajectory and evaluated on the second.
6. **Tip swaps nearly vanish** (2.1% → 0.1%), because the tips are tied to the tool model. That's
   the failure the per-keypoint Kalman filter couldn't catch.
7. **Reference noise:** the GT tips are triangulated from hand labels with the same 5.5 mm baseline
   (about 1.8 mm depth std on tune). Part of the 3.86 mm on standard instruments is the reference.

## Next

- **Long-jaw tool model (post hoc, exploratory):** jaw curvature and flex, or tip positions
  that depend on the opening angle, fitted on the long-jaw train trajectory 17. Any gain on 28 and
  29 would be post hoc, because this run has now shown them.
- In production, the instrument type is known, so a per-type geometry library would replace the
  online estimation. That can't be tested with SurgPose.
- Then v3 Phase B (learned stereo + SERV-CT, fresh data), then Phase A.
