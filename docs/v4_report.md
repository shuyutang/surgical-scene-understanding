# v4: masks as a measurement, and a jaw-length constraint for long-jaw tools, 2026-09-29

Plan: [plans/plan_v4.md](plans/plan_v4.md).
Pre-registration: [validation_plan.md](validation_plan.md), "v4", committed in `9a98371` before
the one test run (`uv run python scripts/eval_v4.py --split test2`, output `runs/v4_test2/`).
**This is test2's fourth use**, and v4's direction was motivated partly by v3's test2 result on
long-jaw tools. Everything here is supportive evidence at most.

## Summary

1. **SAM 2 tracks instruments well but didn't earn a place as a measurement.** It tracked the
   instrument body on 99.9% of train keypoint-frames with no identity swaps, from one automatic
   prompt. But all three uses tried in development failed on train and tune: instrument-type
   recognition, depth from shaft width, and a visibility gate. The method under test uses no
   masks.
2. **The real cause of the long-jaw failure** is that the tip offset in the kinematic tool frame
   isn't rigid on long jaws. Its deviation follows the last wrist joint (cable-driven error), so no
   single tool geometry fits, even the right type's.
3. **The fix is a jaw-length constraint.** The tip lies on its detected viewing ray at a known
   distance from the (well-localised) pivot. This uses kinematics for the pivot only, not for the
   jaw orientation.
4. **Test2:**
   - The primary endpoint M1 fails: −2.15 [−5.34, +1.03] mm on arms classified long-jaw.
   - The classifier mislabelled one standard arm as long-jaw (27/PSM3), and the constraint made
     that arm 3 mm worse.
   - On the four truly long-jaw arms, the error fell by 0.2–6.7 mm.
   - The mean 3D tip error, 4.82 [3.63, 6.21] mm, is the first point estimate under R1's 5 mm,
     but the upper bound fails.

## Pre-specified endpoints (test2)

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| M1 (primary) | Arms classified long (causal): 3D tip error, v4 − v3 (paired, 5 arm-trajectories) | −2.15 [−5.34, 1.03] mm | UB < 0 | **FAIL** |
| M2 (R1) | Mean 3D tip error, v4, all arms | 4.82 [3.63, 6.21] mm | UB ≤ 5.0 | **FAIL** |
| M3 | Causal type = GT type | 0.958 (23 of 24) | ≥ 0.90 | PASS |
| M4 (R7) | Coverage of the κ-inflated 95% ellipsoid (κ = 12.79 from tune) | 0.957 [0.920, 0.987] | LB ≥ 0.85, point ≤ 0.99 | PASS |
| M5 | 2D left tip error, v4 − v2 | −0.25 [−0.84, 0.13] px | UB ≤ +0.5 | PASS |

**Reported:**
- **All arms, v4 − v3 (paired):** −0.45 [−1.41, 0.25] mm (5.27 → 4.82). Standard arms are
  identical to v3 by construction.
- **Along the viewing ray, v4 / v3:** 4.70 / 5.14 mm. Lateral error for v4: 0.56 mm.

| Arm-trajectory | Type (causal) | Type (GT) | v3 (mm) | v4 (mm) |
|---|---|---|---|---|
| 28/PSM1 | long | long | 7.54 | 7.16 |
| 28/PSM3 | long | long | 18.45 | **11.91** |
| 29/PSM1 | long | long | 6.53 | 6.36 |
| 29/PSM3 | long | long | 16.83 | **10.18** |
| 27/PSM3 | **long** | standard | 7.54 | **10.51** |
| the other 19 | standard | standard | 3.67 (mean) | identical |

## Post-hoc findings (after the run; exploratory, not claims)

- **Why 27/PSM3 was misclassified.** The type feature (90th percentile of lateral jaw length in
  mm) was calibrated on train using GT 2D keypoints, but it runs on detections. On 27/PSM3,
  outlier tip detections push it to 13.6–14.0 mm (threshold 13), while GT 2D gives 10.2 mm. The
  threshold should have been calibrated on detections, over train or tune. That was checkable
  before freezing and wasn't checked.
- **On the four GT long-jaw arms,** v4 − v3 is −0.4, −6.5, −0.2 and −6.7 mm (mean −3.4). Four
  clusters are too few for a meaningful interval. The large gains are on PSM3, the arm with the
  larger cable-driven scatter on train (4.8 vs 2.9 mm).
- **Without the misclassification,** the all-arm mean would be about 4.70 mm. It's arithmetic
  on a post-hoc correction, stated only to size the effect.
- **Hypothesis for fresh data:** the length constraint on long-jaw arms, with the type threshold
  calibrated on detections (and possibly tip detections gated by the SAM 2 mask, the one place
  masks might pay off), reduces long-jaw 3D tip error. It needs data none of this has seen.

## Development record (train and tune)

| Idea | Evidence | Kept? |
|---|---|---|
| SAM 2.1 masks, causal, one automatic prompt (step 0–1) | Body coverage 0.999 train / 0.976 tune; tips 0.93 / 0.80; no swaps; 59 of 68 videos started (9 had no confident frame in the first 90) | pipeline kept |
| Type from mask ratio (jaw length / shaft width) | Classes overlap (standard ≤ 1.93, long ≥ 1.86) | no |
| Depth from shaft width | 3–15% within-trajectory noise; implied diameter 5.0–8.2 mm across trajectories | no |
| Mask visibility gate | Neutral on tune (5.97 vs 5.97 mm) | no |
| Type library (train) + correct-type prior in the v3 fit | Oracle type: 6.87 → 6.86 mm on tune: no effect | no |
| Joint-dependent tip offset c(q) | Transfer 17 → 18/19 partial (4.7 → 2.9, 5.8 → 5.1 mm) | no (not pursued) |
| Jaw-length constraint, all arms | Tune 6.87 → 5.89 (oracle type); hurt 20/PSM1 (buried tips) | long arms only |
| Kinematic type feature (jaw mm from GT 2D, first 10 s) | Perfect separation on train + tune (≤ 11.1 vs ≥ 14.9) | yes, but see the post-hoc finding |

## Engineering notes

- **Hugging Face SAM 2 streaming:** it keeps every frame (12.5 GB over 1,001 frames). Prune by
  setting old `processed_frames` entries to `None`; deleting them corrupts frame indexing.
  Latency is about 8 ms per object, whatever the backbone size. Details are in
  [v4_step0_sam2_feasibility.md](v4_step0_sam2_feasibility.md).
- **SurgPose's `bbox_*.json` obj1/obj2 aren't tied to PSM1/PSM3** (reversed on trajectory 0).
- **The v4 method adds a closed-form ray–sphere solve per tip,** with no measurable latency. Since
  masks aren't in the method, plan step 6 (real-time SAM 2) was not pursued.

## Where this leaves R1

v3 Phase C gave 5.27 mm, and v4 gives 4.82 mm, still with an upper bound above 5. Standard
instruments sit at 3.67 mm. Long-jaw instruments are the remaining gap: 9.2 mm with v4. Their
kinematics are least reliable exactly where the tip is (24–29 mm from the pivot, cable-driven wrist
error).
