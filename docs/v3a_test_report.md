# v3 Phase A on test2: DINOv2 keypoint network (pre-specified), 2026-09-28

Pre-registration: `docs/validation_plan.md`, "v3 Phase A", committed in `d4c31a1` before this run.
The run was `uv run python scripts/eval_kp_vit.py --split test2`, once; output is in
`runs/v3_kp_vit_test2/`. The seed-0 DINOv2 model is compared with the seed-0 v2 U-Net; test2
left, every 5th frame, 12 trajectories.

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| A1 (primary) | PCK@10, DINOv2 − v2 (clean) | −0.049 [−0.069, −0.028] | LB > 0 | **FAIL** |
| A2 | Mean error, DINOv2 − v2 (clean) | +0.20 [−0.65, 1.07] px | UB < 0 | FAIL |
| A7 | Occluded keypoints, mean error DINOv2 − v2 | −5.09 [−6.77, −3.41] px | UB < 0 | PASS |
| A5 | Spearman(σ, error): learned σ vs Laplace σ | +0.53 vs −0.25 | learned > Laplace | PASS |
| A8 | Coverage of the 95% radius, learned σ × 6.086 (k from tune) | 0.952 [0.927, 0.975] | LB ≥ 0.85, point ≤ 0.99 | PASS |

**Both seeds agree:**

| | v2 (s0, s1) | DINOv2 (s0, s1) |
|---|---|---|
| Clean mean error | 12.96, 14.52 px | 13.16, 13.00 px |
| Median | 5.00, 5.12 px | 4.94, 4.95 px |
| PCK@10 | 0.717, 0.696 | 0.668, 0.671 |
| Occluded keypoints | 22.2, 24.6 px | 17.1, 18.4 px |

## Reading

1. **A stronger pretrained backbone doesn't fix the background-shift failure (K1).**
   - Clean accuracy is unchanged on average, and PCK@10 is 5 points lower.
   - The per-trajectory pattern is the same as v2's: 24–27, the trajectories with label-convention
     offsets and tissue shift, stay at 19–27 px with either network.
   - That's consistent with the v2 finding that a large share of this error comes from how the
     labels were placed, not from the detector.
2. **It is much more robust to occlusion**, by 5.1 px (about 23%). Foundation features let the
   network place a hidden keypoint from its context.
3. **The learned uncertainty is the clearest gain.**
   - The heatmap-width σ is anti-correlated with error (−0.25).
   - The learned σ ranks errors well (+0.53), and one scalar fitted on tune gives 95.2% coverage
     on test2.
   - For the downstream Kalman filter and triangulation, this is the missing ingredient: v2 needed
     κ = 6.9 and v3 κ = 10.5 because the σ they used didn't track error.
4. The raw learned σ under-covers (59%): it was trained on in-domain errors. The tune-fitted
   k = 6.1 absorbs the domain gap. It isn't a property of the network.

## What this means for v3

- Swapping in the DINOv2 keypoints is **not** justified by 2D accuracy. It *is* justified for
  occlusion robustness and for a usable per-keypoint σ.
- The next test is downstream: feed the learned σ into the Kalman noise and the fusion weights
  (plan step 5), selected on tune.
- The label-convention trajectories remain the main 2D error. That's a data and labelling problem.
