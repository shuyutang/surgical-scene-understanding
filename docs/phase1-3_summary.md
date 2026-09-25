# Phases 1–3: results and lessons (2026-09-25)

## Phase 1: segmentation (SISVSE → EndoVis18 zero-shot)
U-Net/ResNet-34 at 640×512, 32 classes, 60 epochs (~23 min on an RTX 4090). Frozen test set:
10 patients, patient-clustered CIs.

- **All 7 pre-specified criteria pass**: instrument Dice 0.928 [0.909, 0.941], tip Dice
  0.829 [0.794, 0.856], liver 0.850, stomach 0.814, zero-shot EndoVis18 instrument Dice
  0.778 [0.748, 0.810]. All 4 image-condition subgroups are within 0.10 (blurry frames worst, −0.051).
- **Calibration** failed on dev (ECE 0.068). Temperature scaling fit on dev (T = 1.56) brought
  test ECE to 0.021 [0.007, 0.046]. It passes, but the upper bound is close to 0.05.
- **Weak spots:** spleen 0.505 and gallbladder 0.663 with a very wide CI (few patients have
  them). The zero-shot tip Dice collapses to 0.571 (the Xi porcine domain has different
  instruments and lighting), so parts transfer much worse than the binary instrument mask.

## Phases 2–3: keypoints + shape-prior MAP (SurgPose)
Heatmap U-Net, 10 keypoints (5 per arm). Test = 14 trajectories on different tissue backgrounds.

- **K1 fails:** dev median error 2.1 px, but test PCK@10 is 0.69. The background shift is
  real, as the SurgPose authors warn. The *shaft* keypoint dominates (median 15–20 px): a point
  on a featureless cylinder is ambiguous along its axis, so it's a label-definition problem
  as much as a model one.
- **Shape model:** Procrustes + PCA needs 4–5 of the 6 possible modes for 5 keypoints. The
  instruments articulate a lot, so the prior is necessarily loose.
- **MAP, after fixing the leak** (post hoc): on occluded keypoints −1.47 [−2.09, −0.83] px
  (23.0 → 21.5), non-inferior on clean frames, and implausible shapes 17% → 1.4%, at 0.47 ms
  per instrument in NumPy. But **PCK@5 drops** (0.485 → 0.418): the prior pulls
  already-precise points. MAP trades fine precision for protection against gross errors.
- **Discrete top-N candidate MAP** was worse than continuous MAP everywhere except occluded
  keypoints, where it tied, and it was 15× slower. With a unimodal network the candidate set
  rarely contains a better answer than the argmax.

## The main lesson: shortcut learning in the occlusion protocol
The first run failed the primary hypothesis. Covering a keypoint made it *easier* to find,
because training occluders were always centered on keypoints and the network learned to use
the blob as a pointer. The diagnosis followed from one anomaly in the table (occluded error <
visible error), was confirmed by a targeted intervention (off-center occluders), and was
fixed at the data level (decentered occluders + distractors). The pre-specified result stands
as a FAIL; the fix is reported as a post-hoc amendment.

## Next steps
1. Gated fusion: keep argmax when confidence is high, use MAP when low. The reliability table
   shows confidence is informative (0.9+ → 99.8% within 10 px; <0.1 → 19%).
2. Shaft keypoint: replace the point with a line (shaft axis) observation in the MAP energy.
3. Background robustness: stronger color/texture augmentation, or train on SurgPose's
   green-channel videos.
4. Phase 4: the Kalman filter on shape parameters, using the MAP's Laplace covariance as the
   measurement noise.
