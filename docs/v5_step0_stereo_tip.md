# v5 step 0: instrument tip depth from dense learned stereo (development, tune only), 2026-09-29

**Question.** Keypoint triangulation fails in depth because two independent 2–3 px detections
give a few pixels of disparity error. On this rig (f ≈ 1812 px, baseline 6.0 mm), 1 px of
disparity is about 3.7 mm of depth at 200 mm (7.4 mm at half resolution). RAFT-Stereo reached
1.6 mm on SERV-CT *tissue* (B1). Can dense stereo on the jaw pixels replace or beat the kinematic
tip depth, whose cable-driven error is behind the long-jaw failure?

**Answer: no, not zero-shot.** Test2 wasn't used.

## 1. Jaw-pixel estimator (`scripts/v5_dev_stereo_tip.py`)

- **Jaw pixels:** SAM 2 mask ∩ a band around the detected pivot-to-tip segment.
- **Tip depth:** robust affine fit of disparity along the jaw, evaluated at the tip midpoint.
- **Frames:** tune, every 10th frame, paired (all methods have an estimate).

| Mean 3D tip-midpoint error (mm), mean over 8 arm-trajectories | |
|---|---|
| v2 keypoint triangulation | 15.06 |
| v3 kinematic fusion point | 6.72 |
| v3 hybrid | 6.65 |
| v4 hybrid | **5.50** |
| Stereo on jaw pixels, RAFT `realtime` @ 4 | 23.20 |
| Stereo on jaw pixels, RAFT `middlebury` @ 32 | 29.07 |

Only one arm-trajectory worked (23/PSM3: 2.0 mm, a short jaw seen side-on).

## 2. Upper bound: stereo depth read at the GT pixel locations

This takes pixel selection and extrapolation out of the question: the disparity is read at the
projections of the GT pivot and GT tips (3 × 3 median). Absolute depth error, mm (median / mean):

| Source | Pivot | Tips |
|---|---|---|
| RAFT `realtime` @ 4, half res | 6.0 / 8.3 | 16.7 / 19.5 |
| RAFT `realtime` @ 4, full res | 6.4 / 8.5 | 12.1 / 17.5 |
| RAFT `middlebury` @ 32, half res | 2.9 / 5.8 | 12.9 / 19.5 |
| v3 kinematic fusion (along the ray) | **2.0 / 2.7** | **4.8 / 6.3** |

On long jaws, RAFT's disparity at the tips reads 3–5 px (half res) *below* the truth. The thin
metal jaw takes on the disparity of the tissue behind it: the classic thin-structure failure of
learned stereo. Its disparity at the pivot, on the instrument body, is 0.3–1.4 px off.

## Reading

- **Learned stereo is right for tissue and wrong for thin instruments.** The tissue result (B1)
  doesn't transfer. The instrument is thin, specular and in front of textured tissue; the network
  smooths across the depth edge.
- **Kinematics remains the best depth source for the instrument,** even with its cable-driven
  error. That's the same conclusion as v3, from a stronger competitor.
- **Not tested:** stereo models with a monocular foundation prior (MonSter, FoundationStereo,
  both built on Depth Anything V2 features), which are reported to do better on thin structures.
  The oracle-pixel test above is the fair first check for them.
  - Instrument-specific fine-tuning of the stereo network would also need depth ground truth on
    instruments. SurgPose's GT is triangulated keypoints only, which is sparse but would be
    enough to supervise disparity at those pixels.

Outputs: `runs/v5_dev/{results.json, report.md, oracle_pixel_depth.json}` (the last from
`scripts/v5_dev_stereo_tip.py --oracle --models realtime@4 middlebury@32`; the full-resolution
`middlebury@32` row was skipped at the time).
