# v4 step 0: SAM 2 instrument tracking on SurgPose (development, tune only), 2026-09-28

Development check, not a pre-specified test: tune trajectories 18, 19, 20, 23 only (test2 untouched).
Script: `scripts/v4_sam2_feasibility.py` (`uv run --group sam ...`); outputs in `runs/v4_sam2_feas/`.

## Setup

- **Model:** SAM 2.1 through Hugging Face `Sam2VideoModel` (transformers 5.17), bf16, RTX 4090,
  sizes tiny, small and base-plus.
- **Causal streaming:** one frame at a time, left eye, 1,001 frames (33 s) per trajectory.
- **Prompts:** once, at frame 0, from our own v2 U-Net detections (keypoints with confidence
  ≥ 0.5), never refreshed. No ground truth is used for prompting. Two prompt modes:
  - `distal`: the detected keypoints only;
  - `shaft`: plus 3 points extrapolated along the shaft (shaft + k·(shaft − wrist), k = 1..3).
- **Proxy metrics**, since SurgPose has no mask ground truth:
  - a keypoint counts as covered if it's within 5 px of its own instrument's mask;
  - a swap is a keypoint within 5 px of the *other* instrument's mask;
  - box IoU compares the mask's bounding box with SurgPose's per-frame box. Where those boxes come
    from isn't documented, so this is a sanity check.

## Results

| Model | Prompt | GT keypoints covered, mean (worst arm) | Swaps, worst arm | Box IoU, median (worst) | Frames lost, worst arm | ms/frame p50 (2 objects) |
|---|---|---|---|---|---|---|
| tiny | distal | 0.905 (0.592) | 0.000 | 0.95 (0.22) | 0.106 | 27.4 |
| tiny | shaft | 0.906 (0.591) | 0.020 | 0.98 (0.93) | 0.105 | 27.6 |
| small | distal | 0.917 (0.598) | 0.000 | 0.99 (0.94) | 0.064 | 28.8 |
| small | shaft | 0.916 (0.600) | 0.000 | 0.98 (0.95) | 0.064 | 28.3 |
| base-plus | distal | 0.909 (0.604) | 0.000 | 0.98 (0.27) | 0.020 | 32.8 |
| base-plus | shaft | 0.922 (0.603) | 0.000 | 0.98 (0.93) | 0.021 | 33.3 |

"Frames lost" means fewer than half of the instrument's labelled keypoints are near its mask.

**Per keypoint** (small, shaft prompts), fraction covered:

| Trajectory, arm | shaft | wrist | jaw pivot | tip a | tip b |
|---|---|---|---|---|---|
| 18 PSM1 / PSM3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.97 / 0.92 |
| 20 PSM1 / PSM3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.94 | 0.65 / 0.07 | 0.72 / 0.05 |

## Reading

1. **SAM 2 tracks the instrument body reliably from one automatic prompt.** Shaft, wrist and jaw
   pivot stay inside the mask about 100% of the time over 33 s. There are no identity swaps (one
   run touches 2% on one arm), and the track is never lost for good.
2. **The jaw tips are its weak point.** All the shortfall in the table is tips. On trajectory 20,
   PSM3's tips are covered only 5–7% of the time: they're thin, dark and often buried in tissue
   (ground truth still labels them via UV paint). Masks can constrain the tool's pose and
   geometry, but not the tip position on their own.
3. **Model size barely matters for accuracy.** base-plus loses fewer frames on trajectory 20
   (2% vs 6–11%), and is otherwise equal to small.
4. **Prompt mode:** `shaft` gives a whole-instrument mask consistently (box IoU ≥ 0.93).
   `distal` sometimes segments only the wrist and jaws (IoU 0.22–0.27 on trajectory 18).

## Engineering findings

- **Memory grows without bound in the HF streaming session:** every processed frame and every
  per-frame output is kept, so about 12 MB per frame, and 12.5 GB after 1,001 frames. The model
  only reads the last 6 memory frames, 15 object pointers and the prompted frame. Dropping entries
  older than 16 frames caps memory at 0.6–0.7 GB, with masks identical to the unpruned run (IoU
  1.0000 on 300 frames, all three sizes).
- **The trap:** the session numbers each new frame as `len(processed_frames)`. Deleting old frames
  makes new frames reuse old indices, and the masks go wrong (mean IoU 0.23). Set the old entries
  to `None` instead of deleting them.
- **Latency scales with objects, not backbone size:** tiny, small and base-plus all take 23–29 ms
  for two objects and 15–20 ms for one, plus 4.3 ms preprocessing. The cost is the eager
  per-object memory path (memory attention and mask decoding per object), not the image encoder.
  On two eyes this is about 50–60 ms per stereo pair: **over the 33 ms budget** as it stands.
  Paths to fit it:
  - batch the objects;
  - compile or TensorRT the memory attention (the official repo's compiled VOS predictor);
  - run SAM 2 at a lower rate than keypoint tracking, or on one eye only.

## Implications for v4

- **Masks are good enough to use as a measurement of the instrument body** (shaft axis, wrist,
  silhouette) for render-and-compare against the kinematically posed tool model. That's the
  intended fix for tool-geometry errors like the long-jaw offset.
- **Tips must still come from the keypoint detector.** A mask can't be used to veto tip keypoints,
  because on trajectory 20 the mask misses tips that are really there.
- **A visibility signal:** a jaw pivot or wrist far outside the mask is a cheap occlusion or
  detector-failure flag for the Kalman filter update. Tips are excluded, for the same reason.
- **Real-time use needs the latency work first.** Offline, SAM 2 is ready now, for pseudo-labels
  or for fitting tool geometry once per trajectory.

## Limits

- Proxy metrics only. Coverage of 5 of the instrument's points says nothing about the mask's
  boundary accuracy elsewhere.
- 4 trajectories, 8 instrument tracks, left eye only. No specular, smoke or blood stress (ex vivo
  beef, no smoke).
- A single prompt from our detections at frame 0; a bad frame-0 detection would propagate. Not
  tested here.
