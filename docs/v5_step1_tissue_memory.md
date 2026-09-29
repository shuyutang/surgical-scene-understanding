# v5 step 1: occlusion-aware tissue memory for instrument-to-tissue distance (development, tune only), 2026-09-29

**Question.** Per-frame stereo can't see the tissue under the instrument: exactly where the
tip-to-tissue distance matters. Phase 5 interpolates it with a robust plane fitted in an annulus
around the tip. Does a tissue model built over time do better? Is it worth adopting NeRF or
3D Gaussian Splatting for this?

**Answer, from proxies on tune:** a simple causal tissue memory halves the error on hidden tissue
(1.28 vs 2.54 mm). But the distance error is dominated by the tip estimate (5.35 mm), not the
tissue model. Test2 wasn't used.

## Method (`src/surgscene/tissue_memory.py`, `scripts/v5_tissue_memory.py`)

- **The SurgPose endoscope is static** (SIFT drift < 0.6 px over 33 s on 18/19/20; 1.3 px on 23,
  where tissue was manipulated). So the memory lives in the rectified left image; a moving camera
  would need registration.
- **Every frame (30 fps):** RAFT-Stereo `realtime` @ 4 at half resolution, with the B3 row shift.
  The instrument mask is the SAM 2 masks (rectified) ∪ the detected-keypoint capsules, dilated by
  8 px against the disparity bleed at instrument edges (v5 step 0).
- **Memory:** per pixel, a ring of the last 15 valid tissue disparities. It's written only where
  there's no instrument, so it keeps the tissue seen before an instrument covered it. The median
  is the estimate, and it adapts to moving tissue within about 8 observations.
- **Cost:** 3.9 ms per update, 3 ms per tip query (numpy on CPU, not optimised). RAFT `realtime`
  @ 4 is 10.7 ms on GPU.

## Proxies (no tissue GT on SurgPose; tune 18, 19, 20, 23; every 5th frame)

| Proxy | Memory | Plane, current frame (Phase 5) | Raw current frame |
|---|---|---|---|
| **Hidden tissue:** depth error on tissue under the instrument, scored when the same pixel reappears (≤ 3 s later) | **1.28 mm** (coverage 0.97) | 2.54 mm | 4.18 mm |
| GT tip > 3 mm behind the tissue surface | 0.001 | 0.001 | 0.010 |

- **Per trajectory**, hidden tissue, memory / plane: 1.32 / 3.80, 1.52 / 2.25, 1.21 / 1.60 and
  1.08 / 2.49 mm. The memory is better on all four.
- **The two tissue models disagree** by 2.81 mm on average in the depth gap under the GT tip.
- **Distance error from the tip estimate alone** (v4 hybrid tip vs GT tip, same surface):
  5.35 mm. That's larger than the whole tissue-model difference.

## Reading and caveats

1. **Occlusion memory is what a tissue model adds.** On static-to-slow tissue, the memory
   predicts what's under the instrument twice as well as the local plane, and 3× better than
   reading the current frame (which sees the instrument).
2. **The proxy favours the memory.** The reference is the same stereo sensor observing the pixel
   later, so it checks temporal consistency on quasi-static tissue, not absolute accuracy (which
   stays RAFT's, 1.6–1.9 mm on SERV-CT).
   - It also can't see **deformation under contact**: the memory holds the pre-contact surface,
     and the reference is the surface after the instrument leaves. A pushed-in tissue surface is
     exactly what a deformable model (NeRF/3DGS or a deformation graph) would add, and this data
     can't evaluate it.
3. **The penetration proxy doesn't discriminate.** The GT tips almost never sit behind any
   surface estimate.
4. **The tip dominates the distance error** (5.35 mm vs a 2.8 mm tissue-model difference). For
   the proximity feature, the tip estimate (R1) is still the first-order problem.

## On NeRF / 3D Gaussian Splatting

A deformable NeRF or 3DGS model would be supervised by the same stereo depth, so on this data its
geometry is at best a smoothed version of the memory. It adds value only for deformation and
view synthesis, which can't be evaluated on SurgPose (no tissue GT; static camera; ex vivo tissue).
The current generation also optimises per video, mostly offline.

**Recommendation:**
- keep the memory as the real-time tissue model;
- compare a 3DGS reconstruction against it offline only if a dataset with depth GT over time
  (e.g. SCARED) is available.

Outputs: `runs/v5_tissue/{results.json, report.md}`.
