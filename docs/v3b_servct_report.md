# v3 Phase B: learned stereo vs SGM on SERV-CT (pre-specified), 2026-09-28

Pre-registration: `docs/validation_plan.md`, "v3 Phase B", committed in `16fd1d3` before this
run. The run was `uv run --group stereo python scripts/eval_servct.py`, once; output is in
`runs/v3_servct/`. **SERV-CT is fresh data**, unused for any choice before this run. The
RAFT-Stereo checkpoint (`middlebury`) was selected on SurgPose tune.

## Pre-specified endpoint

| ID | Endpoint | Result [95% CI] | Criterion | Verdict |
|---|---|---|---|---|
| B1 | Mean absolute depth error, learned − SGM (pixels valid for both; 16 pairs) | −15.48 [−23.46, −9.06] mm | UB < 0 | **PASS** |

**Reported:**

- Depth MAE, SGM / learned (common pixels): 17.06 / 1.58 mm.
- Learned on all valid pixels: 1.86 mm. SGM covers 59% of the valid pixels.
- Per experiment (SGM → learned): specimen 1: 7.19 → 1.31 mm; specimen 2: 26.93 → 1.85 mm. The
  result holds in both.
- Bad pixels (> 3 px): 25 points fewer with the learned model. Disparity EPE: 4.8 px lower.

## Post-hoc checks (after the run)

1. **Is the comparison unfair to SGM?** Without the B3 shift, SGM's depth MAE is 16.8 mm; with
   9×9 or 11×11 blocks, 16.7–16.8 mm. The result doesn't hinge on those choices.
2. **What drives SGM's error?** Its *typical* disparity error is 1.1–1.9 px (median, with a slight
   −1 px bias), but 3–11% of its pixels are gross errors (> 10 px too small). Depth ∝ 1/disparity,
   so those dominate the mean depth error.
3. The honest reading: learned stereo mostly removes SGM's gross-error tail and fills its holes.
   On typical pixels, the gap is about 1 px.
4. **For the proximity feature, the gain should be smaller.** The v2 tissue-plane fit is already
   robust (Huber IRLS in disparity space) and down-weights SGM outliers. On SurgPose tune, the
   plane scatter falls from 5.4 to 1.5 mm, but there's no tissue GT there.

## Limits

- 16 pairs from 2 porcine specimens, imaged ex vivo with a da Vinci endoscope.
- Latency: RAFT-Stereo (32 iterations, PyTorch) takes about 410 ms per half-resolution pair on
  the RTX 4090, and the `realtime` checkpoint 74 ms. Neither fits the 33 ms budget without
  TensorRT, fewer iterations, a region of interest around the tips, or running stereo at a lower
  rate than the tip tracking (tissue moves slowly). That's Phase E work.
- License: SERV-CT is CC BY-NC-SA 4.0; RAFT-Stereo is MIT.
