# Conventional vs modern: what each choice bought, 2026-09-28 (v4, v5 and scene-semantics rows added 2026-09-29)

This is a consolidation of results already reported elsewhere; no new evaluation was run for it.
Every number links back to a pre-specified endpoint in [validation_plan.md](validation_plan.md)
unless it's marked **post hoc**. CIs are 95% clustered bootstrap (trajectory for SurgPose, stereo
pair for SERV-CT, case for GraSP). Requirements R1–R7 are defined in [traceability.md](traceability.md).

"Conventional vs modern" doesn't split this project into two pipelines. Even the v2 baseline
detects keypoints with a network (ResNet-34 U-Net, ImageNet weights). The classical parts are
the estimation around it: stereo matching, triangulation, the shape prior, the Kalman filter and
kinematics fusion. So the comparison is made **per component**, each swapped with everything
else held fixed.

## Summary

| Component | Conventional | Modern | Result [95% CI] | Endpoint, data | Winner |
|---|---|---|---|---|---|
| Tissue depth | SGM (OpenCV, CPU) | RAFT-Stereo, `middlebury` @ 32 | Depth MAE 17.06 → **1.58 mm**; difference −15.48 [−23.46, −9.06] | B1, SERV-CT (fresh) | **Modern** |
| Tissue depth, real-time | SGM | RAFT-Stereo, `realtime` @ 4 | → 1.88 mm; difference −15.18 [−23.02, −8.75] | B5, SERV-CT | **Modern** |
| Tissue depth, newer model (v5 step 2) | RAFT-Stereo, `middlebury` @ 32 (2021) | Fast-FoundationStereo `23-36-37` @ 8 (monocular foundation prior, distilled) | 1.86 → **1.37 mm**; difference −0.49 [−0.95, −0.09]; 104 → 47 ms | B6, SERV-CT (3rd) | **Newer modern** |
| 2D keypoints, clean | ResNet-34 U-Net | DINOv2 ViT-S/14 + U-Net decoder | PCK@10 −0.049 [−0.069, −0.028]; mean error +0.20 [−0.65, 1.07] px | A1, A2, test2 | **Neither** (v2 slightly better on PCK) |
| 2D keypoints, occluded | ResNet-34 U-Net | DINOv2 | −5.09 [−6.77, −3.41] px | A7, test2 | **Modern** |
| Keypoint uncertainty | σ from the heatmap peak shape (Laplace) | Learned log-variance head | Spearman(σ, error) −0.25 → **+0.53**; coverage 0.952 [0.927, 0.975] | A5, A8, test2 | **Modern** |
| Structured 2D inference | Argmax | Shape prior + robust MAP (classical) | −0.30 [−1.06, 0.20] px clean; −0.09 [−0.44, 0.29] occluded | G1, G3, test2 | **Neither** |
| Temporal smoothing | Frame by frame | Kalman filter (classical) | 3D −1.68 [−2.56, −0.95] mm; occluded −1.95 [−2.30, −1.62] px; alert toggles × 0.54 | S2, T3, S5, test2 | **Classical** |
| 3D tip | Vision-only triangulation: 10.05 [7.62, 12.85] mm | Kinematics-fused EKF, causal hand-eye (classical) | **5.27** [3.56, 7.43] mm; difference −4.78 [−6.93, −2.71] | S1, C1, C2, test2 | **Classical** |
| 3D tip, front end swapped | v2 keypoints into the Phase C fusion | DINOv2 keypoints + learned σ | 5.27 → 5.03 [3.95, 6.29] mm; difference −0.24 [−1.30, 0.65] | V1, V2, test2 | **Neither** |
| Uncertainty inflation needed | κ = 10.5 (heatmap σ) | κ = 5.1 (learned σ) | Same coverage (0.971 vs 0.967) with half the inflation | C3, V4, test2 | **Modern** |
| Instrument masks as a 3D measurement (v4) | Keypoints + kinematics only | SAM 2.1 masks (type ratio, shaft-width depth, visibility gate) | All three no better than without masks on train/tune; a kinematic jaw-length feature beat the mask ratio | v4 development | **Classical** |
| Tissue under the instrument (v5 step 1) | Local plane in an annulus, current frame (Phase 5) | Temporal tissue memory with SAM 2 masks (not NeRF/3DGS) | Hidden-tissue proxy 2.54 → 1.28 mm; distance error still dominated by the tip (5.35 mm) | tune, development | **Memory** (simple, temporal) |
| Instrument tip depth (v5 step 0) | Kinematic fusion | RAFT-Stereo on jaw pixels (zero-shot) | Even read at GT pixels: tips 12–17 mm median vs kinematics 4.8 mm; jaw disparity bleeds to the tissue behind | tune, development | **Classical** |
| Instrument tip depth (v5 step 2) | Kinematic fusion | Fast-FoundationStereo on jaw pixels (zero-shot) | At GT pixels: tips 4.0 mm median (10.3 mean) vs kinematics 4.8 (6.3); jaw estimator 12.1 mm vs v4 hybrid 5.5 mm | tune, development | **Classical**, gap narrowed: a candidate fusion measurement |
| Long-jaw tip depth (v4) | v3 rigid tool model | Jaw-length constraint (geometric, classical) | Arms classified long −2.15 [−5.34, 1.03] mm (1 misclassified); mean 5.27 → 4.82 mm | M1, M2, test2 (4th) | **Neither** (primary fails) |
| Scene semantics: steps, backbone (GraSP) | ResNet-50 (ImageNet), frozen, + causal MS-TCN | DINOv2 ViT-B/14, frozen, same MS-TCN | Step macro-F1 0.401 → **0.469**; +0.067 [+0.039, +0.096], 5/5 cases | S1, GraSP test (1st) | **Modern** |
| Scene semantics: steps, temporal model | Per-frame linear probe | Causal MS-TCN (learned temporal model) | +0.137 [+0.079, +0.224] (ResNet-50); +0.136 on DINOv2 | S2, GraSP test | **Temporal model** |
| Scene semantics: phases, backbone | ResNet-50 + MS-TCN | DINOv2 + MS-TCN | Phase macro-F1 +0.046 [−0.036, +0.120], 4/5 cases | S4, GraSP test | **Neither shown** |
| Scene semantics: VLM | DINOv2 + MS-TCN (0.466 on VLM frames) | Qwen3-VL-8B per frame, QLoRA-fine-tuned (zero-shot: 0.025) | 0.206; −0.259 [−0.363, −0.159], 0/5 cases | S3, GraSP test | **Frozen features + temporal model** |

## Cost side

| Stage | Conventional | Modern |
|---|---|---|
| Keypoint network, TensorRT FP16, per stereo pair | U-Net 1.5 ms | DINOv2 4.4 ms (17.6 ms FP32) |
| FP16 accuracy cost vs GT | +0.008 px (D2) | +0.076 [+0.057, +0.095] px (E1) |
| Stereo, half resolution | SGM 19 ms (CPU) | RAFT-Stereo 10.7 ms @ 4 iterations, 98 ms @ 32 (PyTorch, not yet TensorRT) |
| Stereo, 720×576 (SERV-CT) | — | Fast-FoundationStereo 47 ms @ 8 vs RAFT-Stereo `middlebury` 104 ms @ 32 (PyTorch fp16; readme reports 14–23 ms with TensorRT on a 3090 at 640×480) |
| Kalman + fusion + triangulation | < 1 ms (C++, D3/D4) | — |
| End-to-end p99 (C++) | 2.59 ms without stereo (D4) | **not measured**: about 16 ms summed from separately timed stages |
| Deployment traps found | Argmax flips on near-tied peaks in FP16 (0.4% of keypoints > 2 px) | Argmax-index overflow in FP16 (all NaN, fixed by pinning the decode to FP32); an unexplained 0.88 px TensorRT-only shift |

## What the evidence says

1. **Where the modern network replaced a hand-engineered matcher, it won outright.** Learned
   stereo cut the depth error by an order of magnitude. Almost all of the gain comes from removing
   SGM's gross-error tail and filling its holes; on typical pixels the gap is about 1 px (post-hoc
   analysis in [v3b_servct_report.md](v3b_servct_report.md)).
2. **Where a network replaced a network, the foundation backbone didn't buy accuracy.** DINOv2
   helps where appearance is ambiguous (occluded keypoints) and gives a much better uncertainty,
   but it doesn't beat an ImageNet CNN on clean keypoints, with 18 training trajectories.
3. **The largest 3D gain came from classical estimation with a new sensor, not a bigger network.**
   The robot's own kinematics, fused with an EKF, halved the 3D tip error, and they removed the
   depth-along-the-ray error that no 2D detector can fix at 150–230 mm working distance.
4. **The best use of the modern network downstream was its uncertainty**, which the classical
   filter can consume directly: κ halved, and long-jaw trajectories improved by 3–3.5 mm because
   bad detections were down-weighted. That's the hybrid pattern: learned measurements, classical
   state estimation.
5. **The modern parts cost 3–5× the latency and bring new FP16 failure modes**, all still inside
   the 33 ms budget on paper.

## Not compared (gaps)

- **A truly classical keypoint detector** (colour or template tracking, marker-less edge models).
  Every keypoint comparison here is network vs network.
- **Learned stereo inside the proximity pipeline on SurgPose.** SurgPose has no tissue depth GT;
  on tune, the plane-fit scatter falls from 5.4 to 1.5 mm (development only).
- **A classical segmentation baseline** for Phase 1.
- **A learned temporal model or a learned fusion model** in place of the Kalman filter or EKF; see
  the next section.
- Test2 has been used four times and SERV-CT three times, so any new comparison needs fresh data
  to be confirmatory.

## Learned alternatives to the classical estimators (literature; not tested here)

This section is a reading of the field, not a result. Citations are from memory: check each one
before quoting it.

### Instead of the Kalman filter

| Family | Examples | Where it beats a constant-velocity Kalman filter | Why it isn't a drop-in here |
|---|---|---|---|
| Point-tracking transformers | CoTracker / CoTracker3 (Meta, 2023–24), TAPIR / BootsTAP (DeepMind, 2023–24) | Long occlusions and re-identification: attention across frames recovers a point from appearance, not just from a motion model | Best results are from offline, bidirectional windows; the causal online mode is weaker. Heavy for 30 fps on two eyes. No calibrated covariance |
| Temporal pose-lifting transformers | VideoPose3D (temporal conv, 2019), PoseFormer (2021), MixSTE, MotionBERT (2022–23) | Learn a motion prior for the whole skeleton, not one point at a time | Trained on large human-pose corpora; 34 SurgPose trajectories is too little. Most use non-causal windows |
| Video segmentation with memory | SAM 2 memory attention (2024), and surgical fine-tunes of it | Mask tracking through occlusion and re-entry | Gives masks, not keypoints or 3D; would feed the tracker, not replace it |
| Learned filters | KalmanNet (2022), backprop Kalman filter (2016), differentiable particle filters | Keep the state-space structure but learn the gain or noise from data, so the filter is right when the noise model isn't | Needs labelled sequences to train; loses some of the Kalman filter's interpretability and guarantees |

**Assessment.** A causal temporal transformer over DINOv2 features would plausibly improve
occluded keypoints beyond V3's −4.3 px, since that's where appearance memory helps. It's unlikely
to beat the Kalman filter on clean frames, where the error is already dominated by detector noise,
not motion. The version most likely to pay off here is a **hybrid**: a short causal temporal
model as the *measurement* front end (keypoints + σ), with the Kalman filter kept as the state
estimator that produces calibrated covariances. That keeps R7 (calibrated uncertainty) testable.

### Instead of the kinematics-fused EKF

For dVRK instrument tracking, the strongest published systems are themselves hybrids: a learned
detector feeding a model-based estimator that uses the kinematic chain.

| Family | Examples | What it adds |
|---|---|---|
| Kinematics + vision with a particle filter or optimizer | Richter, Lu, Yip (Yip lab, UCSD, about 2021) on partially visible kinematic chains; the SuPer framework (2020) for tool and tissue together | Same structure as Phase C, with multimodal posteriors instead of an EKF's Gaussian |
| Learned camera-to-robot registration | Keypoint-based pose with sim-to-real transfer; CtRNet (CVPR 2023) | Replaces the hand-eye step with a network; still optimizes over the kinematic chain |
| Render-and-compare | Differentiable rendering of the instrument CAD model posed by kinematics, matched to masks or features | Dense constraints instead of 10 keypoints; directly fixes tool-geometry errors like the long-jaw offset |
| Learned kinematic error compensation | Recurrent networks that correct cable-driven joint errors (e.g. Hwang, Goldberg et al., about 2020) | Learns hysteresis and cable stretch that a rigid kinematic model misses |
| Sliding-window smoothing / factor graphs | iSAM2 / GTSAM-style fixed-lag smoothing | Re-linearizes over a window; better than a single-step EKF when hand-eye and tool geometry are still being learned online |

I don't know of a published end-to-end network (image + joint tokens → 3D tip) that beats a
kinematics-plus-vision estimator on dVRK data. The reason is structural: kinematics give a precise,
nearly free measurement of the part vision is worst at (depth along the ray), and the chain's
geometry is known exactly. A network would have to learn that geometry from a few dozen
trajectories.

**Assessment.** The upgrades most likely to move R1 here, in order:
1. **Render-and-compare against segmentation masks** for the tool geometry. It targets the
   long-jaw failure (12.3 mm) that the per-frame EKF can't fix.
2. **A fixed-lag factor graph** over the hand-eye, tool geometry and tip states. This was in the
   v3 plan and the EKF stood in for it.
3. **A learned correction on the kinematics** (joint-error compensation) if residuals are
   configuration-dependent.
4. **End-to-end training through a differentiable filter**, so the detector learns σ for what
   the filter needs rather than for 2D error alone.

Each would need fresh SurgPose-like data for a confirmatory test.
