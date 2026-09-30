# Scene semantics on GraSP: test report (phase and step recognition), 2026-09-29

Pre-registration: `docs/validation_plan.md`, "scene semantics on GraSP" (commit `31f51d7`). **First
use of the GraSP test split** (5 cases, 42,897 frames at 1 fps). Plan and arms:
[plans/plan_scene_semantics.md](plans/plan_scene_semantics.md). Run once:
`scripts/eval_grasp.py test` → `runs/grasp_test/`.

**Question.** For the "what is happening" part of scene understanding, what does each modern
ingredient buy: a self-supervised foundation backbone, a temporal model, or a vision-language
model?

**Answer.** A temporal model is the largest gain (+0.14 step macro-F1). A DINOv2 backbone adds
+0.07 on top. A fine-tuned 8B VLM, run per frame, is far behind both (−0.26); zero-shot it is
near zero. All four results match the direction and rough size seen in development.

## Endpoints (unit = case; paired case-level bootstrap, 2,000 replicates)

| ID | Endpoint | Result [95% CI] | Cases > 0 | Criterion | Verdict |
|---|---|---|---|---|---|
| S1 (primary) | Step macro-F1, B − A (DINOv2 vs ResNet-50, both causal MS-TCN) | **+0.067** [+0.039, +0.096] | 5/5 | LB > 0 | **PASS** |
| S2 | Step macro-F1, A − A0 (causal MS-TCN vs per-frame linear probe) | **+0.137** [+0.079, +0.224] | 5/5 | LB > 0 | **PASS** |
| S3 | Step macro-F1, fine-tuned VLM − B, VLM frames | −0.259 [−0.363, −0.159] | 0/5 | report | reported |
| S4 | Phase macro-F1, B − A | +0.046 [−0.036, +0.120] | 4/5 | LB > 0 | **FAIL** |

- S3's label in the output says "smoothed"; the frozen smoothing window was 0 s (development
  showed smoothing hurt), so the VLM answers are scored raw.
- S4 fails on one case: CASE041, where DINOv2's phase F1 is 0.10 lower. On the other four it is
  0.03–0.17 higher. As pre-registered, a FAIL here means "not shown on 5 cases".

## All arms (mean over the 5 test cases)

| Arm | Frames | Step F1 | Phase F1 | Step acc | Phase acc |
|---|---|---|---|---|---|
| A0: ResNet-50, linear probe | all | 0.264 | 0.419 | 0.373 | 0.507 |
| B0: DINOv2 ViT-B/14, linear probe | all | 0.333 | 0.510 | 0.473 | 0.604 |
| A: ResNet-50 + causal MS-TCN | all | 0.401 | 0.614 | 0.544 | 0.699 |
| **B: DINOv2 + causal MS-TCN** | all | **0.469** | **0.660** | **0.589** | **0.720** |
| B, on the VLM's frames | every 5th | 0.466 | 0.663 | 0.592 | 0.723 |
| C: Qwen3-VL-8B, zero-shot (bf16) | every 5th | 0.025 | 0.053 | 0.099 | 0.121 |
| C: Qwen3-VL-8B, QLoRA fine-tuned | every 5th | 0.206 | 0.394 | 0.416 | 0.534 |

Reported, not pre-specified as a test: **B − B0** (temporal model on DINOv2 features) +0.136
[+0.078, +0.235], 5/5 cases.

Per-case step macro-F1 (CASE041, 047, 050, 051, 053):

| Arm | Per case |
|---|---|
| A | 0.376, 0.441, 0.290, 0.466, 0.433 |
| A0 | 0.292, 0.328, 0.225, 0.159, 0.315 |
| B | 0.400, 0.508, 0.329, 0.575, 0.530 |
| B0 | 0.318, 0.420, 0.261, 0.241, 0.424 |
| C fine-tuned (VLM frames) | 0.227, 0.222, 0.216, 0.135, 0.231 |

## Development vs test

| Difference (step F1) | Development (fold1 ↔ fold2) | Test |
|---|---|---|
| B − A (S1) | +0.054 | +0.067 |
| A − A0 (S2) | +0.064 | +0.137 |
| VLM fine-tuned − B (S3) | −0.174 (fold1 → fold2) | −0.259 |
| B − A, phase (S4) | +0.064 | +0.046 |

Test scores are higher than development for the temporal arms (B: 0.403 → 0.469), as expected
from training on 8 cases instead of 4.

## Reading

- **Temporal context matters most.** Steps last from seconds to minutes and many look alike frame
  by frame; a causal MS-TCN over 1 fps features uses the history the per-frame models don't have.
  It is the largest gain, positive on every case, with either backbone.
- **The foundation backbone helps, frozen.** DINOv2 features beat ImageNet ResNet-50 features
  under the same temporal model, on every case. That is the opposite of SurgPose keypoints, where
  DINOv2 did not help on clean frames (A1): whole-frame semantics is closer to what
  self-supervised pretraining learns than sub-pixel localization.
- **A general VLM is not a workflow recognizer.** Zero-shot, Qwen3-VL-8B answers almost every
  frame with the same phase and step (e.g. "Denonvilliers_Fascia, Denon_Dissection"): it
  cannot tell prostatectomy steps apart from one frame and the label names. QLoRA fine-tuning on
  7.4k train frames lifts it to 0.21, still below a per-frame linear probe on frozen features
  (B0: 0.35 on the same frames), at far more compute per frame (73 ms per frame for the 4-bit
  8B model at batch 16, against a ViT-B/14 forward pass). Its value in this project would
  be elsewhere (answering questions about a scene), not in replacing a phase recognizer.
- **Caveats.** Five test cases. One center, one procedure. GraSP labels were revised in December
  2024, and SurgMLLMBench's GraSP answers disagree with the official labels on 18–23% of frames:
  this report uses the official labels throughout. The VLM saw one frame per question and no
  history; a video-input VLM was not tested.
