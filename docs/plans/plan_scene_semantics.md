# Plan: surgical scene semantics on GraSP (phase and step recognition), 2026-09-29

**Status (2026-09-29): done.** Endpoints frozen in `docs/validation_plan.md` and run once on
test: S1 and S2 pass, S4 fails, S3 reported. Results: [grasp_test_report.md](../grasp_test_report.md).
Round 2 (EndoSSL backbone S5; instrument and action recognition ST1–ST4): S5, ST1, ST2, ST4 fail,
ST3 passes. Results: [grasp_round2_report.md](../grasp_round2_report.md).

A separate track from the 3D pipeline: GraSP is monocular, with no stereo, depth or kinematics,
and SurgPose has no workflow labels, so the two can't be evaluated jointly. The question is the
same as in the rest of the project, asked for the "scene semantics" component: **what does each
modern ingredient buy, measured against a strong conventional baseline?**

## Data (verified 2026-09-29)

- **GraSP 1 fps** (robot-assisted radical prostatectomy, da Vinci, 1280 × 800): 13 cases,
  11 phases, 21 steps, one label per second. Split and labels are frozen in
  `splits/grasp_split.json` (`scripts/grasp_prepare.py`), from the official December 2024 labels.
  - **Train:** 8 cases, 73,618 frames, with two official development folds (4 + 4 cases).
  - **Test:** 5 cases, 42,897 frames. Untouched until the pre-registration is committed.
- **Class imbalance:** "Idle" is 28–33% of frames. Two steps have < 250 train frames
  (Hold_Prostate 103, Pack_Prostate 200).
- **SurgMLLMBench's GraSP answers disagree with the official labels** on 18% of test and 23% of
  train frames. The most common mismatch: official Cut / Prevessical_Dissection / Idle →
  benchmark "Id_Illiac_Vein_Artery". The likely cause is the pre-2024 annotations (not
  confirmed: the old files are behind a Drive quota). **Official labels are ground truth
  everywhere**; for the language-model arm, questions in SurgMLLMBench's style are regenerated
  from the official labels.
- **Terms:** no data license is stated (code MIT; SurgMLLMBench CC BY-NC). Treated as research,
  non-commercial use, with citation.

## Arms

All arms are **online (causal)**: a prediction at second t uses frames ≤ t, as a live system would.

| Arm | Frame features | Temporal model | What it isolates |
|---|---|---|---|
| A. Conventional | ResNet-50, ImageNet, frozen | Causal multi-stage temporal convolution (MS-TCN) | Baseline recipe |
| B. Foundation backbone | DINOv2 ViT-B/14, frozen | Same MS-TCN, same training | Backbone effect (A vs B) |
| A0 / B0 | Same features | None (per-frame linear probe) | Temporal-model effect |
| C. Vision-language model | Open 7–8B VLM, zero-shot (label list in the prompt) and LoRA-fine-tuned on train frames | None, then causal smoothing of its answers | What an MLLM adds, per frame and smoothed |

Features are extracted once (a few minutes per backbone on the RTX 4090). The temporal models
train in minutes. The VLM arm is the expensive one: about 16 GB of weights, LoRA in 4-bit on a
subsample of train frames, and inference on every 5th test frame (about 8.6k frames).

## Metrics and protocol

- **Primary metric:** step recognition, macro-F1 over the 21 steps (balanced across classes).
  Phase macro-F1, frame accuracy and per-class F1 are also reported.
- **Unit of analysis = case.** With 5 test cases, the case-clustered bootstrap gives wide
  intervals; that's stated up front and not hidden by bootstrapping frames.
- **Development:** train on one official fold, validate on the other, then swap. Every
  selection (epochs, learning rate, smoothing window, prompt) happens there.
- **Pre-registration before test:** the endpoints below, the frozen configs, and the exact
  prompt.

## Draft endpoints (to be frozen after development)

| ID | Endpoint | Criterion |
|---|---|---|
| S1 (primary) | Step macro-F1, B − A (paired over test cases) | LB > 0 |
| S2 | Step macro-F1, A − A0 (does the temporal model help?) | LB > 0 |
| S3 | Step macro-F1, fine-tuned VLM (smoothed) − B | report with CI |
| S4 | Phase macro-F1, B − A | LB > 0 |
| Reported | Zero-shot VLM; per-class F1; latency per frame for each arm | report only |

**Expectation, stated before the data:** a temporal model on frozen features beats any
per-frame model, including a VLM, on step recognition. The backbone effect (S1) is the uncertain
one: DINOv2 didn't help keypoints on SurgPose, but phase recognition is closer to what
self-supervised features are good at.

## Risks

- **5 test cases:** any small difference will be inside the CI. The comparison is still worth
  pre-registering, but its power is low.
- **Rare steps:** macro-F1 is sensitive to two classes with a few hundred frames.
- **VLM output parsing:** answers must map to a label name. Unparseable answers count as wrong,
  and their rate is reported.
