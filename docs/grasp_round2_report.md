# Scene semantics on GraSP, round 2: surgical SSL backbone (S5) and short-term recognition (ST1–ST4), 2026-09-29

Pre-registration: `docs/validation_plan.md`, "scene semantics round 2" (commit `2e69e18`). Round 1:
[grasp_test_report.md](grasp_test_report.md). Outputs: `runs/grasp_test_s5/`, `runs/grasp_st_test/`.

**Questions.**
1. Does a surgical, self-supervised backbone beat DINOv2 for step recognition?
2. For "what is each instrument, and what is it doing", does a fine-tuned VLM (SurgMLLM-style) beat
   a frozen-feature classifier?

**Answers.**
1. **No.** EndoSSL (laparoscopy MSN) is far below DINOv2, and even below ImageNet ResNet-50.
2. **No.** The VLM nearly matches the classifier on instrument type but is well behind on actions.
   Every verdict matches the direction and rough size seen in development.

## What was tested, and what couldn't be

- **SurgVISTA** (surgical video foundation model): the backbone isn't released (its Hugging Face
  repo is empty). Not tested.
- **SurgMotion** (V-JEPA 2 on 15M surgical frames): the weights are gated; access was requested and
  hadn't been granted when this ran. Not tested.
- **EndoSSL ViT-L/16** (Hirsch et al., MICCAI 2023; SurgVISTA's image teacher): released and
  tested.
  - Its PyTorch conversion equals the official JAX checkpoint.
  - With raw 0–255 input it reproduces the official TF model's output (cosine 0.9998).
- **SurgMLLM** (InternVL2.5-4B on cholecystectomy): no released weights found. The VLM arm is
  Qwen3-VL-8B fine-tuned the same way (QLoRA), prompted with the instance's box drawn on the frame.

## S5: EndoSSL vs DINOv2 for steps (2nd use of the GraSP test cases)

| ID | Endpoint | Result [95% CI] | Cases > 0 | Criterion | Verdict |
|---|---|---|---|---|---|
| S5 | Step macro-F1, E (EndoSSL + causal MS-TCN) − B (DINOv2 + MS-TCN) | **−0.230** [−0.300, −0.156] | 0/5 | LB > 0 | **FAIL** |
| reported | Phase macro-F1, E − B | −0.236 [−0.345, −0.117] | 0/5 | | |
| reported | Step macro-F1, E − E0 (temporal model on EndoSSL) | +0.056 [+0.024, +0.093] | 5/5 | | |

| Arm (5 test cases) | Step F1 | Phase F1 | Step acc | Phase acc |
|---|---|---|---|---|
| B: DINOv2 ViT-B/14 + MS-TCN | **0.469** | **0.660** | **0.589** | **0.720** |
| A: ResNet-50 (ImageNet) + MS-TCN (round 1) | 0.401 | 0.614 | 0.544 | 0.699 |
| E: EndoSSL ViT-L/16 + MS-TCN | 0.238 | 0.424 | 0.473 | 0.607 |
| E0: EndoSSL, per-frame | 0.182 | 0.340 | 0.295 | 0.451 |

Development predicted this (E 0.271 vs B 0.403). The temporal model still helps on EndoSSL
features, but the features themselves separate prostatectomy steps poorly: accuracy is moderate,
and macro-F1 (the rare steps) collapses.

## ST1–ST4: instrument type and atomic actions per instance (first use of the short-term labels)

**Task.** 2,861 test instances on 1,125 keyframes (every 35 s), with the ground-truth box given:
- instrument type: 7 classes;
- atomic actions: 14 classes, multi-label.

| ID | Endpoint | Result [95% CI] | Cases > 0 | Criterion | Verdict |
|---|---|---|---|---|---|
| ST1 (primary) | Action macro-F1, fine-tuned VLM − DINOv2 head | **−0.095** [−0.117, −0.065] | 0/5 | LB > 0 | **FAIL** |
| ST2 | Instrument macro-F1, fine-tuned VLM − DINOv2 head | −0.033 [−0.063, +0.012] | 1/5 | LB > 0 | **FAIL** |
| ST3 | Action macro-F1, DINOv2 head − ResNet-50 head | +0.027 [+0.001, +0.050] | 4/5 | LB > 0 | **PASS** |
| ST4 | Action macro-F1, DINOv2 with the previous second's crop − single frame | +0.002 [−0.021, +0.025] | 2/5 | LB > 0 | **FAIL** |

| Arm (mean over 5 test cases) | Instrument F1 | Action F1 | Action mAP |
|---|---|---|---|
| ResNet-50 head, single / +prev | 0.694 / 0.691 | 0.246 / 0.234 | 0.254 / 0.273 |
| **DINOv2 head, single** / +prev | **0.818** / 0.778 | 0.273 / **0.275** | 0.310 / **0.327** |
| EndoSSL head, single / +prev | 0.592 / 0.578 | 0.211 / 0.221 | 0.254 / 0.269 |
| Qwen3-VL-8B, zero-shot | 0.187 | 0.084 | — |
| Qwen3-VL-8B, QLoRA fine-tuned | 0.785 | 0.178 | — |

## Reading

- **Surgical pretraining isn't automatically in-domain.** EndoSSL was trained on private
  laparoscopy video. GraSP is da Vinci prostatectomy: different optics,
  anatomy and instruments. The large, general DINOv2 corpus transfers better than the smaller,
  "surgical" one. The fair check of EndoSSL itself is on its own domain: Cholec80, where it
  reports 84 F1. That hasn't been run here, so this result is "doesn't transfer to robotic
  prostatectomy", not "EndoSSL is weak".
- **A fine-tuned VLM learns what instruments look like, not what they're doing.**
  - Instrument type: 0.785, close to the DINOv2 head (0.818).
  - Actions: 0.178 against 0.273. On test it uses only 4 of the 14 actions: Hold (56% of
    instances vs 32% in GT), Still (81% vs 61%), Suction and Travel. It never predicts the ten
    rare ones (Cut, Grasp, Pull, Push…, each 0.3–6% of GT), and macro-F1 counts every one of them.
    Actions are about motion and contact, and the VLM answers from one image.
- **Actions are hard for everyone.** Action macro-F1 is below 0.3 for every arm. The motion cue
  (the same box one second earlier) didn't help as a concatenated feature (ST4). A proper video
  model over a few seconds is the obvious next step. That is what SurgMotion would test.
- **The backbone effect repeats.** DINOv2 > ResNet-50 on actions (ST3), steps (S1), and
  instruments (0.818 vs 0.694, reported).

## Caveats

- Five test cases, one centre, one procedure.
- The GraSP test cases have now been used twice (long-term labels in S1–S5, short-term labels here
  for the first time).
- GT boxes are given, so this measures recognition only; GraSP's official task is detection.
- The short-term development grid was widened once after its first run chose edge values (both
  reports kept, stated in the pre-registration).
