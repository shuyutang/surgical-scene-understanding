# Scripts

Each script's docstring gives its exact usage, inputs and outputs. The prefix says which stage it
belongs to (the plans are in [docs/plans/](../docs/plans/)). Scripts that evaluate a pre-registered
test say so; they were run once on the test split, after the pre-registration was committed.

| Stage | Script | What it does |
|---|---|---|
| Data | `prepare_seg_data.py` | SISVSE split manifest and caches; EndoVis18 cache |
| | `prepare_surgpose.py` | SurgPose frame cache and keypoint manifest |
| | `check_rectification.py` | Residual vertical offset after rectification (B3) and its effect on SGM |
| **v2** Phase 1 | `train_seg.py`, `calibrate_seg.py`, `eval_seg.py` | Segmentation: train, temperature-scale, evaluate |
| Phases 2–3 | `train_kp.py`, `eval_kp.py` | Keypoint heatmaps; shape prior + MAP ablation |
| Phases 3b–5 | `cache_stage2_obs.py`, `tune_stage2.py`, `cache_tissue_planes.py`, `eval_stage2.py` | Full-rate observations, parameter selection on tune, SGM tissue planes, stage-2 endpoints |
| Phase 6 | `export_trt.py`, `eval_deploy.py`, `prepare_cpp_bench.py` | TensorRT export, parity and FP16 checks, C++ benchmark inputs |
| **v3** A0 | `eval_a0.py` | Focal-loss positives fix (tune) |
| Phase C | `fit_tool_geometry.py`, `eval_fusion.py`, `eval_v3c.py` | Train tool geometry; kinematics fusion on tune; pre-registered C1–C6 |
| Phase B | `eval_stereo_tune.py`, `eval_servct.py` | Stereo checkpoint selection on tune; pre-registered B1/B5 (RAFT-Stereo) and B6 (Fast-FoundationStereo) on SERV-CT |
| Phase A | `train_kp_vit.py`, `eval_kp_vit.py`, `eval_downstream_vit.py` | DINOv2 keypoints; pre-registered A and V endpoints |
| Phase E | `export_trt_vit.py`, `eval_deploy_vit.py`, `eval_e1_vit.py` | DINOv2 TensorRT export, development checks, pre-registered E1 |
| **v4** | `v4_sam2_feasibility.py`, `v4_make_masks.py`, `v4_mask_qa.py` | SAM 2 feasibility, masks for all videos, mask QA |
| | `v4_tool_library.py`, `v4_dev_fusion.py`, `eval_v4.py` | Instrument-type library (train), development variants (tune), pre-registered M1–M5 |
| Scene semantics | `grasp_prepare.py` | Frozen GraSP split and label arrays (official labels) |
| | `grasp_features.py`, `grasp_tcn.py` | Frozen ResNet-50 / DINOv2 frame features; causal MS-TCN and linear probes (development grid, final training) |
| | `grasp_vlm.py`, `eval_grasp.py` | Qwen3-VL-8B zero-shot and QLoRA; development comparison and pre-registered S1–S4 |
| **v5** | `v5_dev_stereo_tip.py`, `v5_tissue_memory.py` | Development only: stereo (RAFT or Fast-FoundationStereo) on instrument jaws; tissue memory |

Shared logic lives in the package (`surgscene.pipeline`, `surgscene.kp_eval`,
`surgscene.rectification`, `surgscene.video`, `surgscene.phase`, `surgscene.learned_stereo`), not in the scripts.
