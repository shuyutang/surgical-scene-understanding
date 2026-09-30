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
| | `grasp_vlm.py`, `eval_grasp.py` | Qwen3-VL-8B zero-shot and QLoRA (steps, or instances with `--task instances`); development comparison, pre-registered S1–S5 |
| | `grasp_shortterm.py`, `eval_grasp_shortterm.py` | Short-term: crop features, MLP heads for instrument and actions per GT instance; pre-registered ST1–ST4 |
| **v5** | `v5_dev_stereo_tip.py`, `v5_tissue_memory.py` | Development only: stereo (RAFT or Fast-FoundationStereo) on instrument jaws; tissue memory |

Shared logic lives in the package (`surgscene.pipeline`, `surgscene.kp_eval`,
`surgscene.rectification`, `surgscene.video`, `surgscene.phase`, `surgscene.learned_stereo`, `surgscene.backbones`, `surgscene.grasp_st`), not in the scripts.

## Reproducing

Test-set commands are marked *once*: each was run a single time, after its pre-registration
(`docs/validation_plan.md`) was committed. Setup and data: the top-level README.

```bash
# data caches
uv run python scripts/prepare_seg_data.py && uv run python scripts/prepare_surgpose.py

# v2: segmentation, keypoints, structured inference, temporal, stereo
uv run python scripts/train_seg.py configs/seg_unet_r34.yaml
uv run python scripts/train_kp.py configs/kp_unet_r34_decentered.yaml
uv run python scripts/cache_stage2_obs.py runs/<kp_run>/best.pt
uv run python scripts/tune_stage2.py && uv run python scripts/cache_tissue_planes.py
uv run python scripts/eval_stage2.py --split tune            # --split test2: once

# v3: kinematics fusion (C), learned stereo (B), DINOv2 keypoints (A)
uv run python scripts/fit_tool_geometry.py
uv run python scripts/eval_v3c.py --split tune               # --split test2: once
uv run --group stereo python scripts/eval_stereo_tune.py && uv run --group stereo python scripts/eval_servct.py
uv run python scripts/train_kp_vit.py configs/kp_vit_s.yaml
uv run python scripts/eval_kp_vit.py --split tune            # --split test2: once

# v4: SAM 2 masks, instrument-type library, jaw-length constraint
uv run --group sam python scripts/v4_make_masks.py && uv run python scripts/v4_mask_qa.py
uv run python scripts/v4_tool_library.py
uv run python scripts/eval_v4.py --split tune                # --split test2: once

# v5 (development, tune only): stereo on instruments, tissue memory
uv run --group stereo --group sam python scripts/v5_tissue_memory.py
# v5 step 2: Fast-FoundationStereo (selection on tune, jaw check on tune, B6 on SERV-CT once)
uv run --group stereo python scripts/eval_stereo_tune.py --out runs/v5_ffs_tune --select-prefix ffs: \
    --methods middlebury realtime@4 ffs:23-36-37@8 ffs:23-36-37@4 ffs:20-26-39@8 ffs:20-30-48@4
uv run --group stereo --group sam python scripts/v5_dev_stereo_tip.py --models middlebury@32 ffs:23-36-37@8 --out runs/v5_ffs_tip
uv run --group stereo --group sam python scripts/v5_dev_stereo_tip.py --oracle --models ffs:23-36-37@8 --out runs/v5_ffs_tip
uv run --group stereo python scripts/eval_servct.py --method ffs:23-36-37@8 --reference middlebury@32

# scene semantics (GraSP): features, temporal models, VLM, endpoints
uv run python scripts/grasp_prepare.py
uv run python scripts/grasp_features.py --backbone resnet50 && uv run python scripts/grasp_features.py --backbone dinov2_b14
uv run python scripts/grasp_tcn.py dev && uv run python scripts/grasp_tcn.py final
uv run --group vlm python scripts/grasp_vlm.py train --tag final_ft --cases <8 train cases> --stride 10
uv run --group vlm python scripts/grasp_vlm.py predict --tag test_ft --adapter runs/grasp_vlm/final_ft/adapter --cases <test cases> --stride 5
uv run python scripts/eval_grasp.py test                     # once
# round 2: EndoSSL backbone (S5), short-term instrument and action recognition (ST1-ST4)
uv run python scripts/grasp_features.py --backbone endossl_l16   # weights: see src/surgscene/backbones.py
uv run python scripts/grasp_tcn.py dev --backbones endossl_l16 && uv run python scripts/grasp_tcn.py final --backbones endossl_l16
uv run python scripts/grasp_shortterm.py feats && uv run python scripts/grasp_shortterm.py dev && uv run python scripts/grasp_shortterm.py final
uv run --group vlm python scripts/grasp_vlm.py train --task instances --tag st_final_ft --split train
uv run --group vlm python scripts/grasp_vlm.py predict --task instances --tag st_test_ft --adapter runs/grasp_vlm/st_final_ft/adapter --split test
uv run python scripts/eval_grasp.py test-s5 && uv run python scripts/eval_grasp_shortterm.py test   # once

# deployment: TensorRT FP16 engines, C++ runtime and latency benchmark
uv sync --group deploy && cpp/scripts/fetch_deps.sh
uv run --group deploy python scripts/export_trt.py runs/<kp_run>/best.pt
uv run --group deploy python scripts/eval_deploy.py
cmake -S cpp -B cpp/build && cmake --build cpp/build -j && cpp/build/tests
```
