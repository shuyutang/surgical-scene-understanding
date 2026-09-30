# Documents

**Start here:** [traceability.md](traceability.md) maps each requirement to its tests and
results, and [conventional_vs_modern.md](conventional_vs_modern.md) compares the approaches
component by component.

## Plans

| Plan | Scope |
|---|---|
| [plans/plan_v1.md](plans/plan_v1.md) | First outline |
| [plans/plan_v2.md](plans/plan_v2.md) | Phases 1–6: segmentation, keypoints, shape prior + MAP, Kalman filter, stereo, TensorRT/C++ |
| [plans/plan_v3.md](plans/plan_v3.md) | Kinematics fusion, learned stereo, DINOv2 keypoints, deployment |
| [plans/plan_v4.md](plans/plan_v4.md) | SAM 2 masks as a measurement |
| [plans/plan_scene_semantics.md](plans/plan_scene_semantics.md) | Separate track: phase and step recognition on GraSP |

## Validation

- [validation_plan.md](validation_plan.md): every pre-registration, in order, with its prior
  exposure stated, plus dated post-hoc amendments.

## Reports, in order

| Stage | Report |
|---|---|
| v2 Phase 1 | [phase1_test_report.md](phase1_test_report.md) |
| v2 Phases 2–3 | [phase23_test_report.md](phase23_test_report.md), post hoc: [decentered](phase23_posthoc_decentered_report.md), [off-center diagnostic](phase23_posthoc_diagnostic_offcenter_report.md) |
| v2 Phases 3b–5 | [stage2_tune_report.md](stage2_tune_report.md), [stage2_test_report.md](stage2_test_report.md) |
| v2 Phase 6 | [phase6_deploy_report.md](phase6_deploy_report.md) |
| v2 summaries | [phase1-3_summary.md](phase1-3_summary.md), [phase3b-6_summary.md](phase3b-6_summary.md) |
| v3 | [v3_progress.md](v3_progress.md) (steps 1–7), [v3c_test_report.md](v3c_test_report.md), [v3b_servct_report.md](v3b_servct_report.md), [v3a_test_report.md](v3a_test_report.md), [v3_step5_e1_b5_report.md](v3_step5_e1_b5_report.md) |
| v4 | [v4_step0_sam2_feasibility.md](v4_step0_sam2_feasibility.md), [v4_report.md](v4_report.md) |
| v5 | [v5_step0_stereo_tip.md](v5_step0_stereo_tip.md), [v5_step1_tissue_memory.md](v5_step1_tissue_memory.md) (development); [v5_step2_fast_foundationstereo.md](v5_step2_fast_foundationstereo.md) (B6 on SERV-CT) |
| Scene semantics | [grasp_test_report.md](grasp_test_report.md) (GraSP phase and step recognition) |

The `*_results.json` files next to some reports are the raw outputs the reports quote.
