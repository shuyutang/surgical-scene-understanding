# Practice Project v3: Foundation Features + Kinematics-Fused 3D Tool State

> **Status (2026-09-27):** plan only; nothing built. v2 (Phases 1–6) is complete at `0ca0675`:
> [v2 plan](plan_v2.md),
> [Phases 1–3 summary](../phase1-3_summary.md), [Phases 3b–6 summary](../phase3b-6_summary.md).

## Why v3, and why not the whole "SOTA diagram"

The starting sketch was the full modern stack:
- a ViT/SAM2 backbone with anatomy, tool and landmark heads;
- learned stereo;
- object tokens, a temporal transformer and a scene graph (action, target);
- geometry optimization;
- TensorRT/C++.

v3 keeps the parts that attack **measured v2 failures** and defers the rest. The deferred parts
have no public dataset that labels them together with stereo and kinematics, so they can't be
validated end to end.

| v2 finding | Root cause | v3 change | Endpoint |
|---|---|---|---|
| S1 fails: 3D tip error 10.0 mm vs 5 mm required | Stereo physics: 5.5 mm baseline at ~200 mm, so the error lies along the viewing ray | **Kinematics-fused factor graph** (Phase C) | C1, C2 |
| Kinematics + vision-fitted hand-eye gives 4.1 mm for the jaw pivot (S4b), but only as an offline fit on the first half of each trajectory | Not causal; tips not covered | Online hand-eye as a graph variable; tips from pivot + tool frame + vision jaw opening | C1, C4 |
| K1 fails: dev PCK@10 0.97 → test 0.69 | Background shift; ImageNet ResNet-34 features | **DINOv2 backbone** (Phase A) | A1, A2 |
| Focal loss: only 33.6% of training keypoints ever get a positive pixel | Sub-pixel Gaussian peak < 0.99 | Positive = argmax pixel of each target channel | A3 |
| Heatmap σ barely tracks error (Spearman 0.26 on traj 21); κ = 6.9 needed | σ is peak width, not error | Learned per-keypoint variance (NLL), then κ / conformal | A5, C3 |
| Per-keypoint Kalman locks onto a swapped tip; reports std 9 px while 59 px off | Keypoints filtered independently; gate widens when conf is low | Joint tool model: tips are tied to the pivot and tool frame | C5 |
| Phase 1 seg model doesn't transfer to SurgPose; instrument mask comes from keypoint capsules | Domain gap (beef labeled as instrument) | Instrument-mask head on the shared backbone, SAM2 pseudo-labels | A4 |
| SGM per-pixel depth scatter 5.5–7 mm; no tissue depth GT | Classical matcher on textureless, specular tissue | **Learned stereo, zero-shot** (Phase B), validated on SERV-CT | B1, B2 |
| Traj 21 has ~1.3 px (half-res) residual vertical misalignment after rectification | Per-trajectory calibration error | Rectification check + vertical correction before any stereo | B3 |
| Proximity alert uses the point estimate | Not built in v2 | Lower-confidence-bound alert (Phase P) | P1, P2 |

**Deferred to a roadmap section, not built:**
- the anatomy heads inside the 3D pipeline;
- object tokens, the temporal transformer, and the scene graph with action/target triplets;
- SAM2 video-memory tracking.

The "Alternatives considered" section below gives the reasons.

---

## Clinical use case and requirements (unchanged from v2)

> **Instrument proximity awareness:** real-time 3D distance from each instrument tip to the
> nearest tissue surface, with an alert below a threshold.

R1–R6 carry over from v2. v3 adds R7, because the alert now consumes the covariance.

| ID | Requirement | v2 status | v3 target |
|---|---|---|---|
| R1 | 3D tip ≤ 5 mm | **FAIL** (10.0 mm) | C1 |
| R2 | Critical-structure segmentation for proximity | Open: no dataset has anatomy + stereo | Still open (Phase F is monocular only) |
| R3 | p99 < 33 ms | PASS (2.59 ms, without SGM) | E3, **with stereo inside the budget** |
| R4 | No alert flicker | PASS (S5) | P2 |
| R5 | Graceful degradation | Partial (synthetic occluders) | Same protocol, plus C5 |
| R6 | FP16 doesn't degrade accuracy | PASS | E1 (a ViT in FP16 is riskier) |
| R7 (new) | Reported uncertainty is calibrated | PASS after κ (S3) | C3 |

---

## Data

Research-only practice data, as in v2.

| Source | Role in v3 | Local status (checked 2026-09-27) |
|---|---|---|
| SurgPose trajectories 0–33 | Phases A, C, P, E | Local: videos (L/R), keypoints (L/R), `bbox_{left,right}.json`, `api_cp_data.yaml` (tool Cartesian pose), `api_jp_data.yaml` (**6 arm joints per PSM, no jaw angle**), calibration |
| SurgPose SAM2 instance masks | Mask head (A4) | **Not in the local archives.** The v2 plan assumed them. Check Zenodo 15278516; otherwise generate them (next row) |
| SAM2.1 Hiera-B+ (`facebook/sam2.1-hiera-base-plus`) | Pseudo-masks, prompted with the GT bbox + GT keypoints | Cached locally (309 MB) |
| DINOv2 ViT-S/14 (`facebook/dinov2-small`) | Phase A backbone | Cached locally (85 MB). ViT-B/L and DINOv3 need download; DINOv3 is licence-gated |
| Learned stereo checkpoint | Phase B | **Not local.** Candidates: RAFT-Stereo, IGEV-Stereo/IGEV++, FoundationStereo. Check licence (some are non-commercial research only) and TensorRT exportability before choosing |
| SERV-CT | B1: tissue depth with CT ground truth | Not local; 16 stereo pairs. Download to confirm access |
| SCARED | Optional, larger depth GT | Access by request; optional |
| SISVSE, EndoVis18 | Phase F (monocular semantic track) | Local (v2 Phase 1) |

**Kinematics detail that shapes Phase C:**
- The tool frame in `api_cp_data` tracks the jaw pivot, as v2 S4 established.
- `api_jp_data` has no jaw joint, so **jaw opening must come from vision**.
- Jaw length L is estimated once on the tune set, from GT-triangulated tips in the tool frame.

### Splits and the test-reuse problem

- **Train:** 0–17. **Tune:** 18, 19, 20, 23, the v2 stage-2 tune split, which includes the background shift.
- **test2:** 21, 22, 24–33. The v2 stage-2 evaluation already used it once.
- **All 34 local trajectories have now been seen.** v3 still selects every design and
  hyperparameter on tune only, and pre-registers its endpoints before touching test2. But
  the v3 design was *motivated* by knowing test2's failure modes (the label-convention
  trajectories, tip swaps). So:
  - **v3 − v2 comparisons on test2 are reported as a second use of test2, not a fresh
    confirmatory test.** Every report says this in its first paragraph.
  - A confirmatory claim needs new data: new SurgPose trajectories if any are released, or other video
    variants of the same trajectories if they are published and usable. Look before
    Phase C's test run.
- **SERV-CT (B1) is fresh.** It's the only confirmatory test in v3.

---

## Architecture

```text
stereo frames (L, R) ─► rectify (+ per-trajectory vertical check, B3)
        │
        ├─► DINOv2 ViT (shared, L and R as a batch of 2)
        │      ├─ keypoint heatmap head (10 ch, fixed focal positives)
        │      ├─ per-keypoint log-variance head (NLL-trained)
        │      └─ instrument-mask head (SAM2 pseudo-labels)
        │
        └─► learned stereo (zero-shot) ─► disparity ─► local tissue plane (disparity space, v2 method)
                                                         │
robot joints ─► api_cp tool pose ──┐                     │
                                   ▼                     │
       per-instrument sliding-window factor graph        │
         variables: hand-eye T_cam←base (shared),        │
                    per-frame kinematic correction δ_t,  │
                    jaw opening θ_t                      │
         factors:   L/R reprojection of pivot + tips     │
                    (robust, learned σ), kinematic       │
                    prior, motion smoothness             │
                                   │                     │
                                   ▼                     ▼
                    3D tips + calibrated covariance ─► signed distance ± σ
                                                         │
                                                         ▼
                                     LCB alert with hysteresis ─► TensorRT FP16 + C++
```

---

## Phase A: Foundation backbone for keypoints and instrument mask (size: L)

**Build**
- **A0, the focal-loss fix alone, on the v2 U-Net/ResNet-34.** Positive = argmax pixel of each
  target channel. This isolates the bug from the backbone change. Cheap, so do it first.
- **The DINOv2 ViT-S/14 backbone**:
  - Input at the v2 cache resolution, padded to a multiple of 14 (e.g. 714×518, 51×37 tokens).
  - A light conv decoder that upsamples to half resolution, with one high-resolution skip from a
    shallow conv stem, because patch-14 tokens alone lose sub-pixel precision.
  - Heads: 10 heatmaps; 10 log-variances; 2 instrument masks (PSM1, PSM3).
  - Variants: frozen backbone; LoRA/partial fine-tune; ViT-B if ViT-S underfits.
- **Learned σ:** a Gaussian NLL on the decoded sub-pixel offset. Detach the heatmap path so the
  variance head can't game localization. Keep the Laplace σ as the baseline.
- **Mask pseudo-labels:**
  - SAM2.1 prompted with the GT bbox + GT keypoints on train and tune.
  - Hand-check about 50 frames, spread over trajectories and backgrounds.
  - Reject trajectories where SAM2 fails systematically.
- Augmentation as in v2 (decentered occluders + distractors), plus stronger color/texture
  augmentation for the background shift.

**Endpoints (test2 clean, left eye, every 5th frame, as in v2 D2)**

| ID | Endpoint | Criterion |
|---|---|---|
| A1 (primary) | PCK@10, v3 − v2 (paired per trajectory) | lower 95% bound > 0 |
| A2 | Mean argmax error, v3 − v2 | upper bound < 0 px |
| A3 | A0 (focal fix only) − v2: PCK@10 and mean error | report (ablation) |
| A4 | Mask IoU vs the hand-checked pseudo-labels; the mask replaces capsules in the SGM exclusion | report; replace only if tune IoU ≥ 0.85 |
| A5 | Spearman(σ, error): learned σ vs Laplace σ | learned > Laplace (point estimate) |
| A6 | Per-trajectory error, with the label-convention trajectories listed separately | report (no model is expected to fix label offsets) |

---

## Phase B: Learned stereo for tissue depth (size: M)

**Build**
- **B3 first:** measure the residual vertical offset per trajectory after rectification. Use
  two independent estimates: GT label pairs and image features. Fix the correction rule on tune
  (e.g. shift the right image when |offset| > 0.5 half-res px) and apply it to SGM and to
  learned stereo alike.
- Run the chosen learned stereo model **zero-shot** at half resolution and at native resolution.
- Reuse the v2 local-plane fit (annulus, instrument mask, disparity-space IRLS). Only the
  disparity source changes.

**Endpoints**

| ID | Endpoint | Criterion |
|---|---|---|
| B1 (confirmatory, fresh data) | SERV-CT depth error, learned − SGM (paired over the 16 pairs) | upper bound < 0 mm |
| B2 | SurgPose proxies (no GT): per-pixel scatter about the local plane, valid-pixel fraction, frame-to-frame plane stability | report |
| B3 | Per-trajectory vertical misalignment, before and after correction | report; correction rule fixed on tune |
| B4 | Latency at the chosen resolution (TensorRT FP16) | fits the E3 budget, or is reported as the dominant stage |

---

## Phase C: Kinematics-fused factor graph for the 3D tool state (size: L, the core of v3)

**Model (per instrument, sliding window of W frames, causal)**

- **Variables:**
  - hand-eye `T_cam←base` ∈ SE(3), shared across the trajectory;
  - per-frame kinematic correction `δ_t` ∈ se(3), which absorbs cable stretch and kinematic error;
  - jaw opening `θ_t`.
- **Geometry:**
  - pivot = `T_cam←base · T_base←tool(t) · exp(δ_t) · o_pivot`;
  - tips = pivot + R_tool,t · L · (±sin θ_t/2, 0, cos θ_t/2), with the axis convention checked on tune.
  - L, `o_pivot` and (optionally) the shaft/wrist offsets are fitted once on tune.
- **Factors:**
  - reprojection of pivot and tips into L and R, with a Cauchy loss weighted by learned σ
    (Laplace σ in the ablation);
  - kinematic prior δ_t ~ N(0, Σ_kin);
  - random-walk smoothness on δ_t and θ_t;
  - a weak prior on the hand-eye transform.
- **Hand-eye, online:**
  - Initialize from the first N seconds, with the v2 S4 robust registration run causally.
  - Then refine as a graph variable with a slow random walk.
  - Never use future frames.
- **Solver:** hand-written LM on the window, in the style of `shape.map_fit`, so the Eigen port
  is direct. Marginalize or fix frames that leave the window. Use GTSAM only if
  marginalization gets messy.
- **Output:** current-frame tips + covariance (inverse Hessian block), × κ fitted on tune.
- **Ablations:**
  - vision-only graph: same model without the kinematic factors;
  - v2 Kalman + triangulation;
  - kinematics only (no reprojection after hand-eye init).

**Endpoints (test2, all frames)**

| ID | Endpoint | Criterion |
|---|---|---|
| C1 (primary) | 3D tip error, factor graph (reference: GT-triangulated tips, as in v2 S1) | upper 95% bound ≤ 5.0 mm (R1) |
| C2 | 3D tip error, factor graph − v2 Kalman pipeline (paired) | upper bound < 0 mm |
| C3 | 95%-ellipsoid coverage after κ | 95% CI overlaps [0.90, 0.99] |
| C4 | Jaw pivot error, online causal hand-eye (vs the v2 S4b offline 4.07 mm) | report; UB ≤ 5.0 mm |
| C5 | Tip-swap rate (tip_a closer to GT tip_b by > 3 mm), graph vs v2 | report |
| C6 | Contribution of kinematics: vision-only graph − full graph | report |

Watch the reference noise: GT stereo labels agree to 0.46 px, which is about 1–2 mm of depth. Report C1 next to
that floor.

---

## Phase P: Proximity with a lower-confidence-bound alert (size: S)

- σ_d = √(nᵀ Σ_tip n + σ_plane²), with σ_plane from the IRLS fit (bootstrap over annulus pixels,
  or the analytic covariance).
- Alert on when d − k·σ_d < 10 mm; off with the v2 3 mm hysteresis. Choose k on tune for a target
  sensitivity.
- The v2 caveat stays: the reference distance uses the GT tip on the *same* plane, so this
  measures the tip's contribution only. B1 covers the surface's contribution separately.

| ID | Endpoint | Criterion |
|---|---|---|
| P1 | Sensitivity at matched specificity: LCB alert vs point-estimate alert | report (ROC over k) |
| P2 | Toggles per minute, v3 vs v2 Kalman + hysteresis | ratio UB ≤ 1.0 (non-inferior on flicker) |

---

## Phase E: Deployment (size: M)

- **ViT to TensorRT FP16.** Known risks: LayerNorm and softmax overflow in FP16. Keep those in
  FP32 if D1-style parity fails, and document which layers and why. Keep the v2 design:
  preprocessing and decoding inside the engine, no custom kernels.
- **Learned stereo to TensorRT**: this is the new dominant stage.
- **Factor graph in C++/Eigen**, with golden-output parity as in v2 D3.

| ID | Endpoint | Criterion |
|---|---|---|
| E1 (R6) | TensorRT FP16 − PyTorch FP32 mean keypoint error, test2 | upper bound ≤ +0.25 px (same margin as D2) |
| E2 | C++ vs Python factor graph on goldens | ≤ 1e-6 mm, covariance ≤ 1e-4 relative |
| E3 (R3) | End-to-end p99 per stereo pair, **including stereo** | < 33 ms |
| E4 | Embedded proxy: clock-locked or reduced-resolution run, or a stated scaling estimate | report |

**Latency budget (allocations, not measurements):**

| Stage | Budget (ms) |
|---|---|
| Decode L+R | 3 |
| H2D | 1 |
| Backbone + heads, L+R | 8 |
| Learned stereo | 14 |
| Plane fit + factor graph + proximity | 2 |
| Slack | 5 |
| **Total** | **< 33** |

If stereo doesn't fit, the options are: run stereo at lower resolution or every other frame
(tissue moves slowly), or run it in a region of interest around the tips. Measure these, don't
assume them.

---

## Phase F (separate track, optional): Monocular semantic segmentation (size: M)

This covers "semantic scene understanding" with a modern backbone, without pretending
it plugs into the 3D pipeline.
- DINOv2 + segmentation head on SISVSE with the Phase 1 harness, compared against the Phase 1 U-Net.
- Zero-shot on EndoVis18.
- The SISVSE test set was used once in Phase 1, so the same second-use caveat applies.

| ID | Endpoint | Criterion |
|---|---|---|
| F1 | SISVSE instrument Dice, DINOv2 − U-Net | report with CI |
| F2 | EndoVis18 zero-shot instrument Dice, DINOv2 − U-Net (the transfer gap) | lower bound > 0 |

---

## Alternatives considered

| Option | Decision | Why |
|---|---|---|
| SAM2-style video memory as the backbone | Deferred | Heavy memory attention for a 33 ms budget; DINOv2 + heads is simpler to export. Use SAM2 offline for pseudo-labels |
| Temporal transformer over object tokens | Deferred | Clip latency; no measured failure it fixes that the factor graph's motion factors don't; Kalman (v2) already gives −68% jitter cheaply |
| Scene graph / action-target triplets | Deferred | A different problem (workflow recognition); the data (CholecT50 family) is monocular, with no stereo or kinematics, so it can't be joined to the 3D state or validated end to end |
| Anatomy heads in the 3D pipeline | Deferred | No dataset has anatomy + stereo in the same frames (v2's known gap); R2 stays open |
| End-to-end learned 3D tool pose | Not now | No 3D pose GT beyond triangulated labels; the factor graph uses the robot's own 3D information, which a network would have to relearn |
| Render-and-compare with a CAD tool model | Stretch | Highest ceiling for vision-only disparity precision, but needs tool meshes and a differentiable renderer; revisit if C6 shows vision still dominates |
| GTSAM/Ceres instead of a hand-written LM | Hand-written first | Small problem; matches the v2 solver style and the Eigen port |

---

## Validation rules (carry over from v2, plus)

1. Pre-register this file's endpoint tables in `docs/validation_plan.md` (a "v3" section) and
   commit **before** any test2 or SERV-CT number is computed.
2. Unit of analysis = trajectory (test2) or stereo pair (SERV-CT). Clustered bootstrap, 2,000
   replicates, percentile CIs.
3. All selection on tune; record the selected config and its hash, as `configs/stage2_selected.json` does.
4. Every v3 report opens with the test2 second-use statement.
5. Post-hoc analyses go in dated amendments, labelled exploratory.
6. Traceability matrix R1–R7 → endpoint → result, written at the end (it's also v2 next step 4).

---

## Order, sizes and cut list

| Order | Work | Size | Depends on |
|---|---|---|---|
| 1 | A0 focal fix on the v2 U-Net (tune only) | S | — |
| 2 | B3 rectification check + correction rule | S | — |
| 3 | C: factor graph with **v2 keypoints** as input (tune) | L | 2 |
| 4 | A: DINOv2 backbone + heads (tune) | L | 1 |
| 5 | C with Phase A keypoints and learned σ (tune) | S | 3, 4 |
| 6 | B: learned stereo, SERV-CT (B1 can run as soon as it's downloaded) | M | 2 |
| 7 | P: LCB alert (tune) | S | 3, 6 |
| 8 | **Pre-register, then run test2 once** | S | all above |
| 9 | E: TensorRT + C++ | M | 8 (freeze the models first) |
| 10 | F: monocular semantic track | M | independent |

Step 3 comes before step 4 on purpose: the kinematics fusion is the largest expected gain and
doesn't need the new backbone.

**Minimum story to protect:** A0 → B3 → C (with v2 keypoints) → pre-registered test2 → E2/E3.
**Cut in this order:** F, render-and-compare, E4, learned σ (keep Laplace σ + κ), mask head
(keep capsules), learned stereo in TensorRT (keep it as a Python-evaluated result).

---

## Risks

| Risk | Mitigation |
|---|---|
| Patch-14 ViT features lose sub-pixel precision | High-resolution conv stem skip; compare PCK@5, not just PCK@10 |
| SAM2 pseudo-masks wrong on some backgrounds | Hand-check sample; A4 gate before replacing capsules |
| Kinematics/video time offset | Check the best lag on tune (v2 found 0 for labels; re-check for `api_cp`) |
| Tool-frame axis conventions for the jaw model | Fit and visualize on tune before building the graph |
| Learned stereo too slow, or not TensorRT-exportable | Pick the checkpoint for exportability; fall back to ROI or reduced rate |
| Test2 reuse weakens the claims | Stated up front; SERV-CT is the fresh test; look for new trajectories |

---

## Coverage map (what v3 adds)

| Topic | v3 |
|---|---|
| Modern vision architectures | Phase A (DINOv2), Phase B (learned stereo), Phase F |
| Probabilistic modeling, graphical models, MAP | Phase C factor graph (explicit, with marginalization) |
| SO(3)/SE(3), registration, camera models | Phase C (hand-eye on SE(3), se(3) corrections, reprojection) |
| Depth estimation / 3D reconstruction | Phase B with real GT (SERV-CT) |
| Clinically meaningful metrics + regulatory-style validation | R7, LCB alert, pre-registration, the test-reuse policy |
| ONNX/TensorRT, mixed precision, C++ | Phase E (ViT FP16 pitfalls, stereo in the budget, C++ graph parity) |
