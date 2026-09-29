# Practice Project v4: Instrument Masks as a Measurement (draft outline, 2026-09-28)

v3 is closed: its results are reported against pre-registrations in [docs/traceability.md](docs/traceability.md).
v4 adds one new measurement modality, SAM 2 instrument masks, and asks one question:
**do masks fix the 3D error that 10 keypoints can't, especially on the long-jaw tool?**

## Why

- R1 (3D tip ≤ 5 mm) fails at 5.0–5.3 mm overall. It's met on standard instruments (3.86 mm)
  and fails on the long-jaw tool (12.3 mm), where the per-trajectory tool geometry, fitted from
  10 keypoints, is the suspected cause.
- Masks give dense constraints on the instrument's silhouette and shaft axis. Render-and-compare
  (project the kinematically posed tool model; match it to the mask) uses them directly.
- Step 0 ([docs/v4_step0_sam2_feasibility.md](docs/v4_step0_sam2_feasibility.md)) showed that
  SAM 2 tracks the instrument body about 100% of the time from one automatic prompt, with no
  swaps, but misses thin jaw tips. So masks go on the body and tool geometry; tips stay with the
  keypoint detector.

## Steps (sizes S/M/L)

| Step | What | Size | Data |
|---|---|---|---|
| 0 | SAM 2 feasibility, prompt modes, latency, memory | S | tune (done) |
| 1 | Masks for all trajectories, offline, both eyes; the same automatic prompt | S | all |
| 2 | Tool silhouette model: a cylinder shaft + wrist + jaw primitives from `ToolGeometry`, projected with the Phase C hand-eye | M | tune |
| 3 | Render-and-compare term in the online tool-geometry fit: a chamfer or distance-transform residual between the projected silhouette and the mask edge, added to the reprojection LM | L | tune |
| 4 | Mask visibility gate for the Kalman update (wrist and jaw pivot only) | S | tune |
| 5 | Pre-register, then evaluate on **fresh data** | M | see below |
| 6 | Latency: batched objects, compiled memory attention, SAM 2 at a reduced rate | M | tune |

## Endpoints (drafted; to be frozen before any test run)

- **M1 (primary, R1):** mean 3D tip error with mask-augmented tool geometry vs v3 Phase C,
  paired; success is an upper bound < 0.
- **M2:** long-jaw subgroup, 3D tip error ≤ 5 mm.
- **M3:** 2D non-inferiority on standard instruments, margin +0.5 px.
- **M4:** coverage of the κ-inflated 95% ellipsoid.
- **M5 (R3):** end-to-end latency with masks at the chosen rate.

## The data problem (decide before step 5)

SurgPose test2 has been used three times, and all 34 trajectories have been seen. A confirmatory
test needs one of:
1. new SurgPose-like recordings (none public that we know of);
2. an honestly labelled "fourth use" of test2, reported as supportive rather than confirmatory;
3. a held-out set carved from what hasn't been used for any choice: nothing qualifies cleanly.

**Current lean:** option 2, disclosed up front, with the hypothesis and analysis frozen first.

## Not in v4

Learned temporal models and learned fusion stay in
[docs/conventional_vs_modern.md](docs/conventional_vs_modern.md) as literature. Real-time SAM 2
on both eyes is step 6, and optional.
