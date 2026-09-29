"""Stage-2 evaluation: Phase 3b (gated MAP), Phase 4 (temporal), Phase 5 (3D) on full-rate SurgPose.

  uv run python scripts/eval_stage2.py --split tune     # sanity / threshold setting
  uv run python scripts/eval_stage2.py --split test2    # once, after docs/validation_plan.md is committed

All parameters come from configs/stage2_selected.json (tune only). Unit of analysis = trajectory;
CIs are trajectory-clustered bootstrap percentile intervals (2,000 replicates).
"""

import argparse
import json
import time

import numpy as np
import yaml
from tqdm import tqdm

from surgscene.evaluation import (
    boot,
    ci,
)
from surgscene.geometry import register_rigid_robust
from surgscene.pipeline import cached_gated
from surgscene.stage2 import RAW, ROOT, SPLITS, TIP, load_obs, load_rig, load_selected, shape_models, triangulate_seq
from surgscene.temporal import KFParams, filter_keypoints, jitter

EST = ROOT / "data/cache/stage2_est"
PLANES = ROOT / "data/cache/tissue_planes"
CHI2_3_95 = 7.8147
LAGS = [0, 3, 6]
ALERT_MM, HYST_MM = 10.0, 3.0  # proximity alert: tip within 10 mm of the local tissue plane
FPS = 30.0


def err(P, gt):
    return np.linalg.norm(P - gt, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test2"], required=True)
    args = ap.parse_args()
    trajs = SPLITS[args.split]
    sel = load_selected()
    hp, kfp, kappa = sel["gated"], KFParams(**sel["kf"]), sel["cov_inflation_kappa"]
    models, plaus_max = shape_models()
    out_dir = ROOT / "runs" / f"eval_stage2_{args.split}"
    out_dir.mkdir(exist_ok=True)

    per = {k: [] for k in [
        # phase 3b (every 5th frame, left)
        "arg_clean", "gat_clean", "arg_occ_kp", "gat_occ_kp", "d_gat_clean", "d_gat_occ_kp",
        "pck5_arg", "pck5_gat",
        # phase 4 (all frames, left)
        "kf_clean", "d_kf_clean", "jit_gat", "jit_kf", "kf_occ_kp", "d_kf_occ_kp", "recover_gat", "recover_kf",
        "unknown_frac", "err_unknown", "err_known", "nis",
        *[f"kf_lag{L}" for L in LAGS], *[f"jit_lag{L}" for L in LAGS],
        # phase 5 (every 5th frame)
        "tip3d_kf", "tip3d_gat", "d_tip3d_kf_minus_gat", "tip3d_depth_err", "tip3d_lateral_err", "cover95", "gt_reproj",
        # phase 5 hand-eye (jaw pivot, second half of each trajectory)
        "pivot_vision", "pivot_kin", "d_pivot_kin_minus_vision",
        # phase 5b proximity (all frames)
        "dist_err_kf", "dist_err_fw", "alert_sens_kf", "alert_spec_kf", "alert_sens_fw", "alert_spec_fw",
        "alert_frac_ref",
    ]}
    toggles = {k: [] for k in ["ref", "fw", "kf", "kf_hyst"]}
    he_rot = []
    timing = []
    for t in tqdm(trajs, desc=args.split):
        O = load_obs(t, "left")
        OR = load_obs(t, "right")
        gt = O["gt"]
        g_clean = cached_gated(t, "left", "", models, hp)
        g_occ = cached_gated(t, "left", "_occ", models, hp)
        g_right = cached_gated(t, "right", "", models, hp)
        f5 = slice(0, None, 5)

        # --- Phase 3b
        Ea, Eg = err(O["kp"], gt), err(g_clean, gt)
        occ = O["occluded"]
        Ea_o, Eg_o = err(O["kp_occ"], gt), err(g_occ, gt)
        per["arg_clean"].append(np.nanmean(Ea[f5], 1))
        per["gat_clean"].append(np.nanmean(Eg[f5], 1))
        per["d_gat_clean"].append(np.nanmean(Eg[f5] - Ea[f5], 1))
        per["arg_occ_kp"].append(np.where(occ, Ea_o, np.nan)[f5].ravel())
        per["gat_occ_kp"].append(np.where(occ, Eg_o, np.nan)[f5].ravel())
        per["d_gat_occ_kp"].append(np.where(occ, Eg_o - Ea_o, np.nan)[f5].ravel())
        fin = np.isfinite(Ea[f5])
        per["pck5_arg"].append(np.where(fin, (Ea[f5] <= 5).astype(float), np.nan).ravel())
        per["pck5_gat"].append(np.where(fin, (Eg[f5] <= 5).astype(float), np.nan).ravel())

        # --- Phase 4 (causal filter + fixed-lag variants)
        t0 = time.perf_counter()
        kf, std, unknown, tracks = filter_keypoints(g_clean, O["sigma"], O["conf"], kfp)
        timing.append((time.perf_counter() - t0) / len(gt) * 1e3)
        Ek = err(kf, gt)
        per["kf_clean"].append(np.nanmean(Ek, 1))
        per["d_kf_clean"].append(np.nanmean(Ek - Eg, 1))
        per["jit_gat"].append(jitter(g_clean, gt).ravel())
        per["jit_kf"].append(jitter(kf, gt).ravel())
        per["unknown_frac"].append(np.where(np.isfinite(gt).all(-1), unknown.astype(float), np.nan).ravel())
        per["err_unknown"].append(np.where(unknown, Ek, np.nan).ravel())
        per["err_known"].append(np.where(~unknown, Ek, np.nan).ravel())
        per["nis"].append(np.concatenate([tr.nis for tr in tracks]))
        kf_o, *_ = filter_keypoints(g_occ, O["sigma_occ"], O["conf_occ"], kfp)
        Ek_o = err(kf_o, gt)
        per["kf_occ_kp"].append(np.where(occ, Ek_o, np.nan).ravel())
        per["d_kf_occ_kp"].append(np.where(occ, Ek_o - Eg_o, np.nan).ravel())
        # recovery: the 15 frames right after an episode ends (occluder gone)
        after = np.zeros_like(occ)
        for k in range(occ.shape[1]):
            ends = np.flatnonzero(occ[:-1, k] & ~occ[1:, k]) + 1
            for e in ends:
                after[e:e + 15, k] = True
        per["recover_gat"].append(np.where(after, Eg_o, np.nan).ravel())
        per["recover_kf"].append(np.where(after, Ek_o, np.nan).ravel())
        for L in LAGS:
            pL, *_ = filter_keypoints(g_clean, O["sigma"], O["conf"], kfp, lag=L)
            per[f"kf_lag{L}"].append(np.nanmean(err(pL, gt), 1))
            per[f"jit_lag{L}"].append(jitter(pL, gt).ravel())

        # --- Phase 5 (every 5th frame): 3D tips from the causal filter of each eye
        rig = load_rig(t)
        kfR, stdR, *_ = filter_keypoints(g_right, OR["sigma"], OR["conf"], kfp)
        X, C, _ = triangulate_seq(rig, kf[f5], kfR[f5], std[f5], stdR[f5])
        Xg, _, _ = triangulate_seq(rig, g_clean[f5], g_right[f5], O["sigma"][f5], OR["sigma"][f5])
        one = np.ones(gt[f5].shape[:2])
        G, _, rp = triangulate_seq(rig, gt[f5], OR["gt"][f5], one, one)
        per["gt_reproj"].append(rp.ravel())
        good = rp <= 3.0  # GT label pairs inconsistent across eyes are excluded (pre-specified)
        for arm, (a, b) in TIP.items():
            ok = good[:, a] & good[:, b]
            g3 = (G[:, a] + G[:, b]) / 2
            for name, P in [("tip3d_kf", X), ("tip3d_gat", Xg)]:
                e = np.where(ok[:, None], (P[:, a] + P[:, b]) / 2 - g3, np.nan)
                per[name].append(np.linalg.norm(e, axis=1))
                if name == "tip3d_kf":
                    ray = g3 / np.linalg.norm(g3, axis=1, keepdims=True)  # viewing ray ~ depth direction
                    along = np.abs((e * ray).sum(1))
                    per["tip3d_depth_err"].append(along)
                    per["tip3d_lateral_err"].append(np.sqrt(np.maximum(np.linalg.norm(e, axis=1) ** 2 - along**2, 0)))
                    S = kappa * (C[:, a] + C[:, b]) / 4
                    d2 = np.full(len(e), np.nan)
                    for i in np.flatnonzero(np.isfinite(e).all(1) & np.isfinite(S).all((1, 2))):
                        d2[i] = e[i] @ np.linalg.solve(S[i], e[i])
                    per["cover95"].append(np.where(np.isfinite(d2), (d2 <= CHI2_3_95).astype(float), np.nan))
            per["d_tip3d_kf_minus_gat"].append(per["tip3d_kf"][-1] - per["tip3d_gat"][-1])

        # --- Phase 5 hand-eye: robot kinematics -> camera, registered from *vision* estimates on the
        # first half of the trajectory, evaluated on the second half against GT (jaw pivot keypoint,
        # which the kinematic tool frame tracks; chosen on tune)
        cp = yaml.load(open(RAW / f"{t:06d}/api_cp_data.yaml"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        fr = np.arange(len(gt))[f5]
        first = fr < len(gt) // 2
        for arm, (a, _) in TIP.items():
            piv = a - 1
            Kin = np.array([np.asarray(cp[str(i)][arm]["t"]) * 1000 for i in fr])
            ok = np.isfinite(X[:, piv]).all(1) & np.isfinite(G[:, piv]).all(1) & good[:, piv]
            Rhe, the = register_rigid_robust(Kin[first & ok], X[first & ok, piv])
            he_rot.append(Rhe)
            ev = ~first & ok
            e_kin = np.linalg.norm(Kin @ Rhe.T + the - G[:, piv], axis=1)
            e_vis = np.linalg.norm(X[:, piv] - G[:, piv], axis=1)
            per["pivot_kin"].append(np.where(ev, e_kin, np.nan))
            per["pivot_vision"].append(np.where(ev, e_vis, np.nan))
            per["d_pivot_kin_minus_vision"].append(np.where(ev, e_kin - e_vis, np.nan))

        # --- Phase 5b: tip-to-tissue distance (all frames) against the SGM tissue plane
        P = np.load(PLANES / f"{t:06d}.npz")
        tipi = [3, 4, 8, 9]
        onef = np.ones(gt.shape[:2])[:, tipi]
        G4, _, _ = triangulate_seq(rig, gt[:, tipi], OR["gt"][:, tipi], onef, onef)
        K4, _, _ = triangulate_seq(rig, kf[:, tipi], kfR[:, tipi], std[:, tipi], stdR[:, tipi])
        F4, _, _ = triangulate_seq(rig, g_clean[:, tipi], g_right[:, tipi], O["sigma"][:, tipi], OR["sigma"][:, tipi])
        for i, arm in enumerate(TIP):
            n, c = P["plane"][:, i, :3], P["plane"][:, i, 3]
            dist = lambda X4: ((((X4[:, 2 * i] + X4[:, 2 * i + 1]) / 2) @ P["R1"].T) * n).sum(1) - c
            d_ref, d_kf, d_fw = dist(G4), dist(K4), dist(F4)
            valid = np.isfinite(d_ref)
            per["dist_err_kf"].append(np.where(valid, np.abs(d_kf - d_ref), np.nan))
            per["dist_err_fw"].append(np.where(valid, np.abs(d_fw - d_ref), np.nan))
            a_ref = d_ref < ALERT_MM
            per["alert_frac_ref"].append(np.where(valid, a_ref.astype(float), np.nan))
            for name, dd in [("kf", d_kf), ("fw", d_fw)]:
                a = np.nan_to_num(dd, nan=np.inf) < ALERT_MM  # no estimate -> no alert
                per[f"alert_sens_{name}"].append(np.where(valid & a_ref, a.astype(float), np.nan))
                per[f"alert_spec_{name}"].append(np.where(valid & ~a_ref, (~a).astype(float), np.nan))
            minutes = len(d_ref) / FPS / 60
            toggles["ref"].append(np.abs(np.diff(np.nan_to_num(d_ref, nan=np.inf) < ALERT_MM)).sum() / minutes)
            toggles["fw"].append(np.abs(np.diff(np.nan_to_num(d_fw, nan=np.inf) < ALERT_MM)).sum() / minutes)
            toggles["kf"].append(np.abs(np.diff(np.nan_to_num(d_kf, nan=np.inf) < ALERT_MM)).sum() / minutes)
            state, st = False, []
            for v in np.nan_to_num(d_kf, nan=np.inf):
                state = v < ALERT_MM if not state else v < ALERT_MM + HYST_MM
                st.append(state)
            toggles["kf_hyst"].append(np.abs(np.diff(np.array(st))).sum() / minutes)

    R = {"split": args.split, "trajectories": trajs, "selected": {k: sel[k] for k in ["gated", "kf", "cov_inflation_kappa"]},
         "kf_ms_per_frame_10kp_python": float(np.mean(timing))}
    R["metrics"] = {k: boot(v) for k, v in per.items() if k not in ("nis", "gt_reproj")}
    # toggles were appended per (trajectory, arm); cluster them by trajectory
    tog = {k: [np.array(v[2 * i:2 * i + 2]) for i in range(len(v) // 2)] for k, v in toggles.items()}
    R["toggles_per_min"] = {k: boot(v) for k, v in tog.items()}
    R["derived"] = {
        "jitter_ratio_kf_over_gated": ratio_boot(per["jit_kf"], per["jit_gat"]),
        "d_pck5_gated_minus_argmax": diff_boot(per["pck5_gat"], per["pck5_arg"]),
        "d_recover_kf_minus_gated": diff_boot(per["recover_kf"], per["recover_gat"]),
        "toggle_ratio_kfhyst_over_fw": ratio_boot(tog["kf_hyst"], tog["fw"]),
    }
    R["alert_mm"], R["hysteresis_mm"] = ALERT_MM, HYST_MM
    nis = np.concatenate(per["nis"])
    R["nis_mean"] = float(np.nanmean(nis))
    rp = np.concatenate(per["gt_reproj"])
    rp = rp[np.isfinite(rp)]
    R["gt_reproj_rms"] = {"median": float(np.median(rp)), "p95": float(np.percentile(rp, 95)),
                          "frac_gt3px": float(np.mean(rp > 3))}
    R["criteria"] = criteria(R)
    (out_dir / "results.json").write_text(json.dumps(R, indent=1))
    txt = render(R)
    (out_dir / "report.md").write_text(txt)
    print(txt)


def criteria(R):
    """Pre-specified in docs/validation_plan.md (stage 2). Each: (id, description, value, rule, pass)."""
    m, d = R["metrics"], R["derived"]
    c = [
        ("G1", "gated − argmax mean error, clean (px)", m["d_gat_clean"], "UB < 0", m["d_gat_clean"]["hi"] < 0),
        ("G2", "gated − argmax PCK@5", d["d_pck5_gated_minus_argmax"], "LB ≥ −0.02",
         d["d_pck5_gated_minus_argmax"]["lo"] >= -0.02),
        ("G3", "gated − argmax, occluded keypoints (px)", m["d_gat_occ_kp"], "UB < 0", m["d_gat_occ_kp"]["hi"] < 0),
        ("T1", "jitter ratio KF / gated", d["jitter_ratio_kf_over_gated"], "UB ≤ 0.6",
         d["jitter_ratio_kf_over_gated"]["hi"] <= 0.6),
        ("T2", "KF − gated mean error, clean (px)", m["d_kf_clean"], "UB ≤ +0.5", m["d_kf_clean"]["hi"] <= 0.5),
        ("T3", "KF − gated, occlusion episodes (px)", m["d_kf_occ_kp"], "UB < 0", m["d_kf_occ_kp"]["hi"] < 0),
        ("T4", "KF − gated, 15 frames after episodes (px)", d["d_recover_kf_minus_gated"], "UB ≤ +1.0",
         d["d_recover_kf_minus_gated"]["hi"] <= 1.0),
        ("S1", "3D tip error, KF pipeline (mm)", m["tip3d_kf"], "UB ≤ 5.0", m["tip3d_kf"]["hi"] <= 5.0),
        ("S2", "3D tip error KF − frame-wise (mm)", m["d_tip3d_kf_minus_gat"], "UB < 0",
         m["d_tip3d_kf_minus_gat"]["hi"] < 0),
        ("S3", "95%-ellipsoid coverage", m["cover95"], "LB ≥ 0.85 and point ≤ 0.99",
         m["cover95"]["lo"] >= 0.85 and m["cover95"]["point"] <= 0.99),
        ("S4a", "hand-eye pivot: kinematics − vision (mm)", m["d_pivot_kin_minus_vision"], "UB < 0",
         m["d_pivot_kin_minus_vision"]["hi"] < 0),
        ("S4b", "hand-eye pivot error, kinematics (mm)", m["pivot_kin"], "UB ≤ 5.0", m["pivot_kin"]["hi"] <= 5.0),
        ("S5", "alert toggles ratio (KF + hysteresis) / frame-wise", d["toggle_ratio_kfhyst_over_fw"], "UB ≤ 0.7",
         d["toggle_ratio_kfhyst_over_fw"]["hi"] <= 0.7),
    ]
    return [{"id": i, "endpoint": e, "value": v, "rule": r, "pass": bool(p)} for i, e, v, r, p in c]


def ratio_boot(num, den, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([np.nansum(v) for v in num])
    ca = np.array([np.isfinite(v).sum() for v in num])
    b = np.array([np.nansum(v) for v in den])
    cb = np.array([np.isfinite(v).sum() for v in den])
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(num), len(num))
        reps.append((a[i].sum() / ca[i].sum()) / (b[i].sum() / cb[i].sum()))
    lo, hi = ci(np.asarray(reps))
    return {"point": float((a.sum() / ca.sum()) / (b.sum() / cb.sum())), "lo": float(lo), "hi": float(hi)}


def diff_boot(x, y, n_boot=2000, seed=0):
    """Paired difference of means (x - y), same (trajectory, frame, keypoint) entries."""
    return boot([xi - yi for xi, yi in zip(x, y)], n_boot, seed)


def f(v, p=2):
    return f"{v['point']:.{p}f} [{v['lo']:.{p}f}, {v['hi']:.{p}f}]"


def render(R):
    m, d = R["metrics"], R["derived"]
    L = [f"# Stage 2 (Phases 3b-5), split `{R['split']}` (trajectories {R['trajectories']})", "",
         "## Pre-specified criteria", "", "| ID | Endpoint | Value [95% CI] | Rule | Result |", "|---|---|---|---|---|"]
    for c in R["criteria"]:
        L.append(f"| {c['id']} | {c['endpoint']} | {f(c['value'], 3)} | {c['rule']} | {'PASS' if c['pass'] else 'FAIL'} |")
    L += ["",
         f"Selected on tune: gate `{R['selected']['gated']}`, Kalman `{R['selected']['kf']}`, "
         f"3D covariance inflation kappa = {R['selected']['cov_inflation_kappa']:.2f}. "
         "Errors in native pixels (1400x986) or mm; 95% CIs are trajectory-clustered bootstrap.", "",
         "## Phase 3b: gated shape-prior MAP (left eye, every 5th frame)", "",
         "| Metric | Value [95% CI] |", "|---|---|",
         f"| Argmax mean error, clean | {f(m['arg_clean'])} |",
         f"| Gated mean error, clean | {f(m['gat_clean'])} |",
         f"| Gated − argmax, clean | {f(m['d_gat_clean'])} |",
         f"| Gated − argmax PCK@5 | {f(d['d_pck5_gated_minus_argmax'], 3)} |",
         f"| Argmax / gated PCK@5 | {m['pck5_arg']['point']:.3f} / {m['pck5_gat']['point']:.3f} |",
         f"| Occluded keypoints: argmax / gated | {f(m['arg_occ_kp'])} / {f(m['gat_occ_kp'])} |",
         f"| Occluded keypoints: gated − argmax | {f(m['d_gat_occ_kp'])} |", "",
         "## Phase 4: temporal filter (left eye, all frames, 30 fps)", "",
         "| Metric | Value [95% CI] |", "|---|---|",
         f"| Causal KF mean error, clean | {f(m['kf_clean'])} |",
         f"| KF − gated, clean | {f(m['d_kf_clean'])} |",
         f"| Jitter (acceleration error, px/frame²): gated / KF | {f(m['jit_gat'])} / {f(m['jit_kf'])} |",
         f"| Jitter ratio KF / gated | {f(d['jitter_ratio_kf_over_gated'], 3)} |",
         f"| Occlusion episodes, occluded keypoints: KF | {f(m['kf_occ_kp'])} |",
         f"| Occlusion episodes: KF − gated | {f(m['d_kf_occ_kp'])} |",
         f"| 15 frames after an episode: gated / KF | {f(m['recover_gat'])} / {f(m['recover_kf'])} |",
         f"| Flagged 'unknown' (position std > {R['selected']['kf']['unknown_px']:g} px) | {f(m['unknown_frac'], 3)} |",
         f"| Error when flagged / not flagged | {f(m['err_unknown'])} / {f(m['err_known'])} |",
         f"| Mean NIS of accepted updates (consistent filter: 2) | {R['nis_mean']:.2f} |",
         f"| KF cost (Python, 10 keypoints) | {R['kf_ms_per_frame_10kp_python']:.2f} ms/frame |", "",
         "Fixed-lag smoother (accuracy vs latency):", "",
         "| Lag (frames / ms) | Mean error | Jitter |", "|---|---|---|"]
    for Lg in LAGS:
        L.append(f"| {Lg} / {Lg * 33} | {f(m[f'kf_lag{Lg}'])} | {f(m[f'jit_lag{Lg}'])} |")
    L += ["", "## Phase 5: stereo 3D tips (every 5th frame)", "",
          "| Metric | Value [95% CI] |", "|---|---|",
          f"| 3D tip error, KF per eye then triangulate (mm) | {f(m['tip3d_kf'])} |",
          f"| 3D tip error, frame-wise gated (mm) | {f(m['tip3d_gat'])} |",
          f"| KF − frame-wise (paired) | {f(m['d_tip3d_kf_minus_gat'])} |",
          f"| Depth-direction / lateral component (mm) | {f(m['tip3d_depth_err'])} / {f(m['tip3d_lateral_err'])} |",
          f"| 95%-ellipsoid coverage (kappa-inflated) | {f(m['cover95'], 3)} |",
          f"| GT left/right label consistency: reprojection RMS median / p95 | {R['gt_reproj_rms']['median']:.2f} / {R['gt_reproj_rms']['p95']:.2f} px |",
          f"| GT pairs excluded (> 3 px) | {R['gt_reproj_rms']['frac_gt3px']:.4f} |", "",
          "Hand-eye (robot kinematics → camera), registered from vision estimates on the first half of each "
          "trajectory, evaluated on the second half; jaw-pivot keypoint:", "",
          "| Metric | Value [95% CI] |", "|---|---|",
          f"| Vision-only 3D pivot error (mm) | {f(m['pivot_vision'])} |",
          f"| Kinematics + vision-fitted registration (mm) | {f(m['pivot_kin'])} |",
          f"| Kinematics − vision | {f(m['d_pivot_kin_minus_vision'])} |", "",
          f"## Phase 5b: tip-to-tissue distance and the proximity alert (all frames; alert < {R['alert_mm']:g} mm)", "",
          "Tissue = robust local plane from SGM around the GT tip; d_ref uses the GT tip on the same plane, so these "
          "numbers isolate the error added by tip perception (SGM's own depth error is not measured: no depth GT).", "",
          "| Metric | Value [95% CI] |", "|---|---|",
          f"| Frames with reference alert on | {f(m['alert_frac_ref'], 3)} |",
          f"| abs(d − d_ref), KF pipeline (mm) | {f(m['dist_err_kf'])} |",
          f"| abs(d − d_ref), frame-wise (mm) | {f(m['dist_err_fw'])} |",
          f"| Alert sensitivity / specificity, KF | {f(m['alert_sens_kf'], 3)} / {f(m['alert_spec_kf'], 3)} |",
          f"| Alert sensitivity / specificity, frame-wise | {f(m['alert_sens_fw'], 3)} / {f(m['alert_spec_fw'], 3)} |",
          "", "Alert toggles per minute (per instrument):", "", "| Source | Toggles/min [95% CI] |", "|---|---|"]
    for k, lab in [("ref", "reference (GT tip)"), ("fw", "frame-wise"), ("kf", "Kalman"),
                   ("kf_hyst", f"Kalman + {R['hysteresis_mm']:g} mm hysteresis")]:
        L.append(f"| {lab} | {f(R['toggles_per_min'][k], 1)} |")
    L += ["", f"Toggle ratio (Kalman + hysteresis) / frame-wise: {f(d['toggle_ratio_kfhyst_over_fw'], 3)}", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
