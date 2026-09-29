"""v3 Phase C endpoints (C1-C6 + pre-specified subgroup), pre-registered in docs/validation_plan.md.

  uv run python scripts/eval_v3c.py --split tune     # sanity; numbers quoted in the pre-registration
  uv run python scripts/eval_v3c.py --split test2    # once, after the pre-registration is committed

Frozen configuration: configs/v3_fusion_selected.json (fusion parameters, depth-prior sd, kappa)
and configs/tool_geometry.json (train-fitted priors). The method under test is the *hybrid*: v2
Kalman 2D detections of each jaw tip, triangulated with a depth prior along the viewing ray from
the causal kinematics fusion (surgscene.fusion). Frames before the first causal calibration (1 s)
use the v2 estimate. Unit of analysis = trajectory; trajectory-clustered bootstrap, 2,000
replicates, percentile CIs (eval_stage2.boot).
Writes runs/v3c_<split>/{results.json, report.md}.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_fusion import CHI2_3_95, hybrid, prepare, run_variant  # noqa: E402
from eval_stage2 import boot  # noqa: E402

from surgscene.fusion import ARMS, FusionParams  # noqa: E402
from surgscene.stage2 import SPLITS, TIP, load_selected, shape_models  # noqa: E402
from surgscene.temporal import KFParams  # noqa: E402

CONFIG = ROOT / "configs/v3_fusion_selected.json"
SWAP_MARGIN_MM = 3.0
LONG_JAW_MM = 15.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def per_trajectory(D, cfg, p):
    tracks = run_variant(D, p)
    Xh, Ch = hybrid(D, tracks, cfg["hybrid_sd_depth_mm"])
    kappa = cfg["kappa"]
    rig, f5 = D["rig"], D["f5"]
    out = {k: [] for k in ["c1", "c2", "c3", "c4", "c5_h", "c5_v2", "c6", "v2", "hyb_post", "depth_h", "depth_v2",
                           "lat_h", "lat_v2", "tip2d_h", "tip2d_v2"]}
    arms = {}
    for arm, (a, b) in TIP.items():
        sl = ARMS[arm]
        G = D["G"]
        g = (G[:, a] + G[:, b]) / 2
        X = Xh[arm]
        h = (X[:, 2] + X[:, 3]) / 2
        v = (D["V2"][:, a] + D["V2"][:, b]) / 2
        filled = np.where(np.isfinite(h), h, v)
        e_h, e_v = np.linalg.norm(filled - g, axis=1), np.linalg.norm(v - g, axis=1)
        post = np.isfinite(h).all(1)
        out["c1"].append(e_h)
        out["v2"].append(e_v)
        out["c2"].append(e_h - e_v)
        out["hyb_post"].append(np.where(post, e_h, np.nan))
        ray = g / np.linalg.norm(g, axis=1, keepdims=True)
        for name, P in (("h", filled), ("v2", v)):
            e = P - g
            along = np.abs((e * ray).sum(1))
            out[f"depth_{name}"].append(along)
            out[f"lat_{name}"].append(np.sqrt(np.maximum(np.linalg.norm(e, axis=1) ** 2 - along**2, 0)))
        # C3: kappa-inflated 95% ellipsoid, post-warm-up (the hybrid has a covariance only there)
        S = (Ch[arm][:, 2] + Ch[arm][:, 3]) / 4
        d2 = np.full(len(g), np.nan)
        for i in np.flatnonzero(post & np.isfinite(g).all(1) & np.isfinite(S).all((1, 2))):
            e = h[i] - g[i]
            d2[i] = e @ np.linalg.solve(kappa * S[i], e)
        out["c3"].append(np.where(np.isfinite(d2), (d2 <= CHI2_3_95).astype(float), np.nan))
        # C4: fused jaw pivot, post-warm-up
        piv = tracks[arm].X[f5][:, 0]
        out["c4"].append(np.where(post, np.linalg.norm(piv - G[:, sl][:, 2], axis=1), np.nan))
        # C5: tip swaps (individual tips closer to the other GT tip by > margin)
        for name, A, B in (("h", X[:, 2], X[:, 3]), ("v2", D["V2"][:, a], D["V2"][:, b])):
            straight = np.linalg.norm(A - G[:, a], axis=1) + np.linalg.norm(B - G[:, b], axis=1)
            crossed = np.linalg.norm(A - G[:, b], axis=1) + np.linalg.norm(B - G[:, a], axis=1)
            sw = crossed + SWAP_MARGIN_MM < straight
            ok = np.isfinite(straight) & np.isfinite(crossed) & (post if name == "h" else True)
            out[f"c5_{name}"].append(np.where(ok, sw.astype(float), np.nan))
        # C6: 2D left-image tip error (px), hybrid projection vs v2 Kalman 2D, both against GT labels
        gl = D["O"]["gt"][f5][:, [a, b]]
        proj = np.stack([rig.left.project(np.nan_to_num(X[:, j], nan=1.0)) for j in (2, 3)], 1)
        e2h = np.linalg.norm(proj - gl, axis=-1).mean(1)
        e2h[~np.isfinite(X[:, 2:4]).all((1, 2))] = np.nan
        e2v = np.linalg.norm(D["kfL"][f5][:, [a, b]] - gl, axis=-1).mean(1)
        out["c6"].append(e2h - e2v)
        out["tip2d_h"].append(e2h)
        out["tip2d_v2"].append(e2v)
        geom = tracks[arm].geom[-1] if tracks[arm].geom else None
        arms[arm] = {"tip_offset_mm": float(np.linalg.norm(geom.tip_mid)) if geom else None,
                     "c1": e_h, "v2": e_v, "c2": e_h - e_v,
                     "calib_first_frame": tracks[arm].calib_time[0] if tracks[arm].calib_time else None}
    return {k: np.concatenate(v) for k, v in out.items()}, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test2"], required=True)
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    p = FusionParams(**cfg["fusion"])
    sel = load_selected()
    models, _ = shape_models()
    kfp = KFParams(**sel["kf"])
    out_dir = ROOT / "runs" / f"v3c_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per, sub = {}, {"long": {"c1": [], "v2": [], "c2": []}, "standard": {"c1": [], "v2": [], "c2": []}}
    offsets = {}
    for t in SPLITS[args.split]:
        D = prepare(t, models, sel["gated"], kfp)
        r, arms = per_trajectory(D, cfg, p)
        for k, v in r.items():
            per.setdefault(k, []).append(v)
        for arm, a in arms.items():
            grp = "long" if (a["tip_offset_mm"] or 0) > LONG_JAW_MM else "standard"
            offsets[f"{t}/{arm}"] = {"tip_offset_mm": a["tip_offset_mm"], "group": grp,
                                     "calib_first_frame": a["calib_first_frame"]}
            for k in ("c1", "v2", "c2"):
                sub[grp][k].append(a[k])  # cluster = trajectory-arm within a subgroup
        print(f"traj {t}: C1 {np.nanmean(r['c1']):.2f} mm vs v2 {np.nanmean(r['v2']):.2f}", flush=True)

    m = {k: boot(v) for k, v in per.items()}
    m["c5_diff"] = boot([a - b for a, b in zip(per["c5_h"], per["c5_v2"])])
    subgroups = {g: {k: boot(v) for k, v in d.items() if v} for g, d in sub.items()}
    ends = [
        ("C1 (primary, R1)", "Mean 3D tip error, hybrid (mm)", m["c1"], "UB ≤ 5.0", m["c1"]["hi"] <= 5.0),
        ("C2", "3D tip error, hybrid − v2 (paired, mm)", m["c2"], "UB < 0", m["c2"]["hi"] < 0),
        ("C3 (R7)", "Coverage of the κ-inflated 95% ellipsoid", m["c3"], "LB ≥ 0.85 and point ≤ 0.99",
         m["c3"]["lo"] >= 0.85 and m["c3"]["point"] <= 0.99),
        ("C4", "Jaw pivot 3D error, fused (mm)", m["c4"], "UB ≤ 5.0", m["c4"]["hi"] <= 5.0),
        ("C6", "2D left tip error, hybrid − v2 (px)", m["c6"], "UB ≤ +0.5", m["c6"]["hi"] <= 0.5),
    ]
    R = {"split": args.split, "config": cfg, "config_sha": sha(CONFIG),
         "tool_geometry_sha": sha(ROOT / "configs/tool_geometry.json"), "fusion_params": asdict(p),
         "metrics": m, "subgroups": subgroups, "arm_trajectories": offsets,
         "verdicts": {e[0]: bool(e[4]) for e in ends}}
    (out_dir / "results.json").write_text(json.dumps(R, indent=1))

    f = lambda d, n=2: f"{d['point']:.{n}f} [{d['lo']:.{n}f}, {d['hi']:.{n}f}]"
    L = [f"# v3 Phase C — {args.split}", "",
         f"Config `configs/v3_fusion_selected.json` (sha256 {R['config_sha']}), tool geometry sha256 "
         f"{R['tool_geometry_sha']}. Trajectory-clustered bootstrap, 2,000 replicates.", "",
         "| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|"]
    for i, name, d, crit, ok in ends:
        L.append(f"| {i} | {name} | {f(d, 3 if i.startswith('C3') else 2)} | {crit} | {'PASS' if ok else 'FAIL'} |")
    L += ["", "Reported (no criterion):", "",
          f"- v2 3D tip error on the same frames: {f(m['v2'])} mm",
          f"- Hybrid after the warm-up only: {f(m['hyb_post'])} mm",
          f"- Along the viewing ray, hybrid / v2: {f(m['depth_h'])} / {f(m['depth_v2'])} mm",
          f"- Lateral, hybrid / v2: {f(m['lat_h'])} / {f(m['lat_v2'])} mm",
          f"- 2D left tips, hybrid / v2: {f(m['tip2d_h'])} / {f(m['tip2d_v2'])} px",
          f"- C5 tip-swap rate (> {SWAP_MARGIN_MM:g} mm), hybrid / v2: {f(m['c5_h'], 3)} / {f(m['c5_v2'], 3)}; "
          f"difference {f(m['c5_diff'], 3)}", "",
          f"Pre-specified subgroup: long-jaw instruments (online-estimated tip offset > {LONG_JAW_MM:g} mm at the "
          "last calibration) vs standard; cluster = trajectory-arm.", "",
          "| Subgroup | n arm-trajectories | Hybrid (mm) | v2 (mm) | Hybrid − v2 (mm) |", "|---|---|---|---|---|"]
    for g, d in subgroups.items():
        if d:
            L.append(f"| {g} | {len(sub[g]['c1'])} | {f(d['c1'])} | {f(d['v2'])} | {f(d['c2'])} |")
    L += ["", "Arm-trajectories (online tip offset, subgroup):", "",
          ", ".join(f"{k}: {v['tip_offset_mm']:.1f} mm ({v['group']})" for k, v in offsets.items() if v["tip_offset_mm"])]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
