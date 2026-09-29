"""v4 endpoints (M1-M5), pre-registered in docs/validation_plan.md, section "v4".

  uv run python scripts/eval_v4.py --split tune --fit-kappa   # writes kappa into configs/v4_selected.json
  uv run python scripts/eval_v4.py --split tune               # sanity; numbers quoted in the pre-registration
  uv run python scripts/eval_v4.py --split test2              # once, after the pre-registration is committed

Method under test ("v4"): the frozen v3 Phase C fusion (configs/v3_fusion_selected.json) and its
hybrid, except on arms classified as long-jaw: there each tip's depth prior is the inverse-variance
combination of the kinematic tip depth (std = the type's train tip-offset scatter) and a jaw-length
constraint (the tip midpoint lies on its detected left viewing ray at the type's pivot-to-tip
length from the fused pivot; fusion.length_constrained_depth). The type comes from a causal
classifier (fusion.classify_type: running 90th percentile of the lateral jaw length in mm,
threshold 13 mm, from 30 valid frames on). Lengths, spreads and scatters come from the train-only
library (configs/v4_tool_library.json). Standard-type arms, undecided frames and the warm-up are
exactly v3 (which uses v2 during the warm-up).
Comparator: v3 Phase C on the same frames (frozen config, κ = 10.4649).
Unit = trajectory; trajectory-clustered bootstrap (eval_stage2.boot), 2,000 replicates.
Writes runs/v4_<split>/{results.json, report.md}.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from surgscene.fusion import ARMS, FusionParams  # noqa: E402
from surgscene.stage2 import SPLITS, TIP  # noqa: E402
from surgscene.evaluation import boot  # noqa: E402
from surgscene.pipeline import CHI2_3_95, causal_types, hybrid, hybrid_length, load_prepared, run  # noqa: E402

CONFIG = ROOT / "configs/v4_selected.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def gt_type(D, arm):
    """Post-hoc label for the report only (GT pivot -> tip-midpoint distance > 15 mm, as the v3 subgroup)."""
    a, b = TIP[arm]
    v = (D["G"][:, a] + D["G"][:, b]) / 2 - D["G"][:, ARMS[arm]][:, 2]
    return "long" if np.nanmedian(np.linalg.norm(v, axis=1)) > 15.0 else "standard"


def per_trajectory(D, cfg):
    p = FusionParams(**cfg["fusion"])
    tracks = run(D, p, None)
    types = causal_types(D, tracks)
    X3, C3 = hybrid(D, tracks, cfg["hybrid_sd_depth_mm"])
    X4, C4 = hybrid_length(D, tracks, types, cfg["hybrid_sd_depth_mm"], kin_sd_by_type=True,
                           apply_types=tuple(cfg["length_constraint_types"]))
    rig, f5 = D["rig"], D["f5"]
    out = {k: [] for k in ["m1", "v4", "v3", "d2", "depth_v4", "depth_v3", "tip2d_v4", "tip2d_v2", "m5", "lat_v4"]}
    arms, d2s = {}, []
    for arm, (a, b) in TIP.items():
        g = (D["G"][:, a] + D["G"][:, b]) / 2
        v2 = (D["V2"][:, a] + D["V2"][:, b]) / 2
        e = {}
        for name, X in (("v4", X4), ("v3", X3)):
            m = (X[arm][:, 2] + X[arm][:, 3]) / 2
            e[name] = np.linalg.norm(np.where(np.isfinite(m), m, v2) - g, axis=1)
            ray = g / np.linalg.norm(g, axis=1, keepdims=True)
            err = np.where(np.isfinite(m), m, v2) - g
            out[f"depth_{name}"].append(np.abs((err * ray).sum(1)))
        m4 = (X4[arm][:, 2] + X4[arm][:, 3]) / 2
        err = np.where(np.isfinite(m4), m4, v2) - g
        along = np.abs((err * (g / np.linalg.norm(g, axis=1, keepdims=True))).sum(1))
        out["lat_v4"].append(np.sqrt(np.maximum(np.linalg.norm(err, axis=1) ** 2 - along**2, 0)))
        out["v4"].append(e["v4"])
        out["v3"].append(e["v3"])
        out["m1"].append(e["v4"] - e["v3"])
        # coverage: squared Mahalanobis distance under the (uninflated) v4 covariance, post-warm-up
        S = (C4[arm][:, 2] + C4[arm][:, 3]) / 4
        d2 = np.full(len(g), np.nan)
        for i in np.flatnonzero(np.isfinite(m4).all(1) & np.isfinite(g).all(1) & np.isfinite(S).all((1, 2))):
            d = m4[i] - g[i]
            d2[i] = d @ np.linalg.solve(S[i], d)
        out["d2"].append(d2)
        # M5: 2D left tips, v4 projection vs v2 Kalman 2D, against GT labels
        gl = D["O"]["gt"][f5][:, [a, b]]
        proj = np.stack([rig.left.project(np.nan_to_num(X4[arm][:, j], nan=1.0)) for j in (2, 3)], 1)
        e2 = np.linalg.norm(proj - gl, axis=-1).mean(1)
        e2[~np.isfinite(X4[arm][:, 2:4]).all((1, 2))] = np.nan
        e2v = np.linalg.norm(D["kfL"][f5][:, [a, b]] - gl, axis=-1).mean(1)
        out["m5"].append(e2 - e2v)
        out["tip2d_v4"].append(e2)
        out["tip2d_v2"].append(e2v)
        final = [t for t in types[arm] if t is not None]
        arms[arm] = {"type_final": final[-1] if final else None, "type_gt_posthoc": gt_type(D, arm),
                     "type_first_frame": int(np.argmax([t is not None for t in types[arm]])) if final else None,
                     "v4": e["v4"], "v3": e["v3"], "m1": e["v4"] - e["v3"]}
    return {k: np.concatenate(v) for k, v in out.items()}, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test2"], required=True)
    ap.add_argument("--fit-kappa", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    per, sub, arm_rows = {}, {}, {}
    for t in SPLITS[args.split]:
        D = load_prepared(t)
        r, arms = per_trajectory(D, cfg)
        for k, v in r.items():
            per.setdefault(k, []).append(v)
        for arm, a in arms.items():
            grp = a["type_final"] or "undecided"
            for k in ("v4", "v3", "m1"):
                sub.setdefault(grp, {}).setdefault(k, []).append(a[k])
            arm_rows[f"{t}/{arm}"] = {k: a[k] for k in ("type_final", "type_gt_posthoc", "type_first_frame")}
            arm_rows[f"{t}/{arm}"].update({"v4": float(np.nanmean(a["v4"])), "v3": float(np.nanmean(a["v3"]))})
        print(f"traj {t}: v4 {np.nanmean(r['v4']):.2f} mm vs v3 {np.nanmean(r['v3']):.2f}", flush=True)

    if args.fit_kappa:
        assert args.split == "tune", "kappa is fitted on tune only"
        pooled = np.concatenate(per["d2"])
        pooled = pooled[np.isfinite(pooled)]
        cfg["kappa"] = float(np.quantile(pooled, 0.95) / CHI2_3_95)
        CONFIG.write_text(json.dumps(cfg, indent=1) + "\n")
        print("kappa", cfg["kappa"])
    kappa = cfg["kappa"]
    per["m4"] = [np.where(np.isfinite(d), (d <= kappa * CHI2_3_95).astype(float), np.nan) for d in per["d2"]]
    m = {k: boot(v) for k, v in per.items() if k != "d2"}
    subgroups = {g: {k: boot(v) for k, v in d.items()} for g, d in sub.items()}
    longm = subgroups.get("long", {}).get("m1")
    nan = {"point": np.nan, "lo": np.nan, "hi": np.nan}
    agree = [a["type_final"] == a["type_gt_posthoc"] for a in arm_rows.values()]
    m["m3"] = {"point": float(np.mean(agree)), "lo": np.nan, "hi": np.nan, "n": len(agree)}
    m["overall_m1"] = m.pop("m1")
    ends = [
        ("M1 (primary)", "Long-type arms (causal classifier): 3D tip error, v4 − v3 (paired, mm)", longm or nan,
         "UB < 0", bool(longm and longm["hi"] < 0)),
        ("M2 (R1)", "Mean 3D tip error, v4, all arms (mm)", m["v4"], "UB ≤ 5.0", m["v4"]["hi"] <= 5.0),
        ("M3", "Causal type = GT type (post-hoc label), fraction of arm-trajectories", m["m3"], "≥ 0.90",
         m["m3"]["point"] >= 0.90),
        ("M4 (R7)", "Coverage of the κ-inflated 95% ellipsoid", m["m4"], "LB ≥ 0.85 and point ≤ 0.99",
         m["m4"]["lo"] >= 0.85 and m["m4"]["point"] <= 0.99),
        ("M5", "2D left tip error, v4 − v2 (px)", m["m5"], "UB ≤ +0.5", m["m5"]["hi"] <= 0.5),
    ]
    out_dir = ROOT / f"runs/v4_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    R = {"split": args.split, "config": cfg, "config_sha": sha(CONFIG), "library_sha": sha(ROOT / "configs/v4_tool_library.json"),
         "v3_config_sha": sha(ROOT / "configs/v3_fusion_selected.json"), "fusion_params": asdict(FusionParams(**cfg["fusion"])),
         "metrics": m, "subgroups": subgroups, "arm_trajectories": arm_rows, "verdicts": {e[0]: bool(e[4]) for e in ends}}
    (out_dir / "results.json").write_text(json.dumps(R, indent=1, default=str))
    f = lambda d, n=2: f"{d['point']:.{n}f} [{d['lo']:.{n}f}, {d['hi']:.{n}f}]"
    L = [f"# v4 — {args.split}", "",
         f"Config `configs/v4_selected.json` (sha256 {R['config_sha']}), library sha256 {R['library_sha']}, "
         f"v3 config sha256 {R['v3_config_sha']}. Trajectory-clustered bootstrap, 2,000 replicates.", "",
         "| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|"]
    for i, name, d, crit, ok in ends:
        val = f"{d['point']:.3f} ({d['n']} arm-trajectories)" if i == "M3" else f(d, 3 if i.startswith("M4") else 2)
        L.append(f"| {i} | {name} | {val} | {crit} | {'PASS' if ok else 'FAIL'} |")
    L += ["", "Reported (no criterion):", "",
          f"- v3 Phase C on the same frames: {f(m['v3'])} mm; all arms v4 − v3 (paired): {f(m['overall_m1'])} mm "
          "(zero on standard arms by construction)",
          f"- Along the viewing ray, v4 / v3: {f(m['depth_v4'])} / {f(m['depth_v3'])} mm; lateral v4 {f(m['lat_v4'])} mm",
          f"- 2D left tips, v4 / v2: {f(m['tip2d_v4'])} / {f(m['tip2d_v2'])} px", "",
          "Subgroups by the causal classifier's final type (cluster = trajectory-arm):", "",
          "| Type | n | v4 (mm) | v3 (mm) | v4 − v3 (mm) |", "|---|---|---|---|---|"]
    for g, d in subgroups.items():
        L.append(f"| {g} | {len(sub[g]['v4'])} | {f(d['v4'])} | {f(d['v3'])} | {f(d['m1'])} |")
    L += ["", "| Arm-trajectory | Type (causal, final) | Type (GT, post hoc) | Decided at frame | v4 | v3 |",
          "|---|---|---|---|---|---|"]
    for k, a in arm_rows.items():
        L.append(f"| {k} | {a['type_final']} | {a['type_gt_posthoc']} | {a['type_first_frame']} | {a['v4']:.2f} | {a['v3']:.2f} |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
