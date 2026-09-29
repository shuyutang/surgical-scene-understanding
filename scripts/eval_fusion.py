"""v3 Phase C on the tune trajectories: kinematics-fused 3D tool state vs the v2 pipeline.

  uv run python scripts/eval_fusion.py [--grid]

Same reference and frames as v2 S1: GT tips triangulated from the GT label pairs (pairs with
reprojection RMS > 3 px excluded), every 5th frame. Tip = jaw-tip midpoint. Variants:
  v2        per-keypoint Kalman + triangulation (the v2 pipeline, tune-selected parameters)
  fusion    causal hand-eye + kinematics + vision (fusion.fuse_arm)
  kin_only  causal hand-eye, then kinematics alone (delta = 0)
  oracle    hand-eye fitted to the GT pivot over the whole trajectory (diagnostic upper bound)
Frames before the first hand-eye fit fall back to v2 in the "all frames" number.
Writes runs/v3_fusion/{results.json, report.md}.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from surgscene.fusion import ARMS, FusionParams
from surgscene.pipeline import CHI2_3_95, hybrid, prepare, run_variant
from surgscene.stage2 import TIP, load_selected, shape_models
from surgscene.temporal import KFParams

ROOT = Path(__file__).resolve().parents[1]

TUNE = [18, 19, 20, 23]
OUT = ROOT / "runs/v3_fusion"


def coverage_table(data, sd_depth):
    """Squared Mahalanobis distance of the GT tip midpoint under the hybrid covariance (tips assumed
    independent: midpoint cov = (Ca + Cb) / 4), per trajectory. kappa = variance inflation that gives
    95% coverage over the pooled tune frames (fitted in-sample here; test2 is the real check)."""
    d2 = {}
    for t, D in data.items():
        X, C = hybrid(D, D["_fusion_tracks"], sd_depth)
        v = []
        for arm, (a, b) in TIP.items():
            g = (D["G"][:, a] + D["G"][:, b]) / 2
            m = (X[arm][:, 2] + X[arm][:, 3]) / 2
            S = (C[arm][:, 2] + C[arm][:, 3]) / 4
            for i in range(len(g)):
                if np.isfinite(g[i]).all() and np.isfinite(m[i]).all() and np.isfinite(S[i]).all():
                    e = m[i] - g[i]
                    v.append(float(e @ np.linalg.solve(S[i], e)))
        d2[t] = np.array(v)
    pooled = np.concatenate(list(d2.values()))
    kappa = float(np.quantile(pooled, 0.95) / CHI2_3_95)
    return {"kappa": kappa, "coverage_raw": float((pooled <= CHI2_3_95).mean()),
            "coverage_after_kappa_per_traj": {int(t): float((v <= kappa * CHI2_3_95).mean()) for t, v in d2.items()}}


def tip_errors(D, tracks, X_override=None):
    """Per arm: (F5,) tip-midpoint error for fusion (NaN before hand-eye), v2, with the depth split."""
    res = {}
    for arm, (a, b) in TIP.items():
        sl = ARMS[arm]
        g = (D["G"][:, a] + D["G"][:, b]) / 2
        v2 = (D["V2"][:, a] + D["V2"][:, b]) / 2
        X = tracks[arm].X[D["f5"]] if X_override is None else X_override[arm]
        fu = (X[:, 2] + X[:, 3]) / 2
        piv_g = D["G"][:, sl][:, 2]
        ray = g / np.linalg.norm(g, axis=1, keepdims=True)
        e_fu, e_v2 = fu - g, v2 - g
        rig = D["rig"]
        gl = D["O"]["gt"][D["f5"]]
        tip2d = lambda P3: np.stack([rig.left.project(np.nan_to_num(P3[:, j], nan=1.0)) for j in (0, 1)], 1)
        fu2 = tip2d(np.stack([X[:, 2], X[:, 3]], 1))
        e2_fu = np.linalg.norm(fu2 - gl[:, [a, b]], axis=-1).mean(1)
        e2_v2 = np.linalg.norm(D["kfL"][D["f5"]][:, [a, b]] - gl[:, [a, b]], axis=-1).mean(1)
        e2_fu[~np.isfinite(X[:, 2:4]).all((1, 2))] = np.nan
        res[arm] = {"fusion2d": e2_fu, "v22d": e2_v2, "ref_depth_sd": D["ref_sd"][arm],
                    "fusion": np.linalg.norm(e_fu, axis=1), "v2": np.linalg.norm(e_v2, axis=1),
                    "fusion_depth": np.abs((e_fu * ray).sum(1)), "v2_depth": np.abs((e_v2 * ray).sum(1)),
                    "pivot_fusion": np.linalg.norm(X[:, 0] - piv_g, axis=1),
                    "pivot_v2": np.linalg.norm(D["V2"][:, sl][:, 2] - piv_g, axis=1),
                    "tip_a": np.linalg.norm(X[:, 2] - D["G"][:, a], axis=1),
                    "tip_a_v2": np.linalg.norm(D["V2"][:, a] - D["G"][:, a], axis=1)}
    return res


def summarize(per_traj):
    """per_traj: {traj: {arm: {...}}} -> pooled means, post-warmup and all-frames-with-fallback."""
    acc = {}
    for traj, arms in per_traj.items():
        for arm, r in arms.items():
            ok = np.isfinite(r["v2"])
            fu_ok = ok & np.isfinite(r["fusion"])
            filled = np.where(np.isfinite(r["fusion"]), r["fusion"], r["v2"])
            for k, v in [("fusion_post", r["fusion"][fu_ok]), ("v2_post", r["v2"][fu_ok]),
                         ("fusion_all", filled[ok]), ("v2_all", r["v2"][ok]),
                         ("fusion_depth_post", r["fusion_depth"][fu_ok]), ("v2_depth_post", r["v2_depth"][fu_ok]),
                         ("pivot_fusion_post", r["pivot_fusion"][fu_ok & np.isfinite(r["pivot_fusion"])]),
                         ("pivot_v2_post", r["pivot_v2"][fu_ok & np.isfinite(r["pivot_v2"])]),
                         ("tip_a_post", r["tip_a"][fu_ok & np.isfinite(r["tip_a"])]),
                         ("tip_a_v2_post", r["tip_a_v2"][fu_ok & np.isfinite(r["tip_a_v2"])]),
                         ("tip2d_fusion_post", r["fusion2d"][fu_ok & np.isfinite(r["fusion2d"]) & np.isfinite(r["v22d"])]),
                         ("tip2d_v2_post", r["v22d"][fu_ok & np.isfinite(r["fusion2d"]) & np.isfinite(r["v22d"])]),
                         ("ref_depth_sd", r["ref_depth_sd"][ok & np.isfinite(r["ref_depth_sd"])])]:
                acc.setdefault(k, []).append(v)
                acc.setdefault(f"{k}@{traj}", []).append(v)
    return {k: {"mean": float(np.concatenate(v).mean()), "median": float(np.median(np.concatenate(v))),
                "n": int(sum(len(x) for x in v))} for k, v in acc.items() if sum(len(x) for x in v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="small grid over the fusion noise parameters")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = load_selected()
    models, _ = shape_models()
    kfp = KFParams(**sel["kf"])
    data = {t: prepare(t, models, sel["gated"], kfp) for t in TUNE}
    base = FusionParams()
    variants = {"fusion": (base, False), "fusion_unconstrained": (replace(base, sig_ray=None), False),
                "kin_only": (replace(base, use_vision=False), False),
                "fusion_tip3d": (replace(base, tip_mode="3d"), False),
                "oracle": (base, True),
                "oracle_kin_only": (replace(base, use_vision=False), True)}
    if args.grid:
        for sk in [1.0, 2.0, 4.0]:
            for tau in [10.0, 30.0, 100.0]:
                variants[f"fusion_sk{sk:g}_tau{tau:g}"] = (replace(base, sig_kin=sk, tau=tau), False)
        for wu in [15, 60]:
            variants[f"fusion_warmup{wu}"] = (replace(base, warmup=wu), False)
        for sr in [0.2, 2.0]:
            variants[f"fusion_sigray{sr:g}"] = (replace(base, sig_ray=sr), False)
        for re in [5, 45]:
            variants[f"fusion_refit{re}"] = (replace(base, refit_every=re), False)
    R = {"params": asdict(base), "variants": {}}
    hyb = {f"hybrid_sd{sd:g}": sd for sd in (2.0, 5.0)}
    hyb3 = {"hybrid_tip3d_sd5": 5.0}
    for name, (p, oracle) in list(variants.items()) + [(h, (base, False)) for h in hyb] + [(h, (base, False)) for h in hyb3]:
        per = {}
        he_info = {}
        for t, D in data.items():
            if name in hyb3:
                tracks = D["_tip3d_tracks"]
                per[t] = tip_errors(D, tracks, hybrid(D, tracks, hyb3[name])[0])
            elif name in hyb:
                tracks = D.setdefault("_fusion_tracks", run_variant(D, base, False))
                per[t] = tip_errors(D, tracks, hybrid(D, tracks, hyb[name])[0])
            else:
                tracks = run_variant(D, p, oracle)
                if name == "fusion":
                    D["_fusion_tracks"] = tracks
                if name == "fusion_tip3d":
                    D["_tip3d_tracks"] = tracks
                per[t] = tip_errors(D, tracks)
            he_info[t] = {arm: {"first_fit_frame": tr.calib_time[0] if tr.calib_time else None,
                                "last_rms_px": tr.calib_rms[-1] if tr.calib_rms else None,
                                "last_tip_mid": tr.geom[-1].tip_mid.round(2).tolist() if tr.geom else None,
                                "last_wrist_L": round(tr.geom[-1].wrist_L, 2) if tr.geom else None}
                          for arm, tr in tracks.items()}
        R["variants"][name] = {"params": asdict(p), "oracle": oracle, "summary": summarize(per), "handeye": he_info}
        s = R["variants"][name]["summary"]
        print(f"{name:28s} tip post-warmup: fusion {s['fusion_post']['mean']:.2f} (med {s['fusion_post']['median']:.2f}) "
              f"vs v2 {s['v2_post']['mean']:.2f} (med {s['v2_post']['median']:.2f}); all frames {s['fusion_all']['mean']:.2f} "
              f"vs {s['v2_all']['mean']:.2f}; depth {s['fusion_depth_post']['mean']:.2f} vs {s['v2_depth_post']['mean']:.2f}; "
              f"pivot {s['pivot_fusion_post']['mean']:.2f} vs {s['pivot_v2_post']['mean']:.2f}; "
              f"tip_a {s['tip_a_post']['mean']:.2f} vs {s['tip_a_v2_post']['mean']:.2f}; "
              f"2D left tips px {s['tip2d_fusion_post']['mean']:.2f} (med {s['tip2d_fusion_post']['median']:.2f}) vs "
              f"{s['tip2d_v2_post']['mean']:.2f} (med {s['tip2d_v2_post']['median']:.2f}); ref depth sd {s['ref_depth_sd']['median']:.2f}", flush=True)
    R["coverage_hybrid_sd5"] = coverage_table(data, 5.0)
    print("hybrid_sd5 covariance:", json.dumps(R["coverage_hybrid_sd5"]))
    (OUT / ("results_grid.json" if args.grid else "results.json")).write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
