"""v3 plan step 5: the Phase A front end (DINOv2 keypoints + learned sigma) through the downstream
pipeline: per-keypoint Kalman filter -> Phase C kinematics fusion -> hybrid 3D tips.

  uv run python scripts/eval_downstream_vit.py --split tune            # select the Kalman scale, fit kappa
  uv run python scripts/eval_downstream_vit.py --split test2           # once, after pre-registration

Only the inputs change against the frozen v3 pipeline (configs/v3_fusion_selected.json):
  2D per frame  DINOv2 argmax (no gated MAP: its thresholds were tuned on the v2 confidence scale)
  noise         sigma_eff = k * learned sigma (k from configs/v3_kp_vit_selected.json); confidence
                only gates (conf >= conf_min -> 1, else 0: predict only). So the Kalman R and the
                fusion pixel std come from the learned sigma instead of Laplace sigma / conf.
  Kalman r0     selected on tune (grid), all other Kalman parameters as stage 2
  kappa         refitted on tune (the covariance now comes from a different sigma)
The v2-input pipeline (the pre-registered v3 Phase C method) is computed alongside as the paired
comparator. Also: Kalman 2D error on occluded keypoints during the occlusion episodes (left).
Writes runs/v3_downstream_vit_<split>/{results.json, report.md}.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from surgscene.evaluation import boot
from surgscene.fusion import FusionParams
from surgscene.pipeline import CHI2_3_95, hybrid, prepare, run_variant
from surgscene.pipeline import v3c_per_trajectory as per_trajectory
from surgscene.stage2 import SPLITS, TIP, load_selected, shape_models, triangulate_seq
from surgscene.temporal import KFParams, filter_keypoints

ROOT = Path(__file__).resolve().parents[1]

VIT_OBS = ROOT / "data/cache/surgpose_obs_vit"
CFG_C = ROOT / "configs/v3_fusion_selected.json"
CFG_A = ROOT / "configs/v3_kp_vit_selected.json"
CFG_OUT = ROOT / "configs/v3_downstream_vit_selected.json"


def vit_inputs(traj, k, conf_min):
    """Observation dicts shaped like the v2 cache, with sigma = k * learned sigma and conf in {0, 1}."""
    out = {}
    for eye in ("left", "right"):
        d = dict(np.load(VIT_OBS / f"{traj:06d}_{eye}.npz"))
        for sfx in ("", "_occ") if eye == "left" else ("",):
            d["sigma" + sfx] = k * d["sigma_learned" + sfx]
            d["conf" + sfx] = (d["conf" + sfx] >= conf_min).astype(float)
        out[eye] = d
    return out


def prepare_vit(traj, kfp, k):
    """Same dict as eval_fusion.prepare, with the DINOv2 front end."""
    obs = vit_inputs(traj, k, kfp.conf_min)
    O, OR = obs["left"], obs["right"]
    D = prepare(traj, *_V2_ARGS)  # GT, rig, kinematics, reference noise: identical to the v2 run
    kf, std, *_ = filter_keypoints(O["kp"], O["sigma"], O["conf"], kfp)
    kfR, stdR, *_ = filter_keypoints(OR["kp"], OR["sigma"], OR["conf"], kfp)
    V2, _, _ = triangulate_seq(D["rig"], kf[D["f5"]], kfR[D["f5"]], std[D["f5"]], stdR[D["f5"]])
    D.update(O=O, OR=OR, gL=O["kp"], gR=OR["kp"], kfL=kf, kfR=kfR, std=std, stdR=stdR, V2=V2)
    kfo, *_ = filter_keypoints(O["kp_occ"], O["sigma_occ"], O["conf_occ"], kfp)
    D["kf_occ"] = kfo
    return D


def occ_errors(D, kf_occ):
    occ = D["O"]["occluded"] if "occluded" in D["O"] else None
    e = np.linalg.norm(kf_occ - D["O"]["gt"], axis=-1)
    return np.where(occ, e, np.nan).ravel()


def kappa_fit(rows):
    """rows: list of per-trajectory d2 arrays (unscaled) -> kappa for 95% coverage."""
    d2 = np.concatenate([r[np.isfinite(r)] for r in rows])
    return float(np.quantile(d2, 0.95) / CHI2_3_95)


_V2_ARGS = None


def main():
    global _V2_ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test2"], required=True)
    args = ap.parse_args()
    sel = load_selected()
    models, _ = shape_models()
    kfp_v2 = KFParams(**sel["kf"])
    _V2_ARGS = (models, sel["gated"], kfp_v2)
    cfgC, cfgA = json.loads(CFG_C.read_text()), json.loads(CFG_A.read_text())
    p = FusionParams(**cfgC["fusion"])
    k = cfgA["learned_sigma_scale"]
    out_dir = ROOT / "runs" / f"v3_downstream_vit_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs = SPLITS[args.split]

    # reference pipeline: the pre-registered v3 Phase C method on v2 inputs
    ref = {}
    for t in trajs:
        D = prepare(t, *_V2_ARGS)
        r, _ = per_trajectory(D, cfgC, p)
        kfo, *_ = filter_keypoints(D["O"]["kp_occ"], D["O"]["sigma_occ"], D["O"]["conf_occ"], kfp_v2)
        ref[t] = {"c1": r["c1"], "occ2d": occ_errors(D, kfo), "tip2d": r["tip2d_h"]}
        print(f"traj {t} v2 inputs: {np.nanmean(r['c1']):.2f} mm", flush=True)

    def run(r0, kappa, need_d2=False):
        kfp = replace(kfp_v2, r0=r0)
        res = {}
        for t in trajs:
            D = prepare_vit(t, kfp, k)
            cfg = dict(cfgC, kappa=kappa)
            r, _ = per_trajectory(D, cfg, p)
            res[t] = {"c1": r["c1"], "c3": r["c3"], "occ2d": occ_errors(D, D["kf_occ"]), "tip2d": r["tip2d_h"]}
            if need_d2:
                res[t]["d2"] = _d2(D, p, cfgC["hybrid_sd_depth_mm"])
        return res

    if args.split == "tune":
        grid = {}
        for r0 in (0.5, 1.0, 2.0):
            res = run(r0, 1.0)
            grid[r0] = {"c1": float(np.nanmean(np.concatenate([v["c1"] for v in res.values()]))),
                        "occ2d": float(np.nanmean(np.concatenate([v["occ2d"] for v in res.values()]))), "res": res}
            print(f"r0 {r0}: 3D {grid[r0]['c1']:.2f} mm, occluded 2D {grid[r0]['occ2d']:.2f} px", flush=True)
        r0 = min(grid, key=lambda x: grid[x]["c1"])
        kappa = kappa_fit([v["d2"] for v in run(r0, 1.0, need_d2=True).values()])
        CFG_OUT.write_text(json.dumps({"note": "v3 step 5, frozen from tune: Kalman r0 selected by mean 3D hybrid tip error; "
                                       "kappa for 95% coverage of the hybrid tip midpoint.",
                                       "kalman_r0": r0, "kappa": round(kappa, 4), "learned_sigma_scale": k,
                                       "grid": {str(x): {kk: g[kk] for kk in ("c1", "occ2d")} for x, g in grid.items()}},
                                      indent=1) + "\n")
        print(f"selected r0 {r0}, kappa {kappa:.3f}")
    frozen = json.loads(CFG_OUT.read_text())
    res = run(frozen["kalman_r0"], frozen["kappa"])
    m = {"v1_3d_vit_minus_v2in": boot([res[t]["c1"] - ref[t]["c1"] for t in trajs]),
         "v2_3d_vit": boot([res[t]["c1"] for t in trajs]), "3d_v2in": boot([ref[t]["c1"] for t in trajs]),
         "v3_occ2d_vit_minus_v2in": boot([res[t]["occ2d"] - ref[t]["occ2d"] for t in trajs]),
         "occ2d_vit": boot([res[t]["occ2d"] for t in trajs]), "occ2d_v2in": boot([ref[t]["occ2d"] for t in trajs]),
         "v4_cover": boot([res[t]["c3"] for t in trajs]),
         "tip2d_vit_minus_v2in": boot([res[t]["tip2d"] - ref[t]["tip2d"] for t in trajs])}
    ends = [("V1 (primary)", "3D tip error, DINOv2 inputs − v2 inputs (paired, mm)", m["v1_3d_vit_minus_v2in"], "UB < 0",
             m["v1_3d_vit_minus_v2in"]["hi"] < 0),
            ("V2 (R1)", "Mean 3D tip error, DINOv2 inputs (mm)", m["v2_3d_vit"], "UB ≤ 5.0", m["v2_3d_vit"]["hi"] <= 5.0),
            ("V3", "Occlusion episodes: Kalman 2D error on occluded keypoints, DINOv2 − v2 (px)", m["v3_occ2d_vit_minus_v2in"],
             "UB < 0", m["v3_occ2d_vit_minus_v2in"]["hi"] < 0),
            ("V4 (R7)", "Coverage of the κ-inflated 95% ellipsoid", m["v4_cover"], "LB ≥ 0.85 and point ≤ 0.99",
             m["v4_cover"]["lo"] >= 0.85 and m["v4_cover"]["point"] <= 0.99)]
    R = {"split": args.split, "frozen": frozen, "metrics": m, "verdicts": {e[0]: bool(e[4]) for e in ends},
         "per_traj": {int(t): {"vit": float(np.nanmean(res[t]["c1"])), "v2in": float(np.nanmean(ref[t]["c1"]))} for t in trajs}}
    (out_dir / "results.json").write_text(json.dumps(R, indent=1))
    f = lambda d, n=2: f"{d['point']:.{n}f} [{d['lo']:.{n}f}, {d['hi']:.{n}f}]"
    L = [f"# v3 step 5 — {args.split}: DINOv2 front end through the Phase C pipeline", "",
         f"Kalman r0 = {frozen['kalman_r0']}, κ = {frozen['kappa']}, learned-σ scale k = {k} (all from tune).", "",
         "| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|"]
    for i, name, d, crit, ok in ends:
        L.append(f"| {i} | {name} | {f(d, 3 if i.startswith('V4') else 2)} | {crit} | {'PASS' if ok else 'FAIL'} |")
    L += ["", f"- 3D tip error, v2 inputs (the pre-registered method): {f(m['3d_v2in'])} mm",
          f"- Occluded keypoints (Kalman, 2D), DINOv2 / v2 inputs: {f(m['occ2d_vit'])} / {f(m['occ2d_v2in'])} px",
          f"- 2D left tips, DINOv2 − v2 inputs: {f(m['tip2d_vit_minus_v2in'])} px", "",
          "| Trajectory | " + " | ".join(str(t) for t in trajs) + " |", "|---|" + "---|" * len(trajs),
          "| DINOv2 inputs (mm) | " + " | ".join(f"{R['per_traj'][t]['vit']:.1f}" for t in trajs) + " |",
          "| v2 inputs (mm) | " + " | ".join(f"{R['per_traj'][t]['v2in']:.1f}" for t in trajs) + " |"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _d2(D, p, sd):
    """Unscaled squared Mahalanobis distances of the GT tip midpoint under the hybrid covariance."""
    tracks = run_variant(D, p)
    X, C = hybrid(D, tracks, sd)
    v = []
    for arm, (a, b) in TIP.items():
        g = (D["G"][:, a] + D["G"][:, b]) / 2
        mid = (X[arm][:, 2] + X[arm][:, 3]) / 2
        S = (C[arm][:, 2] + C[arm][:, 3]) / 4
        for i in range(len(g)):
            if np.isfinite(g[i]).all() and np.isfinite(mid[i]).all() and np.isfinite(S[i]).all():
                e = mid[i] - g[i]
                v.append(float(e @ np.linalg.solve(S[i], e)))
    return np.array(v)


if __name__ == "__main__":
    main()
