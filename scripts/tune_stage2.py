"""Select every stage-2 parameter on the tune trajectories (18, 19, 20, 23). Never touches test2.

  uv run python scripts/tune_stage2.py

1. Gated MAP (Phase 3b): solver hyperparameters x gate threshold tau. Objective: mean keypoint
   error over clean + occluded left frames, subject to PCK@5 >= argmax PCK@5 - 0.01, i.e. the
   gate may not give back the fine precision that plain MAP loses.
2. Kalman filter (Phase 4): q (process noise), r0 (measurement-noise scale), conf_min. Objective:
   mean error over clean + occluded left frames (full rate), causal filter.
3. 3D covariance inflation kappa (Phase 5): one scalar so that 95% of tune tip errors fall inside
   the 95% ellipsoid (chi2_3), the analog of temperature scaling for the 3D uncertainty.
4. Hand-eye target keypoint (Phase 5, exploratory): which keypoint the kinematic tool frame tracks.
Writes configs/stage2_selected.json and caches estimates under data/cache/stage2_est/.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from surgscene.stage2 import (ROOT, TUNE, framewise, load_obs, load_rig, shape_models, tip_points,
                              triangulate_seq)
from surgscene.temporal import KFParams, filter_keypoints

EST = ROOT / "data/cache/stage2_est"
CHI2_3_95 = 7.8147


def err(P, gt):
    return np.linalg.norm(P - gt, axis=-1)


def pck(E, thr):
    f = E[np.isfinite(E)]
    return float((f <= thr).mean())


def gated_estimates(traj, eye, sfx, models, hp):
    """Full-rate gated estimates, cached per (traj, eye, condition, hp)."""
    EST.mkdir(parents=True, exist_ok=True)
    key = "_".join(f"{k}{hp[k]:g}" for k in sorted(hp))
    path = EST / f"{traj:06d}_{eye}{sfx}_{key}.npy"
    if path.exists():
        return np.load(path)
    O = load_obs(traj, eye)
    P = framewise(O["kp" + sfx], O["conf" + sfx], O["sigma" + sfx], models, "gated", hp)
    np.save(path, P)
    return P


def main():
    models, _ = shape_models()
    obs = {(t, c): load_obs(t, "left") for t in TUNE for c in ["clean"]}
    sel = {"tune_trajectories": TUNE}

    # 1. gated MAP on every 5th frame
    sub = slice(0, None, 5)
    data = []
    for t in TUNE:
        O = obs[(t, "clean")]
        for sfx in ["", "_occ"]:
            data.append({k: O[k + sfx][sub] for k in ["kp", "conf", "sigma"]} | {"gt": O["gt"][sub]})
    E_arg = np.concatenate([err(d["kp"], d["gt"]).ravel() for d in data])
    base_pck5 = pck(E_arg, 5)
    print(f"argmax: mean {np.nanmean(E_arg):.2f}px pck5 {base_pck5:.3f}")
    best, rows = None, []
    for conf_min, beta, c in tqdm(list(itertools.product([0.05, 0.2], [0.3, 1.0, 3.0], [3.0, 5.0])), desc="gate"):
        hp = {"conf_min": conf_min, "beta": beta, "cauchy_c": c}
        maps = [framewise(d["kp"], d["conf"], d["sigma"], models, "map", hp) for d in data]
        for tau in [0.1, 0.2, 0.3, 0.5, 1.01]:  # 1.01 = plain MAP
            E = np.concatenate([err(np.where((d["conf"] >= tau)[..., None], d["kp"], m), d["gt"]).ravel()
                                for d, m in zip(data, maps)])
            r = {**hp, "tau": tau, "mean": float(np.nanmean(E)), "pck5": pck(E, 5)}
            rows.append(r)
            if r["pck5"] >= base_pck5 - 0.01 and (best is None or r["mean"] < best["mean"]):
                best = r
    print("selected gate:", best)
    sel["gated"] = {k: best[k] for k in ["conf_min", "beta", "cauchy_c", "tau"]}
    sel["gated_tune_grid"] = rows
    sel["argmax_tune"] = {"mean": float(np.nanmean(E_arg)), "pck5": base_pck5}
    hp = sel["gated"]

    # full-rate gated estimates for tune (left clean/occ, right clean)
    est = {}
    for t in tqdm(TUNE, desc="gated full-rate"):
        for eye, sfx in [("left", ""), ("left", "_occ"), ("right", "")]:
            est[(t, eye, sfx)] = gated_estimates(t, eye, sfx, models, hp)

    # 2. Kalman filter
    best_kf, kf_rows = None, []
    for q, r0, cm in tqdm(list(itertools.product([0.5, 1.0, 2.0, 4.0], [0.5, 1.0, 2.0, 4.0], [0.05, 0.2])), desc="kf"):
        p = KFParams(q=q, r0=r0, conf_min=cm)
        E = []
        for t in TUNE:
            O = obs[(t, "clean")]
            for sfx in ["", "_occ"]:
                pos, *_ = filter_keypoints(est[(t, "left", sfx)], O["sigma" + sfx], O["conf" + sfx], p)
                E.append(err(pos, O["gt"]).ravel())
        E = np.concatenate(E)
        kf_rows.append({"q": q, "r0": r0, "conf_min": cm, "mean": float(np.nanmean(E))})
        if best_kf is None or kf_rows[-1]["mean"] < best_kf["mean"]:
            best_kf = kf_rows[-1]
    print("selected kf:", best_kf)
    sel["kf"] = {"q": best_kf["q"], "r0": best_kf["r0"], "conf_min": best_kf["conf_min"], "max_rejects": 3,
                 "unknown_px": 20.0}
    sel["kf_tune_grid"] = kf_rows
    p = KFParams(**sel["kf"])

    # 3. 3D covariance inflation, on every 5th frame
    d2_all, gt_rp = [], []
    for t in tqdm(TUNE, desc="3d"):
        rig = load_rig(t)
        OL, OR = load_obs(t, "left"), load_obs(t, "right")
        pl, sl, *_ = filter_keypoints(est[(t, "left", "")], OL["sigma"], OL["conf"], p)
        pr, sr, *_ = filter_keypoints(est[(t, "right", "")], OR["sigma"], OR["conf"], p)
        f = slice(0, None, 5)
        X, C, _ = triangulate_seq(rig, pl[f], pr[f], sl[f], sr[f])
        one = np.ones(OL["gt"][f].shape[:2])
        G, _, rp = triangulate_seq(rig, OL["gt"][f], OR["gt"][f], one, one)
        gt_rp.append(rp.ravel())
        good = rp <= 3.0
        for arm, (a, b) in {"PSM1": (3, 4), "PSM3": (8, 9)}.items():
            ok = good[:, a] & good[:, b]
            e = (X[:, a] + X[:, b]) / 2 - (G[:, a] + G[:, b]) / 2
            S = (C[:, a] + C[:, b]) / 4
            for i in np.flatnonzero(ok & np.isfinite(e).all(1) & np.isfinite(S).all((1, 2))):
                d2_all.append(float(e[i] @ np.linalg.solve(S[i], e[i])))
    d2_all = np.array(d2_all)
    kappa = float(np.quantile(d2_all, 0.95) / CHI2_3_95)
    gt_rp = np.concatenate(gt_rp)
    gt_rp = gt_rp[np.isfinite(gt_rp)]
    print(f"kappa {kappa:.2f}; GT stereo reprojection RMS median {np.median(gt_rp):.2f}px, "
          f"frac > 3px {np.mean(gt_rp > 3):.3f}")
    sel["cov_inflation_kappa"] = kappa
    sel["gt_reproj_rms_tune"] = {"median": float(np.median(gt_rp)), "p95": float(np.percentile(gt_rp, 95)),
                                 "frac_gt3px": float(np.mean(gt_rp > 3))}

    (ROOT / "configs/stage2_selected.json").write_text(json.dumps(sel, indent=1))
    print("wrote configs/stage2_selected.json")


if __name__ == "__main__":
    main()
