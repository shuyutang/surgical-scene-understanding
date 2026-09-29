"""The 3D tool-state pipelines evaluated in v3 and v4, shared by the evaluation scripts.

  prepare / cached_gated   stage-2 inputs per trajectory (v2 gated MAP + Kalman 2D, GT triangulation)
  run_variant, hybrid      v3 Phase C: kinematics fusion + hybrid tip triangulation
  v3c_per_trajectory       v3 Phase C endpoint quantities (C1-C6) for one trajectory
  run, causal_types,
  hybrid_length            v4: causal instrument type + jaw-length constraint
Moved verbatim from the scripts that pre-registered them (eval_fusion.py, eval_stage2.py,
eval_v3c.py, v4_dev_fusion.py); behaviour is unchanged.
"""

import json
import pickle

import numpy as np

from .fusion import (ARMS, FusionParams, ToolGeometry, calib_observations, classify_type, fit_calibration, fuse_arm,
                     length_constrained_depth, load_kinematics, observations, pixel_std, triangulate_with_depth_prior)
from .geometry import register_rigid_robust
from .sam2_track import load_masks
from .stage2 import ROOT, TIP, framewise, load_obs, load_rig, load_selected, shape_models, triangulate_seq
from .temporal import KFParams, filter_keypoints

EST = ROOT / "data/cache/stage2_est"
PREPARED = ROOT / "runs/v4_dev"  # cache of prepare() outputs


def cached_gated(traj, eye, sfx, models, hp):
    key = "_".join(f"{k}{hp[k]:g}" for k in sorted(hp))
    path = EST / f"{traj:06d}_{eye}{sfx}_{key}.npy"
    if path.exists():
        return np.load(path)
    EST.mkdir(parents=True, exist_ok=True)
    O = load_obs(traj, eye)
    P = framewise(O["kp" + sfx], O["conf" + sfx], O["sigma" + sfx], models, "gated", hp)
    np.save(path, P)
    return P


LABEL_SD = 0.46 / 0.6745 * 0.7071  # px per eye: median |L-R label disagreement| 0.46 px -> per-eye std


def prepare(traj, models, hp, kfp):
    O, OR = load_obs(traj, "left"), load_obs(traj, "right")
    gL, gR = cached_gated(traj, "left", "", models, hp), cached_gated(traj, "right", "", models, hp)
    rig = load_rig(traj)
    f5 = np.arange(0, len(O["gt"]), 5)
    one = np.ones((len(f5), 10))
    G, _, rp = triangulate_seq(rig, O["gt"][f5], OR["gt"][f5], one, one)
    G[rp > 3] = np.nan
    kf, std, *_ = filter_keypoints(gL, O["sigma"], O["conf"], kfp)
    kfR, stdR, *_ = filter_keypoints(gR, OR["sigma"], OR["conf"], kfp)
    V2, _, _ = triangulate_seq(rig, kf[f5], kfR[f5], std[f5], stdR[f5])
    # reference noise: depth std of the GT-triangulated tip midpoint, from the label disagreement
    # between eyes (vertical residual after triangulation ~ label noise; median 0.46 px on tune)
    Gc, Cc, _ = triangulate_seq(rig, O["gt"][f5], OR["gt"][f5], one * LABEL_SD, one * LABEL_SD)
    ref_sd = {}
    for arm, (a, b) in TIP.items():
        g = (G[:, a] + G[:, b]) / 2
        ray = g / np.linalg.norm(g, axis=1, keepdims=True)
        C = (Cc[:, a] + Cc[:, b]) / 4
        ref_sd[arm] = np.sqrt(np.einsum("fi,fij,fj->f", ray, C, ray))
    return dict(O=O, OR=OR, gL=gL, gR=gR, rig=rig, f5=f5, G=G, V2=V2, kin=load_kinematics(traj), kfL=kf, kfR=kfR, std=std, stdR=stdR,
                ref_sd=ref_sd)


def run_variant(D, p: FusionParams, oracle=False):
    """oracle: hand-eye registered to the GT-triangulated pivot, geometry fitted to the GT 2D labels,
    both over the whole trajectory (not causal; diagnostic upper bound)."""
    """-> {arm: FusionTrack}"""
    out = {}
    sL = pixel_std(D["O"]["sigma"], D["O"]["conf"], p)
    sR = pixel_std(D["OR"]["sigma"], D["OR"]["conf"], p)
    for arm, sl in ARMS.items():
        Z, S = observations(D["gL"][:, sl], D["gR"][:, sl], sL[:, sl], sR[:, sl])
        prior = ToolGeometry.load(arm)
        calib = None
        if oracle:  # calibration fitted to the GT labels over the whole trajectory (not causal)
            k = D["kin"][arm]
            Zg, Sg = observations(D["O"]["gt"][:, sl], D["OR"]["gt"][:, sl], np.ones((len(Z), 5)), np.ones((len(Z), 5)))
            Zc, Sc = calib_observations(Zg, Sg)
            f = np.arange(0, len(Z), 3)
            piv = D["G"][:, sl][:, 2]
            ok = np.isfinite(piv).all(1)
            R0, t0 = register_rigid_robust(k["t"][D["f5"]][ok], piv[ok])
            R, t, g, _ = fit_calibration(D["rig"], k["R"][f], k["t"][f], k["q"][f, 5], Zc[f], Sc[f], R0, t0,
                                         prior, prior, fix_handeye=True)
            calib = (R, t, g)
        out[arm] = fuse_arm(D["rig"], D["kin"][arm], prior, Z, S, p, calib_fixed=calib)
    return out


def hybrid(D, tracks, sd_depth):
    """Per tip: triangulate the v2 Kalman 2D detections with a depth prior from the fused point.
    Falls back to the fused point where a detection is missing."""
    out, cov = {}, {}
    for arm, (a, b) in TIP.items():
        X = tracks[arm].X[D["f5"]].copy()
        C = tracks[arm].cov[D["f5"]].copy()
        for j, kp in ((2, a), (3, b)):
            for i, fr in enumerate(D["f5"]):
                uL, uR = D["kfL"][fr, kp], D["kfR"][fr, kp]
                sL, sR = D["std"][fr, kp], D["stdR"][fr, kp]
                if np.isfinite(X[i, j]).all() and np.isfinite(uL).all() and np.isfinite(uR).all() and np.isfinite([sL, sR]).all():
                    X[i, j], C[i, j] = triangulate_with_depth_prior(D["rig"], uL, uR, max(sL, 0.5), max(sR, 0.5),
                                                                     X[i, j], sd_depth)
        out[arm], cov[arm] = X, C
    return out, cov


CHI2_3_95 = 7.8147


SWAP_MARGIN_MM = 3.0


def v3c_per_trajectory(D, cfg, p):
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


MASKS = ROOT / "data/cache/surgpose_masks"


GATE_PX = 10.0  # tip-midpoint detection farther than this outside its instrument mask: no length constraint


def load_prepared(traj):
    path = PREPARED / f"prepared_{traj}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    sel = load_selected()
    models, _ = shape_models()
    D = prepare(traj, models, sel["gated"], KFParams(**sel["kf"]))
    PREPARED.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(D))
    return D


def oracle_type(D, arm):
    a, b = TIP[arm]
    G = D["G"]
    v = (G[:, a] + G[:, b]) / 2 - G[:, ARMS[arm]][:, 2]
    return "long" if np.nanmedian(np.linalg.norm(v, axis=1)) > 15.0 else "standard"


def run(D, p: FusionParams, types: dict | None):
    """types: {arm: (T,) type names or None} or None (v3 behaviour)."""
    sL = pixel_std(D["O"]["sigma"], D["O"]["conf"], p)
    sR = pixel_std(D["OR"]["sigma"], D["OR"]["conf"], p)
    tracks = {}
    for arm, sl in ARMS.items():
        Z, S = observations(D["gL"][:, sl], D["gR"][:, sl], sL[:, sl], sR[:, sl])
        lib = {t: ToolGeometry.from_library(arm, t) for t in ("standard", "long")} if types else None
        tracks[arm] = fuse_arm(D["rig"], D["kin"][arm], ToolGeometry.load(arm), Z, S, p,
                               type_priors=lib, type_seq=types[arm] if types else None)
    return tracks


LIB = json.loads((ROOT / "configs/v4_tool_library.json").read_text())


def causal_types(D, tracks):
    f = D["rig"].left.K[0, 0]
    out = {}
    for arm, (a, b) in TIP.items():
        piv = tracks[arm].X[:, 0]
        uP = D["kfL"][:, ARMS[arm]][:, 2]
        uM = (D["kfL"][:, a] + D["kfL"][:, b]) / 2
        out[arm] = classify_type(piv, uP, uM, f)
    return out


def mask_gate(traj, D):
    """{arm: (T,) bool} True where both left tip detections are within GATE_PX of their instrument's
    mask (not the midpoint: with open jaws it sits in the gap between them, outside the mask)."""
    import cv2
    d = load_masks(MASKS / f"{traj:06d}_left.npz")
    M, valid = d["masks"], d["valid"]
    out = {}
    for k, (arm, (a, b)) in enumerate(TIP.items()):
        u = D["kfL"][:, [a, b]]
        ok = np.ones(len(u), bool)
        for fr in D["f5"]:
            if not valid[fr] or not np.isfinite(u[fr]).all():
                continue
            dist = cv2.distanceTransform((~M[fr, k]).astype(np.uint8), cv2.DIST_L2, 5)
            xy = np.clip(np.round(u[fr]).astype(int), 0, [M.shape[-1] - 1, M.shape[-2] - 1])
            ok[fr] = bool((dist[xy[:, 1], xy[:, 0]] <= GATE_PX).all())
        out[arm] = ok
    return out


def hybrid_length(D, tracks, types, sd_kin, gate=None, kin_sd_by_type=False, apply_types=("standard", "long")):
    """v3 hybrid, but each tip's depth prior is shifted along the tip-midpoint ray by the
    length-constrained correction (fusion.length_constrained_depth), with the type's train length
    and length spread. types: {arm: (T,) type names or None}; None -> the v3 prior."""
    rig, f5 = D["rig"], D["f5"]
    out, cov = {}, {}
    Kinv = np.linalg.inv(rig.left.K)
    for arm, (a, b) in TIP.items():
        X = tracks[arm].X[f5].copy()
        C = tracks[arm].cov[f5].copy()
        for i, fr in enumerate(f5):
            if not np.isfinite(X[i]).all():
                continue
            uL, uR = D["kfL"][fr, [a, b]], D["kfR"][fr, [a, b]]
            sL, sR = D["std"][fr, [a, b]], D["stdR"][fr, [a, b]]
            if not (np.isfinite(uL).all() and np.isfinite(uR).all() and np.isfinite(sL).all() and np.isfinite(sR).all()):
                continue
            centres, sd = X[i, 2:4].copy(), sd_kin
            typ = types[arm][fr] if types else None
            if typ in apply_types and (gate is None or gate[arm][fr]):
                g = LIB["types"][typ]
                r = Kinv @ np.r_[uL.mean(0), 1.0]
                r /= np.linalg.norm(r)
                mid = X[i, 2:4].mean(0)
                sd_lat = float(r @ mid) * np.hypot(*sL) / 2 / rig.left.K[0, 0]  # tip-midpoint 2D std -> mm
                sk = g[arm]["scatter_mm_median"] if kin_sd_by_type else sd_kin
                c, sd = length_constrained_depth(r, X[i, 0], g[arm]["tip_offset_mm"], g["length_sd_mm"], mid, sk,
                                                 sd_lat)
                centres = centres + (c - mid)
            for j in range(2):
                X[i, 2 + j], C[i, 2 + j] = triangulate_with_depth_prior(rig, uL[j], uR[j], max(sL[j], 0.5), max(sR[j], 0.5),
                                                                        centres[j], sd)
        out[arm], cov[arm] = X, C
    return out, cov
