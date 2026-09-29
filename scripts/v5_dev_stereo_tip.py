"""v5 step 0 (development, tune only): instrument tip depth from dense learned stereo on jaw pixels.

  uv run --group stereo --group sam python scripts/v5_dev_stereo_tip.py [--every 10]

Keypoint triangulation is limited by independent 2-3 px detections in each eye (1 px of disparity
~ 3.7 mm of depth at 200 mm on this rig); dense stereo matches sub-pixel over whole regions. Per
frame and arm:
  1. rectify (half resolution, B3 row shift), RAFT-Stereo disparity;
  2. jaw pixels = SAM 2 mask of the arm (left, native) within a band of +-BAND px around the
     detected pivot -> tip-midpoint segment (v2 Kalman 2D), t in [T0, 1] along it; the mask keeps
     out the tissue seen between open jaws;
  3. robust fit of disparity as an affine function of the rectified pixel position along the jaw
     (exact for a straight 3D segment: inverse depth is affine in image coordinates), evaluated at
     the tip-midpoint's rectified position -> 3D point on its viewing ray.
Compared with the GT tip midpoint (GT label triangulation, as every 3D endpoint here) against:
v2 keypoint triangulation, the v3 kinematic fusion point, v3 hybrid, v4 hybrid.
Writes runs/v5_dev/{results.json, report.md}.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from surgscene.fusion import ARMS, FusionParams
from surgscene.learned_stereo import LearnedStereo
from surgscene.pipeline import causal_types, hybrid, hybrid_length, load_prepared, run
from surgscene.proximity import Rectifier
from surgscene.rectification import shift_rows, sift_dy
from surgscene.sam2_track import load_masks
from surgscene.stage2 import TIP

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/surgpose/raw"
MASKS = ROOT / "data/cache/surgpose_masks"
OUT = ROOT / "runs/v5_dev"
TUNE = [18, 19, 20, 23]
BAND, T0 = 50.0, 0.2
V4 = json.loads((ROOT / "configs/v4_selected.json").read_text())
MODELS = {"realtime@4": ("realtime", 4), "middlebury@32": ("middlebury", 32)}


def read_pair(traj, t):
    out = []
    for eye in ("left", "right"):
        cap = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/{eye}_video.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        out.append(cap.read()[1])
    return out


def jaw_tip_from_disparity(d, rect, mask, u_piv, u_mid):
    """-> (X_cam (3,), n_pixels) or (None, n)."""
    seg = u_mid - u_piv
    L = np.linalg.norm(seg)
    if not np.isfinite(L) or L < 5:
        return None, 0
    e = seg / L
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float) - u_piv
    t, q = P @ e / L, P @ np.array([-e[1], e[0]])
    sel = (t >= T0) & (t <= 1.05) & (np.abs(q) <= BAND)
    if sel.sum() < 30:
        return None, int(sel.sum())
    pts = np.stack([xs[sel], ys[sel]], 1).astype(float)
    r = rect.left_px_to_rect(pts)
    H, W = d.shape
    xi, yi = np.round(r[:, 0]).astype(int), np.round(r[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi, yi, r = xi[ok], yi[ok], r[ok]
    dv = d[yi, xi]
    ok = np.isfinite(dv)
    if ok.sum() < 30:
        return None, int(ok.sum())
    A = np.c_[np.ones(ok.sum()), r[ok]]
    y = dv[ok]
    w = np.ones(len(y))
    for _ in range(10):  # Huber IRLS, scale from the MAD
        beta = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        res = y - A @ beta
        s = 1.4826 * np.median(np.abs(res)) + 1e-3
        w = np.sqrt(np.minimum(1.0, 1.345 * s / np.maximum(np.abs(res), 1e-9)))
    rm = rect.left_px_to_rect(u_mid[None])[0]
    dm = float(np.r_[1.0, rm] @ beta)
    if dm <= 0:
        return None, int(ok.sum())
    Z = rect.f * rect.B / dm
    Xr = np.array([(rm[0] - rect.cx) * Z / rect.f, (rm[1] - rect.cy) * Z / rect.f, Z])
    return rect.R1.T @ Xr, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nets = {k: LearnedStereo(c, iters=i) for k, (c, i) in MODELS.items()}
    sift = cv2.SIFT_create(4000)
    p = FusionParams(**V4["fusion"])
    rows = {}
    for traj in TUNE:
        D = load_prepared(traj)
        tracks = run(D, p, None)
        types = causal_types(D, tracks)
        X3, _ = hybrid(D, tracks, V4["hybrid_sd_depth_mm"])
        X4, _ = hybrid_length(D, tracks, types, V4["hybrid_sd_depth_mm"], kin_sd_by_type=True,
                              apply_types=tuple(V4["length_constraint_types"]))
        M = load_masks(MASKS / f"{traj:06d}_left.npz")
        rect = Rectifier(D["rig"])
        f5 = D["f5"]
        sel_idx = np.arange(0, len(f5), args.every // 5)
        dy = float(np.median(np.concatenate([sift_dy(sift, *rect.rectify(*read_pair(traj, f5[i])))
                                             for i in sel_idx[:: max(1, len(sel_idx) // 6)]])))
        dy = dy if abs(dy) > 0.5 else 0.0
        errs = {}
        for i in sel_idx:
            fr = f5[i]
            a, b = rect.rectify(*read_pair(traj, fr))
            b = shift_rows(b, dy) if dy else b
            disp = {k: net.disparity(a, b) for k, net in nets.items()}
            for k_arm, (arm, (ka, kb)) in enumerate(TIP.items()):
                g = (D["G"][i, ka] + D["G"][i, kb]) / 2
                if not np.isfinite(g).all():
                    continue
                ray = g / np.linalg.norm(g)
                e = errs.setdefault(arm, {})
                cands = {"v2_triangulation": (D["V2"][i, ka] + D["V2"][i, kb]) / 2,
                         "v3_kinematic": (tracks[arm].X[fr, 2] + tracks[arm].X[fr, 3]) / 2,
                         "v3_hybrid": (X3[arm][i, 2] + X3[arm][i, 3]) / 2,
                         "v4_hybrid": (X4[arm][i, 2] + X4[arm][i, 3]) / 2}
                u_piv = D["kfL"][fr, ARMS[arm]][2]
                u_mid = (D["kfL"][fr, ka] + D["kfL"][fr, kb]) / 2
                valid = bool(M["valid"][fr])
                for k, dmap in disp.items():
                    X, n = jaw_tip_from_disparity(dmap, rect, M["masks"][fr, k_arm], u_piv, u_mid) if valid else (None, 0)
                    cands[f"stereo_jaw_{k}"] = X if X is not None else np.full(3, np.nan)
                if not all(np.isfinite(c).all() for c in cands.values()):
                    continue  # paired: only frames where every method has an estimate
                for k, c in cands.items():
                    err = c - g
                    e.setdefault(k, []).append((float(np.linalg.norm(err)), float(abs(err @ ray))))
        rows[traj] = {arm: {k: {"n": len(v), "err": float(np.mean([x[0] for x in v])), "depth": float(np.mean([x[1] for x in v])),
                                "median": float(np.median([x[0] for x in v]))} for k, v in e.items()}
                      for arm, e in errs.items()}
        print(traj, json.dumps({arm: {k: round(v["err"], 2) for k, v in r.items()} for arm, r in rows[traj].items()}), flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    methods = list(next(iter(next(iter(rows.values())).values())))
    L = ["# v5 step 0: tip depth from dense stereo on jaw pixels (tune, development)", "",
         f"Every {args.every}th frame; frames where all methods have an estimate (paired). Mean 3D tip-midpoint error, mm "
         "(along the viewing ray in brackets).", "",
         "| Arm-trajectory | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for traj, r in rows.items():
        for arm, m in r.items():
            L.append(f"| {traj}/{arm} | {m[methods[0]]['n']} | " +
                     " | ".join(f"{m[k]['err']:.2f} ({m[k]['depth']:.2f})" for k in methods) + " |")
    means = {k: np.mean([m[k]["err"] for r in rows.values() for m in r.values()]) for k in methods}
    L.append("| **mean over arm-trajectories** | | " + " | ".join(f"**{means[k]:.2f}**" for k in methods) + " |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
