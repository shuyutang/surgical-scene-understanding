"""v3 Phase B, selection on SurgPose tune (18, 19, 20, 23): SGM vs RAFT-Stereo checkpoints.

  uv run --group stereo python scripts/eval_stereo_tune.py

No depth GT on SurgPose, so proxies on tissue pixels in the annulus around each GT tip (the region
the proximity feature uses), instruments masked, the B3 rectification correction applied to all:
  photo   mean |gray_left - gray_right warped by the disparity| (0-255) on pixels valid for the
          method: whether matches are *correct* (the primary selection proxy)
  scatter per-pixel 3D scatter about the local tissue plane on pixels valid for both SGM and the
          method (low can also mean over-smoothing: secondary)
  valid   fraction of annulus pixels with a disparity
The checkpoint with the lowest mean photometric error is selected for the SERV-CT test (B1).
Writes runs/v3_stereo_tune/{results.json, report.md}.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from surgscene.geometry import load_stereo_ini
from surgscene.learned_stereo import CHECKPOINTS, LearnedStereo
from surgscene.proximity import TIPS, Rectifier, disparity, instrument_mask, make_sgbm, tissue_plane
from surgscene.rectification import load_gt, shift_rows, sift_dy

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/surgpose/raw"
TUNE = [18, 19, 20, 23]
N_FRAMES = 8
R_IN, R_OUT = 20, 60
OUT = ROOT / "runs/v3_stereo_tune"


def photometric(gl, gr, d, sel):
    H, W = gl.shape
    ys, xs = np.nonzero(sel)
    xr = xs - d[ys, xs]
    ok = (xr >= 0) & (xr <= W - 1)
    x0 = np.floor(xr[ok]).astype(int)
    a = xr[ok] - x0
    x1 = np.minimum(x0 + 1, W - 1)
    warped = (1 - a) * gr[ys[ok], x0] + a * gr[ys[ok], x1]
    return np.abs(gl[ys[ok], xs[ok]] - warped)


def scatter(d, sel, rect, plane):
    H, W = d.shape
    ys, xs = np.nonzero(sel)
    Z = rect.f * rect.B / d[ys, xs]
    X = np.stack([(xs - rect.cx) * Z / rect.f, (ys - rect.cy) * Z / rect.f, Z], 1)
    res = X @ plane[0] - plane[1]
    return 1.4826 * np.median(np.abs(res - np.median(res)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    methods = ["sgm"] + list(CHECKPOINTS)
    models = {}
    acc = {m: {"photo": [], "scatter": [], "valid": [], "ms": []} for m in methods}
    sgbm, sift = make_sgbm(), cv2.SIFT_create(4000)
    frames_data = []
    for traj in TUNE:
        rig = load_stereo_ini(RAW / f"{traj:06d}/StereoCalibrationDVRK.ini")
        rect = Rectifier(rig)
        cL = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/left_video.mp4"))
        cR = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/right_video.mp4"))
        n = int(cL.get(cv2.CAP_PROP_FRAME_COUNT))
        gl = load_gt(traj, "left", n)
        pairs = []
        for t in np.linspace(30, n - 30, N_FRAMES).astype(int):
            cL.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            cR.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            _, fl = cL.read()
            _, fr = cR.read()
            pairs.append((t, *rect.rectify(fl, fr)))
        dy = float(np.median(np.concatenate([sift_dy(sift, a, b) for _, a, b in pairs])))
        dy = dy if abs(dy) > 0.5 else 0.0  # B3 rule
        for t, a, b in pairs:
            b = shift_rows(b, dy) if dy else b
            g = gl[t]
            gt_r = rect.left_px_to_rect(np.nan_to_num(g, nan=-1e4)).reshape(10, 2)
            gt_r[~np.isfinite(g).all(1)] = np.nan
            frames_data.append((traj, t, rect, a, b, gt_r))

    for m in methods:
        if m != "sgm":
            models[m] = LearnedStereo(m)
        for traj, t, rect, a, b, gt_r in frames_data:
            t0 = time.perf_counter()
            d = disparity(sgbm, a, b) if m == "sgm" else models[m].disparity(a, b)
            if m != "sgm":
                torch.cuda.synchronize()
            acc[m]["ms"].append((time.perf_counter() - t0) * 1e3)
            acc[m].setdefault("_d", []).append(d)
        if m != "sgm":
            del models[m]
            torch.cuda.empty_cache()

    for i, (traj, t, rect, a, b, gt_r) in enumerate(frames_data):
        mask = instrument_mask(a.shape[:2], [gt_r], 30, 15)
        ga, gb = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (a, b)]
        H, W = mask.shape
        ys, xs = np.mgrid[0:H, 0:W]
        d_sgm = acc["sgm"]["_d"][i]
        for arm, (ka, kb) in TIPS.items():
            c = (gt_r[ka] + gt_r[kb]) / 2
            if not np.isfinite(c).all():
                continue
            r = np.hypot(xs - c[0], ys - c[1])
            ann = (r >= R_IN) & (r <= R_OUT) & ~mask
            if ann.sum() < 300:
                continue
            for m in methods:
                d = acc[m]["_d"][i]
                sel = ann & np.isfinite(d)
                acc[m]["valid"].append(float(sel.sum() / ann.sum()))
                if sel.sum() > 50:
                    acc[m]["photo"].append(float(photometric(ga, gb, d, sel).mean()))
                both = ann & np.isfinite(d) & np.isfinite(d_sgm)
                pl = tissue_plane(d, mask, c, rect, R_IN, R_OUT)
                if pl is not None and both.sum() > 50:
                    acc[m]["scatter"].append(float(scatter(d, both, rect, pl)))
    R = {m: {k: float(np.mean(v)) for k, v in acc[m].items() if not k.startswith("_") and v} |
         {"ms_median": float(np.median(acc[m]["ms"]))} for m in methods}
    learned = [m for m in methods if m != "sgm"]
    R["selected"] = min(learned, key=lambda m: R[m]["photo"])
    (OUT / "results.json").write_text(json.dumps(R, indent=1))
    L = ["# Phase B selection on SurgPose tune (proxies; no depth GT)", "",
         f"{len(frames_data)} frames (8 per trajectory, 18/19/20/23), half resolution, B3 correction applied.", "",
         "| Method | Photometric error (gray levels) | Plane scatter (mm) | Valid fraction | ms / pair |", "|---|---|---|---|---|"]
    for m in methods:
        L.append(f"| {m} | {R[m]['photo']:.2f} | {R[m]['scatter']:.2f} | {R[m]['valid']:.3f} | {R[m]['ms_median']:.0f} |")
    L += ["", f"Selected for SERV-CT (lowest photometric error): **{R['selected']}**"]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
