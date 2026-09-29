"""v5 step 1 (development, tune only): occlusion-aware tissue memory vs the Phase 5 local plane.

  uv run --group stereo --group sam python scripts/v5_tissue_memory.py [--trajs 18 19 20 23]

Pass 1 (all 1,001 frames, 30 fps): rectify (half res, B3 row shift), RAFT-Stereo realtime@4
disparity, instrument mask = SAM 2 masks (left, rectified) + detected-keypoint capsules, dilated
DILATE px. Pass 2 (causal): TissueMemory updated every frame with non-instrument pixels; at every
5th frame, per arm, tissue near the tip from (a) the memory, (b) the Phase 5 plane (Huber fit in an
annulus of the *current* frame's disparity), (c) the raw current-frame disparity.

No tissue GT on SurgPose, so proxies:
  hidden   tissue under the instrument now (masked pixels within R_EVAL of the tip), predicted, then
           scored against the same pixel when it reappears (median of its first 5 valid
           disparities within the next 3 s; offline reference, causal predictions). mm, per frame.
  penetr   fraction of frames where the GT tip is > 3 mm behind the tissue surface along its ray
           (tips rest on or above the tissue; lower is more plausible).
  tipshare |gap(pipeline tip) - gap(GT tip)| with the memory surface: the part of the distance error
           that comes from the tip estimate (v4 hybrid), not the tissue.
Writes runs/v5_tissue/{results.json, report.md}.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from surgscene.fusion import FusionParams  # noqa: E402
from surgscene.learned_stereo import LearnedStereo  # noqa: E402
from surgscene.proximity import Rectifier, instrument_mask, tissue_plane  # noqa: E402
from surgscene.sam2_track import load_masks  # noqa: E402
from surgscene.stage2 import TIP  # noqa: E402
from surgscene.tissue_memory import TissueMemory, distances, pixel_depth, plane_depth  # noqa: E402
from surgscene.pipeline import causal_types, hybrid_length, load_prepared, run  # noqa: E402
from surgscene.rectification import shift_rows, sift_dy  # noqa: E402

RAW = ROOT / "data/surgpose/raw"
MASKS = ROOT / "data/cache/surgpose_masks"
OUT = ROOT / "runs/v5_tissue"
TUNE = [18, 19, 20, 23]
DILATE = 8          # half-res px
R_EVAL = 40         # half-res px around the tip projection
R_IN, R_OUT = 20, 60  # Phase 5 annulus (half-res px)
FUTURE = 90         # frames (3 s) to find a reappearance reference
PENETRATION_MM = 3.0
V4 = json.loads((ROOT / "configs/v4_selected.json").read_text())


def frames(traj):
    cL = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/left_video.mp4"))
    cR = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/right_video.mp4"))
    while True:
        ok1, a = cL.read()
        ok2, b = cR.read()
        if not (ok1 and ok2):
            return
        yield a, b


def pass1(traj, D, rect, net):
    """-> disparity (T, H, W) float16, instrument mask (T, H, W) bool (both rectified half res)."""
    M = load_masks(MASKS / f"{traj:06d}_left.npz")
    sift = cv2.SIFT_create(4000)
    dys = []
    for i, (a, b) in enumerate(frames(traj)):
        if i % 150 == 0:
            dys.append(np.median(sift_dy(sift, *rect.rectify(a, b))))
    dy = float(np.median(dys))
    dy = dy if abs(dy) > 0.5 else 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
    disp, mask = [], []
    for i, (a, b) in enumerate(frames(traj)):
        ra, rb = rect.rectify(a, b)
        rb = shift_rows(rb, dy) if dy else rb
        disp.append(net.disparity(ra, rb).astype(np.float16))
        m = np.zeros(ra.shape[:2], bool)
        if M["valid"][i]:
            for k in range(2):
                m |= cv2.remap(M["masks"][i, k].astype(np.uint8), *rect.mapL, cv2.INTER_NEAREST).astype(bool)
        kp = D["kfL"][i]
        kr = rect.left_px_to_rect(np.nan_to_num(kp, nan=-1e4)).reshape(10, 2)
        kr[~np.isfinite(kp).all(1)] = np.nan
        m |= instrument_mask(m.shape, [kr], 30 * rect.s * 2, 15 * rect.s * 2)
        mask.append(cv2.dilate(m.astype(np.uint8), kernel).astype(bool))
    return np.stack(disp), np.stack(mask), dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", nargs="+", type=int, default=TUNE)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    net = LearnedStereo("realtime", iters=4)
    p = FusionParams(**V4["fusion"])
    R = {}
    for traj in args.trajs:
        D = load_prepared(traj)
        rect = Rectifier(D["rig"])
        tracks = run(D, p, None)
        types = causal_types(D, tracks)
        X4, _ = hybrid_length(D, tracks, types, V4["hybrid_sd_depth_mm"], kin_sd_by_type=True,
                              apply_types=tuple(V4["length_constraint_types"]))
        disp, mask, dy = pass1(traj, D, rect, net)
        T, H, W = disp.shape
        mem = TissueMemory((H, W))
        f5 = {int(fr): i for i, fr in enumerate(D["f5"])}
        acc = {k: [] for k in ["hidden_mem", "hidden_plane", "hidden_raw", "hidden_cov_mem", "pen_mem", "pen_plane", "pen_raw",
                               "tipshare", "gap_mem_minus_plane"]}
        for t in range(T):
            if t in f5:
                i = f5[t]
                d_now = disp[t].astype(np.float32)
                for arm, (ka, kb) in TIP.items():
                    g = (D["G"][i, ka] + D["G"][i, kb]) / 2
                    if not np.isfinite(g).all():
                        continue
                    Xg = rect.R1 @ g
                    u = rect.f * Xg[:2] / Xg[2] + np.array([rect.cx, rect.cy])
                    x0, x1 = int(max(u[0] - R_OUT, 0)), int(min(u[0] + R_OUT + 1, W))
                    y0, y1 = int(max(u[1] - R_OUT, 0)), int(min(u[1] + R_OUT + 1, H))
                    if x1 - x0 < 20 or y1 - y0 < 20:
                        continue
                    Zm = pixel_depth(mem.disparity(y0, y1, x0, x1), rect)
                    plane = tissue_plane(d_now, mask[t], u, rect, R_IN, R_OUT)
                    ys, xs = np.mgrid[y0:y1, x0:x1]
                    Zp = plane_depth(xs.astype(float), ys.astype(float), plane, rect) if plane is not None else np.full(Zm.shape, np.nan)
                    Zr = pixel_depth(d_now[y0:y1, x0:x1], rect)  # raw: sees the instrument where it is
                    # hidden: masked pixels near the tip that reappear within FUTURE frames
                    near = np.hypot(xs - u[0], ys - u[1]) <= R_EVAL
                    hid = near & mask[t, y0:y1, x0:x1]
                    hy, hx = np.nonzero(hid)
                    if len(hy) > 20:
                        fut = slice(t + 1, min(t + 1 + FUTURE, T))
                        vis = ~mask[fut, y0 + hy, x0 + hx] & np.isfinite(disp[fut, y0 + hy, x0 + hx])
                        dfut = np.where(vis, disp[fut, y0 + hy, x0 + hx].astype(np.float32), np.nan)
                        ref = np.full(len(hy), np.nan)
                        for j in np.flatnonzero(vis.sum(0) >= 5):
                            ref[j] = np.median(dfut[:, j][np.isfinite(dfut[:, j])][:5])
                        ok = np.isfinite(ref)
                        if ok.sum() > 20:
                            Zref = pixel_depth(ref[ok], rect)
                            for name, Z in (("mem", Zm), ("plane", Zp), ("raw", Zr)):
                                z = Z[hy[ok], hx[ok]]
                                if name == "mem":
                                    acc["hidden_cov_mem"].append(float(np.isfinite(z).mean()))
                                if np.isfinite(z).sum() > 10:
                                    acc[f"hidden_{name}"].append(float(np.nanmedian(np.abs(z - Zref))))
                    # penetration and gaps, GT tip
                    gaps = {}
                    for name, Z in (("mem", Zm), ("plane", Zp), ("raw", Zr)):
                        gaps[name] = distances(Xg, Z, x0, y0, rect)[0]
                        if np.isfinite(gaps[name]):
                            acc[f"pen_{name}"].append(float(gaps[name] < -PENETRATION_MM))
                    if np.isfinite(gaps["mem"]) and np.isfinite(gaps["plane"]):
                        acc["gap_mem_minus_plane"].append(float(abs(gaps["mem"] - gaps["plane"])))
                    X4m = (X4[arm][i, 2] + X4[arm][i, 3]) / 2
                    if np.isfinite(X4m).all() and np.isfinite(gaps["mem"]):
                        gp = distances(rect.R1 @ X4m, Zm, x0, y0, rect)[0]
                        if np.isfinite(gp):
                            acc["tipshare"].append(float(abs(gp - gaps["mem"])))
            mem.update(disp[t].astype(np.float32), ~mask[t])
        R[traj] = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()} | {"n_hidden": len(acc["hidden_mem"]), "dy": dy}
        print(traj, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[traj].items()}), flush=True)
        del disp, mask
    (OUT / "results.json").write_text(json.dumps(R, indent=1))
    f = lambda k: np.mean([r[k] for r in R.values() if r[k] is not None])
    L = ["# v5 step 1: tissue memory vs Phase 5 local plane (tune, development proxies)", "",
         "RAFT-Stereo realtime@4, half res, every frame; evaluated every 5th frame. Means over trajectories.", "",
         "| Proxy | Memory | Plane (current frame) | Raw current frame |", "|---|---|---|---|",
         f"| Hidden tissue under the instrument: depth error vs reappearance (mm, median per frame) | {f('hidden_mem'):.2f} "
         f"(coverage {f('hidden_cov_mem'):.2f}) | {f('hidden_plane'):.2f} | {f('hidden_raw'):.2f} |",
         f"| GT tip > {PENETRATION_MM:g} mm behind the tissue surface (fraction of frames) | {f('pen_mem'):.3f} | {f('pen_plane'):.3f} | "
         f"{f('pen_raw'):.3f} |", "",
         f"- |gap(memory) − gap(plane)| with the GT tip: {f('gap_mem_minus_plane'):.2f} mm",
         f"- Distance error from the tip estimate alone (v4 tip vs GT tip, memory surface): {f('tipshare'):.2f} mm", "",
         "| Trajectory | hidden mem / plane / raw (mm) | memory coverage | penetration mem / plane / raw | tip share (mm) |",
         "|---|---|---|---|---|"]
    for t, r in R.items():
        L.append(f"| {t} | {r['hidden_mem']:.2f} / {r['hidden_plane']:.2f} / {r['hidden_raw']:.2f} | {r['hidden_cov_mem']:.2f} | "
                 f"{r['pen_mem']:.3f} / {r['pen_plane']:.3f} / {r['pen_raw']:.3f} | {r['tipshare']:.2f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
