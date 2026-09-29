"""SGM tissue planes around each instrument tip, every frame of the stage-2 trajectories.

  uv run python scripts/cache_tissue_planes.py

The plane is centered on the ground-truth tip and masks both GT and predicted instruments, so
the reference distance and the predicted distance share one surface (see proximity.py).
Writes data/cache/tissue_planes/<traj>.npz: plane (T, 2, 4) = [n, c] in the rectified left frame,
npts (T, 2), R1 (3, 3).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from surgscene.proximity import TIPS, Rectifier, disparity, instrument_mask, make_sgbm, tissue_plane
from surgscene.stage2 import RAW, ROOT, TEST2, TUNE, load_obs, load_rig, load_selected, shape_models
from surgscene.temporal import KFParams, filter_keypoints

OUT = ROOT / "data/cache/tissue_planes"
R_IN, R_OUT = 20, 60        # half-res px annulus around the tip
R_SHAFT, R_JAW = 30, 15     # half-res px instrument capsule radii


def gated(traj, eye):
    from surgscene.pipeline import cached_gated
    sel = load_selected()
    models, _ = shape_models()
    return cached_gated(traj, eye, "", models, sel["gated"])


def work(traj: int):
    cv2.setNumThreads(1)
    sel = load_selected()
    kfp = KFParams(**sel["kf"])
    O = load_obs(traj, "left")
    key = "_".join(f"{k}{sel['gated'][k]:g}" for k in sorted(sel["gated"]))
    g = np.load(ROOT / f"data/cache/stage2_est/{traj:06d}_left_{key}.npy")
    kf, *_ = filter_keypoints(g, O["sigma"], O["conf"], kfp)
    rig = load_rig(traj)
    rect = Rectifier(rig)
    sgbm = make_sgbm()
    cL = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/left_video.mp4"))
    cR = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/right_video.mp4"))
    T = len(O["gt"])
    plane = np.full((T, 2, 4), np.nan)
    npts = np.zeros((T, 2), int)
    for t in range(T):
        okl, fl = cL.read()
        okr, fr = cR.read()
        if not (okl and okr):
            break
        rl, rr = rect.rectify(fl, fr)
        disp = disparity(sgbm, rl, rr)
        gt_r = rect.left_px_to_rect(np.nan_to_num(O["gt"][t], nan=-1e4)).reshape(10, 2)
        gt_r[~np.isfinite(O["gt"][t]).all(1)] = np.nan
        kf_r = rect.left_px_to_rect(np.nan_to_num(kf[t], nan=-1e4)).reshape(10, 2)
        kf_r[~np.isfinite(kf[t]).all(1)] = np.nan
        mask = instrument_mask(disp.shape, [gt_r, kf_r], R_SHAFT, R_JAW)
        for i, (arm, (a, b)) in enumerate(TIPS.items()):
            c = (gt_r[a] + gt_r[b]) / 2
            if not np.isfinite(c).all():
                continue
            p = tissue_plane(disp, mask, c, rect, R_IN, R_OUT)
            if p is not None:
                plane[t, i, :3], plane[t, i, 3], npts[t, i] = p[0], p[1], p[2]
    np.savez_compressed(OUT / f"{traj:06d}.npz", plane=plane, npts=npts, R1=rect.R1)
    return traj, float(np.isfinite(plane[..., 3]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [t for t in TUNE + TEST2 if not (OUT / f"{t:06d}.npz").exists()]
    for t in todo:  # gated estimates (no ground truth involved) are needed for the predicted-instrument mask
        gated(t, "left")
        gated(t, "right")
    with ProcessPoolExecutor(args.workers) as ex:
        for traj, frac in ex.map(work, todo):
            print(f"{traj}: plane found in {frac:.3f} of (frame, arm)", flush=True)


if __name__ == "__main__":
    main()
