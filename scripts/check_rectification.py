"""v3 step B3: residual vertical misalignment after stereo rectification, and whether correcting
it helps SGM. Train + tune trajectories only (0-20, 23); test2 is not touched.

  uv run python scripts/check_rectification.py

Rectification should put a 3D point on the same row in both images. The residual row offset
dy = y_left - y_right (rectified half-res px) is measured two independent ways:
  labels  GT keypoint pairs, rectified with each eye's (R, P)
  images  SIFT matches between the rectified frames (no labels: usable at run time)
Then, on every trajectory, SGM with and without shifting the right image by the image-based dy:
valid-pixel fraction and the per-pixel scatter about a local tissue plane at the GT tips.
Writes runs/rectification/check.json and report.md.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from surgscene.geometry import load_stereo_ini
from surgscene.proximity import TIPS, Rectifier, disparity, instrument_mask, make_sgbm, tissue_plane

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/surgpose/raw"
OUT = ROOT / "runs/rectification"
TRAJS = list(range(0, 21)) + [23]
KP_IDS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]  # as in prepare_surgpose.py
N_FRAMES = 8           # frames per trajectory for the image measure and the SGM comparison
R_IN, R_OUT = 20, 60   # annulus, as in cache_tissue_planes.py
R_SHAFT, R_JAW = 30, 15


def load_gt(traj: int, eye: str, n: int) -> np.ndarray:
    """(n, 10, 2) native px, NaN where unlabeled (same keypoint order as the frame cache)."""
    raw = yaml.load(open(RAW / f"{traj:06d}/keypoints_{eye}.yaml"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    gt = np.full((n, len(KP_IDS), 2), np.nan)
    for t, kps in raw.items():
        if int(t) < n and kps:
            for j, k in enumerate(KP_IDS):
                if k in kps:
                    gt[int(t), j] = kps[k]
    return gt


def sift_dy(sift, rl, rr):
    ka, da = sift.detectAndCompute(cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.array([])
    good = [m for m, n2 in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n2.distance]
    pa = np.array([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pb = np.array([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
    ok = (pa[:, 0] - pb[:, 0] > 0) & (np.abs(pa[:, 1] - pb[:, 1]) < 8)  # positive disparity, near-epipolar
    return (pa[ok, 1] - pb[ok, 1])


def shift_rows(img, dy):
    """new(y) = img(y - dy): moves content down by dy rows (sub-pixel, bilinear)."""
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def plane_stats(disp, gt_r, rect, common):
    """Scatter over `common` pixels only (valid with and without the correction), so a correction
    that validates extra, harder pixels isn't penalized for including them."""
    mask = instrument_mask(disp.shape, [gt_r], R_SHAFT, R_JAW)
    H, W = disp.shape
    ys, xs = np.mgrid[0:H, 0:W]
    Z = rect.f * rect.B / disp
    out = []
    for a, b in TIPS.values():
        c = (gt_r[a] + gt_r[b]) / 2
        if not np.isfinite(c).all():
            continue
        p = tissue_plane(disp, mask, c, rect, R_IN, R_OUT)
        if p is None:
            continue
        r = np.hypot(xs - c[0], ys - c[1])
        sel = (r >= R_IN) & (r <= R_OUT) & ~mask & common
        if sel.sum() < 50:
            continue
        X = np.stack([(xs[sel] - rect.cx) * Z[sel] / rect.f, (ys[sel] - rect.cy) * Z[sel] / rect.f, Z[sel]], 1)
        res = X @ p[0] - p[1]
        out.append(1.4826 * np.median(np.abs(res - np.median(res))))
    return out


def work(traj: int) -> dict:
    cv2.setNumThreads(1)
    rig = load_stereo_ini(RAW / f"{traj:06d}/StereoCalibrationDVRK.ini")
    rect = Rectifier(rig)
    cL = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/left_video.mp4"))
    cR = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/right_video.mp4"))
    n = int(cL.get(cv2.CAP_PROP_FRAME_COUNT))
    gl, gr = load_gt(traj, "left", n), load_gt(traj, "right", n)

    # 1. labels
    ok = np.isfinite(gl).all(-1) & np.isfinite(gr).all(-1)
    yl = rect.left_px_to_rect(gl[ok])[:, 1]
    yr = rect.right_px_to_rect(gr[ok])[:, 1]
    d_lab = yl - yr

    # 2. images, and 3. SGM with / without the correction
    sift, sgbm = cv2.SIFT_create(4000), make_sgbm()
    frames = np.linspace(30, n - 30, N_FRAMES).astype(int)
    pairs, d_img, per_frame = [], [], []
    for t in frames:
        cL.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        cR.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        okl, fl = cL.read()
        okr, fr = cR.read()
        if not (okl and okr):
            continue
        rl, rr = rect.rectify(fl, fr)
        d = sift_dy(sift, rl, rr)
        d_img.append(d)
        per_frame.append(float(np.median(d)) if len(d) else np.nan)
        pairs.append((t, rl, rr))
    d_img_all = np.concatenate(d_img) if d_img else np.array([])
    dy_hat = float(np.median(d_img_all)) if len(d_img_all) else 0.0

    sgm = {"raw": {"valid": [], "scatter": []}, "corrected": {"valid": [], "scatter": []}}
    for t, rl, rr in pairs:
        g = gl[t]
        gt_r = rect.left_px_to_rect(np.nan_to_num(g, nan=-1e4)).reshape(10, 2)
        gt_r[~np.isfinite(g).all(1)] = np.nan
        disps = {"raw": disparity(sgbm, rl, rr), "corrected": disparity(sgbm, rl, shift_rows(rr, dy_hat))}
        common = np.isfinite(disps["raw"]) & np.isfinite(disps["corrected"])
        for name, disp in disps.items():
            sgm[name]["valid"].append(float(np.isfinite(disp).mean()))
            sgm[name]["scatter"] += plane_stats(disp, gt_r, rect, common)

    return {
        "traj": traj,
        "labels": {"n": int(ok.sum()), "median": float(np.median(d_lab)),
                   "iqr": float(np.subtract(*np.percentile(d_lab, [75, 25])))},
        "images": {"n": int(len(d_img_all)), "median": dy_hat,
                   "iqr": float(np.subtract(*np.percentile(d_img_all, [75, 25]))) if len(d_img_all) else None,
                   "per_frame_median": per_frame},
        "sgm": {k: {"valid": float(np.mean(v["valid"])), "scatter_mm": float(np.median(v["scatter"])),
                    "n_planes": len(v["scatter"])} for k, v in sgm.items()},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        res = sorted(ex.map(work, TRAJS), key=lambda r: r["traj"])
    (OUT / "check.json").write_text(json.dumps(res, indent=1))
    L = ["# B3: residual vertical misalignment after rectification (train + tune trajectories)", "",
         "dy = y_left − y_right in rectified half-res px. SGM: valid-pixel fraction and median per-pixel "
         "scatter (mm, over pixels valid in both) about the local tissue plane at the GT tips, without / with shifting the right "
         "image by the image-based dy.", "",
         "| traj | dy labels (IQR) | dy images (IQR) | per-frame image dy range | valid raw → corr | scatter mm raw → corr |",
         "|---|---|---|---|---|---|"]
    for r in res:
        pf = [x for x in r["images"]["per_frame_median"] if np.isfinite(x)]
        s = r["sgm"]
        L.append(f"| {r['traj']} | {r['labels']['median']:+.2f} ({r['labels']['iqr']:.2f}) | "
                 f"{r['images']['median']:+.2f} ({r['images']['iqr']:.2f}) | "
                 f"{min(pf):+.2f} … {max(pf):+.2f} | {s['raw']['valid']:.3f} → {s['corrected']['valid']:.3f} | "
                 f"{s['raw']['scatter_mm']:.2f} → {s['corrected']['scatter_mm']:.2f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
