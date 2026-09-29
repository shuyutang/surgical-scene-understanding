"""SurgPose rectification helpers: GT keypoint loading, the B3 vertical-offset measure and fix.

B3 (v3): 18 of 22 SurgPose trajectories have a 1-2 px row offset after rectification; the right
image is shifted when the SIFT-measured median |dy| > 0.5 px (scripts/check_rectification.py).
"""

import cv2
import numpy as np
import yaml

from .stage2 import RAW

KP_IDS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]  # as in prepare_surgpose.py


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
