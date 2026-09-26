"""Phase 5b: tissue surface from stereo (SGM) and the tip-to-tissue distance.

Per frame: rectify the stereo pair, SGM disparity at half resolution, and for each instrument a
local tissue plane fitted (robustly) to the 3D points in an annulus around the tip, with the
instruments masked out. Distance = signed distance of the 3D tip from that plane, positive on the
camera side.

The instrument mask comes from keypoints (capsules along the keypoint chain, the shaft extended to
the image border), not from the Phase 1 segmentation model: zero-shot on SurgPose that model
labels beef tissue as instrument and misses the dVRK jaws.

There is no tissue depth ground truth in SurgPose. The reference distance d_ref uses the
ground-truth tip on the *same* SGM plane, so d_pred - d_ref isolates the error that tip perception
adds; it says nothing about SGM's own depth error.
"""

import cv2
import numpy as np

from surgscene.geometry import StereoRig

CHAIN = {"PSM1": [0, 1, 2, 3, 2, 4], "PSM3": [5, 6, 7, 8, 7, 9]}
TIPS = {"PSM1": (3, 4), "PSM3": (8, 9)}


class Rectifier:
    def __init__(self, rig: StereoRig, half: bool = True):
        W, H = rig.left.size
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(rig.left.K, rig.left.dist, rig.right.K, rig.right.dist, (W, H),
                                                    rig.R, rig.T.reshape(3, 1), alpha=0)
        self.s = 0.5 if half else 1.0
        S = np.diag([self.s, self.s, 1.0])
        self.P1, self.P2 = S @ P1, S @ P2
        self.R1 = R1
        size = (int(W * self.s), int(H * self.s))
        self.mapL = cv2.initUndistortRectifyMap(rig.left.K, rig.left.dist, R1, self.P1, size, cv2.CV_32FC1)
        self.mapR = cv2.initUndistortRectifyMap(rig.right.K, rig.right.dist, R2, self.P2, size, cv2.CV_32FC1)
        self.f = self.P1[0, 0]
        self.cx, self.cy = self.P1[0, 2], self.P1[1, 2]
        self.B = abs(self.P2[0, 3] / self.P2[0, 0])  # baseline, mm
        self.rig = rig

    def rectify(self, fl, fr):
        return (cv2.remap(fl, *self.mapL, cv2.INTER_LINEAR), cv2.remap(fr, *self.mapR, cv2.INTER_LINEAR))

    def left_px_to_rect(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(u, self.rig.left.K, self.rig.left.dist, R=self.R1, P=self.P1).reshape(-1, 2)

    def cam_to_rect(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R1.T


def make_sgbm(num_disp: int = 96, block: int = 5):
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                                 P1=8 * 3 * block**2, P2=32 * 3 * block**2, disp12MaxDiff=1, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def disparity(sgbm, rl, rr) -> np.ndarray:
    d = sgbm.compute(rl, rr).astype(np.float32) / 16.0
    d[d <= 0.5] = np.nan
    return d


def instrument_mask(shape, kps_rect: list[np.ndarray], r_shaft: float, r_jaw: float) -> np.ndarray:
    """kps_rect: list of (10, 2) keypoint sets in rectified half-res px (NaN allowed); union of capsules."""
    m = np.zeros(shape, np.uint8)
    H, W = shape
    for kp in kps_rect:
        for arm, ch in CHAIN.items():
            pts = kp[ch]
            for a, b in zip(pts[:-1], pts[1:]):
                if np.isfinite(a).all() and np.isfinite(b).all():
                    cv2.line(m, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 1, int(2 * r_jaw))
            s, w = kp[ch[0]], kp[ch[1]]  # shaft continues away from the wrist, out of the image
            if np.isfinite(s).all() and np.isfinite(w).all() and np.linalg.norm(s - w) > 1:
                d = (s - w) / np.linalg.norm(s - w)
                far = s + d * 2 * (H + W)
                cv2.line(m, tuple(np.round(w).astype(int)), tuple(np.round(far).astype(int)), 1, int(2 * r_shaft))
    return m.astype(bool)


def tissue_plane(disp, mask, center, rect: Rectifier, r_in: float, r_out: float, min_pts: int = 150):
    """Robust local tissue plane from SGM in an annulus around `center` (rectified half-res px).

    The fit is done in *disparity* space, d(x, y) = a x + b y + e, which is exact for a 3D plane
    and puts the noise where it actually is (in d). A total-least-squares fit on 3D points is
    ill-conditioned here: at ~20 cm with a 5.5 mm baseline, per-pixel depth noise (~6 mm) is as
    large as the annulus itself (~13 mm), so the smallest-variance direction is not the normal.
    Returns (n, c, n_points) with n.X = c, |n| = 1, the camera side positive; or None."""
    H, W = disp.shape
    x0, x1 = int(max(center[0] - r_out, 0)), int(min(center[0] + r_out + 1, W))
    y0, y1 = int(max(center[1] - r_out, 0)), int(min(center[1] + r_out + 1, H))
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xs - center[0], ys - center[1])
    d = disp[y0:y1, x0:x1]
    sel = (r >= r_in) & (r <= r_out) & ~mask[y0:y1, x0:x1] & np.isfinite(d)
    if sel.sum() < min_pts:
        return None
    A = np.stack([xs[sel] - center[0], ys[sel] - center[1], np.ones(sel.sum())], 1)
    y = d[sel]
    w = np.ones(len(y))
    for _ in range(5):  # IRLS with Huber weights (scale from the MAD of the residuals)
        coef = np.linalg.lstsq(A * w[:, None] ** 0.5, y * w**0.5, rcond=None)[0]
        res = y - A @ coef
        s = 1.4826 * np.median(np.abs(res)) + 1e-6
        w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(res), 1e-9))
    a, b, e0 = coef
    e = e0 - a * center[0] - b * center[1]  # back to absolute pixel coordinates
    # d = f B / Z and x = f X / Z + cx  =>  a f X + b f Y + (a cx + b cy + e) Z = f B
    n = np.array([a * rect.f, b * rect.f, a * rect.cx + b * rect.cy + e])
    k = np.linalg.norm(n)
    return -n / k, float(-rect.f * rect.B / k), int(sel.sum())


def signed_distance(X_rect: np.ndarray, plane) -> float:
    n, c = plane[0], plane[1]
    return float(X_rect @ n - c)
