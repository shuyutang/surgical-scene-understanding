"""v5: occlusion-aware tissue memory for instrument-to-tissue distance.

Per-frame stereo can't see the tissue under the instrument, which is where the distance matters
most. SurgPose's endoscope is static (< 1 px drift over 33 s), so the memory lives in the rectified
left image: per pixel, a ring of the last K valid tissue disparities, written only where no
instrument is (mask dilated against the disparity bleed at instrument edges). Its median is the
tissue estimate; it adapts to tissue motion within ~K/2 observations, and keeps the tissue seen
before an instrument covered it. A moving camera would need registration first (not done here).
"""

import numpy as np

from .proximity import Rectifier


class TissueMemory:
    def __init__(self, shape: tuple[int, int], K: int = 15):
        self.buf = np.full((K,) + shape, np.nan, np.float32)
        self.slot = np.zeros(shape, np.int16)
        self.K = K

    def update(self, disp: np.ndarray, valid: np.ndarray) -> None:
        ys, xs = np.nonzero(valid & np.isfinite(disp))
        s = self.slot[ys, xs]
        self.buf[s, ys, xs] = disp[ys, xs]
        self.slot[ys, xs] = (s + 1) % self.K

    def disparity(self, y0: int, y1: int, x0: int, x1: int, min_obs: int = 3) -> np.ndarray:
        """Median disparity over the ring in a window; NaN where fewer than min_obs observations."""
        w = self.buf[:, y0:y1, x0:x1]
        n = np.isfinite(w).sum(0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(w, 0)
        m[n < min_obs] = np.nan
        return m


def pixel_depth(d: np.ndarray, rect: Rectifier) -> np.ndarray:
    return rect.f * rect.B / d


def backproject(xs, ys, Z, rect: Rectifier) -> np.ndarray:
    return np.stack([(xs - rect.cx) * Z / rect.f, (ys - rect.cy) * Z / rect.f, Z], -1)


def plane_depth(xs, ys, plane, rect: Rectifier) -> np.ndarray:
    """Depth where each pixel's viewing ray meets the plane n.X = c (rectified frame)."""
    n, c = plane[0], plane[1]
    ray = np.stack([(xs - rect.cx) / rect.f, (ys - rect.cy) / rect.f, np.ones_like(xs, float)], -1)
    return c / (ray @ n)


def distances(X_tip: np.ndarray, Zmap: np.ndarray, x0: int, y0: int, rect: Rectifier, q: float = 2.0):
    """X_tip (3,) rectified frame; Zmap tissue depth over a window starting at (x0, y0).
    Returns (signed depth gap along the tip's viewing ray: tissue Z - tip Z, positive = tip in front;
    near-surface distance: q-th percentile of |X - X_tip| over the window's tissue points)."""
    u = rect.f * X_tip[:2] / X_tip[2] + np.array([rect.cx, rect.cy])
    xi, yi = int(round(u[0])) - x0, int(round(u[1])) - y0
    H, W = Zmap.shape
    gap = np.nan
    if 1 <= xi < W - 1 and 1 <= yi < H - 1:
        z = Zmap[yi - 1:yi + 2, xi - 1:xi + 2]
        if np.isfinite(z).any():
            gap = float(np.nanmedian(z) - X_tip[2])
    ys, xs = np.nonzero(np.isfinite(Zmap))
    if len(xs) < 20:
        return gap, np.nan
    P = backproject(xs + x0, ys + y0, Zmap[ys, xs], rect)
    return gap, float(np.percentile(np.linalg.norm(P - X_tip, axis=1), q))
