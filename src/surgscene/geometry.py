"""Camera geometry for Phase 5: calibration, distortion, triangulation with covariance, SO(3), rigid frames.

Written from first principles (numpy only; OpenCV is used in tests as the reference) so the C++
runtime can port it line by line.

Conventions
- Pixels u = (x, y). Normalized image coordinates m = (X/Z, Y/Z) before distortion.
- Distortion is Brown-Conrady, OpenCV / Bouguet order (k1, k2, p1, p2, k3).
- Stereo extrinsics from SurgPose's StereoCalibrationDVRK.ini: X_right = R @ X_left + T (mm).
  The left camera frame is the reference frame for everything 3D.
- Rigid transforms are 4x4 homogeneous matrices named T_a_b: they map points in frame b to frame a.
"""

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    K: np.ndarray      # (3, 3)
    dist: np.ndarray   # (5,) k1, k2, p1, p2, k3
    size: tuple[int, int]

    def distort(self, m: np.ndarray) -> np.ndarray:
        """Normalized undistorted (N, 2) -> normalized distorted (N, 2)."""
        k1, k2, p1, p2, k3 = self.dist
        x, y = m[:, 0], m[:, 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], 1)

    def project(self, X: np.ndarray) -> np.ndarray:
        """3D points (N, 3) in this camera's frame -> pixels (N, 2)."""
        md = self.distort(X[:, :2] / X[:, 2:3])
        return md @ self.K[:2, :2].T + self.K[:2, 2]

    def undistort(self, u: np.ndarray, iters: int = 20) -> np.ndarray:
        """Pixels (N, 2) -> normalized undistorted (N, 2). Fixed-point iteration, as in OpenCV."""
        md = (u - self.K[:2, 2]) @ np.linalg.inv(self.K[:2, :2]).T
        m = md.copy()
        k1, k2, p1, p2, k3 = self.dist
        for _ in range(iters):
            x, y = m[:, 0], m[:, 1]
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            m = np.stack([(md[:, 0] - dx) / radial, (md[:, 1] - dy) / radial], 1)
        return m


@dataclass
class StereoRig:
    left: Camera
    right: Camera
    R: np.ndarray  # (3, 3)  X_right = R X_left + T
    T: np.ndarray  # (3,) mm

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))


def load_stereo_ini(path: Path) -> StereoRig:
    cp = configparser.ConfigParser()
    cp.read(path)

    def cam(sec):
        s = cp[sec]
        K = np.array([[float(s["fc_x"]), 0, float(s["cc_x"])], [0, float(s["fc_y"]), float(s["cc_y"])], [0, 0, 1]])
        dist = np.array([float(s[f"kc_{i}"]) for i in range(5)])
        return Camera(K, dist, (int(s["res_x"]), int(s["res_y"])))

    s = cp["StereoRight"]
    R = np.array([float(s[f"R_{i}"]) for i in range(9)]).reshape(3, 3)
    T = np.array([float(s[f"T_{i}"]) for i in range(3)])
    return StereoRig(cam("StereoLeft"), cam("StereoRight"), R, T)


# ---------------------------------------------------------------- triangulation

def triangulate_dlt(mL: np.ndarray, mR: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation from normalized undistorted coords. (N, 2) x2 -> (N, 3) left frame."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, T[:, None]])
    out = np.empty((len(mL), 3))
    for i, (a, b) in enumerate(zip(mL, mR)):
        A = np.stack([a[0] * P1[2] - P1[0], a[1] * P1[2] - P1[1], b[0] * P2[2] - P2[0], b[1] * P2[2] - P2[1]])
        X = np.linalg.svd(A)[2][-1]
        out[i] = X[:3] / X[3]
    return out


def _proj_both(rig: StereoRig, X: np.ndarray) -> np.ndarray:
    """(3,) -> (4,) = [uL, vL, uR, vR]."""
    return np.concatenate([rig.left.project(X[None])[0], rig.right.project((rig.R @ X + rig.T)[None])[0]])


def _jac_both(rig: StereoRig, X: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """d[uL, vL, uR, vR]/dX by central differences, (4, 3). h in mm."""
    J = np.empty((4, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = (_proj_both(rig, X + e) - _proj_both(rig, X - e)) / (2 * h)
    return J


@dataclass
class Triangulated:
    X: np.ndarray        # (3,) left camera frame, mm
    cov: np.ndarray      # (3, 3) first-order covariance, mm^2
    reproj_px: float     # RMS reprojection residual over the 4 coordinates


def triangulate(rig: StereoRig, uL: np.ndarray, uR: np.ndarray, sigL: float = 1.0, sigR: float = 1.0,
                iters: int = 5) -> Triangulated:
    """DLT initialization, then Gauss-Newton on the pixel reprojection error (with distortion).
    The covariance is (J^T W J)^-1 at the optimum: pixel noise sigL, sigR (std, px) pushed to 3D."""
    X = triangulate_dlt(rig.left.undistort(uL[None]), rig.right.undistort(uR[None]), rig.R, rig.T)[0]
    z = np.concatenate([uL, uR])
    W = np.array([1 / sigL**2] * 2 + [1 / sigR**2] * 2)
    for _ in range(iters):
        J = _jac_both(rig, X)
        r = _proj_both(rig, X) - z
        dX = np.linalg.solve(J.T @ (W[:, None] * J), -J.T @ (W * r))
        X = X + dX
        if np.abs(dX).max() < 1e-6:
            break
    J = _jac_both(rig, X)
    r = _proj_both(rig, X) - z
    return Triangulated(X, np.linalg.inv(J.T @ (W[:, None] * J)), float(np.sqrt((r**2).mean())))


# ---------------------------------------------------------------- SO(3) and rigid transforms

def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Axis-angle (rotation vector) -> rotation matrix (Rodrigues)."""
    th = np.linalg.norm(w)
    W = hat(w)
    if th < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th**2 * W @ W


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector, stable near 0 and pi."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-6:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - th < 1e-2:
        # near pi, sin(th) -> 0: read a a^T off the symmetric part, (R + R^T)/2 = cos(th) I + (1 - cos(th)) a a^T,
        # then take the sign from the skew part, which is 2 sin(th) [a]x
        aaT = ((R + R.T) / 2 - c * np.eye(3)) / (1 - c)
        k = int(np.argmax(np.diag(aaT)))
        a = aaT[:, k] / np.sqrt(aaT[k, k])
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return (a if a @ v >= 0 else -a) * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """-> unit quaternion (w, x, y, z) with w >= 0 (Shepperd's method)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rigid(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rigid_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return rigid(R.T, -R.T @ t)


def apply(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(so3_log(R))))


def register_rigid_robust(A: np.ndarray, B: np.ndarray, iters: int = 10):
    """Rigid (R, t) with B ~= R A + t, robust to outliers: Umeyama inside IRLS with Cauchy weights
    (scale = 2x the MAD-based sigma of the current residuals). Used for hand-eye registration from
    noisy per-frame vision estimates, where depth outliers of several cm are common."""
    from surgscene.shape import umeyama
    w = np.ones(len(A))
    for _ in range(iters):
        _, R, t = umeyama(A, B, with_scale=False, weights=w)
        r = np.linalg.norm(A @ R.T + t - B, axis=1)
        s = 1.4826 * np.median(r) + 1e-6
        w = 1 / (1 + (r / (2 * s)) ** 2)
    return R, t
