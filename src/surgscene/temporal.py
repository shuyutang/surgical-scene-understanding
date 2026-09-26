"""Phase 4: temporal filtering of per-frame keypoint estimates.

One constant-velocity Kalman filter per keypoint, state s = (x, y, vx, vy) in native pixels,
one step = one frame (30 fps). The measurement is the per-frame structured estimate, with a
per-frame noise that comes from the network itself:

    R_t = (r0 * sigma_t)^2 / max(conf_t, eps) * I

where sigma_t is the Laplace std of the heatmap peak and conf_t its peak probability, so a blurry
or low-confidence peak is automatically trusted less. No fixed "measurement noise" knob to hand-tune
per scene.

Robustness:
- conf_t < conf_min: no update (predict only), so the covariance grows through occlusions.
- innovation gate: if the normalized innovation squared (NIS) exceeds chi2_2 at 99.9% the
  measurement is rejected as an outlier; after `max_rejects` consecutive rejections the track is
  re-initialized at the measurement (the target really moved, or the track was wrong).
- "unknown": if the position std exceeds `unknown_px`, the output is flagged instead of reporting
  a stale estimate as if it were current.

A fixed-lag RTS smoother (lag L frames) is provided to measure the accuracy-vs-latency tradeoff.
"""

from dataclasses import dataclass

import numpy as np

CHI2_2_999 = 13.8155


@dataclass
class KFParams:
    q: float = 2.0            # white-noise acceleration std, px / frame^2
    r0: float = 1.0           # measurement noise multiplier on the Laplace sigma
    conf_min: float = 0.05
    max_rejects: int = 3
    unknown_px: float = 25.0  # flag the output as unknown above this position std


def _F():
    F = np.eye(4)
    F[0, 2] = F[1, 3] = 1.0
    return F


def _Q(q: float):
    return q**2 * np.array([[0.25, 0, 0.5, 0], [0, 0.25, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]])


H = np.hstack([np.eye(2), np.zeros((2, 2))])


@dataclass
class Track:
    x_pred: np.ndarray   # (T, 4)
    P_pred: np.ndarray   # (T, 4, 4)
    x_filt: np.ndarray   # (T, 4)
    P_filt: np.ndarray   # (T, 4, 4)
    updated: np.ndarray  # (T,) bool: a measurement was accepted
    nis: np.ndarray      # (T,) NIS of accepted updates (NaN otherwise)
    alive: np.ndarray    # (T,) bool: track initialized


def kalman_track(z: np.ndarray, sigma: np.ndarray, conf: np.ndarray, p: KFParams) -> Track:
    """One keypoint. z (T, 2), sigma (T,), conf (T,)."""
    T = len(z)
    F, Q = _F(), _Q(p.q)
    xp, Pp = np.full((T, 4), np.nan), np.full((T, 4, 4), np.nan)
    xf, Pf = np.full((T, 4), np.nan), np.full((T, 4, 4), np.nan)
    upd, nis, alive = np.zeros(T, bool), np.full(T, np.nan), np.zeros(T, bool)
    x, P, rejects = None, None, 0
    for t in range(T):
        valid = np.isfinite(z[t]).all() and conf[t] >= p.conf_min
        R = np.eye(2) * (p.r0 * sigma[t]) ** 2 / max(conf[t], 1e-3) if valid else None
        if x is None:
            if valid:  # initialize: position from the measurement, velocity unknown
                x = np.array([*z[t], 0.0, 0.0])
                P = np.diag([R[0, 0], R[1, 1], 100.0, 100.0])
                xp[t], Pp[t], xf[t], Pf[t], upd[t], alive[t] = x, P, x, P, True, True
            continue
        x, P = F @ x, F @ P @ F.T + Q
        xp[t], Pp[t], alive[t] = x, P, True
        if valid:
            nu = z[t] - H @ x
            S = H @ P @ H.T + R
            d2 = float(nu @ np.linalg.solve(S, nu))
            if d2 <= CHI2_2_999:
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ nu
                P = (np.eye(4) - K @ H) @ P
                upd[t], nis[t], rejects = True, d2, 0
            else:
                rejects += 1
                if rejects >= p.max_rejects:  # re-initialize at the measurement
                    x = np.array([*z[t], 0.0, 0.0])
                    P = np.diag([R[0, 0], R[1, 1], 100.0, 100.0])
                    upd[t], rejects = True, 0
        xf[t], Pf[t] = x, P
    return Track(xp, Pp, xf, Pf, upd, nis, alive)


def fixed_lag_smooth(tr: Track, lag: int, q: float) -> np.ndarray:
    """Position at time t from an RTS backward pass over (t, t+lag]. Output for t is available at
    t+lag, i.e. it adds `lag` frames of latency. Returns (T, 2)."""
    if lag == 0:
        return tr.x_filt[:, :2].copy()
    T = len(tr.x_filt)
    F = _F()
    out = tr.x_filt[:, :2].copy()
    for t in range(T):
        end = min(t + lag, T - 1)
        if not tr.alive[t] or end == t:
            continue
        xs, Ps = tr.x_filt[end], tr.P_filt[end]
        if not np.isfinite(xs).all():
            continue
        ok = True
        for k in range(end - 1, t - 1, -1):
            if not (tr.alive[k] and tr.alive[k + 1]) or not np.isfinite(tr.P_pred[k + 1]).all():
                ok = False
                break
            G = tr.P_filt[k] @ F.T @ np.linalg.inv(tr.P_pred[k + 1])
            xs = tr.x_filt[k] + G @ (xs - tr.x_pred[k + 1])
            Ps = tr.P_filt[k] + G @ (Ps - tr.P_pred[k + 1]) @ G.T
        if ok:
            out[t] = xs[:2]
    return out


def filter_keypoints(Z: np.ndarray, sigma: np.ndarray, conf: np.ndarray, p: KFParams, lag: int = 0):
    """All keypoints of one sequence. Z (T, K, 2). Returns positions (T, K, 2), position std (T, K),
    unknown flags (T, K), per-keypoint Tracks."""
    T, K = Z.shape[:2]
    pos, std, tracks = np.full((T, K, 2), np.nan), np.full((T, K), np.nan), []
    for k in range(K):
        tr = kalman_track(Z[:, k], sigma[:, k], conf[:, k], p)
        tracks.append(tr)
        pos[:, k] = fixed_lag_smooth(tr, lag, p.q)
        std[:, k] = np.sqrt(np.maximum(tr.P_filt[:, 0, 0], tr.P_filt[:, 1, 1]))
    unknown = ~np.isfinite(std) | (std > p.unknown_px)
    return pos, std, unknown, tracks


def jitter(pos: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per (frame, keypoint) acceleration error ||d2 pos - d2 gt|| (px/frame^2), NaN where undefined.
    Measures frame-to-frame shake that the ground-truth motion does not have."""
    d2p = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    d2g = gt[2:] - 2 * gt[1:-1] + gt[:-2]
    e = np.linalg.norm(d2p - d2g, axis=-1)
    return np.concatenate([np.full((1,) + e.shape[1:], np.nan), e, np.full((1,) + e.shape[1:], np.nan)])
