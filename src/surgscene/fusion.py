"""v3 Phase C: kinematics-fused 3D instrument state (causal).

v2 triangulated each keypoint from its two 2D detections. With a 5.5 mm baseline at ~200 mm, that
estimate is good laterally but poor along the viewing ray (10 mm tip error, S1). The robot already
knows where the tool is in its own base frame; what's missing is the base -> camera transform
(hand-eye) and the small kinematic error. This module estimates both from the 2D detections:

  X_cam = R_he (p_base(t)) + t_he + delta_t

  p_base(t): tool points from forward kinematics and a fixed tool geometry (configs/tool_geometry.json,
             fitted on train trajectories): pivot, wrist (via the wrist-yaw joint), and the two jaw
             tips (pivot-frame offset +- h_t along the jaw opening direction)
  (R_he, t_he): hand-eye, re-fitted causally every `refit_every` frames by robust LM on the
             *reprojection* error of pivot and wrist over the frames seen so far. Fitting in pixels
             weights depth correctly: a single frame says little about depth, hundreds say a lot.
  delta_t, h_t: per-frame state, an iterated EKF with robust (Cauchy) measurement weights.
             delta is mean-reverting (the kinematic error is small and slowly varying); h is a
             random walk. h is signed: tip_a/tip_b labels are swapped on ~5% of train frames.

Everything is numpy with numeric Jacobians, written to port to Eigen like the v2 solvers.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from surgscene.geometry import StereoRig, register_rigid_robust, so3_exp, triangulate

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/surgpose/raw"
ARMS = {"PSM1": slice(0, 5), "PSM3": slice(5, 10)}
# tool points modelled here -> keypoint index within an arm's 5 (shaft, wrist, pivot, tip_a, tip_b)
POINTS = ["pivot", "wrist", "tip_a", "tip_b"]
KP_INDEX = {"pivot": 2, "wrist": 1, "tip_a": 3, "tip_b": 4}


def load_kinematics(traj: int) -> dict:
    """{arm: {"R": (T,3,3), "t": (T,3) mm, "q": (T,6) rad}} in each arm's kinematic base frame."""
    load = lambda name: yaml.load(open(RAW / f"{traj:06d}/{name}"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    cp, jp = load("api_cp_data.yaml"), load("api_jp_data.yaml")
    n = len(cp)
    out = {}
    for arm in ARMS:
        out[arm] = {"R": np.array([np.asarray(cp[str(i)][arm]["R"]).reshape(3, 3) for i in range(n)]),
                    "t": np.array([np.asarray(cp[str(i)][arm]["t"]) * 1000.0 for i in range(n)]),
                    "q": np.array([jp[str(i)][arm] for i in range(n)])}
    return out


@dataclass
class ToolGeometry:
    wrist_b: float
    wrist_L: float
    wrist_ax: float
    tip_mid: np.ndarray   # (3,) tool frame, mm
    open_dir: np.ndarray  # (3,) unit
    h0: float             # typical half-opening, mm

    @classmethod
    def load(cls, arm: str, path: Path = ROOT / "configs/tool_geometry.json") -> "ToolGeometry":
        g = json.loads(path.read_text())["arms"][arm]
        return cls(g["wrist"]["b"], g["wrist"]["L"], g["wrist"]["ax"], np.array(g["tip_mid"]),
                   np.array(g["open_dir"]), g["h_median"])

    def local(self, q5: float, h: float) -> np.ndarray:
        """(4, 3) tool-frame points in POINTS order."""
        a = q5 + self.wrist_b
        wrist = np.array([self.wrist_ax, self.wrist_L * np.sin(a), -self.wrist_L * np.cos(a)])
        return np.stack([np.zeros(3), wrist, self.tip_mid + h * self.open_dir, self.tip_mid - h * self.open_dir])


def project_both(rig: StereoRig, X: np.ndarray) -> np.ndarray:
    """(N, 3) left-camera points -> (N, 4) [uL, vL, uR, vR]."""
    return np.concatenate([rig.left.project(X), rig.right.project(X @ rig.R.T + rig.T)], 1)


def _cauchy_w(r2: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1.0 + r2 / c**2)


def _wrist_local(q5: np.ndarray, L: float, b: float, ax: float) -> np.ndarray:
    a = q5 + b
    return np.stack([np.full_like(a, ax), L * np.sin(a), -L * np.cos(a)], -1)


def fit_calibration(rig: StereoRig, Rk: np.ndarray, tk: np.ndarray, q5: np.ndarray, Z: np.ndarray, S: np.ndarray,
                    R0: np.ndarray, t0: np.ndarray, geom: "ToolGeometry", prior: "ToolGeometry",
                    prior_std=(4.0, 3.0, 0.3, 2.0), iters: int = 25, cauchy: float = 3.0, fix_handeye: bool = False):
    """Per-trajectory calibration by robust LM on reprojection error over past frames:
    hand-eye (R, t) plus the tool geometry the robot doesn't report here (tip-midpoint offset c,
    wrist L, b, ax). SurgPose mixes instrument types whose jaw lengths differ by up to 3x.
    Rk (M,3,3), tk (M,3), q5 (M,), Z/S (M, 3, 4) for [pivot, wrist, tip midpoint] (NaN = missing).
    The geometry has Gaussian priors at the train values (std: c mm, L mm, b rad, ax mm).
    fix_handeye: only the geometry is fitted. Fitting the hand-eye by reprojection turned out to be
    ill-conditioned along depth on tune (small view-dependent label offsets are "explained" by a
    large, cheap depth shift), so the default pipeline registers the hand-eye in 3D instead.
    Returns (R, t, geom, rms_px)."""
    valid = np.isfinite(Z) & np.isfinite(S)
    Zf, Sf = np.where(valid, Z, 0.0), np.where(valid, S, 1.0)
    pc, pL, pb, pax = prior_std
    mu0 = np.concatenate([prior.tip_mid, [prior.wrist_L, prior.wrist_b, prior.wrist_ax]])
    sd = np.array([pc, pc, pc, pL, pb, pax])

    def data_resid(R, t, gv):
        c, (L, b, ax) = gv[:3], gv[3:]
        loc = np.stack([np.zeros_like(tk), _wrist_local(q5, L, b, ax), np.broadcast_to(c, tk.shape)], 1)  # (M,3,3)
        Xb = tk[:, None] + np.einsum("mij,mkj->mki", Rk, loc)
        Xc = Xb.reshape(-1, 3) @ R.T + t
        r = (project_both(rig, Xc).reshape(Z.shape) - Zf) / Sf
        return np.where(valid, r, 0.0).ravel()

    def energy(rd, gv):
        return float((cauchy**2 * np.log1p(rd**2 / cauchy**2)).sum() + (((gv - mu0) / sd) ** 2).sum())

    R, t = R0.copy(), t0.copy()
    gv = np.concatenate([geom.tip_mid, [geom.wrist_L, geom.wrist_b, geom.wrist_ax]])
    rd = data_resid(R, t, gv)
    E, mu = energy(rd, gv), 1e-3
    steps = np.array([1e-6] * 3 + [1e-3] * 3 + [1e-3] * 4 + [1e-5, 1e-3])
    for _ in range(iters):
        w = np.where(valid.ravel(), _cauchy_w(rd**2, cauchy), 0.0)
        J = np.empty((rd.size, 12))
        for k in range(12):
            d = np.zeros(12)
            d[k] = steps[k]
            Rp, Rm = so3_exp(d[:3]) @ R, so3_exp(-d[:3]) @ R
            J[:, k] = (data_resid(Rp, t + d[3:6], gv + d[6:]) - data_resid(Rm, t - d[3:6], gv - d[6:])) / (2 * steps[k])
        Hp = np.zeros((12, 12))
        Hp[6:, 6:] = np.diag(1 / sd**2)
        H = J.T @ (w[:, None] * J) + Hp
        g = J.T @ (w * rd)
        g[6:] += (gv - mu0) / sd**2
        if fix_handeye:
            H[:6, :], H[:, :6], g[:6] = 0.0, 0.0, 0.0
            H[:6, :6] = np.eye(6)
        improved = False
        for _ in range(8):
            step = np.linalg.solve(H + mu * np.diag(np.diag(H) + 1e-9), -g)
            Rn, tn, gn = so3_exp(step[:3]) @ R, t + step[3:6], gv + step[6:]
            rn = data_resid(Rn, tn, gn)
            En = energy(rn, gn)
            if En < E:
                R, t, gv, rd, E, mu, improved = Rn, tn, gn, rn, En, max(mu / 3, 1e-7), True
                break
            mu *= 4
        if not improved or np.abs(step).max() < 1e-7:
            break
    out = replace(geom, tip_mid=gv[:3].copy(), wrist_L=float(gv[3]), wrist_b=float(gv[4]), wrist_ax=float(gv[5]))
    rms = float(np.sqrt((rd[valid.ravel()] ** 2).mean())) if valid.any() else np.nan
    return R, t, out, rms


def _robust_mean(X: np.ndarray, iters: int = 10) -> np.ndarray:
    """Cauchy-IRLS mean of (N, 3) points (scale from the MAD of the distances to the current mean)."""
    m = np.median(X, 0)
    for _ in range(iters):
        r = np.linalg.norm(X - m, axis=1)
        s = 1.4826 * np.median(r) + 1e-6
        w = 1 / (1 + (r / (2 * s)) ** 2)
        m = (w[:, None] * X).sum(0) / w.sum()
    return m


def calib_observations(Z: np.ndarray, S: np.ndarray):
    """(T, 4pts, 4) in POINTS order -> (T, 3, 4) for [pivot, wrist, tip midpoint]. The midpoint of the
    two projected tips stands in for the projection of the 3D midpoint (sub-pixel at this scale)."""
    mid = (Z[:, 2] + Z[:, 3]) / 2
    smid = np.sqrt(S[:, 2] ** 2 + S[:, 3] ** 2) / 2
    return np.stack([Z[:, 0], Z[:, 1], mid], 1), np.stack([S[:, 0], S[:, 1], smid], 1)


@dataclass
class FusionParams:
    r0: float = 2.0            # pixel std = r0 * Laplace sigma / sqrt(conf), as in the v2 Kalman filter
    conf_min: float = 0.05
    min_px: float = 1.0
    cauchy: float = 3.0        # in units of the pixel std
    sig_kin: float = 2.0       # stationary std of the kinematic error delta, mm (per axis)
    tau: float = 30.0          # delta mean-reversion time, frames
    q_h: float = 0.5           # random-walk std of the jaw half-opening, mm / frame
    warmup: int = 30           # frames before the first hand-eye fit
    refit_every: int = 15
    he_max_frames: int = 300   # frames used per hand-eye refit (evenly subsampled from the past)
    use_vision: bool = True    # False: kinematics + hand-eye only (delta = 0, h = h0)
    handeye_mode: str = "3d"   # "3d": robust rigid registration of kinematic to triangulated pivots;
                               # "reproj": joint reprojection LM with the geometry (ill-conditioned)
    tip_mode: str = "reproj"   # tip-midpoint offset c: "reproj" = fitted with the wrist by reprojection LM;
                               # "3d" = robust mean of per-frame triangulated tip midpoints in the tool frame
                               # (post hoc, v3 step 4: the reprojection fit of c is depth-ill-conditioned)
    sig_ray: float | None = 0.5  # std (mm) of a pseudo-measurement delta . ray = 0: vision corrects the
                                 # kinematics laterally, depth stays with kinematics. None: unconstrained


def pixel_std(sigma, conf, p: FusionParams):
    s = p.r0 * sigma / np.sqrt(np.maximum(conf, 1e-3))
    return np.where(conf >= p.conf_min, np.maximum(s, p.min_px), np.nan)


def observations(zL, zR, sL, sR):
    """Per-frame (T, 5, 2) x2 and (T, 5) x2 -> Z (T, 4pts, 4), S (T, 4pts, 4) in POINTS order."""
    idx = [KP_INDEX[n] for n in POINTS]
    Z = np.concatenate([zL[:, idx], zR[:, idx]], -1)
    S = np.stack([sL[:, idx], sL[:, idx], sR[:, idx], sR[:, idx]], -1)
    Z = np.where(np.isfinite(S), Z, np.nan)
    return Z, S


def initial_handeye(rig, Pb_pivot, Z_pivot, S_pivot):
    """Rigid fit of kinematic pivots to per-frame triangulated pivots (v2 S4 style), as the LM start."""
    A, B = [], []
    for p, z, s in zip(Pb_pivot, Z_pivot, S_pivot):
        if np.isfinite(z).all() and np.isfinite(s).all():
            X = triangulate(rig, z[:2], z[2:], s[0], s[2]).X
            if 20 <= X[2] <= 2000:
                A.append(p)
                B.append(X)
    if len(A) < 10:
        return None
    return register_rigid_robust(np.array(A), np.array(B))


@dataclass
class FusionTrack:
    X: np.ndarray        # (T, 4, 3) camera-frame points in POINTS order (NaN before the first calibration)
    cov: np.ndarray      # (T, 4, 3, 3)
    h: np.ndarray        # (T,)
    delta: np.ndarray    # (T, 3)
    calib_time: list     # frames at which the calibration was refitted
    calib_rms: list
    geom: list           # ToolGeometry after each refit


def fuse_arm(rig: StereoRig, kin_arm: dict, prior: ToolGeometry, Z: np.ndarray, S: np.ndarray,
             p: FusionParams, calib_fixed: tuple | None = None) -> FusionTrack:
    """Causal fusion for one arm. Z, S: (T, 4, 4) from `observations`. calib_fixed: (R, t, geom) used
    throughout instead of the causal refits (diagnostics only, e.g. an oracle)."""
    T = len(Z)
    Rk, tk, q5 = kin_arm["R"][:T], kin_arm["t"][:T], kin_arm["q"][:T, 5]
    Zc, Sc = calib_observations(Z, S)
    Xv = np.full((T, 3), np.nan)   # per-frame vision pivot (uses frame i only, so causal)
    Xtm = np.full((T, 3), np.nan)  # per-frame vision tip midpoint (tip_mode "3d")
    if calib_fixed is None and p.handeye_mode == "3d":
        for i in range(T):
            z, s_ = Z[i, 0], S[i, 0]
            if np.isfinite(z).all() and np.isfinite(s_).all():
                X = triangulate(rig, z[:2], z[2:], s_[0], s_[2]).X
                if 20 <= X[2] <= 2000:
                    Xv[i] = X
            if p.tip_mode == "3d" and np.isfinite(Z[i, 2:4]).all() and np.isfinite(S[i, 2:4]).all():
                Xa = triangulate(rig, Z[i, 2, :2], Z[i, 2, 2:], S[i, 2, 0], S[i, 2, 2]).X
                Xb = triangulate(rig, Z[i, 3, :2], Z[i, 3, 2:], S[i, 3, 0], S[i, 3, 2]).X
                if 20 <= Xa[2] <= 2000 and 20 <= Xb[2] <= 2000:
                    Xtm[i] = (Xa + Xb) / 2
    out = FusionTrack(np.full((T, 4, 3), np.nan), np.full((T, 4, 3, 3), np.nan), np.full(T, np.nan),
                      np.full((T, 3), np.nan), [], [], [])
    calib = calib_fixed
    rho = 1.0 - 1.0 / p.tau
    q_delta = p.sig_kin**2 * (1 - rho**2)
    x = np.array([0.0, 0.0, 0.0, prior.h0])
    P = np.diag([p.sig_kin**2] * 3 + [2.0**2])
    for i in range(T):
        # --- calibration: hand-eye + tool geometry, from frames 0..i only
        if calib_fixed is None and i >= p.warmup and (calib is None or (i - out.calib_time[-1]) >= p.refit_every):
            past = np.arange(i + 1)
            if len(past) > p.he_max_frames:
                past = np.linspace(0, i, p.he_max_frames).astype(int)
            if p.handeye_mode == "3d":
                okv = np.isfinite(Xv[: i + 1]).all(1)
                if okv.sum() >= 10:
                    R0, t0 = register_rigid_robust(tk[: i + 1][okv], Xv[: i + 1][okv])
                    init = (R0, t0, prior if calib is None else calib[2])
                else:
                    init = None
            elif calib is None:
                he0 = initial_handeye(rig, tk[past], Z[past, 0], S[past, 0])
                init = None if he0 is None else (he0[0], he0[1], prior)
            else:
                init = calib
            if init is not None:
                prior_i, pstd = prior, (4.0, 3.0, 0.3, 2.0)
                if p.tip_mode == "3d":
                    okm = np.isfinite(Xtm[: i + 1]).all(1)
                    if okm.sum() >= 10:
                        R0, t0 = init[0], init[1]
                        c_i = np.einsum("fji,fj->fi", Rk[: i + 1][okm], (Xtm[: i + 1][okm] - t0) @ R0 - tk[: i + 1][okm])
                        c3 = _robust_mean(c_i)
                        prior_i = replace(prior, tip_mid=c3)
                        init = (R0, t0, replace(init[2], tip_mid=c3))
                        pstd = (1e-3, 3.0, 0.3, 2.0)  # c pinned at the 3D estimate; the wrist is still fitted
                R_he, t_he, geom, rms = fit_calibration(rig, Rk[past], tk[past], q5[past], Zc[past], Sc[past],
                                                        init[0], init[1], init[2], prior_i, prior_std=pstd,
                                                        cauchy=p.cauchy, fix_handeye=p.handeye_mode == "3d")
                calib = (R_he, t_he, geom)
                out.calib_time.append(i)
                out.calib_rms.append(rms)
                out.geom.append(geom)
        if calib is None:
            continue
        R_he, t_he, geom = calib
        # --- predict
        F = np.diag([rho] * 3 + [1.0])
        x = F @ x
        P = F @ P @ F.T + np.diag([q_delta] * 3 + [p.q_h**2])

        def points(xx):
            loc = geom.local(q5[i], xx[3])
            return (tk[i] + loc @ Rk[i].T) @ R_he.T + t_he + xx[:3]

        if p.use_vision:
            z, s = Z[i].ravel(), S[i].ravel()
            valid = np.isfinite(z) & np.isfinite(s)
            if valid.any():
                zf, sf = np.where(valid, z, 0.0), np.where(valid, s, 1.0)
                res = lambda xx: np.where(valid, (project_both(rig, points(xx)).ravel() - zf) / sf, 0.0)
                ray = points(x)[0] / np.linalg.norm(points(x)[0])
                Pinv = np.linalg.inv(P)
                xp, xi = x.copy(), x.copy()
                for _ in range(5):
                    r = res(xi)
                    J = np.empty((r.size, 4))
                    for k in range(4):
                        d = np.zeros(4)
                        d[k] = 1e-3
                        J[:, k] = (res(xi + d) - res(xi - d)) / 2e-3
                    w = np.where(valid, _cauchy_w(r**2, p.cauchy), 0.0)
                    H = Pinv + J.T @ (w[:, None] * J)
                    gr = Pinv @ (xi - xp) + J.T @ (w * r)
                    if p.sig_ray is not None:
                        a = np.concatenate([ray, [0.0]]) / p.sig_ray
                        H = H + np.outer(a, a)
                        gr = gr + a * (a @ xi)
                    step = np.linalg.solve(H, -gr)
                    xi = xi + step
                    if np.abs(step).max() < 1e-4:
                        break
                x, P = xi, np.linalg.inv(H)
        else:
            x = np.array([0.0, 0.0, 0.0, geom.h0])
            P = np.diag([p.sig_kin**2] * 3 + [2.0**2])
        u = R_he @ Rk[i] @ geom.open_dir
        Jp = np.zeros((4, 3, 4))
        Jp[:, :, :3] = np.eye(3)
        Jp[2, :, 3], Jp[3, :, 3] = u, -u
        out.X[i] = points(x)
        out.cov[i] = Jp @ P @ Jp.transpose(0, 2, 1)
        out.h[i], out.delta[i] = x[3], x[:3]
    return out


def triangulate_with_depth_prior(rig: StereoRig, uL: np.ndarray, uR: np.ndarray, sL: float, sR: float,
                                 X_prior: np.ndarray, sd_depth: float, iters: int = 8):
    """Stereo triangulation plus a Gaussian prior on depth along the viewing ray from X_prior (the
    kinematics-fused point). Each source is used where it is informative: the 2D detections fix
    the point laterally (sub-mm), the prior fixes it along the ray, where a 5.5 mm baseline is weak.
    Minimizes |pi(X) - u|^2 / s^2 over both eyes + ((X - X_prior) . n)^2 / sd_depth^2.
    Returns (X, cov)."""
    z = np.concatenate([uL, uR])
    Wd = np.array([1 / sL**2] * 2 + [1 / sR**2] * 2)
    n = X_prior / np.linalg.norm(X_prior)
    X = X_prior.copy()
    for _ in range(iters):
        r = project_both(rig, X[None])[0] - z
        J = np.empty((4, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1e-4
            J[:, k] = (project_both(rig, (X + e)[None])[0] - project_both(rig, (X - e)[None])[0]) / 2e-4
        H = J.T @ (Wd[:, None] * J) + np.outer(n, n) / sd_depth**2
        g = J.T @ (Wd * r) + n * ((X - X_prior) @ n) / sd_depth**2
        dX = np.linalg.solve(H, -g)
        X = X + dX
        if np.abs(dX).max() < 1e-6:
            break
    return X, np.linalg.inv(H)
