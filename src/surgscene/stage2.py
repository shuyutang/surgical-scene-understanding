"""Stage-2 pipeline (Phases 3b-5) on full-rate SurgPose observations.

    per eye, per frame:  heatmap observations -> gated shape-prior MAP  (Phase 3b)
    per eye, over time:  constant-velocity Kalman filter per keypoint    (Phase 4)
    across eyes:         triangulation with covariance -> 3D keypoints   (Phase 5)
"""

import json
from pathlib import Path

import numpy as np

from surgscene.frontend import TEST2, TUNE
from surgscene.geometry import StereoRig, load_stereo_ini, triangulate
from surgscene.keypoints import INSTRUMENTS, load_records
from surgscene.shape import ShapeModel
from surgscene.structured import fit_gated, fit_map

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "data/cache/surgpose_obs"
RAW = ROOT / "data/surgpose/raw"
TIP = {"PSM1": (3, 4), "PSM3": (8, 9)}  # jaw tips A/B; the "tip" is their midpoint
Z_RANGE_MM = (20.0, 2000.0)  # plausible working depths; tissue is at ~150-250 mm
SPLITS = {"tune": TUNE, "test2": TEST2}


def load_obs(traj: int, eye: str) -> dict[str, np.ndarray]:
    return dict(np.load(OBS / f"{traj:06d}_{eye}.npz"))


def load_rig(traj: int) -> StereoRig:
    return load_stereo_ini(RAW / f"{traj:06d}/StereoCalibrationDVRK.ini")


def shape_models() -> tuple[dict[str, ShapeModel], dict[str, float]]:
    """Per-arm shape models from train-split ground truth (as in Phase 3). The energy is
    similarity-invariant, so models fit in half-res pixels apply unchanged to native pixels."""
    tr = np.array([r["kp"] for r in load_records("train")])
    models, plaus_max = {}, {}
    for arm, sl in INSTRUMENTS.items():
        X = tr[:, sl]
        X = X[np.isfinite(X).all((1, 2))]
        models[arm] = ShapeModel.fit(X)
        pl = np.array([models[arm].plausibility(x) for x in X[:: max(1, len(X) // 3000)]])
        plaus_max[arm] = float(np.percentile(pl, 99))
    return models, plaus_max


def framewise(kp, conf, sigma, models, method: str, hp: dict | None = None, frames=None) -> np.ndarray:
    """Per-frame structured estimate. method: argmax | map | gated. hp: conf_min, beta, cauchy_c (+ tau)."""
    out = kp.copy()
    if method == "argmax":
        return out
    idx = range(len(kp)) if frames is None else frames
    solver_hp = {k: hp[k] for k in ("conf_min", "beta", "cauchy_c")}
    for i in idx:
        for arm, sl in INSTRUMENTS.items():
            if method == "map":
                out[i, sl] = fit_map(models[arm], kp[i, sl], conf[i, sl], sigma[i, sl], **solver_hp)
            else:
                out[i, sl] = fit_gated(models[arm], kp[i, sl], conf[i, sl], sigma[i, sl], tau=hp["tau"], **solver_hp)
    return out


def tip_points(P: np.ndarray) -> dict[str, np.ndarray]:
    """(T, 10, d) keypoints -> {arm: (T, d) jaw-tip midpoint}."""
    return {arm: (P[:, a] + P[:, b]) / 2 for arm, (a, b) in TIP.items()}


def triangulate_seq(rig: StereoRig, uL: np.ndarray, uR: np.ndarray, sL: np.ndarray, sR: np.ndarray):
    """Triangulate every finite (frame, keypoint). uL/uR (T, K, 2) native px, sL/sR (T, K) px std.
    Returns X (T, K, 3) mm, cov (T, K, 3, 3), reprojection RMS (T, K).
    Points outside Z_RANGE_MM are returned as NaN: with a 5.5 mm baseline, a left/right pair with
    near-zero (or negative) disparity triangulates to meters or behind the camera."""
    T, K = uL.shape[:2]
    X, C, rp = np.full((T, K, 3), np.nan), np.full((T, K, 3, 3), np.nan), np.full((T, K), np.nan)
    for t in range(T):
        for k in range(K):
            if np.isfinite(uL[t, k]).all() and np.isfinite(uR[t, k]).all() and np.isfinite(sL[t, k]) and np.isfinite(sR[t, k]):
                r = triangulate(rig, uL[t, k], uR[t, k], max(sL[t, k], 0.5), max(sR[t, k], 0.5))
                if Z_RANGE_MM[0] <= r.X[2] <= Z_RANGE_MM[1]:
                    X[t, k], C[t, k], rp[t, k] = r.X, r.cov, r.reproj_px
    return X, C, rp


def load_selected() -> dict:
    return json.loads((ROOT / "configs/stage2_selected.json").read_text())
