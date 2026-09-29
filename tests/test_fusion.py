import numpy as np

from surgscene.fusion import ToolGeometry, calib_observations, fit_calibration, observations, project_both, \
    triangulate_with_depth_prior
from surgscene.geometry import so3_exp
from test_geometry import rig


def _geom(**kw):
    g = dict(wrist_b=0.0, wrist_L=6.5, wrist_ax=0.0, tip_mid=np.array([-2.0, -0.7, 9.0]),
             open_dir=np.array([0.0, -1.0, 0.0]), h0=2.0)
    g.update(kw)
    return ToolGeometry(**g)


def _sequence(n, seed=0):
    """Random smooth tool motion in a kinematic base frame, plus a hand-eye into the camera."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    tk = np.stack([30 * np.sin(2 * np.pi * t), 20 * np.cos(3 * t), 15 * t], 1) + rng.normal(0, 0.1, (n, 3))
    Rk = np.stack([so3_exp(np.array([0.4 * np.sin(5 * s), 0.3 * np.cos(4 * s), 0.8 * s])) for s in t])
    q5 = 0.6 * np.sin(7 * t)
    R_he, t_he = so3_exp(np.array([2.5, 0.3, -0.2])), np.array([5.0, -3.0, 190.0])
    return Rk, tk, q5, R_he, t_he


def test_fit_calibration_recovers_tool_geometry_with_known_handeye():
    r = rig()
    Rk, tk, q5, R_he, t_he = _sequence(200)
    true = _geom(tip_mid=np.array([3.0, 0.5, 24.0]), wrist_L=8.8)  # a long-jaw instrument
    loc = np.stack([true.local(q, 2.0) for q in q5])                       # (n, 4, 3)
    X = (tk[:, None] + np.einsum("nij,nkj->nki", Rk, loc)) @ R_he.T + t_he
    uv = project_both(r, X.reshape(-1, 3)).reshape(len(X), 4, 4)            # [uL, vL, uR, vR] per point
    zL, zR = np.full((len(X), 5, 2), np.nan), np.full((len(X), 5, 2), np.nan)
    for j, kp in enumerate([2, 1, 3, 4]):                                   # POINTS -> keypoint slots
        zL[:, kp], zR[:, kp] = uv[:, j, :2], uv[:, j, 2:]
    Z, S = observations(zL, zR, np.ones((len(X), 5)), np.ones((len(X), 5)))
    Zc, Sc = calib_observations(Z, S)
    prior = _geom()
    _, _, g, rms = fit_calibration(r, Rk, tk, q5, Zc, Sc, R_he, t_he, prior, prior,
                                   prior_std=(50.0, 50.0, 1.0, 50.0), fix_handeye=True)
    assert rms < 0.05
    assert np.allclose(g.tip_mid, true.tip_mid, atol=0.05)
    assert abs(g.wrist_L - true.wrist_L) < 0.05


def test_depth_prior_triangulation_limits():
    r = rig()
    X = np.array([12.0, -8.0, 180.0])
    u = project_both(r, X[None])[0]
    prior = X + 10.0 * X / np.linalg.norm(X) + np.array([1.0, -1.0, 0.0])  # 10 mm too deep, 1.4 mm lateral
    # weak prior: the exact stereo observations win
    Xw, _ = triangulate_with_depth_prior(r, u[:2], u[2:], 0.5, 0.5, prior, sd_depth=1e4)
    assert np.linalg.norm(Xw - X) < 1e-3
    # strong prior: depth follows the prior. At a wrong depth the two eyes' rays don't meet, so the
    # lateral position is the compromise: the disparity mismatch (~3 px here) is split between eyes
    Xs, C = triangulate_with_depth_prior(r, u[:2], u[2:], 0.5, 0.5, prior, sd_depth=1e-3)
    n = prior / np.linalg.norm(prior)
    assert abs((Xs - prior) @ n) < 0.05
    res = project_both(r, Xs[None])[0] - u
    eL, eR = np.linalg.norm(res[:2]), np.linalg.norm(res[2:])
    assert max(eL, eR) < 2.5 and 0.7 < eL / eR < 1.4
    assert np.sqrt(n @ C @ n) < 0.01
