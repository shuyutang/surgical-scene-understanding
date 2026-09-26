import cv2
import numpy as np
import pytest

from surgscene.geometry import (Camera, StereoRig, apply, quat_from_R, R_from_quat, rigid, rigid_inv, so3_exp,
                                so3_log, triangulate, triangulate_dlt)
from surgscene.shape import umeyama

rng = np.random.default_rng(0)


def random_R():
    return so3_exp(rng.normal(size=3) * 1.5)


def rig():
    K = np.array([[1800.0, 0, 600], [0, 1800, 480], [0, 0, 1]])
    L = Camera(K, np.array([-0.25, 0.5, -0.002, -0.004, -0.25]), (1400, 986))
    Rr = Camera(K + [[0, 0, 190], [0, 0, -40], [0, 0, 0]], np.array([-0.26, 0.1, 0.0008, 0.0007, 2.5]), (1400, 986))
    return StereoRig(L, Rr, so3_exp(np.array([0.006, -0.002, -0.01])), np.array([-5.9, -0.08, -0.8]))


def test_rotation_properties():
    for _ in range(50):
        R = random_R()
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)
        assert np.allclose(R_from_quat(quat_from_R(R)), R, atol=1e-12)


def test_so3_log_near_pi_and_zero():
    for th in [1e-9, 1e-5, np.pi - 1e-6, np.pi]:
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        R = so3_exp(a * th)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-6)


def test_so3_matches_opencv_rodrigues():
    w = rng.normal(size=3)
    assert np.allclose(so3_exp(w), cv2.Rodrigues(w)[0], atol=1e-12)


def test_rigid_compose_invert():
    A, B = rigid(random_R(), rng.normal(size=3)), rigid(random_R(), rng.normal(size=3))
    assert np.allclose(A @ rigid_inv(A), np.eye(4), atol=1e-12)
    assert np.allclose(rigid_inv(A @ B), rigid_inv(B) @ rigid_inv(A), atol=1e-12)
    X = rng.normal(size=(5, 3))
    assert np.allclose(apply(A @ B, X), apply(A, apply(B, X)))


def test_umeyama_3d_rigid():
    R, t = random_R(), rng.normal(size=3)
    X = rng.normal(size=(20, 3))
    s, R2, t2 = umeyama(X, X @ R.T + t, with_scale=False)
    assert np.allclose(R2, R, atol=1e-9) and np.allclose(t2, t, atol=1e-9) and s == 1.0


def test_projection_and_undistort_match_opencv():
    cam = rig().left
    X = np.column_stack([rng.uniform(-20, 20, 50), rng.uniform(-15, 15, 50), rng.uniform(40, 120, 50)])
    u = cam.project(X)
    u_cv = cv2.projectPoints(X, np.zeros(3), np.zeros(3), cam.K, cam.dist)[0][:, 0]
    assert np.allclose(u, u_cv, atol=1e-8)
    m = cam.undistort(u)
    assert np.allclose(m, X[:, :2] / X[:, 2:], atol=1e-6)


def test_triangulation_exact_and_covariance():
    r = rig()
    X = np.array([[5.0, -3.0, 60.0], [-10, 8, 90]])
    uL = r.left.project(X)
    uR = r.right.project(X @ r.R.T + r.T)
    dlt = triangulate_dlt(r.left.undistort(uL), r.right.undistort(uR), r.R, r.T)
    assert np.allclose(dlt, X, atol=1e-5)
    tri = triangulate(r, uL[0], uR[0])
    assert np.allclose(tri.X, X[0], atol=1e-6) and tri.reproj_px < 1e-6
    # Monte Carlo check of the first-order covariance: 1 px noise on all four coordinates
    samples = np.array([triangulate(r, uL[0] + rng.normal(size=2), uR[0] + rng.normal(size=2)).X
                        for _ in range(2000)])
    emp = np.cov(samples.T)
    assert np.allclose(np.sqrt(np.diag(emp)), np.sqrt(np.diag(tri.cov)), rtol=0.1)
    # depth is by far the least certain direction for a 6 mm baseline
    assert tri.cov[2, 2] > 10 * tri.cov[0, 0]


@pytest.mark.parametrize("seed", range(3))
def test_triangulation_matches_opencv(seed):
    r = rig()
    g = np.random.default_rng(seed)
    X = np.column_stack([g.uniform(-15, 15, 10), g.uniform(-10, 10, 10), g.uniform(40, 100, 10)])
    uL, uR = r.left.project(X), r.right.project(X @ r.R.T + r.T)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([r.R, r.T[:, None]])
    mL = cv2.undistortPoints(uL[:, None], r.left.K, r.left.dist)[:, 0]
    mR = cv2.undistortPoints(uR[:, None], r.right.K, r.right.dist)[:, 0]
    Xh = cv2.triangulatePoints(P1, P2, mL.T, mR.T)
    assert np.allclose((Xh[:3] / Xh[3]).T, X, atol=1e-3)


def test_tissue_plane_recovers_synthetic_plane():
    from surgscene.proximity import tissue_plane

    class R:  # minimal rectified camera
        f, cx, cy, B = 900.0, 350.0, 250.0, 5.5
    n_true = np.array([0.2, -0.1, -1.0])
    n_true /= np.linalg.norm(n_true)
    c_true = n_true @ np.array([0, 0, 150.0])  # plane through (0, 0, 150) mm
    H, W = 500, 700
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    # ray through each pixel: X = Z * ((x - cx)/f, (y - cy)/f, 1); intersect with n.X = c
    ray = np.stack([(xs - R.cx) / R.f, (ys - R.cy) / R.f, np.ones_like(xs)], -1)
    Z = c_true / (ray @ n_true)
    disp = R.f * R.B / Z + rng.normal(scale=0.3, size=Z.shape)
    disp[::7, ::5] += 15  # gross SGM outliers
    n, c, npts = tissue_plane(disp, np.zeros((H, W), bool), np.array([350.0, 250.0]), R, 20, 60)
    assert np.allclose(n, n_true, atol=0.02) and abs(c - c_true) < 1.0
    tip = np.array([0, 0, 140.0])  # 10 mm in front of the plane along the optical axis
    assert 8 < tip @ n - c < 10.5


def test_register_rigid_robust_ignores_outliers():
    from surgscene.geometry import register_rigid_robust
    R, t = random_R(), rng.normal(size=3) * 50
    A = rng.normal(size=(300, 3)) * 30
    B = A @ R.T + t + rng.normal(scale=0.5, size=A.shape)
    B[:60] += rng.normal(scale=40, size=(60, 3))  # 20% gross outliers
    R2, t2 = register_rigid_robust(A, B)
    assert np.degrees(np.linalg.norm(so3_log(R2 @ R.T))) < 0.5 and np.linalg.norm(t2 - t) < 1.5
