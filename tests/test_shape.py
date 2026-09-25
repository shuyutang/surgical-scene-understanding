import numpy as np
import pytest

from surgscene.shape import ShapeModel, generalized_procrustes, map_fit, umeyama


def rot(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


BASE = np.array([[0.0, 0.0], [10.0, 0.0], [14.0, 0.0], [20.0, 2.0], [20.0, -2.0]])


def articulated(rng, n):
    """Toy instrument: jaw opening + wrist bend, then random similarity."""
    out = []
    for _ in range(n):
        X = BASE.copy()
        open_ = rng.uniform(0, 4)
        X[3, 1] += open_
        X[4, 1] -= open_
        bend = rng.uniform(-0.4, 0.4)
        X[2:] = (X[2:] - X[1]) @ rot(bend).T + X[1]
        s, th, t = rng.uniform(5, 15), rng.uniform(-np.pi, np.pi), rng.uniform(100, 500, 2)
        out.append(s * X @ rot(th).T + t)
    return np.array(out)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 2))
    s, R, t = 2.5, rot(0.7), np.array([3.0, -1.0])
    s2, R2, t2 = umeyama(X, s * X @ R.T + t)
    assert s2 == pytest.approx(s) and np.allclose(R2, R) and np.allclose(t2, t)
    assert np.isclose(np.linalg.det(R2), 1.0)


def test_umeyama_never_returns_reflection():
    X = np.array([[0.0, 0], [1, 0], [0, 1]])
    Y = X * [-1, 1]  # mirror image
    _, R, _ = umeyama(X, Y)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_procrustes_mean_is_normalized():
    shapes = articulated(np.random.default_rng(1), 200)
    mean, aligned = generalized_procrustes(shapes)
    assert np.allclose(mean.mean(0), 0, atol=1e-9) and np.isclose(np.linalg.norm(mean), 1)


@pytest.fixture(scope="module")
def model():
    return ShapeModel.fit(articulated(np.random.default_rng(2), 2000))


def test_model_captures_articulation_in_few_modes(model):
    assert len(model.lam) <= 4


def test_plausibility_separates_real_from_scrambled(model):
    rng = np.random.default_rng(3)
    real = [model.plausibility(X) for X in articulated(rng, 200)]
    bad = []
    for X in articulated(rng, 200):
        X = X.copy()
        X[[0, 3]] = X[[3, 0]]  # swap shaft and a tip
        bad.append(model.plausibility(X))
    assert np.percentile(real, 95) < np.percentile(bad, 5)


def test_map_fills_missing_keypoint(model):
    X = articulated(np.random.default_rng(4), 1)[0]
    w = np.ones(5)
    w[3] = 0  # tip unobserved
    r = map_fit(model, np.where(w[:, None] > 0, X, 0.0), np.full(5, 1.0), w)
    scale = np.linalg.norm(X[0] - X[1])
    assert np.linalg.norm(r.points[3] - X[3]) < 0.1 * scale


def test_map_robust_to_one_outlier(model):
    rng = np.random.default_rng(5)
    X = articulated(rng, 1)[0]
    z = X + rng.normal(0, 0.5, X.shape)
    z[4] += [200, -150]  # confidently wrong peak
    r = map_fit(model, z, np.full(5, 1.0), np.ones(5))
    good = [0, 1, 2, 3]
    assert np.abs(r.points[good] - X[good]).max() < 2.0
    assert np.linalg.norm(r.points[4] - X[4]) < 0.2 * np.linalg.norm(X[0] - X[1])
