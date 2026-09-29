"""Similarity-invariant statistical shape model + robust MAP fitting (Phase 3).

Shape model (per instrument, K keypoints in 2D):
    X = A(a, c) (mu + P b) + t,   A = [[a, -c], [c, a]] = s R(phi),   b ~ N(0, diag(lam))
The similarity (a, c, t) absorbs zoom, depth, and in-plane rotation, so the prior only has to
explain articulation and out-of-plane foreshortening (the PCA modes of b).

MAP energy, with each keypoint observed as a Gaussian measurement z_k ~ N(x_k, sigma_k^2 I)
(a Laplace approximation of its heatmap peak) and confidence weight w_k:
    E = sum_k w_k rho(||x_k - z_k||^2 / sigma_k^2) + beta * b^T diag(1/lam) b
rho is the Cauchy robust loss, so one confidently wrong peak can't drag the whole shape. It is
solved by Levenberg-Marquardt with IRLS weights. The solver is hand-written (no scipy) so the
C++ port in Phase 6 is a direct translation to Eigen.
"""

from dataclasses import dataclass

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True, weights: np.ndarray | None = None):
    """Least-squares similarity (s, R, t) with dst ~= s R src + t. src, dst: (N, d)."""
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    S, D = src - mu_s, dst - mu_d
    cov = (D * w[:, None]).T @ S
    U, sig, Vt = np.linalg.svd(cov)
    E = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[-1, -1] = -1
    R = U @ E @ Vt
    var_s = (w * (S**2).sum(1)).sum()
    s = (sig * np.diag(E)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def generalized_procrustes(shapes: np.ndarray, iters: int = 20):
    """shapes: (N, K, 2). Returns the mean shape (centered, unit Frobenius norm) and the aligned shapes."""
    mean = shapes[0] - shapes[0].mean(0)
    mean /= np.linalg.norm(mean)
    for _ in range(iters):
        aligned = np.empty_like(shapes)
        for i, X in enumerate(shapes):
            s, R, t = umeyama(X, mean)
            aligned[i] = s * X @ R.T + t
        new = aligned.mean(0)
        new -= new.mean(0)
        new /= np.linalg.norm(new)
        _, R, _ = umeyama(new, mean)  # keep the mean's orientation fixed across iterations (rotation only)
        new = new @ R.T
        if np.abs(new - mean).max() < 1e-9:
            break
        mean = new
    return mean, aligned


@dataclass
class ShapeModel:
    mean: np.ndarray       # (K, 2)
    P: np.ndarray          # (2K, m) PCA modes, rows ordered x0,y0,x1,y1,...
    lam: np.ndarray        # (m,) mode variances
    resid_var: float       # per-coordinate variance outside the subspace

    @property
    def K(self):
        return self.mean.shape[0]

    @classmethod
    def fit(cls, shapes: np.ndarray, var_explained: float = 0.95, max_modes: int = 6) -> "ShapeModel":
        mean, aligned = generalized_procrustes(shapes)
        D = (aligned - mean).reshape(len(shapes), -1)
        U, sig, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
        lam_all = sig**2 / (len(D) - 1)
        m = int(min(max_modes, np.searchsorted(np.cumsum(lam_all) / lam_all.sum(), var_explained) + 1))
        resid = lam_all[m:].sum() / max(D.shape[1] - m, 1)
        return cls(mean, Vt[:m].T, lam_all[:m], float(resid))

    def shape(self, b: np.ndarray) -> np.ndarray:
        return self.mean + (self.P @ b).reshape(self.K, 2)

    def plausibility(self, X: np.ndarray) -> float:
        """Squared Mahalanobis distance of X's shape (after similarity alignment) under the model:
        in-subspace b^T Lam^-1 b + out-of-subspace residual / resid_var. ~chi2(2K-4) for real shapes."""
        s, R, t = umeyama(X, self.mean)
        Y = (s * X @ R.T + t - self.mean).reshape(-1)
        b = self.P.T @ Y
        r = Y - self.P @ b
        return float((b**2 / self.lam).sum() + (r**2).sum() / self.resid_var)


def _params_to_points(model: ShapeModel, p: np.ndarray) -> np.ndarray:
    a, c, tx, ty, b = p[0], p[1], p[2], p[3], p[4:]
    Y = model.shape(b)
    return np.stack([a * Y[:, 0] - c * Y[:, 1] + tx, c * Y[:, 0] + a * Y[:, 1] + ty], 1)


def _jacobian(model: ShapeModel, p: np.ndarray) -> np.ndarray:
    """d(x_k)/dp stacked as (2K, 4+m)."""
    a, c, b = p[0], p[1], p[4:]
    Y = model.shape(b)
    K, m = model.K, len(b)
    J = np.zeros((2 * K, 4 + m))
    J[0::2, 0], J[1::2, 0] = Y[:, 0], Y[:, 1]
    J[0::2, 1], J[1::2, 1] = -Y[:, 1], Y[:, 0]
    J[0::2, 2], J[1::2, 3] = 1, 1
    Px, Py = model.P[0::2], model.P[1::2]
    J[0::2, 4:] = a * Px - c * Py
    J[1::2, 4:] = c * Px + a * Py
    return J


@dataclass
class FitResult:
    points: np.ndarray   # (K, 2) MAP keypoints
    params: np.ndarray   # [a, c, tx, ty, b...]
    energy: float
    iters: int
    ok: bool


def init_params(model: ShapeModel, z: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    use = w > 0
    if use.sum() < 2:
        return None
    s, R, t = umeyama(model.mean[use], z[use], weights=w[use])
    return np.concatenate([[s * R[0, 0], s * R[1, 0]], t, np.zeros(len(model.lam))])


def map_fit(model: ShapeModel, z: np.ndarray, sigma: np.ndarray, w: np.ndarray, beta: float = 1.0,
            cauchy_c: float = 3.0, iters: int = 30, p0: np.ndarray | None = None) -> FitResult | None:
    """Robust LM on E(p). z: (K, 2) observations, sigma: (K,) std in px, w: (K,) weights (0 = missing).
    cauchy_c is in units of sigma: residuals beyond ~c sigma are progressively down-weighted."""
    p = init_params(model, z, w) if p0 is None else p0.copy()
    if p is None:
        return None
    prior_prec = np.concatenate([np.zeros(4), beta / model.lam])

    def energy(p):
        r2 = (((_params_to_points(model, p) - z) / sigma[:, None]) ** 2).sum(1)
        data = (w * cauchy_c**2 * np.log1p(r2 / cauchy_c**2)).sum()
        return data + (prior_prec * p**2).sum(), r2

    E, r2 = energy(p)
    mu = 1e-3
    it = 0
    for it in range(1, iters + 1):
        irls = w / (1 + r2 / cauchy_c**2)  # Cauchy IRLS weight per keypoint
        Wk = np.repeat(irls / sigma**2, 2)
        J = _jacobian(model, p)
        res = (_params_to_points(model, p) - z).reshape(-1)
        H = J.T @ (Wk[:, None] * J) + np.diag(prior_prec)
        g = J.T @ (Wk * res) + prior_prec * p
        improved = False
        for _ in range(10):
            step = np.linalg.solve(H + mu * np.diag(np.diag(H) + 1e-9), -g)
            E_new, r2_new = energy(p + step)
            if E_new < E:
                p, E, r2, mu, improved = p + step, E_new, r2_new, max(mu / 3, 1e-7), True
                break
            mu *= 4
        if not improved or np.abs(step).max() < 1e-6:
            break
    return FitResult(_params_to_points(model, p), p, float(E), it, True)
