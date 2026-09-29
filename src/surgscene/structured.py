"""Heatmaps -> observations -> structured estimates (Phase 3).

Three estimators per instrument (5 keypoints):
  argmax    independent per-keypoint peaks (the Phase 2 baseline)
  map       continuous robust MAP under the shape prior (shape.map_fit), from Laplace observations
  gated     per keypoint: argmax where the network is confident (conf >= tau), MAP elsewhere.
            MAP protects against gross errors but pulls already-precise points; the gate keeps both
  discrete  exact MAP over the top-N peak candidates of every keypoint, scored by
            heatmap evidence + shape plausibility; then refined with the continuous MAP
"""

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from surgscene.keypoints import decode
from surgscene.shape import ShapeModel, map_fit


@dataclass
class Observations:
    kp: np.ndarray        # (B, K, 2) sub-pixel argmax
    conf: np.ndarray      # (B, K) peak probability
    sigma: np.ndarray     # (B, K) Laplace std in px (from log-heatmap curvature at the peak)
    cand: np.ndarray      # (B, K, N, 2) top-N NMS peaks
    cand_conf: np.ndarray  # (B, K, N)


@torch.no_grad()
def observe(hm: torch.Tensor, n_cand: int = 3, nms: int = 9) -> Observations:
    """hm: (B, K, H, W) probabilities on GPU."""
    kp, conf = decode(hm)
    B, K, H, W = hm.shape
    # Laplace approximation: sigma^2 = -1 / (d^2 log h) at the integer peak, averaged over x and y
    lh = torch.log(hm.clamp(min=1e-6))
    idx = hm.flatten(2).argmax(-1)
    yi, xi = (idx // W).clamp(1, H - 2), (idx % W).clamp(1, W - 2)
    b, k = torch.meshgrid(torch.arange(B), torch.arange(K), indexing="ij")
    c = lh[b, k, yi, xi]
    dxx = lh[b, k, yi, xi - 1] - 2 * c + lh[b, k, yi, xi + 1]
    dyy = lh[b, k, yi - 1, xi] - 2 * c + lh[b, k, yi + 1, xi]
    curv = -(dxx + dyy) / 2
    sigma = (1 / curv.clamp(min=1e-3)).sqrt().clamp(1.0, 20.0)
    # candidates: local maxima after NMS
    peaks = hm * (F.max_pool2d(hm, nms, 1, nms // 2) == hm)
    cc, ci = peaks.flatten(2).topk(n_cand, -1)
    cand = torch.stack([(ci % W).float(), (ci // W).float()], -1)
    return Observations(kp, conf, sigma.cpu().numpy(), cand.cpu().numpy(), cc.cpu().numpy())


def fit_map(model: ShapeModel, z, conf, sigma, conf_min: float, beta: float, cauchy_c: float):
    w = np.where(conf >= conf_min, conf, 0.0)
    r = map_fit(model, z, sigma, w, beta=beta, cauchy_c=cauchy_c)
    return z.copy() if r is None else r.points


def fit_discrete(model: ShapeModel, cand, cand_conf, sigma, conf_min, beta, cauchy_c, gamma: float = 1.0,
                 plaus_max: float | None = None):
    """Enumerate one candidate per keypoint (N^K combos), score
        -sum log conf  +  gamma * 0.5 * plausibility(shape),
    keep the best, then refine with the continuous MAP."""
    K, N = cand_conf.shape
    opts = [[n for n in range(N) if cand_conf[k, n] >= 1e-3] or [0] for k in range(K)]
    best, best_E = None, np.inf
    for combo in itertools.product(*opts):
        Z = cand[np.arange(K), combo]
        E = -np.log(cand_conf[np.arange(K), combo] + 1e-6).sum() + gamma * 0.5 * min(
            model.plausibility(Z), plaus_max if plaus_max else np.inf)
        if E < best_E:
            best, best_E = combo, E
    Z = cand[np.arange(K), best]
    C = cand_conf[np.arange(K), best]
    return fit_map(model, Z, C, sigma, conf_min, beta, cauchy_c)


def fit_gated(model: ShapeModel, z, conf, sigma, tau: float, conf_min: float, beta: float, cauchy_c: float):
    m = fit_map(model, z, conf, sigma, conf_min, beta, cauchy_c)
    return np.where((conf >= tau)[:, None], z, m)
