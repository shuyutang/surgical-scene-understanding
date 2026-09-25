"""Instrument keypoint heatmaps on SurgPose: dataset, targets, loss, decoding."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from surgscene.data import normalize

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/cache/surgpose"
N_KP = 10
KP_NAMES = ["shaft", "wrist", "jaw_pivot", "tip_a", "tip_b"]
INSTRUMENTS = {"PSM1": slice(0, 5), "PSM3": slice(5, 10)}
SIGMA = 2.5  # px at cache resolution (= 5 px native)


def load_records(split: str) -> list[dict]:
    return [r for r in json.loads((CACHE / "records.json").read_text()) if r["split"] == split]


def render_heatmaps(kp: np.ndarray, h: int, w: int, sigma: float = SIGMA) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    hm = np.zeros((len(kp), h, w), np.float32)
    for j, (x, y) in enumerate(kp):
        if np.isfinite(x) and 0 <= x < w and 0 <= y < h:
            hm[j] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    return hm


def paste_occluder(img: np.ndarray, center: tuple[float, float], radius: float, rng) -> np.ndarray:
    """Cover a disk around `center` with a tissue patch copied from elsewhere in the same image."""
    h, w = img.shape[:2]
    r = int(radius)
    cx, cy = int(center[0]), int(center[1])
    for _ in range(20):  # find a source patch away from the target
        sx, sy = rng.integers(r, w - r), rng.integers(r, h - r)
        if abs(sx - cx) > 3 * r or abs(sy - cy) > 3 * r:
            break
    patch = img[sy - r:sy + r, sx - r:sx + r].copy()
    mask = np.zeros((2 * r, 2 * r), np.uint8)
    cv2.ellipse(mask, (r, r), (r, int(r * rng.uniform(0.6, 1.0))), float(rng.uniform(0, 180)), 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32)[..., None] / 255
    y0, x0 = cy - r, cx - r
    ty0, tx0, ty1, tx1 = max(y0, 0), max(x0, 0), min(y0 + 2 * r, h), min(x0 + 2 * r, w)
    if ty1 <= ty0 or tx1 <= tx0:
        return img
    m = mask[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    p = patch[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0].astype(np.float32)
    out = img.copy()
    out[ty0:ty1, tx0:tx1] = (m * p + (1 - m) * out[ty0:ty1, tx0:tx1]).astype(np.uint8)
    return out


class KeypointDataset(Dataset):
    """Returns (image, heatmaps, keypoints, index). `occlude` is a fixed list of keypoint indices to
    occlude per item (for the controlled occlusion test) or None."""

    def __init__(self, records: list[dict], augment: bool = False, occlude: list[list[int]] | None = None,
                 occluder_radius: float = 18.0, seed: int = 0):
        self.records = records
        self.augment = augment
        self.occlude = occlude
        self.occluder_radius = occluder_radius
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def _augment(self, img, kp):
        rng = self.rng
        h, w = img.shape[:2]
        s, a = rng.uniform(0.8, 1.25), rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
        M[:, 2] += rng.uniform(-0.1, 0.1, 2) * (w, h)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        kp = kp @ M[:, :2].T + M[:, 2]
        # photometric, including hue shifts: test trajectories have different tissue backgrounds
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
        hsv[..., 1] *= rng.uniform(0.6, 1.4)
        hsv[..., 2] = hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < 0.2:
            k = int(rng.choice([3, 5, 7]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        # random occluders over keypoints teach the net to *not* hallucinate a confident peak;
        # the structured prior (Phase 3) is what should recover them
        for j in np.flatnonzero(rng.random(len(kp)) < 0.1):
            if np.isfinite(kp[j]).all():
                img = paste_occluder(img, kp[j], rng.uniform(10, 25), rng)
        return img, kp

    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.imread(str(CACHE / "images" / f"{r['name']}.jpg"), cv2.IMREAD_COLOR)
        kp = np.asarray(r["kp"], np.float64)
        if self.augment:
            img, kp = self._augment(img, kp)
        if self.occlude is not None:
            rng = np.random.default_rng(self.seed * 100003 + i)
            for j in self.occlude[i]:
                img = paste_occluder(img, kp[j], self.occluder_radius, rng)
        h, w = img.shape[:2]
        hm = render_heatmaps(kp, h, w)
        return normalize(img), torch.from_numpy(hm), torch.from_numpy(kp.astype(np.float32)), i


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss on Gaussian heatmap targets."""
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p**alpha) * torch.log(1 - p) * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def decode(heatmaps: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Argmax + quadratic sub-pixel refinement. heatmaps: (B, K, H, W) probabilities.
    Returns keypoints (B, K, 2) in pixels and peak confidences (B, K)."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.flatten(2)
    conf, idx = flat.max(-1)
    ys, xs = (idx // W).float(), (idx % W).float()
    hm = F.pad(heatmaps, (1, 1, 1, 1), mode="replicate")
    bi = torch.arange(B)[:, None].expand(B, K)
    ki = torch.arange(K)[None, :].expand(B, K)
    yi, xi = (idx // W) + 1, (idx % W) + 1
    lg = lambda dy, dx: torch.log(hm[bi, ki, yi + dy, xi + dx].clamp(min=1e-8))
    c, l, r, u, d = lg(0, 0), lg(0, -1), lg(0, 1), lg(-1, 0), lg(1, 0)
    dx = ((l - r) / (2 * (l - 2 * c + r)).clamp(max=-1e-6)).clamp(-0.5, 0.5)
    dy = ((u - d) / (2 * (u - 2 * c + d)).clamp(max=-1e-6)).clamp(-0.5, 0.5)
    kp = torch.stack([xs + dx, ys + dy], -1)
    return kp.cpu().numpy(), conf.cpu().numpy()
