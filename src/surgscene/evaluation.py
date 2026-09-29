"""Segmentation evaluation: per-frame sufficient statistics + patient-clustered bootstrap.

Every metric is computed from per-frame counts, so the bootstrap can resample *groups*
(patients / sequences) and re-pool counts without re-running the model:

  dice/iou     pooled TP/FP/FN over the sampled frames
  boundary F   pooled boundary-pixel matches within a tolerance (in pixels)
  hd95         per-frame symmetric 95th-percentile boundary distance, median over frames
  ece          pooled confidence-bin histogram
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

N_BINS = 15


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def _dist_to(boundary: np.ndarray) -> np.ndarray:
    # distance from every pixel to the nearest boundary pixel
    return cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)


@dataclass
class FrameStats:
    """Per-frame counts. Arrays are indexed by class id."""
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    bf_pred_hit: np.ndarray   # predicted boundary px within tol of GT boundary
    bf_pred_n: np.ndarray
    bf_gt_hit: np.ndarray     # GT boundary px within tol of predicted boundary
    bf_gt_n: np.ndarray
    hd95: np.ndarray          # NaN where the class is absent from GT or prediction
    conf_count: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS))
    conf_sum: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS))
    correct_sum: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS))


def frame_stats(pred: np.ndarray, gt: np.ndarray, num_classes: int, conf: np.ndarray | None = None,
                ignore: int = 255, tol: float = 3.0, classes: list[int] | None = None) -> FrameStats:
    valid = gt != ignore
    K = num_classes
    idx = gt[valid].astype(np.int64) * K + pred[valid].astype(np.int64)
    cm = np.bincount(idx, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(np.int64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    z = lambda: np.zeros(K, np.int64)
    s = FrameStats(tp, fp, fn, z(), z(), z(), z(), np.full(K, np.nan))
    for c in (classes if classes is not None else range(K)):
        g, p = (gt == c) & valid, (pred == c) & valid
        if not g.any() and not p.any():
            continue
        gb, pb = _boundary(g), _boundary(p)
        s.bf_pred_n[c], s.bf_gt_n[c] = pb.sum(), gb.sum()
        if gb.any() and pb.any():
            dg, dp = _dist_to(gb), _dist_to(pb)
            s.bf_pred_hit[c] = (dg[pb] <= tol).sum()
            s.bf_gt_hit[c] = (dp[gb] <= tol).sum()
            s.hd95[c] = np.percentile(np.concatenate([dg[pb], dp[gb]]), 95)
    if conf is not None:
        c, ok = conf[valid], (pred[valid] == gt[valid])
        b = np.minimum((c * N_BINS).astype(int), N_BINS - 1)
        s.conf_count = np.bincount(b, minlength=N_BINS).astype(float)
        s.conf_sum = np.bincount(b, weights=c, minlength=N_BINS)
        s.correct_sum = np.bincount(b, weights=ok, minlength=N_BINS)
    return s


def stack(stats: list[FrameStats]) -> dict[str, np.ndarray]:
    return {k: np.stack([getattr(s, k) for s in stats]) for k in FrameStats.__dataclass_fields__}


def summarize(S: dict[str, np.ndarray], rows: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    """Pool the per-frame stats over `rows` (with repeats, for the bootstrap)."""
    T = {k: (v if rows is None else v[rows]) for k, v in S.items()}
    tp, fp, fn = T["tp"].sum(0), T["fp"].sum(0), T["fn"].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        bp = T["bf_pred_hit"].sum(0) / T["bf_pred_n"].sum(0)
        br = T["bf_gt_hit"].sum(0) / T["bf_gt_n"].sum(0)
        bf = 2 * bp * br / (bp + br)
    hd = T["hd95"]
    hd95 = np.array([np.nanmedian(hd[:, c]) if np.isfinite(hd[:, c]).any() else np.nan for c in range(hd.shape[1])])
    n, cs, ok = T["conf_count"].sum(0), T["conf_sum"].sum(0), T["correct_sum"].sum(0)
    ece = float(np.sum(np.abs(cs - ok)) / n.sum()) if n.sum() else np.nan
    return {"dice": dice, "iou": iou, "bf": bf, "hd95": hd95, "ece": ece, "gt_pixels": tp + fn}


def cluster_bootstrap(S: dict[str, np.ndarray], groups: list[str], fn, n_boot: int = 2000, seed: int = 0):
    """Resample groups with replacement; returns an array of fn(summary) per replicate."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    rows_of = {g: np.flatnonzero(np.asarray(groups) == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        rows = np.concatenate([rows_of[uniq[i]] for i in pick])
        out.append(fn(summarize(S, rows)))
    return np.asarray(out)


def ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    return np.nanpercentile(samples, 100 * alpha / 2, axis=0), np.nanpercentile(samples, 100 * (1 - alpha / 2), axis=0)


def quality_proxies(img_bgr: np.ndarray) -> dict[str, float]:
    """Cheap, label-free image-condition proxies used to define robustness subgroups."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    dark = cv2.erode(img_bgr.min(axis=2), np.ones((15, 15), np.uint8))
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),                 # low -> motion blur / defocus
        "specular": float(((hsv[..., 2] > 230) & (hsv[..., 1] < 50)).mean()),      # high -> glare
        "haze": float(dark.mean()),                                                # high -> smoke / fog
        "brightness": float(hsv[..., 2].mean()),                                   # low -> underexposed
    }


def boot(values: list[np.ndarray], n_boot=2000, seed=0):
    """values: one array per trajectory (NaN = excluded). Mean over pooled finite entries, with a
    trajectory-clustered bootstrap."""
    rng = np.random.default_rng(seed)
    sums = np.array([np.nansum(v) for v in values])
    cnts = np.array([np.isfinite(v).sum() for v in values])
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(values), len(values))
        reps.append(sums[i].sum() / max(cnts[i].sum(), 1))
    lo, hi = ci(np.asarray(reps))
    return {"point": float(sums.sum() / cnts.sum()), "lo": float(lo), "hi": float(hi), "n": int(cnts.sum())}
