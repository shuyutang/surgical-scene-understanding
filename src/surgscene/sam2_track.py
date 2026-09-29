"""v4: SAM 2.1 instrument masks, causal streaming, prompted from our own keypoint detections.

Object k + 1 = instrument k in `keypoints.INSTRUMENTS` (PSM1 -> 1, PSM3 -> 2). Masks are in native
(unrectified) image pixels, the same frame as the SurgPose keypoints. See
docs/v4_step0_sam2_feasibility.md for the choices (shaft-extended prompts, pruning).
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch

from .keypoints import INSTRUMENTS

ARMS = list(INSTRUMENTS)
CONF_MIN = 0.5
MEMORY_KEEP = 16  # frames of per-frame state kept; SAM 2 reads the last 6 memories and 15 pointers


def prompt_points(kp: np.ndarray, conf: np.ndarray, frame_hw: tuple[int, int], extend_shaft: bool = True):
    """Per instrument: confident detected keypoints, plus 3 points extrapolated along the shaft
    (shaft + k (shaft - wrist), k = 1..3) when both are confident and the point is in the image."""
    h, w = frame_hw
    pts, labels = [], []
    for arm in ARMS:
        sl = INSTRUMENTS[arm]
        k, c = kp[sl], conf[sl]
        keep = c >= CONF_MIN
        p = k[keep].tolist()
        if extend_shaft and keep[0] and keep[1]:
            d = k[0] - k[1]
            for m in (1, 2, 3):
                q = k[0] + m * d
                if 0 <= q[0] < w and 0 <= q[1] < h:
                    p.append(q.tolist())
        pts.append(p)
        labels.append([1] * len(p))
    return pts, labels


def start_frame(conf: np.ndarray, min_kp: int = 3) -> int | None:
    """First frame where every instrument has >= min_kp confident keypoints, preferring frames where
    shaft and wrist are both confident (so the shaft extension applies); None if there is none."""
    ok_any, ok_shaft = [], []
    for c in conf:
        per = [c[INSTRUMENTS[a]] >= CONF_MIN for a in ARMS]
        ok_any.append(all(p.sum() >= min_kp for p in per))
        ok_shaft.append(ok_any[-1] and all(p[0] and p[1] for p in per))
    for flags in (ok_shaft, ok_any):
        if any(flags):
            return int(np.argmax(flags))
    return None


class Sam2Tracker:
    def __init__(self, size: str = "base-plus", device: str = "cuda"):
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        repo = f"facebook/sam2.1-hiera-{size}"
        self.model = Sam2VideoModel.from_pretrained(repo).to(device, dtype=torch.bfloat16).eval()
        self.proc = Sam2VideoProcessor.from_pretrained(repo)
        self.device = device

    @torch.inference_mode()
    def track(self, frames: Iterator[np.ndarray], pts, labels) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """frames: BGR frames starting at the prompted frame. Yields (masks bool [2, H, W], object
        score logits [2]) per frame. Memory is pruned to MEMORY_KEEP frames; old entries are set to
        None rather than deleted, because the session numbers new frames by len(processed_frames)."""
        s = self.proc.init_video_session(inference_device=self.device, dtype=torch.bfloat16)
        for t, bgr in enumerate(frames):
            inp = self.proc(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), device=self.device, return_tensors="pt")
            if t == 0:
                self.proc.add_inputs_to_inference_session(
                    inference_session=s, frame_idx=0, obj_ids=[1, 2], input_points=[pts], input_labels=[labels],
                    original_size=inp.original_sizes[0])
            out = self.model(inference_session=s, frame=inp.pixel_values[0])
            m = self.proc.post_process_masks([out.pred_masks], original_sizes=inp.original_sizes, binarize=True)[0]
            order = [list(out.object_ids).index(k) for k in (1, 2)]
            yield m[order, 0].cpu().numpy().astype(bool), out.object_score_logits.float().reshape(-1)[order].cpu().numpy()
            old = t - MEMORY_KEEP
            if old > 0:
                s.processed_frames[old] = None
                for o in s.output_dict_per_obj.values():
                    o["non_cond_frame_outputs"].pop(old, None)


def save_masks(path: Path, masks: np.ndarray, **meta) -> None:
    """masks: bool [T, 2, H, W], stored bit-packed along W and compressed."""
    np.savez_compressed(path, packed=np.packbits(masks, axis=-1), width=masks.shape[-1], **meta)


def load_masks(path: Path) -> dict[str, np.ndarray]:
    d = dict(np.load(path))
    d["masks"] = np.unpackbits(d.pop("packed"), axis=-1, count=int(d["width"])).astype(bool)
    return d


def shaft_width(mask: np.ndarray, shaft: np.ndarray, wrist: np.ndarray, t0: float = 15.0, t1: float = 115.0,
                half: float = 60.0) -> tuple[float, np.ndarray]:
    """Apparent shaft width (px) from an instrument mask, past the 2D shaft keypoint.

    The shaft axis is the principal axis of the mask pixels in a band beyond the shaft keypoint
    (initial direction wrist -> shaft); the width is the band's mask area / length over
    t in [t0, t1] px along the axis, |perpendicular| < half. Returns (width, unit axis) or
    (nan, axis) when the band leaves the image or holds too little mask."""
    d0 = shaft - wrist
    n0 = np.linalg.norm(d0)
    if not np.isfinite(n0) or n0 < 1:
        return np.nan, np.full(2, np.nan)
    d = d0 / n0
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float) - shaft
    for _ in range(2):  # refine the axis by PCA of the band
        t, q = P @ d, P @ np.array([-d[1], d[0]])
        sel = (t >= t0) & (t <= t1) & (np.abs(q) < half)
        if sel.sum() < 50:
            return np.nan, d
        Q = P[sel] - P[sel].mean(0)
        ev = np.linalg.eigh(Q.T @ Q)[1][:, -1]
        d = ev if ev @ d > 0 else -ev
    ends = shaft + np.outer([t0, t1], d)
    corners = np.concatenate([ends + half * np.array([-d[1], d[0]]), ends - half * np.array([-d[1], d[0]])])
    if (corners < 0).any() or (corners[:, 0] >= w).any() or (corners[:, 1] >= h).any():
        return np.nan, d
    t, q = P @ d, P @ np.array([-d[1], d[0]])
    sel = (t >= t0) & (t <= t1) & (np.abs(q) < half)
    return float(sel.sum() / (t1 - t0)), d
