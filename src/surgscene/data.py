"""Segmentation datasets over the half-resolution caches built by scripts/prepare_seg_data.py."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/cache"
IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sisvse_frames(split: str) -> list[str]:
    return json.loads((ROOT / "splits/sisvse_split.json").read_text())[split]["frames"]


def endovis18_frames(split: str | None = None) -> list[str]:
    names = sorted(p.stem for p in (CACHE / "endovis18/images").glob("*.png"))
    return [n for n in names if split is None or n.startswith(split + "_")]


def sisvse_group(frame: str) -> str:
    return frame.split("_")[0]  # patient id


def endovis18_group(frame: str) -> str:
    return "_".join(frame.split("_")[:3])  # e.g. test_seq_1


def normalize(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


class SegDataset(Dataset):
    def __init__(self, dataset: str, frames: list[str], augment: bool = False, label_lut: np.ndarray | None = None):
        self.dir = CACHE / dataset
        self.frames = frames
        self.augment = augment
        self.label_lut = label_lut
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.frames)

    def _augment(self, img, mask):
        rng = self.rng
        h, w = mask.shape
        # random scale + crop/pad back to the original size
        s = rng.uniform(0.75, 1.5)
        nh, nw = int(h * s), int(w * s)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if s >= 1:
            y, x = rng.integers(0, nh - h + 1), rng.integers(0, nw - w + 1)
            img, mask = img[y:y + h, x:x + w], mask[y:y + h, x:x + w]
        else:
            pi = np.zeros((h, w, 3), img.dtype)
            pm = np.full((h, w), IGNORE, mask.dtype)
            y, x = rng.integers(0, h - nh + 1), rng.integers(0, w - nw + 1)
            pi[y:y + nh, x:x + nw], pm[y:y + nh, x:x + nw] = img, mask
            img, mask = pi, pm
        if rng.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        # photometric: brightness/contrast/saturation, blur (motion-blur proxy)
        f = img.astype(np.float32)
        f = f * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
        hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < 0.2:
            k = int(rng.choice([3, 5, 7]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, i):
        name = self.frames[i]
        img = cv2.imread(str(self.dir / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.dir / "masks" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if self.label_lut is not None:
            mask = self.label_lut[mask]
        if self.augment:
            img, mask = self._augment(img, mask)
        return normalize(img), torch.from_numpy(mask.astype(np.int64)), i


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)
