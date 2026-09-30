"""Scene semantics: frozen frame features for every GraSP 1 fps frame (train and test cases).

  uv run python scripts/grasp_features.py --backbone resnet50|dinov2_b14|endossl_l16

Every backbone sees the same input: the full 1280 x 800 frame resized to 224 x 224 (ImageNet
normalisation, except EndoSSL: raw pixels). Features are computed once and never fine-tuned
(frozen-backbone comparison); backbones are defined in surgscene.backbones:
  resnet50     torchvision ResNet-50, ImageNet (IMAGENET1K_V2), global-average-pooled 2048-d
  dinov2_b14   timm vit_base_patch14_dinov2 (LVD-142M), [CLS, mean patch token] 1536-d
  endossl_l16  EndoSSL ViT-L/16 (MSN on private laparoscopy video; Hirsch et al., MICCAI 2023), the
               PyTorch conversion released with SurgVISTA (third_party/endossl/surgvista_teacher.pth;
               its weights equal the official JAX checkpoint). It takes raw 0-255 RGB, no
               normalisation: that reproduces the official TF SavedModel (cosine 0.9998 on the CLS
               output). [CLS, mean patch token] after the final norm, 2048-d
Frames follow splits/grasp_split.json. Writes data/cache/grasp_feats/<backbone>/<case>.npy (float16).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from surgscene.backbones import BACKBONES, build, preprocess

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data/grasp/GraSP_1fps/frames"
OUT = ROOT / "data/cache/grasp_feats"


class Frames(Dataset):
    def __init__(self, case: str, frames: np.ndarray, backbone: str):
        self.paths = [FRAMES / case / f"{f:05d}.jpg" for f in frames]
        self.backbone = backbone

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return preprocess(cv2.cvtColor(cv2.imread(str(self.paths[i])), cv2.COLOR_BGR2RGB), self.backbone)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    args = ap.parse_args()
    split = json.loads((ROOT / "splits/grasp_split.json").read_text())
    net, fwd = build(args.backbone)
    net = net.cuda().eval()
    out = OUT / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "test"):
        for case in split["splits"][s]["cases"]:
            path = out / f"{case}.npy"
            if path.exists():
                continue
            frames = np.load(ROOT / f"data/cache/grasp/{case}.npz")["frame"]
            t0, feats = time.time(), []
            for x in DataLoader(Frames(case, frames, args.backbone), batch_size=128, num_workers=24, pin_memory=True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats.append(fwd(x.cuda(non_blocking=True)).float().cpu().numpy().astype(np.float16))
            f = np.concatenate(feats)
            assert len(f) == len(frames) and np.isfinite(f).all()
            np.save(path, f)
            print(f"{args.backbone} {case}: {f.shape} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
