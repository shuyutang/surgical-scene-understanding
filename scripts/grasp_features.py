"""Scene semantics: frozen frame features for every GraSP 1 fps frame (train and test cases).

  uv run python scripts/grasp_features.py --backbone resnet50|dinov2_b14

Both backbones see the same input: the full 1280 x 800 frame resized to 224 x 224, ImageNet
normalisation. Features are computed once and never fine-tuned (frozen-backbone comparison):
  resnet50     torchvision ResNet-50, ImageNet (IMAGENET1K_V2), global-average-pooled 2048-d
  dinov2_b14   timm vit_base_patch14_dinov2 (LVD-142M), [CLS, mean patch token] 1536-d
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

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data/grasp/GraSP_1fps/frames"
OUT = ROOT / "data/cache/grasp_feats"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Frames(Dataset):
    def __init__(self, case: str, frames: np.ndarray):
        self.paths = [FRAMES / case / f"{f:05d}.jpg" for f in frames]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(str(self.paths[i])), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1))


def build(backbone: str):
    if backbone == "resnet50":
        import torchvision
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        return net, lambda x: net(x)
    import timm
    net = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0)

    def fwd(x):
        t = net.forward_features(x)  # (B, 1 + N, 768)
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
    return net, fwd


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["resnet50", "dinov2_b14"], required=True)
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
            for x in DataLoader(Frames(case, frames), batch_size=128, num_workers=24, pin_memory=True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats.append(fwd(x.cuda(non_blocking=True)).float().cpu().numpy().astype(np.float16))
            f = np.concatenate(feats)
            assert len(f) == len(frames) and np.isfinite(f).all()
            np.save(path, f)
            print(f"{args.backbone} {case}: {f.shape} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
