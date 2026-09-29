"""Build the frozen SISVSE split manifest and half-resolution caches for SISVSE and EndoVis18.

Split (see the plan's Data section):
  test  = the 10 patients of official real_val_1 (frozen, hashed)
  dev   = 6 patients drawn (seed 0) from real_train_1
  train = remaining 24 patients

Outputs:
  splits/sisvse_split.json      committed; includes a hash of each split's frame list
  data/cache/sisvse/{images,masks}/<frame>.png
  data/cache/endovis18/{images,masks}/<split>_<seq>_<frame>.png
"""

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

SISVSE = ROOT / "data/sisvse/miccai2022_sisvse_dataset"
ENDOVIS = ROOT / "data/endovis18"
CACHE = ROOT / "data/cache"
SIZE = (640, 512)  # (W, H): half of the native 1280x1024 for both datasets


def patient(frame: str) -> str:
    return frame.split("_")[0]


def split_hash(frames: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(frames)).encode()).hexdigest()[:16]


def build_split() -> dict:
    frames = sorted(f[:-4] for f in os.listdir(SISVSE / "images/real"))
    test_p = sorted({patient(f[:-4]) for f in os.listdir(SISVSE / "semantic_masks/real_val_1")})
    rest_p = sorted({patient(f) for f in frames} - set(test_p))
    dev_p = sorted(random.Random(0).sample(rest_p, 6))
    train_p = sorted(set(rest_p) - set(dev_p))
    split = {}
    for name, ps in [("train", train_p), ("dev", dev_p), ("test", test_p)]:
        fs = [f for f in frames if patient(f) in ps]
        split[name] = {"patients": ps, "n_frames": len(fs), "hash": split_hash(fs), "frames": fs}
    assert not (set(train_p) & set(dev_p)) and not (set(rest_p) & set(test_p))
    return split


def _mask_src(frame: str) -> Path:
    for d in sorted((SISVSE / "semantic_masks").glob("real_*")):
        p = d / f"{frame}.png"
        if p.exists():
            return p
    raise FileNotFoundError(frame)


def _resize_pair(args):
    img_src, mask_src, img_dst, mask_dst = args
    if Path(img_dst).exists() and Path(mask_dst).exists():
        return
    img = cv2.imread(str(img_src), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_src), cv2.IMREAD_UNCHANGED)
    assert mask.ndim == 2, mask_src
    cv2.imwrite(str(img_dst), cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(mask_dst), cv2.resize(mask, SIZE, interpolation=cv2.INTER_NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    split = build_split()
    (ROOT / "splits").mkdir(exist_ok=True)
    out = ROOT / "splits/sisvse_split.json"
    if out.exists():
        old = json.loads(out.read_text())
        for k in split:
            assert old[k]["hash"] == split[k]["hash"], f"frozen split '{k}' changed"
    out.write_text(json.dumps(split, indent=1))
    for k, v in split.items():
        print(f"{k:5s} patients={len(v['patients']):2d} frames={v['n_frames']:4d} hash={v['hash']}")

    jobs = []
    for sub in ["images", "masks"]:
        (CACHE / "sisvse" / sub).mkdir(parents=True, exist_ok=True)
        (CACHE / "endovis18" / sub).mkdir(parents=True, exist_ok=True)
    for v in split.values():
        for f in v["frames"]:
            jobs.append((SISVSE / f"images/real/{f}.jpg", _mask_src(f),
                         CACHE / f"sisvse/images/{f}.png", CACHE / f"sisvse/masks/{f}.png"))
    for img in sorted(ENDOVIS.glob("*/seq_*/left_frames/*.png")):
        sp, seq = img.parts[-4], img.parts[-3]
        name = f"{sp}_{seq}_{img.stem}"
        jobs.append((img, img.parents[1] / "class_labels" / img.name,
                     CACHE / f"endovis18/images/{name}.png", CACHE / f"endovis18/masks/{name}.png"))
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_resize_pair, jobs, chunksize=32))
    print(f"cached {len(jobs)} image/mask pairs at {SIZE}")


if __name__ == "__main__":
    main()
