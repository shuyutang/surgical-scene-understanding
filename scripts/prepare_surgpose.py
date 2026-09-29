"""Decode SurgPose trajectories into a half-resolution frame cache + keypoint manifest.

Split (official: 0-19 train, 20-33 val):
  train = 0-17, dev = 18-19 (both on the training backgrounds)
  test  = 20-33 (different ex vivo backgrounds, frozen)

Only keypoints 1-5 (PSM1) and 8-12 (PSM3) are used: the README says 6/7/13/14 are labeled on
a few frames only. Stored order: [PSM1 kp1..5, PSM3 kp8..12] = 10 keypoints,
per instrument: shaft, wrist pivot, jaw pivot, jaw tip A, jaw tip B.

Images are resized 1400x986 -> 700x493 and zero-padded to 704x512 (bottom/right), so
cached pixel coords = native coords * 0.5.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/surgpose/raw"
OUT = ROOT / "data/cache/surgpose"
SCALE = 0.5
PAD_WH = (704, 512)
KP_IDS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
SPLITS = {"train": range(0, 18), "dev": range(18, 20), "test": range(20, 34)}
STRIDE = {"train": 6, "dev": 5, "test": 5}


def load_kp(path: Path) -> dict[int, np.ndarray]:
    raw = yaml.safe_load(open(path))
    out = {}
    for t, kps in raw.items():
        a = np.full((len(KP_IDS), 2), np.nan)
        for j, k in enumerate(KP_IDS):
            if kps and k in kps:
                a[j] = kps[k]
        out[int(t)] = a
    return out


def process(job):
    traj, split, eyes = job
    d = RAW / f"{traj:06d}"
    recs = []
    for eye in eyes:
        kp = load_kp(d / f"keypoints_{eye}.yaml")
        cap = cv2.VideoCapture(str(d / f"regular/{eye}_video.mp4"))
        t = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if t % STRIDE[split] == 0 and t in kp:
                name = f"{traj:06d}_{eye}_{t:04d}"
                dst = OUT / "images" / f"{name}.jpg"
                if not dst.exists():
                    small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
                    canvas = np.zeros((PAD_WH[1], PAD_WH[0], 3), np.uint8)
                    canvas[:small.shape[0], :small.shape[1]] = small
                    cv2.imwrite(str(dst), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
                recs.append({"name": name, "traj": traj, "eye": eye, "t": t, "split": split,
                             "kp": (kp[t] * SCALE).round(2).tolist()})
            t += 1
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, trajs in SPLITS.items():
        for traj in trajs:
            assert (RAW / f"{traj:06d}/regular/left_video.mp4").exists(), f"trajectory {traj} missing"
            # stereo pairs for train/dev; test uses left only (right is kept for Phase 5)
            jobs.append((traj, split, ["left", "right"] if split != "test" else ["left"]))
    with ProcessPoolExecutor(args.workers) as ex:
        recs = [r for rs in ex.map(process, jobs) for r in rs]
    manifest = {"kp_ids": KP_IDS, "scale": SCALE, "pad_wh": PAD_WH, "splits": {}}
    for split in SPLITS:
        rs = [r for r in recs if r["split"] == split]
        h = hashlib.sha256("\n".join(sorted(r["name"] for r in rs)).encode()).hexdigest()[:16]
        manifest["splits"][split] = {"trajectories": list(SPLITS[split]), "n_frames": len(rs), "hash": h}
        print(f"{split:5s} trajectories={len(SPLITS[split]):2d} frames={len(rs):5d} hash={h}")
    (ROOT / "splits").mkdir(exist_ok=True)
    (ROOT / "splits/surgpose_split.json").write_text(json.dumps(manifest, indent=1))
    (OUT / "records.json").write_text(json.dumps(recs))


if __name__ == "__main__":
    main()
