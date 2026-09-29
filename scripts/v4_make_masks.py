"""v4 step 1: SAM 2.1 base-plus instrument masks for every SurgPose trajectory, both eyes, offline.

  uv run --group sam python scripts/v4_make_masks.py [--trajs ...]

The same procedure everywhere, with no labels: the v2 U-Net detector (the checkpoint that built
the stage-2 cache) runs on the first SEARCH frames, the start frame is the first where both
instruments have >= 3 confident keypoints (preferring shaft + wrist), and SAM 2 is prompted there
once and tracks to the end, causally. Frames before the start have `valid = False`.
Writes data/cache/surgpose_masks/<traj>_<eye>.npz (masks bit-packed, object score logits, start,
prompts) and data/cache/surgpose_masks/manifest.json.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from surgscene.frontend import load_kp_model, preprocess, run_batch
from surgscene.sam2_track import Sam2Tracker, prompt_points, save_masks, start_frame

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/surgpose/raw"
OUT = ROOT / "data/cache/surgpose_masks"
SEARCH = 90
SIZE = "base-plus"


def video(traj: int, eye: str) -> Path:
    return RAW / f"{traj:06d}/regular/{eye}_video.mp4"


def frames(traj: int, eye: str, start: int = 0):
    cap = cv2.VideoCapture(str(video(traj, eye)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", nargs="+", type=int, default=sorted(int(p.name) for p in RAW.iterdir() if p.is_dir()))
    ap.add_argument("--eyes", nargs="+", default=["left", "right"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    ckpt = json.loads((ROOT / "splits/surgpose_stage2.json").read_text())["checkpoint"]
    manifest["_meta"] = {"sam2": f"facebook/sam2.1-hiera-{SIZE}", "detector": ckpt, "search_frames": SEARCH}
    det = load_kp_model(ckpt)
    tracker = Sam2Tracker(SIZE)
    for traj in args.trajs:
        for eye in args.eyes:
            head = [f for _, f in zip(range(SEARCH), frames(traj, eye))]
            o = run_batch(det, [preprocess(f) for f in head])
            t0 = start_frame(o["conf"])
            n_total = int(cv2.VideoCapture(str(video(traj, eye))).get(cv2.CAP_PROP_FRAME_COUNT))
            key = f"{traj:06d}_{eye}"
            if t0 is None:
                manifest[key] = {"start": None, "frames": n_total}
                print(key, "no start frame in the first", SEARCH, flush=True)
                continue
            h, w = head[0].shape[:2]
            pts, labels = prompt_points(o["kp"][t0], o["conf"][t0], (h, w))
            masks = np.zeros((n_total, 2, h, w), bool)
            score = np.full((n_total, 2), np.nan, np.float32)
            tic = time.perf_counter()
            n = 0
            for i, (m, s) in enumerate(tracker.track(frames(traj, eye, t0), pts, labels)):
                masks[t0 + i], score[t0 + i] = m, s
                n = i + 1
            valid = np.zeros(n_total, bool)
            valid[t0:t0 + n] = True
            save_masks(OUT / f"{key}.npz", masks, score=score, valid=valid, start=t0,
                       prompts=json.dumps({"points": pts, "labels": labels}))
            area = masks[valid].reshape(n, 2, -1).mean(-1)
            manifest[key] = {"start": t0, "frames": n_total, "tracked": n, "n_prompt_points": [len(p) for p in pts],
                             "empty_frac": (area == 0).mean(0).round(4).tolist(),
                             "area_median": np.median(area, 0).round(5).tolist(),
                             "score_neg_frac": (score[valid] < 0).mean(0).round(4).tolist(),
                             "sec": round(time.perf_counter() - tic, 1)}
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(key, json.dumps(manifest[key]), flush=True)


if __name__ == "__main__":
    main()
