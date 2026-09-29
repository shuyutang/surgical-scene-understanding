"""Run the keypoint front end on every frame (30 fps) of the stage-2 trajectories, both eyes.

  uv run python scripts/cache_stage2_obs.py runs/<kp_run>/best.pt

Left eye is run twice: clean, and with *occlusion episodes*: every 60 frames, one random keypoint
per instrument is covered for 15 consecutive frames (0.5 s) by a tissue patch that moves with the
keypoint, center offset up to 0.6x radius (the de-centered Phase 2-3 protocol, now temporal).
Writes data/cache/surgpose_obs/<traj>_<eye>.npz and splits/surgpose_stage2.json.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from surgscene.frontend import SCALE, TEST2, TUNE, load_kp_model, load_vit_model, preprocess, run_batch, run_batch_vit
from surgscene.keypoints import _disk, paste_occluder

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/surgpose/raw"
OUT = ROOT / "data/cache/surgpose_obs"
KP_IDS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
OCC_RADIUS = 18.0  # half-res px, as in Phase 2-3
OCC_OFFSET = 0.6
EP_EVERY, EP_LEN = 60, 15


def load_gt(path: Path, n: int) -> np.ndarray:
    import yaml
    raw = yaml.load(open(path), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    gt = np.full((n, len(KP_IDS), 2), np.nan)
    for t, kps in raw.items():
        if int(t) < n and kps:
            for j, k in enumerate(KP_IDS):
                if k in kps:
                    gt[int(t), j] = kps[k]
    return gt


def episode_plan(n: int, seed: int):
    """occluded (n, 10) bool, plus a per-frame center offset (n, 10, 2) in units of the radius."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((n, len(KP_IDS)), bool)
    off = np.zeros((n, len(KP_IDS), 2))
    for s in range(30, n - EP_LEN, EP_EVERY):
        for j in (int(rng.integers(0, 5)), 5 + int(rng.integers(0, 5))):
            occ[s:s + EP_LEN, j] = True
            off[s:s + EP_LEN, j] = _disk(rng) * OCC_OFFSET
    return occ, off


def process(net, traj: int, eye: str, bs: int = 32, run=run_batch, out_dir=None):
    out_dir = OUT if out_dir is None else out_dir
    d = RAW / f"{traj:06d}"
    cap = cv2.VideoCapture(str(d / f"regular/{eye}_video.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt = load_gt(d / f"keypoints_{eye}.yaml", n)
    occluded, offset = episode_plan(n, seed=1000 + traj)
    conds = ["clean", "occ"] if eye == "left" else ["clean"]
    out = {c: {} for c in conds}
    buf = {c: [] for c in conds}
    rng = np.random.default_rng(traj)

    def flush():
        for c in conds:
            if buf[c]:
                r = run(net, buf[c])
                for k in r:
                    out[c].setdefault(k, []).append(r[k])
                buf[c] = []

    for t in tqdm(range(n), desc=f"{traj} {eye}", leave=False):
        ok, frame = cap.read()
        if not ok:
            n = t
            break
        img = preprocess(frame)
        buf["clean"].append(img)
        if "occ" in conds:
            o = img
            for j in np.flatnonzero(occluded[t]):
                if np.isfinite(gt[t, j]).all():
                    c = gt[t, j] * SCALE + offset[t, j] * OCC_RADIUS
                    o = paste_occluder(o, c, OCC_RADIUS, rng)
            buf["occ"].append(o)
        if len(buf["clean"]) == bs:
            flush()
    flush()
    res = {"gt": gt[:n]}
    for c in conds:
        sfx = "" if c == "clean" else "_occ"
        for k, v in out[c].items():
            res[k + sfx] = np.concatenate(v)[:n]
    if "occ" in conds:
        res["occluded"] = occluded[:n] & np.isfinite(gt[:n]).all(-1)
    np.savez_compressed(out_dir / f"{traj:06d}_{eye}.npz", **res)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--vit", action="store_true", help="v3 Phase A DINOv2 model; writes data/cache/surgpose_obs_vit")
    args = ap.parse_args()
    if args.vit:  # separate cache; the v2 manifest is left untouched
        out = ROOT / "data/cache/surgpose_obs_vit"
        out.mkdir(parents=True, exist_ok=True)
        net = load_vit_model(args.ckpt)
        for traj in TUNE + TEST2:
            for eye in ["left", "right"]:
                if not (out / f"{traj:06d}_{eye}.npz").exists():
                    print(traj, eye, process(net, traj, eye, run=run_batch_vit, out_dir=out), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    net = load_kp_model(args.ckpt)
    counts = {}
    for traj in TUNE + TEST2:
        for eye in ["left", "right"]:
            if not (OUT / f"{traj:06d}_{eye}.npz").exists():
                counts[f"{traj}_{eye}"] = process(net, traj, eye)
            else:
                counts[f"{traj}_{eye}"] = len(np.load(OUT / f"{traj:06d}_{eye}.npz")["gt"])
            print(traj, eye, counts[f"{traj}_{eye}"], flush=True)
    manifest = {"checkpoint": args.ckpt, "splits": {}}
    for name, trajs in [("tune", TUNE), ("test2", TEST2)]:
        key = "\n".join(f"{t}:{counts[f'{t}_left']}:{counts[f'{t}_right']}" for t in trajs)
        manifest["splits"][name] = {"trajectories": trajs, "hash": hashlib.sha256(key.encode()).hexdigest()[:16],
                                    "frames_per_eye": sum(counts[f"{t}_left"] for t in trajs)}
    (ROOT / "splits/surgpose_stage2.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["splits"], indent=1))


if __name__ == "__main__":
    main()
