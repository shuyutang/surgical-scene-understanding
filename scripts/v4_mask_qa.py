"""v4 step 1 QA: quality of the SAM 2 masks in data/cache/surgpose_masks.

  uv run python scripts/v4_mask_qa.py

Label-based proxies (GT keypoint coverage within TOL px of the own mask, swaps, box IoU with
SurgPose's boxes) are computed on train + tune only. For test2 only label-free statistics are
computed (empty masks, negative object scores, area), so the step doesn't read test2 labels.
Writes runs/v4_masks_qa/{results.json, report.md}.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from surgscene.frontend import TEST2, TUNE
from surgscene.keypoints import KP_NAMES
from surgscene.sam2_track import load_masks

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/surgpose/raw"
MASKS = ROOT / "data/cache/surgpose_masks"
OUT = ROOT / "runs/v4_masks_qa"
KP_IDS = [[1, 2, 3, 4, 5], [8, 9, 10, 11, 12]]  # SurgPose ids per instrument, in KP_NAMES order
TOL = 5.0


def gt_keypoints(traj: int, eye: str, n: int) -> np.ndarray:
    y = yaml.load(open(RAW / f"{traj:06d}/keypoints_{eye}.yaml"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    g = np.full((n, 2, 5, 2), np.nan)
    for t, kps in y.items():
        if int(t) >= n:
            continue
        for a, ids in enumerate(KP_IDS):
            for j, k in enumerate(ids):
                if k in kps:
                    g[int(t), a, j] = kps[k]
    return g


def box_iou(mask, box):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    a = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
    b = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], float)
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def labelled_metrics(traj, eye, d):
    M, valid = d["masks"], d["valid"]
    g = gt_keypoints(traj, eye, len(M))
    boxes = json.loads((RAW / f"{traj:06d}/bbox_{eye}.json").read_text())
    cov = np.full((len(M), 2, 5), np.nan)
    swap = np.full((len(M), 2, 5), np.nan)
    iou = [[], []]
    h, w = M.shape[-2:]
    for t in np.nonzero(valid)[0]:
        dist = [cv2.distanceTransform((~M[t, a]).astype(np.uint8), cv2.DIST_L2, 5) for a in range(2)]
        for a in range(2):
            lab = np.isfinite(g[t, a]).all(-1)
            if lab.any():
                p = g[t, a][lab]
                xi, yi = np.clip(p[:, 0].round().astype(int), 0, w - 1), np.clip(p[:, 1].round().astype(int), 0, h - 1)
                cov[t, a, lab] = dist[a][yi, xi] <= TOL
                swap[t, a, lab] = dist[1 - a][yi, xi] <= TOL
        # SurgPose's obj1/obj2 aren't tied to PSM1/PSM3 (trajectory 0 has them reversed vs 18),
        # so each mask takes the better of the two box assignments for the frame
        bx = boxes.get(str(t), {})
        if "obj1" in bx and "obj2" in bx:
            same = [box_iou(M[t, 0], bx["obj1"]), box_iou(M[t, 1], bx["obj2"])]
            cross = [box_iou(M[t, 0], bx["obj2"]), box_iou(M[t, 1], bx["obj1"])]
            best = same if sum(same) >= sum(cross) else cross
            for a in range(2):
                iou[a].append(best[a])
    return {"coverage_by_kp": np.nanmean(cov, 0).round(4).tolist(), "coverage": float(np.nanmean(cov)),
            "coverage_body": float(np.nanmean(cov[:, :, :3])), "coverage_tips": float(np.nanmean(cov[:, :, 3:])),
            "swap": float(np.nanmean(swap)), "box_iou_median": [float(np.median(i)) if i else None for i in iou]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((MASKS / "manifest.json").read_text())
    R = {}
    for key in sorted(k for k in manifest if not k.startswith("_")):
        traj, eye = int(key[:6]), key[7:]
        split = "test2" if traj in TEST2 else "tune" if traj in TUNE else "train"
        r = {"split": split, "start": manifest[key]["start"]}
        if r["start"] is None:
            R[key] = r
            continue
        d = load_masks(MASKS / f"{key}.npz")
        v = d["valid"]
        area = d["masks"][v].mean((-1, -2))
        r.update({"empty_frac": (area == 0).mean(0).round(4).tolist(), "score_neg_frac": (d["score"][v] < 0).mean(0).round(4).tolist(),
                  "area_median": np.median(area, 0).round(5).tolist()})
        if split != "test2":
            r.update(labelled_metrics(traj, eye, d))
        R[key] = r
        print(key, json.dumps({k: v for k, v in r.items() if k != "coverage_by_kp"}), flush=True)
    (OUT / "results.json").write_text(json.dumps(R, indent=1))

    lab = [r for r in R.values() if "coverage" in r]
    by_kp = np.nanmean(np.array([r["coverage_by_kp"] for r in lab]), (0, 1))
    worst = sorted(((k, r) for k, r in R.items() if "coverage" in r), key=lambda kr: kr[1]["coverage_tips"])[:5]
    started = [r for r in R.values() if r["start"] is not None]
    L = ["# v4 step 1: SAM 2 mask QA", "",
         f"{len(R)} videos; SAM 2 started on {len(started)}. Label-based proxies on train + tune ({len(lab)} videos); "
         "test2 label-free only.", "",
         "| Split | Videos | Coverage, body (shaft/wrist/pivot) | Coverage, tips | Swaps | Box IoU median | Empty mask frac | Negative object score frac |",
         "|---|---|---|---|---|---|---|---|"]
    for split in ("train", "tune", "test2"):
        rr = [r for r in R.values() if r["split"] == split and r["start"] is not None]
        if not rr:
            continue
        f = lambda k: f"{np.mean([r[k] for r in rr]):.3f}" if k in rr[0] else "—"
        iou = f"{np.median([x for r in rr for x in r['box_iou_median'] if x is not None]):.2f}" if "box_iou_median" in rr[0] else "—"
        L.append(f"| {split} | {len(rr)} | {f('coverage_body')} | {f('coverage_tips')} | {f('swap')} | {iou} | "
                 f"{np.mean([r['empty_frac'] for r in rr]):.4f} | {np.mean([r['score_neg_frac'] for r in rr]):.4f} |")
    L += ["", "Coverage by keypoint (train + tune): " + ", ".join(f"{n} {c:.3f}" for n, c in zip(KP_NAMES, by_kp)), "",
          "Worst tip coverage (train + tune): " + "; ".join(f"{k} {r['coverage_tips']:.2f} (body {r['coverage_body']:.2f})" for k, r in worst)]
    no_start = [k for k, r in R.items() if r["start"] is None]
    if no_start:
        L += ["", f"No start frame: {', '.join(no_start)}"]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
