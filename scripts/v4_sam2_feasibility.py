"""v4 step 0 (development, tune only): can SAM 2 track the SurgPose instruments causally?

  uv run --group sam python scripts/v4_sam2_feasibility.py [--models tiny small base-plus] [--frames N]

SAM 2.1 (Hugging Face `Sam2VideoModel`) runs in streaming mode, one frame at a time, left eye,
on tune trajectories 18, 19, 20, 23. It's prompted once, at frame 0, with point prompts from our
own v2 U-Net detections (keypoints with confidence >= CONF_MIN; no ground truth); it is never
re-prompted, so 1,001 frames (33 s) test drift. Object 1 = PSM1, object 2 = PSM3.
Prompt modes: `distal` = the detected keypoints only (SAM 2 then segments the painted wrist and
jaws); `shaft` = plus 3 points extrapolated along the shaft (shaft + k (shaft - wrist), k = 1..3)
so the mask covers the whole visible instrument, as SurgPose's boxes do.

SurgPose has no mask ground truth, so the metrics are proxies:
  in_own     fraction of labelled GT keypoints within TOL px of their own instrument's mask
  in_other   fraction of GT keypoints within TOL px of the *other* instrument's mask (swap/merge)
  empty      fraction of frames where an instrument's mask is empty
  box_iou    IoU of the mask's bounding box with SurgPose's per-frame box (provenance undocumented;
             a sanity check, not ground truth)
  ms         per-frame latency (GPU synchronized; preprocessing + model + mask upsampling)
Writes runs/v4_sam2_feas/{results.json, report.md, overlays/}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from surgscene.keypoints import INSTRUMENTS, KP_NAMES  # noqa: E402
from surgscene.stage2 import RAW, load_obs  # noqa: E402

OUT = ROOT / "runs/v4_sam2_feas"
TUNE = [18, 19, 20, 23]
ARMS = list(INSTRUMENTS)  # PSM1 -> obj 1 (SurgPose "obj1"), PSM3 -> obj 2 ("obj2")
CONF_MIN = 0.5
TOL = 5.0
OVERLAY_FRAMES = (0, 250, 500, 750, 1000)
COLORS = [(0, 200, 255), (255, 120, 0)]


def read_frames(traj: int, n: int | None):
    cap = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/left_video.mp4"))
    i = 0
    while n is None or i < n:
        ok, bgr = cap.read()
        if not ok:
            break
        yield i, bgr
        i += 1


def prompts(obs: dict, mode: str, size: tuple[int, int], t: int = 0):
    pts, labels = [], []
    h, w = size
    for arm in ARMS:
        sl = INSTRUMENTS[arm]
        kp, conf = obs["kp"][t, sl], obs["conf"][t, sl]
        keep = conf >= CONF_MIN
        p = kp[keep].tolist()
        if mode == "shaft" and keep[0] and keep[1]:
            d = kp[0] - kp[1]  # wrist -> shaft direction
            for k in (1, 2, 3):
                q = kp[0] + k * d
                if 0 <= q[0] < w and 0 <= q[1] < h:
                    p.append(q.tolist())
        pts.append(p)
        labels.append([1] * len(p))
    return pts, labels


def box_iou(mask: np.ndarray, box) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    a = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
    b = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], float)
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def near_mask(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """True where a point is within TOL px of the mask (distance transform of the complement)."""
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mask.shape
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return dist[yi, xi] <= TOL


def run(model_name: str, mode: str, traj: int, n_frames: int | None, model, processor, save_overlays: bool):
    obs = load_obs(traj, "left")
    boxes = json.loads((RAW / f"{traj:06d}/bbox_left.json").read_text())
    session = processor.init_video_session(inference_device="cuda", dtype=torch.bfloat16)
    pts, labels = prompts(obs, mode, (986, 1400))  # SurgPose native frame size
    own = {a: [] for a in ARMS}
    other = {a: [] for a in ARMS}
    empty = {a: [] for a in ARMS}
    by_kp = {a: [] for a in ARMS}  # per frame, per keypoint: 1 near own mask, 0 not, nan unlabelled
    iou = {a: [] for a in ARMS}
    ms = []
    torch.cuda.reset_peak_memory_stats()
    for t, bgr in read_frames(traj, n_frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(images=rgb, device="cuda", return_tensors="pt")
        if t == 0:
            processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0, obj_ids=[1, 2], input_points=[pts], input_labels=[labels],
                original_size=inputs.original_sizes[0])
        with torch.inference_mode():
            out = model(inference_session=session, frame=inputs.pixel_values[0])
        masks = processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes, binarize=True)[0]
        masks = masks[:, 0].cpu().numpy().astype(bool)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
        ids = list(out.object_ids) if hasattr(out, "object_ids") else [1, 2]
        for k, arm in enumerate(ARMS):
            m = masks[ids.index(k + 1)]
            m_other = masks[ids.index(2 - k)]
            g = obs["gt"][t, INSTRUMENTS[arm]]
            lab = np.isfinite(g).all(1)
            row = np.full(len(g), np.nan)
            if lab.any():
                row[lab] = near_mask(m, g[lab])
            by_kp[arm].append(row)
            g = g[lab]
            empty[arm].append(not m.any())
            if len(g):
                own[arm].append(near_mask(m, g))
                other[arm].append(near_mask(m_other, g))
            b = boxes.get(str(t), {}).get(f"obj{k + 1}")
            if b is not None:
                iou[arm].append(box_iou(m, b))
        if save_overlays and t in OVERLAY_FRAMES:
            vis = bgr.copy()
            for k in range(2):
                m = masks[ids.index(k + 1)]
                vis[m] = (0.5 * vis[m] + 0.5 * np.array(COLORS[k])).astype(np.uint8)
            for x, y in obs["gt"][t][np.isfinite(obs["gt"][t]).all(1)]:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
            (OUT / "overlays").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT / f"overlays/{model_name}_{mode}_{traj}_{t:04d}.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5))
    res = {"frames": len(ms), "ms_p50": float(np.median(ms[10:])), "ms_p99": float(np.percentile(ms[10:], 99)),
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "n_prompt_points": [len(p) for p in pts]}
    for arm in ARMS:
        o = np.concatenate(own[arm]) if own[arm] else np.array([])
        x = np.concatenate(other[arm]) if other[arm] else np.array([])
        # per-frame "lost" = fewer than half of the labelled keypoints near the own mask
        lost = np.array([f.mean() < 0.5 for f in own[arm]])
        res[arm] = {"in_own": float(o.mean()), "in_other": float(x.mean()), "empty": float(np.mean(empty[arm])),
                    "frames_lost": float(lost.mean()), "box_iou_median": float(np.median(iou[arm])) if iou[arm] else None,
                    "first_lost_frame": int(np.argmax(lost)) if lost.any() else None,
                    "in_own_by_kp": dict(zip(KP_NAMES, np.nanmean(np.array(by_kp[arm]), 0).round(3).tolist())),
                    "in_own_last_quarter": float(np.concatenate(own[arm][3 * len(own[arm]) // 4:]).mean())}
    del session
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tiny", "small", "base-plus"])
    ap.add_argument("--trajs", nargs="+", type=int, default=TUNE)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=["distal", "shaft"])
    args = ap.parse_args()
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    OUT.mkdir(parents=True, exist_ok=True)
    R = {"conf_min": CONF_MIN, "tol_px": TOL, "runs": {}}
    for name in args.models:
        repo = f"facebook/sam2.1-hiera-{name}"
        model = Sam2VideoModel.from_pretrained(repo).to("cuda", dtype=torch.bfloat16).eval()
        processor = Sam2VideoProcessor.from_pretrained(repo)
        for mode in args.modes:
            for traj in args.trajs:
                r = run(name, mode, traj, args.frames, model, processor, save_overlays=True)
                R["runs"][f"{name}/{mode}/{traj}"] = r
                short = {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
                         if isinstance(v, dict) else v for k, v in r.items()}
                print(name, mode, traj, json.dumps(short), flush=True)
                (OUT / "results.json").write_text(json.dumps(R, indent=1))
        del model
        torch.cuda.empty_cache()
    (OUT / "results.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
