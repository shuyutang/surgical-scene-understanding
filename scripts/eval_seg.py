"""Evaluate a segmentation checkpoint against docs/validation_plan.md.

  uv run python scripts/eval_seg.py runs/<run>/best.pt --split dev     # tuning / sanity
  uv run python scripts/eval_seg.py runs/<run>/best.pt --split test    # final, frozen

Writes <run>/eval_<split>/{stats.npz, results.json, report.md}.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from surgscene.data import (CACHE, SegDataset, endovis18_frames, endovis18_group, sisvse_frames,
                            sisvse_group)
from surgscene.evaluation import ci, cluster_bootstrap, frame_stats, quality_proxies, stack, summarize
from surgscene.models import build_seg_model
from surgscene.taxonomy import (ANATOMY, ENDOVIS18_TO_HARMONIZED, HARMONIZED, SISVSE_CLASSES,
                                SISVSE_TO_HARMONIZED)

H = {n: i for i, n in enumerate(HARMONIZED)}
S = {n: i for i, n in enumerate(SISVSE_CLASSES)}
BINARY = (np.arange(256) > 0).astype(np.uint8)  # harmonized {tip,wrist,shaft} -> instrument


def _stats_job(args):
    pred32, conf, gt, dataset = args
    out = {}
    if dataset == "sisvse":
        out["full"] = frame_stats(pred32, gt, 32, conf=conf)
        gt_h = SISVSE_TO_HARMONIZED[gt]
    else:
        gt_h = ENDOVIS18_TO_HARMONIZED[gt]
    pred_h = SISVSE_TO_HARMONIZED[pred32]
    out["harm"] = frame_stats(pred_h, gt_h, 4, classes=[1, 2, 3])
    out["inst"] = frame_stats(BINARY[pred_h], BINARY[gt_h], 2, classes=[1])
    return out


@torch.no_grad()
def predict(model, dataset, frames, temperature=1.0):
    dl = torch.utils.data.DataLoader(SegDataset(dataset, frames), 16, num_workers=8)
    for x, y, idx in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda().to(memory_format=torch.channels_last)).float()
        prob = (logits / temperature).softmax(1)
        conf, pred = prob.max(1)
        for p, c, g in zip(pred.cpu().numpy().astype(np.uint8), conf.cpu().numpy(), y.numpy().astype(np.uint8)):
            yield p, c, g


def run_set(model, dataset, frames, temperature=1.0, workers=16):
    jobs = ((p, c, g, dataset) for p, c, g in predict(model, dataset, frames, temperature))
    with ProcessPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(_stats_job, jobs, chunksize=4), total=len(frames), desc=dataset))
    return {k: stack([r[k] for r in res]) for k in res[0]}


def proxies(dataset, frames):
    return [quality_proxies(cv2.imread(str(CACHE / dataset / "images" / f"{f}.png"))) for f in frames]


def boot(St, groups, fn, n=2000):
    point = fn(summarize(St))
    lo, hi = ci(cluster_bootstrap(St, groups, fn, n_boot=n))
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_seg_model(cfg["arch"], cfg["encoder"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval().to(memory_format=torch.channels_last)
    t_file = Path(args.ckpt).parent / "temperature.json"
    temperature = json.loads(t_file.read_text())["temperature"] if t_file.exists() else 1.0
    out_dir = Path(args.ckpt).parent / f"eval_{args.split}"
    out_dir.mkdir(exist_ok=True)

    frames = sisvse_frames(args.split)
    groups = [sisvse_group(f) for f in frames]
    internal = run_set(model, "sisvse", frames, temperature)
    # The external set is used whole in both modes: it is never trained or tuned on.
    ev_frames = endovis18_frames()
    ev_groups = [endovis18_group(f) for f in ev_frames]
    external = run_set(model, "endovis18", ev_frames, temperature)

    # subgroup cut-points from the train split (pre-specified)
    train_px = proxies("sisvse", sisvse_frames("train"))
    q = {k: np.percentile([p[k] for p in train_px], [25, 75]) for k in train_px[0]}
    px = proxies("sisvse", frames)
    subgroups = {
        "blurry": [p["sharpness"] < q["sharpness"][0] for p in px],
        "glare": [p["specular"] > q["specular"][1] for p in px],
        "hazy": [p["haze"] > q["haze"][1] for p in px],
        "dark": [p["brightness"] < q["brightness"][0] for p in px],
    }
    np.savez_compressed(out_dir / "stats.npz", **{f"internal_{k}_{f}": v for k, d in internal.items() for f, v in d.items()},
                        **{f"external_{k}_{f}": v for k, d in external.items() for f, v in d.items()})

    n = args.n_boot
    R = {"checkpoint": str(args.ckpt), "epoch": ck["epoch"], "split": args.split, "temperature": temperature,
         "n_frames": len(frames), "n_groups": len(set(groups)),
         "external_n_frames": len(ev_frames), "external_n_groups": len(set(ev_groups))}
    I, E = internal, external
    crit = {
        "P1": ("Instrument Dice (internal)", boot(I["inst"], groups, lambda r: r["dice"][1], n), ">=", 0.85),
        "P2": ("Tip Dice (internal)", boot(I["harm"], groups, lambda r: r["dice"][H["tip"]], n), ">=", 0.70),
        "S1": ("Tip boundary F 3px (internal)", boot(I["harm"], groups, lambda r: r["bf"][H["tip"]], n), ">=", 0.60),
        "S2a": ("Liver Dice (internal)", boot(I["full"], groups, lambda r: r["dice"][S["Liver"]], n), ">=", 0.60),
        "S2b": ("Stomach Dice (internal)", boot(I["full"], groups, lambda r: r["dice"][S["Stomach"]], n), ">=", 0.60),
        "S3": ("Pixel ECE, 32 classes (internal)", boot(I["full"], groups, lambda r: r["ece"], n), "<=", 0.05),
        "S4": ("Instrument Dice zero-shot (external)", boot(E["inst"], ev_groups, lambda r: r["dice"][1], n), ">=", 0.70),
    }
    R["criteria"] = {}
    for k, (name, v, op, thr) in crit.items():
        passed = v["lo"] >= thr if op == ">=" else v["hi"] <= thr
        R["criteria"][k] = {"name": name, **v, "criterion": f"{'lower' if op == '>=' else 'upper'} bound {op} {thr}",
                            "pass": bool(passed)}

    # exploratory: per-class tables
    full = summarize(I["full"])
    R["internal_per_class"] = {
        SISVSE_CLASSES[c]: {"dice": float(full["dice"][c]), "iou": float(full["iou"][c]), "bf": float(full["bf"][c]),
                            "hd95_px": float(full["hd95"][c]), "gt_pixels": int(full["gt_pixels"][c])}
        for c in range(32) if full["gt_pixels"][c] > 0}
    R["internal_miou_present"] = float(np.nanmean([v["iou"] for v in R["internal_per_class"].values()]))
    R["anatomy_ci"] = {a: boot(I["full"], groups, lambda r, c=S[a]: r["dice"][c], n) for a in ANATOMY}
    R["harmonized"] = {}
    for setname, St, g in [("internal", I["harm"], groups), ("external", E["harm"], ev_groups)]:
        R["harmonized"][setname] = {p: boot(St, g, lambda r, c=H[p]: r["dice"][c], n) for p in ["tip", "wrist", "shaft"]}

    # robustness subgroups (R1): instrument Dice within 0.10 of overall
    overall = crit["P1"][1]["point"]
    R["subgroups"] = {}
    for name, mask in subgroups.items():
        rows = np.flatnonzero(mask)
        sub = {k: v[rows] for k, v in I["inst"].items()}
        v = boot(sub, [groups[i] for i in rows], lambda r: r["dice"][1], n)
        R["subgroups"][name] = {"n_frames": int(len(rows)), "n_patients": len({groups[i] for i in rows}), **v,
                                "delta": v["point"] - overall, "pass_R1": bool(v["point"] >= overall - 0.10)}
    R["subgroup_cutpoints_train"] = {k: v.tolist() for k, v in q.items()}

    (out_dir / "results.json").write_text(json.dumps(R, indent=1))
    (out_dir / "report.md").write_text(render(R))
    print(render(R))


def fmt(v):
    return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


def render(R):
    L = [f"# Phase 1 segmentation, {R['split']} evaluation", "",
         f"Checkpoint `{R['checkpoint']}` (epoch {R['epoch']}, softmax temperature {R['temperature']:.3f} fit on dev). Internal: {R['n_frames']} frames / "
         f"{R['n_groups']} patients. External (EndoVis18, zero-shot): {R['external_n_frames']} frames / "
         f"{R['external_n_groups']} sequences. 95% CIs: patient/sequence-clustered bootstrap.", "",
         "## Pre-specified criteria", "", "| ID | Endpoint | Estimate [95% CI] | Criterion | Result |", "|---|---|---|---|---|"]
    for k, c in R["criteria"].items():
        L.append(f"| {k} | {c['name']} | {fmt(c)} | {c['criterion']} | {'PASS' if c['pass'] else 'FAIL'} |")
    L += ["", "## Robustness subgroups (R1: within 0.10 of overall instrument Dice)", "",
          "| Subgroup | Frames | Patients | Instrument Dice [95% CI] | Δ vs overall | R1 |", "|---|---|---|---|---|---|"]
    for k, s in R["subgroups"].items():
        L.append(f"| {k} | {s['n_frames']} | {s['n_patients']} | {fmt(s)} | {s['delta']:+.3f} | {'PASS' if s['pass_R1'] else 'FAIL'} |")
    L += ["", "## Harmonized parts (exploratory)", "", "| Set | Tip | Wrist | Shaft |", "|---|---|---|---|"]
    for s, d in R["harmonized"].items():
        L.append(f"| {s} | {fmt(d['tip'])} | {fmt(d['wrist'])} | {fmt(d['shaft'])} |")
    L += ["", "## Anatomy Dice (internal)", "", "| Structure | Dice [95% CI] |", "|---|---|"]
    for a, v in R["anatomy_ci"].items():
        L.append(f"| {a} | {fmt(v)} |")
    L += ["", f"## All classes (internal, pooled; mIoU over present classes = {R['internal_miou_present']:.3f})", "",
          "| Class | Dice | IoU | Boundary F | HD95 (px @640×512) | GT pixels |", "|---|---|---|---|---|---|"]
    for c, v in R["internal_per_class"].items():
        L.append(f"| {c} | {v['dice']:.3f} | {v['iou']:.3f} | {v['bf']:.3f} | {v['hd95_px']:.1f} | {v['gt_pixels']:,} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
