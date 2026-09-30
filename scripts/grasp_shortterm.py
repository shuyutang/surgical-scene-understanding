"""Scene semantics, short-term: frozen-feature heads for instrument type and atomic actions per GT
instance (surgscene.grasp_st).

  uv run python scripts/grasp_shortterm.py feats   # crop features for every backbone (train + test instances)
  uv run python scripts/grasp_shortterm.py dev     # official folds, both directions, 3 seeds
  uv run python scripts/grasp_shortterm.py final   # selected configs on all train instances, 3 seeds

Input per instance, frozen backbone (surgscene.backbones, 224 x 224):
  crop        the GT box padded by 10% (+4 px) at the keyframe, squashed to 224 x 224
  frame       the whole-frame feature of the keyframe (data/cache/grasp_feats, the long-term features)
  box         normalised box centre and size (4)
  crop_prev   (variant "+prev" only) the same box one second earlier: a motion cue that a single
              image lacks; reported, because the VLM sees one image
Head: MLP (512 hidden, ReLU, dropout 0.5) with an instrument softmax head (7) and an action
sigmoid head (14); loss CE + BCE; inputs standardised on the training instances; Adam 1e-3,
weight decay 1e-4, batch 256.
dev: held-out metrics every 2 epochs up to 60. Selection rule, per (backbone, variant): the
epoch count and action threshold (0.05-0.5) with the highest mean held-out action macro-F1
(per case, averaged over cases, seeds and directions). A first grid (every 10 epochs to 150,
thresholds 0.2-0.6) selected its lowest threshold and 10-30 epochs everywhere, so it was widened. Writes runs/grasp_st_dev/{results.json,
report.md, selected.json}. final writes runs/grasp_st_final/<arm>/{seed*.pt, norm.npz, config.json}.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch

from surgscene.backbones import BACKBONES, build, preprocess
from surgscene.grasp_st import FEATS, crop_box, design, labels, load_instances, per_case_scores, predict, train

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data/grasp/GraSP_1fps/frames"
VARIANTS = ["single", "prev"]
SEEDS = [0, 1, 2]
EPOCHS, EVAL_EVERY = 60, 2
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


@torch.no_grad()
def feats():
    """Crop features at t and t - 1 s for every train / test instance, every backbone."""
    for bb in BACKBONES:
        net, fwd = build(bb)
        net = net.cuda().eval()
        for split in ("train", "test"):
            path = FEATS / bb / f"{split}.npz"
            if path.exists():
                continue
            inst = load_instances(split)
            out = {"t": [], "prev": []}
            for i in range(0, len(inst), 64):
                chunk = inst[i:i + 64]
                for key, dt in (("t", 0), ("prev", 1)):
                    ims = []
                    for r in chunk:
                        f = r["frame"] - dt
                        p = FRAMES / r["case"] / f"{f:05d}.jpg"
                        p = p if p.exists() else FRAMES / r["case"] / f"{r['frame']:05d}.jpg"
                        x0, y0, x1, y1 = crop_box(r["bbox"])
                        ims.append(preprocess(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)[y0:y1, x0:x1], bb))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out[key].append(fwd(torch.stack(ims).cuda()).float().cpu().numpy())
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, key=np.array([r["key"] for r in inst]), t=np.concatenate(out["t"]).astype(np.float16),
                     prev=np.concatenate(out["prev"]).astype(np.float16))
            print(bb, split, len(inst), flush=True)
        del net
        torch.cuda.empty_cache()


def dev():
    out_dir = ROOT / "runs/grasp_st_dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    inst = {f: load_instances(f) for f in ("fold1", "fold2")}
    rows = []
    for bb, variant in product(BACKBONES, VARIANTS):
        X = {f: design(inst[f], "train", bb, variant) for f in inst}
        for (tr, va), seed in product([("fold1", "fold2"), ("fold2", "fold1")], SEEDS):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xv = (X[va] - mu) / sd

            def ev(net):
                pi, pa = predict(net, Xv)
                return {t: per_case_scores(inst[va], pi, pa, t) for t in THRESHOLDS}
            _, hist = train((X[tr] - mu) / sd, *labels(inst[tr]), EPOCHS, seed, ev, eval_every=EVAL_EVERY)
            for ep, byt in hist.items():
                for t, m in byt.items():
                    rows.append({"backbone": bb, "variant": variant, "seed": seed, "train": tr, "epoch": ep,
                                 "threshold": t} | m)
        print(bb, variant, "done", flush=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=1))
    L = ["# GraSP short-term development (official folds, both directions, 3 seeds)", "",
         "Held-out per-case macro-F1 averaged over cases, seeds and directions; GT instances given.", "",
         "| Backbone | Variant | Epochs | Threshold | Action F1 | Action mAP | Instrument F1 |", "|---|---|---|---|---|---|---|"]
    selected = {}
    for bb, variant in product(BACKBONES, VARIANTS):
        cands = {}
        for ep, t in product(range(EVAL_EVERY, EPOCHS + 1, EVAL_EVERY), THRESHOLDS):
            r = [x for x in rows if (x["backbone"], x["variant"], x["epoch"], x["threshold"]) == (bb, variant, ep, t)]
            cands[(ep, t)] = {k: float(np.mean([x[k] for x in r])) for k in ("action_f1", "action_map", "instr_f1")}
        (ep, t), m = max(cands.items(), key=lambda kv: kv[1]["action_f1"])
        selected[f"{bb}_{variant}"] = {"backbone": bb, "variant": variant, "epochs": ep, "threshold": t, "dev": m}
        L.append(f"| {bb} | {variant} | {ep} | {t} | {m['action_f1']:.3f} | {m['action_map']:.3f} | {m['instr_f1']:.3f} |")
    (out_dir / "selected.json").write_text(json.dumps(selected, indent=1))
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def final():
    sel = json.loads((ROOT / "runs/grasp_st_dev/selected.json").read_text())
    inst = load_instances("train")
    for name, s in sel.items():
        out = ROOT / f"runs/grasp_st_final/{name}"
        out.mkdir(parents=True, exist_ok=True)
        X = design(inst, "train", s["backbone"], s["variant"])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        np.savez(out / "norm.npz", mu=mu, sd=sd)
        for seed in SEEDS:
            net, _ = train((X - mu) / sd, *labels(inst), s["epochs"], seed)
            torch.save(net.state_dict(), out / f"seed{seed}.pt")
        (out / "config.json").write_text(json.dumps(s | {"d_in": int(X.shape[1])}, indent=1))
        print("trained", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["feats", "dev", "final"])
    {"feats": feats, "dev": dev, "final": final}[ap.parse_args().mode]()
