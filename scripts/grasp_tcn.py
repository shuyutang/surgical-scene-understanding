"""Scene semantics: temporal models (causal MS-TCN) and per-frame linear probes on frozen features.

  uv run python scripts/grasp_tcn.py dev      # official folds: train fold1 -> eval fold2 and back
  uv run python scripts/grasp_tcn.py final    # train the selected configs on all 8 train cases

dev: grid over backbone {resnet50, dinov2_b14} x model {tcn, linear} x class weighting
{none, sqrt_inv} x seeds {0, 1, 2}, both fold directions; the held-out metric is recorded every
25 epochs (up to 300). Selection rule (same for every backbone): per (backbone, model), the class weighting
and epoch count with the highest mean held-out step macro-F1 (per-case, averaged over cases,
seeds and directions). Writes runs/grasp_dev/{results.json, report.md, selected.json}.
final: 3 seeds per selected config on all train cases; the test prediction is the mean of their
probabilities. Writes runs/grasp_final/<backbone>_<model>/{seed*.pt, norm.npz, config.json}.
Features are standardized with mean / std over the training cases of each run.
"""

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

from surgscene.phase import N_PHASE, N_STEP, TrainCfg, macro_f1, predict_proba, train

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "splits/grasp_split.json").read_text())
BACKBONES = ["resnet50", "dinov2_b14"]
MODELS = ["tcn", "linear"]
WEIGHTS = ["none", "sqrt_inv"]
SEEDS = [0, 1, 2]
EPOCHS = 300
EVAL_EVERY = 25


def load(case, backbone):
    lab = np.load(ROOT / f"data/cache/grasp/{case}.npz")
    x = np.load(ROOT / f"data/cache/grasp_feats/{backbone}/{case}.npy").astype(np.float32)
    return x, lab["phase"], lab["step"]


def norm_stats(X):
    allx = np.concatenate(X)
    return allx.mean(0), allx.std(0) + 1e-6


def evaluate(net, X, P, S):
    out = []
    for x, p, s in zip(X, P, S):
        pp, ps = predict_proba(net, x)
        out.append({"phase_f1": macro_f1(p, pp.argmax(1), N_PHASE), "step_f1": macro_f1(s, ps.argmax(1), N_STEP),
                    "phase_acc": float((pp.argmax(1) == p).mean()), "step_acc": float((ps.argmax(1) == s).mean())})
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def dev():
    out_dir = ROOT / "runs/grasp_dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = SPLIT["dev_folds"]
    rows = []
    for bb in BACKBONES:
        data = {c: load(c, bb) for c in folds["fold1"] + folds["fold2"]}
        for (tr, va), model, cw, seed in product([("fold1", "fold2"), ("fold2", "fold1")], MODELS, WEIGHTS, SEEDS):
            Xtr = [data[c][0] for c in folds[tr]]
            mu, sd = norm_stats(Xtr)
            Z = lambda cs: [(data[c][0] - mu) / sd for c in cs]
            Xv, Pv, Sv = Z(folds[va]), [data[c][1] for c in folds[va]], [data[c][2] for c in folds[va]]
            cfg = TrainCfg(model=model, epochs=EPOCHS, class_weight=cw, seed=seed)
            _, hist = train(Z(folds[tr]), [data[c][1] for c in folds[tr]], [data[c][2] for c in folds[tr]], cfg,
                            eval_fn=lambda net: evaluate(net, Xv, Pv, Sv), eval_every=EVAL_EVERY)
            for ep, m in hist.items():
                rows.append({"backbone": bb, "model": model, "class_weight": cw, "seed": seed, "train": tr,
                             "epoch": ep} | m)
            best = max(hist.items(), key=lambda kv: kv[1]["step_f1"])
            print(bb, model, cw, seed, f"{tr}->{va}", f"best ep {best[0]} step F1 {best[1]['step_f1']:.3f}", flush=True)
        (out_dir / "results.json").write_text(json.dumps(rows, indent=1))
    selected, L = {}, ["# GraSP development (official folds, both directions, 3 seeds)", "",
                       "Held-out per-case macro-F1 averaged over cases, seeds and fold directions.", "",
                       "| Backbone | Model | Class weight | Best epoch | Step F1 | Phase F1 | Step acc | Phase acc |",
                       "|---|---|---|---|---|---|---|---|"]
    for bb, model in product(BACKBONES, MODELS):
        cands = {}
        for cw, ep in product(WEIGHTS, range(EVAL_EVERY, EPOCHS + 1, EVAL_EVERY)):
            r = [x for x in rows if (x["backbone"], x["model"], x["class_weight"], x["epoch"]) == (bb, model, cw, ep)]
            cands[(cw, ep)] = {k: float(np.mean([x[k] for x in r])) for k in ("step_f1", "phase_f1", "step_acc", "phase_acc")}
        for cw in WEIGHTS:
            ep, m = max(((ep, m) for (c, ep), m in cands.items() if c == cw), key=lambda t: t[1]["step_f1"])
            L.append(f"| {bb} | {model} | {cw} | {ep} | {m['step_f1']:.3f} | {m['phase_f1']:.3f} | {m['step_acc']:.3f} | "
                     f"{m['phase_acc']:.3f} |")
        (cw, ep), m = max(cands.items(), key=lambda kv: kv[1]["step_f1"])
        selected[f"{bb}_{model}"] = {"backbone": bb, "model": model, "class_weight": cw, "epochs": ep, "dev": m}
    L += ["", "Selected (highest mean held-out step macro-F1): " +
          "; ".join(f"{k}: {v['class_weight']}, {v['epochs']} epochs" for k, v in selected.items())]
    (out_dir / "selected.json").write_text(json.dumps(selected, indent=1))
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def final():
    sel = json.loads((ROOT / "runs/grasp_dev/selected.json").read_text())
    cases = list(SPLIT["splits"]["train"]["cases"])
    for name, s in sel.items():
        out = ROOT / f"runs/grasp_final/{name}"
        out.mkdir(parents=True, exist_ok=True)
        data = [load(c, s["backbone"]) for c in cases]
        mu, sd = norm_stats([d[0] for d in data])
        np.savez(out / "norm.npz", mu=mu, sd=sd)
        for seed in SEEDS:
            cfg = TrainCfg(model=s["model"], epochs=s["epochs"], class_weight=s["class_weight"], seed=seed)
            net, _ = train([(d[0] - mu) / sd for d in data], [d[1] for d in data], [d[2] for d in data], cfg)
            torch.save(net.state_dict(), out / f"seed{seed}.pt")
            (out / "config.json").write_text(json.dumps(asdict(cfg) | {"backbone": s["backbone"]}, indent=1))
        print("trained", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "final"])
    {"dev": dev, "final": final}[ap.parse_args().mode]()
