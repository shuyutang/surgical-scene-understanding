"""Scene semantics, short-term endpoints (ST1-ST4) on GraSP keyframes; development comparison.

  uv run python scripts/eval_grasp_shortterm.py dev    # fold1 -> fold2: frozen-feature heads vs VLM
  uv run python scripts/eval_grasp_shortterm.py test   # once, after the pre-registration is committed

Arms (instrument type + atomic actions per GT instance, surgscene.grasp_st):
  <backbone>_single / <backbone>_prev   frozen-feature MLP heads (scripts/grasp_shortterm.py), mean
                                        probabilities of 3 seeds, the dev-selected action threshold
  VLM zero-shot / fine-tuned            Qwen3-VL-8B-Instruct, box drawn on the frame (scripts/grasp_vlm.py --task instances)
Per case: instrument macro-F1, action macro-F1 (and action mAP for the heads). Unit = case; paired
differences, case-level bootstrap (2,000 replicates), per-case values.
The test configuration (reference arm, VLM tags) is read from configs/grasp_st_selected.json.
Writes runs/grasp_st_<mode>_compare/{results.json, report.md}.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from surgscene.grasp_st import (
    N_INSTR,
    InstanceMLP,
    decide,
    design,
    labels,
    load_instances,
    mean_ap,
    multilabel_macro_f1,
    predict,
    single_label_macro_f1,
    train,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2]


def boot_diff(a, b, n_boot=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    reps = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": float(d.mean()), "lo": float(lo), "hi": float(hi), "per_case": [round(float(x), 4) for x in d],
            "n_positive": int((d > 0).sum()), "n": len(d)}


def case_scores(inst, ins_pred, act_pred, act_score=None):
    """{case: metrics} from per-instance predictions (instrument index, action bool matrix)."""
    yi, ya = labels(inst)
    cases = np.array([r["case"] for r in inst])
    out = {}
    for c in sorted(set(cases)):
        m = cases == c
        out[c] = {"instr_f1": single_label_macro_f1(yi[m], ins_pred[m], N_INSTR),
                  "action_f1": multilabel_macro_f1(ya[m], act_pred[m])}
        if act_score is not None:
            out[c]["action_map"] = mean_ap(ya[m], act_score[m])
    return out


def head_predictions(s, train_inst, eval_inst, source_eval, saved: Path | None):
    """Mean probabilities of 3 seeds -> (instrument index, action decisions, action scores)."""
    Xe = design(eval_inst, source_eval, s["backbone"], s["variant"])
    if saved is None:
        X = design(train_inst, "train", s["backbone"], s["variant"])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        nets = [train((X - mu) / sd, *labels(train_inst), s["epochs"], seed)[0] for seed in SEEDS]
    else:
        n = np.load(saved / "norm.npz")
        mu, sd = n["mu"], n["sd"]
        nets = []
        for seed in SEEDS:
            net = InstanceMLP(Xe.shape[1]).cuda()
            net.load_state_dict(torch.load(saved / f"seed{seed}.pt", map_location="cuda"))
            nets.append(net)
    pr = [predict(net, ((Xe - mu) / sd).astype(np.float32)) for net in nets]
    pi, pa = np.mean([p[0] for p in pr], 0), np.mean([p[1] for p in pr], 0)
    return pi.argmax(1), decide(pa, s["threshold"]), pa


def vlm_predictions(tag, split, inst):
    z = np.load(ROOT / f"runs/grasp_vlm/{tag}/instances_{split}.npz")
    row = {k: i for i, k in enumerate(z["key"])}
    idx = np.array([row[r["key"]] for r in inst])
    return z["instrument"][idx], z["actions"][idx]


def mean_of(per_case):
    ks = next(iter(per_case.values())).keys()
    return {k: float(np.mean([m[k] for m in per_case.values()])) for k in ks}


def table(per):
    L = ["| Arm | Instrument F1 | Action F1 | Action mAP |", "|---|---|---|---|"]
    for name, pc in per.items():
        m = mean_of(pc)
        mAP = f"{m['action_map']:.3f}" if "action_map" in m else "—"
        L.append(f"| {name} | {m['instr_f1']:.3f} | {m['action_f1']:.3f} | {mAP} |")
    return L


def dev():
    sel = json.loads((ROOT / "runs/grasp_st_dev/selected.json").read_text())
    tr, ev = load_instances("fold1"), load_instances("fold2")
    per = {}
    for name, s in sel.items():
        ip, ap_, sc = head_predictions(s, tr, ev, "train", None)
        per[name] = case_scores(ev, ip, ap_, sc)
    for tag in ("st_dev_zs", "st_dev_ft"):
        if (ROOT / f"runs/grasp_vlm/{tag}/instances_fold2.npz").exists():
            per[f"VLM {tag}"] = case_scores(ev, *vlm_predictions(tag, "fold2", ev))
    out = ROOT / "runs/grasp_st_dev_compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(per, indent=1))
    L = ["# GraSP short-term development comparison: train fold1 (4 cases), evaluate fold2 (4 cases)", "",
         "GT instances given; mean over the 4 held-out cases of per-case macro-F1.", ""] + table(per)
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def test():
    sel = json.loads((ROOT / "runs/grasp_st_dev/selected.json").read_text())
    cfg = json.loads((ROOT / "configs/grasp_st_selected.json").read_text())
    ev = load_instances("test")
    per = {}
    for name, s in sel.items():
        ip, ap_, sc = head_predictions(s, None, ev, "test", ROOT / f"runs/grasp_st_final/{name}")
        per[name] = case_scores(ev, ip, ap_, sc)
    for key in ("vlm_zero_shot_tag", "vlm_finetuned_tag"):
        per[f"VLM {cfg[key]}"] = case_scores(ev, *vlm_predictions(cfg[key], "test", ev))
    cases = sorted(per[cfg["reference_arm"]])
    col = lambda arm, k: [per[arm][c][k] for c in cases]
    ref, vlm, bb = cfg["reference_arm"], f"VLM {cfg['vlm_finetuned_tag']}", cfg["reference_arm"].rsplit("_", 1)[0]
    E = {
        "ST1": (f"Action macro-F1, fine-tuned VLM − {ref}", boot_diff(col(vlm, "action_f1"), col(ref, "action_f1"))),
        "ST2": (f"Instrument macro-F1, fine-tuned VLM − {ref}", boot_diff(col(vlm, "instr_f1"), col(ref, "instr_f1"))),
        "ST3": ("Action macro-F1, dinov2_b14_single − resnet50_single",
                boot_diff(col("dinov2_b14_single", "action_f1"), col("resnet50_single", "action_f1"))),
        "ST4": (f"Action macro-F1, {bb}_prev − {bb}_single (motion cue)",
                boot_diff(col(f"{bb}_prev", "action_f1"), col(f"{bb}_single", "action_f1"))),
    }
    verdict = {k: v[1]["lo"] > 0 for k, v in E.items()}
    out = ROOT / "runs/grasp_st_test"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"endpoints": {k: v[1] for k, v in E.items()}, "verdicts": verdict,
                                                  "per_case": per}, indent=1))
    f = lambda d: f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] ({d['n_positive']}/{d['n']} cases > 0)"
    L = ["# GraSP short-term test (5 cases, GT instances given), pre-specified endpoints", "",
         "| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|"]
    for k, (name, d) in E.items():
        L.append(f"| {k} | {name} | {f(d)} | LB > 0 | {'PASS' if verdict[k] else 'FAIL'} |")
    L += ["", "Mean over the 5 test cases:", ""] + table(per)
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    {"dev": dev, "test": test}[ap.parse_args().mode]()
