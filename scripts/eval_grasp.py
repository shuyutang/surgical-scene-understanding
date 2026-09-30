"""Scene semantics endpoints (S1-S4) on GraSP; development comparison on the official folds.

  uv run python scripts/eval_grasp.py dev     # fold1 -> fold2: temporal arms vs VLM, VLM smoothing window
  uv run python scripts/eval_grasp.py test    # once, after the pre-registration is committed

Arms (selected configs in runs/grasp_dev/selected.json; VLM tags in configs/grasp_selected.json):
  A  resnet50 + causal MS-TCN        A0  resnet50 + per-frame linear probe
  B  dinov2_b14 + causal MS-TCN      B0  dinov2_b14 + per-frame linear probe
  C  Qwen3-VL-8B, zero-shot and QLoRA fine-tuned, raw and with causal majority smoothing
Temporal arms: mean probabilities of 3 seeds. Metric per case: macro-F1 over classes present in GT
or prediction, and accuracy; unit of analysis = case; paired differences, case-clustered bootstrap
(2,000 replicates) plus the per-case values. A/B/A0/B0 are scored on every frame; comparisons that
involve the VLM use the frames the VLM was run on (every 5th test frame, every 10th dev frame).
Writes runs/grasp_<mode>/{results.json, report.md}.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from surgscene.phase import N_PHASE, N_STEP, CausalMSTCN, Linear, TrainCfg, causal_mode, macro_f1, predict_proba, train

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "splits/grasp_split.json").read_text())
ARMS = {"A": "resnet50_tcn", "A0": "resnet50_linear", "B": "dinov2_b14_tcn", "B0": "dinov2_b14_linear"}
SEEDS = [0, 1, 2]
WINDOWS_S = [0, 30, 60, 120, 300]  # VLM causal smoothing windows, seconds (0 = none)


def labels(case):
    d = np.load(ROOT / f"data/cache/grasp/{case}.npz")
    return d["phase"], d["step"]


def feats(case, backbone):
    return np.load(ROOT / f"data/cache/grasp_feats/{backbone}/{case}.npy").astype(np.float32)


def scores(p, s, pp, ps):
    return {"step_f1": macro_f1(s, ps, N_STEP), "phase_f1": macro_f1(p, pp, N_PHASE),
            "step_acc": float((ps == s).mean()), "phase_acc": float((pp == p).mean())}


def boot_diff(a, b, n_boot=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    reps = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": float(d.mean()), "lo": float(lo), "hi": float(hi), "per_case": [round(float(x), 4) for x in d],
            "n_positive": int((d > 0).sum()), "n": len(d)}


def temporal_predictions(arm_name, train_cases, eval_cases, sel, saved: Path | None):
    """{case: (phase pred, step pred)} from the mean probabilities of 3 seeds."""
    s = sel[arm_name]
    bb = s["backbone"]
    if saved is None:
        X = [feats(c, bb) for c in train_cases]
        allx = np.concatenate(X)
        mu, sd = allx.mean(0), allx.std(0) + 1e-6
        nets = []
        for seed in SEEDS:
            cfg = TrainCfg(model=s["model"], epochs=s["epochs"], class_weight=s["class_weight"], seed=seed)
            net, _ = train([(x - mu) / sd for x in X], [labels(c)[0] for c in train_cases], [labels(c)[1] for c in train_cases], cfg)
            nets.append(net)
    else:
        n = np.load(saved / "norm.npz")
        mu, sd = n["mu"], n["sd"]
        nets = []
        for seed in SEEDS:
            d = feats(eval_cases[0], bb).shape[1]
            net = (CausalMSTCN(d) if s["model"] == "tcn" else Linear(d)).cuda()
            net.load_state_dict(torch.load(saved / f"seed{seed}.pt", map_location="cuda"))
            nets.append(net)
    out = {}
    for c in eval_cases:
        x = (feats(c, bb) - mu) / sd
        pr = [predict_proba(net, x) for net in nets]
        out[c] = (np.mean([p[0] for p in pr], 0).argmax(1), np.mean([p[1] for p in pr], 0).argmax(1))
    return out


def vlm_predictions(tag, cases, window_s, stride):
    out = {}
    k = max(1, round(window_s / stride)) if window_s else 1
    for c in cases:
        d = np.load(ROOT / f"runs/grasp_vlm/{tag}/{c}.npz")
        p, s = d["phase"].copy(), d["step"].copy()
        # unparsed answers stay wrong: map -1 to an extra "invalid" class that no GT frame has
        p[p < 0], s[s < 0] = N_PHASE, N_STEP
        out[c] = (causal_mode(p, k, N_PHASE + 1), causal_mode(s, k, N_STEP + 1), d["frame"])
    return out


def table(rows):
    L = ["| Arm | Frames | Step F1 | Phase F1 | Step acc | Phase acc |", "|---|---|---|---|---|---|"]
    for name, (frames, m) in rows.items():
        L.append(f"| {name} | {frames} | {m['step_f1']:.3f} | {m['phase_f1']:.3f} | {m['step_acc']:.3f} | {m['phase_acc']:.3f} |")
    return L


def mean_scores(per_case):
    return {k: float(np.mean([m[k] for m in per_case.values()])) for k in next(iter(per_case.values()))}


def dev():
    sel = json.loads((ROOT / "runs/grasp_dev/selected.json").read_text())
    tr, ev = SPLIT["dev_folds"]["fold1"], SPLIT["dev_folds"]["fold2"]
    stride = 10
    per = {}
    for arm, name in ARMS.items():
        pred = temporal_predictions(name, tr, ev, sel, None)
        per[f"{arm} (all frames)"] = {c: scores(*labels(c), *pred[c]) for c in ev}
        per[f"{arm} (every {stride}th)"] = {c: scores(*(x[::stride] for x in labels(c)), pred[c][0][::stride],
                                                      pred[c][1][::stride]) for c in ev}
    vlm = {}
    for tag in ("dev_zs_bf16", "dev_zs_4bit", "dev_ft"):
        if not (ROOT / f"runs/grasp_vlm/{tag}/{ev[-1]}.npz").exists():
            continue
        for w in WINDOWS_S:
            pred = vlm_predictions(tag, ev, w, stride)
            per[f"C {tag}, window {w} s"] = vlm[(tag, w)] = {
                c: scores(*(x[pred[c][2]] for x in labels(c)), pred[c][0], pred[c][1]) for c in ev}
    rows = {k: ("all" if "all frames" in k else f"every {stride}th", mean_scores(v)) for k, v in per.items()}
    best = {tag: max(WINDOWS_S, key=lambda w: mean_scores(vlm[(tag, w)])["step_f1"])
            for tag in {t for t, _ in vlm}}
    out = ROOT / "runs/grasp_dev_compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"per_case": per, "best_window_s": best}, indent=1))
    L = ["# GraSP development comparison: train fold1 (4 cases), evaluate fold2 (4 cases)", "",
         "Mean over the 4 held-out cases of per-case macro-F1 / accuracy.", ""] + table(rows) + [
         "", "Best VLM smoothing window (held-out step F1): " + ", ".join(f"{k}: {v} s" for k, v in best.items())]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def test():
    sel = json.loads((ROOT / "runs/grasp_dev/selected.json").read_text())
    cfg = json.loads((ROOT / "configs/grasp_selected.json").read_text())
    cases, stride = list(SPLIT["splits"]["test"]["cases"]), cfg["vlm_test_stride"]
    pred = {arm: temporal_predictions(name, None, cases, sel, ROOT / f"runs/grasp_final/{name}") for arm, name in ARMS.items()}
    S = {arm: {c: scores(*labels(c), *pred[arm][c]) for c in cases} for arm in ARMS}
    Ssub = {arm: {c: scores(*(x[::stride] for x in labels(c)), pred[arm][c][0][::stride], pred[arm][c][1][::stride])
                  for c in cases} for arm in ARMS}
    for key, (tag, w) in {"C_zs": (cfg["vlm_zero_shot_tag"], cfg["vlm_window_s"]["zero_shot"]),
                          "C_ft": (cfg["vlm_finetuned_tag"], cfg["vlm_window_s"]["finetuned"]),
                          "C_ft_raw": (cfg["vlm_finetuned_tag"], 0)}.items():
        vp = vlm_predictions(tag, cases, w, stride)
        Ssub[key] = {c: scores(*(x[vp[c][2]] for x in labels(c)), vp[c][0], vp[c][1]) for c in cases}
    col = lambda D, arm, k: [D[arm][c][k] for c in cases]
    E = {
        "S1": ("Step macro-F1, B − A (DINOv2 vs ResNet-50, both MS-TCN)", boot_diff(col(S, "B", "step_f1"), col(S, "A", "step_f1"))),
        "S2": ("Step macro-F1, A − A0 (temporal model vs per-frame)", boot_diff(col(S, "A", "step_f1"), col(S, "A0", "step_f1"))),
        "S3": ("Step macro-F1, fine-tuned VLM (smoothed) − B, VLM frames",
               boot_diff(col(Ssub, "C_ft", "step_f1"), col(Ssub, "B", "step_f1"))),
        "S4": ("Phase macro-F1, B − A", boot_diff(col(S, "B", "phase_f1"), col(S, "A", "phase_f1"))),
    }
    verdict = {"S1": E["S1"][1]["lo"] > 0, "S2": E["S2"][1]["lo"] > 0, "S3": None, "S4": E["S4"][1]["lo"] > 0}
    out = ROOT / "runs/grasp_test"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"endpoints": {k: v[1] for k, v in E.items()}, "verdicts": verdict,
                                                  "per_case_all": S, "per_case_vlm_frames": Ssub}, indent=1))
    f = lambda d: f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] ({d['n_positive']}/{d['n']} cases > 0)"
    L = ["# GraSP test (5 cases), pre-specified endpoints", "",
         "| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|"]
    for k, (name, d) in E.items():
        crit = "report with CI" if k == "S3" else "LB > 0"
        v = "report" if verdict[k] is None else ("PASS" if verdict[k] else "FAIL")
        L.append(f"| {k} | {name} | {f(d)} | {crit} | {v} |")
    rows = {arm: ("all", mean_scores(S[arm])) for arm in ARMS}
    rows |= {f"{arm} (VLM frames)": (f"every {stride}th", mean_scores(Ssub[arm])) for arm in Ssub}
    L += ["", "Mean over the 5 test cases:", ""] + table(rows)
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    {"dev": dev, "test": test}[ap.parse_args().mode]()
