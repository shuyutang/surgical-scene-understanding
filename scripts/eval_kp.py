"""Phase 2 + 3 evaluation: keypoint detection and structured inference ablation.

  uv run python scripts/eval_kp.py runs/<kp_run>/best.pt

1. Fit per-arm shape models on train-split ground truth.
2. Run the network on dev and test, clean and with synthetic occluders (one random keypoint per
   instrument per frame, covered by a tissue patch). Cache the observations.
3. Tune (conf_min, beta, cauchy_c) on dev only.
4. Report on test: argmax vs map vs discrete, clean vs occluded, with trajectory-clustered CIs.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from surgscene.evaluation import ci
from surgscene.keypoints import INSTRUMENTS, KP_NAMES, load_records
from surgscene.kp_eval import NATIVE, infer, occlusion_plan
from surgscene.models import build_seg_model
from surgscene.shape import ShapeModel
from surgscene.structured import fit_discrete, fit_map


def estimate(O, models, method, hp, plaus_max):
    n = len(O["kp"])
    pred = O["kp"].copy()
    t0 = time.perf_counter()
    if method != "argmax":
        for i in range(n):
            for arm, sl in INSTRUMENTS.items():
                if method == "map":
                    pred[i, sl] = fit_map(models[arm], O["kp"][i, sl], O["conf"][i, sl], O["sigma"][i, sl], **hp)
                else:
                    pred[i, sl] = fit_discrete(models[arm], O["cand"][i, sl], O["cand_conf"][i, sl],
                                               O["sigma"][i, sl], plaus_max=plaus_max[arm], **hp)
    ms = (time.perf_counter() - t0) / (2 * n) * 1e3
    return pred, ms


def errors(pred, O):
    return np.linalg.norm(pred - O["gt"], axis=-1) * NATIVE  # (n, 10) native px


def boot_mean(err, groups, n_boot=2000, seed=0):
    """err: (n,) per-frame values, NaN = excluded. Cluster bootstrap over trajectories."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    reps = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
        reps.append(np.nanmean(err[pick]))
    lo, hi = ci(np.asarray(reps))
    return {"point": float(np.nanmean(err)), "lo": float(lo), "hi": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--occluder-offset", type=float, default=0.0,
                    help="max occluder-center offset as a fraction of its radius (0 = centered on the keypoint)")
    args = ap.parse_args()
    run = Path(args.ckpt).parent
    out_dir = run / ("eval_structured" if args.occluder_offset == 0 else f"eval_structured_off{args.occluder_offset:g}")
    out_dir.mkdir(exist_ok=True)

    # 1. shape models from train GT (both eyes)
    tr = np.array([r["kp"] for r in load_records("train")])
    models, plaus_max = {}, {}
    for arm, sl in INSTRUMENTS.items():
        X = tr[:, sl]
        X = X[np.isfinite(X).all((1, 2))]
        models[arm] = ShapeModel.fit(X)
        pl = np.array([models[arm].plausibility(x) for x in X[:: max(1, len(X) // 3000)]])
        plaus_max[arm] = float(np.percentile(pl, 99))
        print(f"{arm}: {len(X)} train shapes, {len(models[arm].lam)} modes, "
              f"var={np.round(models[arm].lam / models[arm].lam.sum(), 3)}, plaus99={plaus_max[arm]:.1f}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = build_seg_model(cfg["arch"], cfg["encoder"], cfg["num_keypoints"], pretrained=False)
    net.load_state_dict(ck["model"])
    net = net.cuda().eval()

    sets = {}
    for split in ["dev", "test"]:
        recs = load_records(split)
        groups = np.array([r["traj"] for r in recs])
        sets[(split, "clean")] = (infer(net, recs, None, out_dir / f"obs_{split}_clean.npz"), groups)
        sets[(split, "occluded")] = (infer(net, recs, occlusion_plan(len(recs), 7), out_dir / f"obs_{split}_occ.npz",
                                           args.occluder_offset), groups)

    # 3. tune on dev: minimize mean error over clean + occluded dev (all keypoints)
    grid = list(itertools.product([0.05, 0.1, 0.2, 0.3], [0.3, 1.0, 3.0], [2.0, 3.0, 5.0]))
    sub = slice(0, None, 4)  # every 4th dev frame keeps the grid search fast
    best = None
    for conf_min, beta, c in tqdm(grid, desc="tune map (dev)"):
        hp = {"conf_min": conf_min, "beta": beta, "cauchy_c": c}
        e = []
        for cond in ["clean", "occluded"]:
            O = {k: v[sub] for k, v in sets[("dev", cond)][0].items()}
            e.append(np.nanmean(errors(estimate(O, models, "map", hp, plaus_max)[0], O)))
        if best is None or np.mean(e) < best[0]:
            best = (float(np.mean(e)), hp)
    hp = best[1]
    print("selected on dev:", hp, f"(dev mean err {best[0]:.2f}px)")

    # 4. test
    R = {"checkpoint": args.ckpt, "occluder_offset": args.occluder_offset, "hyperparams_selected_on_dev": hp,
         "shape_models": {a: {"modes": len(m.lam), "var_frac": (m.lam / m.lam.sum()).round(4).tolist(),
                              "plaus99_train": plaus_max[a]} for a, m in models.items()},
         "results": {}}
    for cond in ["clean", "occluded"]:
        O, groups = sets[("test", cond)]
        R["results"][cond] = {}
        errs = {}
        for method in ["argmax", "map", "discrete"]:
            pred, ms = estimate(O, models, method, hp, plaus_max)
            E = errors(pred, O)
            errs[method] = E
            viol = []
            for arm, sl in INSTRUMENTS.items():
                viol += [models[arm].plausibility(p) > plaus_max[arm] for p in pred[:, sl] if np.isfinite(p).all()]
            occ = np.where(O["occluded"], E, np.nan)
            vis = np.where(~O["occluded"], E, np.nan)
            finite = E[np.isfinite(E)]
            R["results"][cond][method] = {
                "mean_err_px": boot_mean(np.nanmean(E, 1), groups),
                "median_err_px": float(np.median(finite)),
                "pck5": float((finite <= 5).mean()), "pck10": float((finite <= 10).mean()),
                "pck20": float((finite <= 20).mean()),
                "occluded_kp_mean_err_px": boot_mean(np.nanmean(occ, 1), groups) if O["occluded"].any() else None,
                "visible_kp_mean_err_px": boot_mean(np.nanmean(vis, 1), groups) if O["occluded"].any() else None,
                "implausible_rate": float(np.mean(viol)),
                "per_keypoint_median_px": {f"{arm}.{KP_NAMES[j]}": float(np.nanmedian(E[:, sl][:, j]))
                                           for arm, sl in INSTRUMENTS.items() for j in range(5)},
                "ms_per_instrument": ms,
            }
        # paired differences vs argmax
        for method in ["map", "discrete"]:
            d = np.nanmean(errs[method] - errs["argmax"], 1)
            R["results"][cond][method]["delta_vs_argmax_px"] = boot_mean(d, groups)
            if O["occluded"].any():
                d_occ = np.nanmean(np.where(O["occluded"], errs[method] - errs["argmax"], np.nan), 1)
                R["results"][cond][method]["occluded_delta_vs_argmax_px"] = boot_mean(d_occ, groups)
        # calibration of confidence: does conf predict error <= 10 px?
        conf, E = O["conf"].ravel(), errs["argmax"].ravel()
        bins = np.linspace(0, 1, 11)
        R["results"][cond]["argmax_reliability"] = [
            {"conf_bin": [float(lo), float(hi)], "n": int(m.sum()), "frac_within_10px": float((E[m] <= 10).mean())}
            for lo, hi in zip(bins[:-1], bins[1:]) if (m := (conf >= lo) & (conf < hi)).sum() > 0]

    (out_dir / "results.json").write_text(json.dumps(R, indent=1))
    (out_dir / "report.md").write_text(render(R))
    print(render(R))


def f(v):
    return "n/a" if v is None else f"{v['point']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"


def render(R):
    L = ["# Phase 2–3: keypoints and structured inference (SurgPose test, trajectories 20–33)", "",
         f"Checkpoint `{R['checkpoint']}`; occluder center offset ≤ {R['occluder_offset']:g}×radius. "
         f"Hyperparameters selected on dev: `{R['hyperparams_selected_on_dev']}`. Errors in native pixels "
         "(1400×986). 95% CIs: trajectory-clustered bootstrap.", "",
         "Shape models (train GT): " + "; ".join(f"{a}: {m['modes']} modes, variance fractions {m['var_frac']}"
                                                  for a, m in R["shape_models"].items()), ""]
    for cond, res in R["results"].items():
        L += [f"## {cond.capitalize()} test frames", "",
              "| Method | Mean err [CI] | Δ vs argmax [CI] | Occluded-kp err [CI] | Δ occluded vs argmax [CI] | Visible-kp err | PCK@5/10/20 | Implausible | ms/instr |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in ["argmax", "map", "discrete"]:
            r = res[m]
            L.append(f"| {m} | {f(r['mean_err_px'])} | {f(r.get('delta_vs_argmax_px'))} | {f(r['occluded_kp_mean_err_px'])} | "
                     f"{f(r.get('occluded_delta_vs_argmax_px'))} | {f(r['visible_kp_mean_err_px'])} | "
                     f"{r['pck5']:.3f}/{r['pck10']:.3f}/{r['pck20']:.3f} | {r['implausible_rate']:.3f} | {r['ms_per_instrument']:.2f} |")
        L += ["", "Argmax confidence reliability (fraction of keypoints within 10 px, per confidence bin):", "",
              "| Confidence | n | Within 10 px |", "|---|---|---|"]
        for b in res["argmax_reliability"]:
            L.append(f"| {b['conf_bin'][0]:.1f}–{b['conf_bin'][1]:.1f} | {b['n']} | {b['frac_within_10px']:.3f} |")
        L += ["", "Per-keypoint median error (px): " + ", ".join(
            f"{k} {v:.1f}" for k, v in res["argmax"]["per_keypoint_median_px"].items()), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
