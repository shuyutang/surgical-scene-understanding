"""v3 step A0: does fixing the focal-loss positives change the keypoint network? Tune only.

  uv run python scripts/eval_a0.py

Compares keypoint runs trained identically except for `focal_pos` (threshold = v2, argmax = fix),
two seeds each, on the tune trajectories (18, 19, 20, 23; left eye, every 5th frame), clean and
with the v2 de-centered occluder protocol. test2 is not touched.
Writes runs/v3_a0/{results.json, report.md}.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_kp import NATIVE, infer, occlusion_plan  # noqa: E402

from surgscene.frontend import TUNE, load_kp_model  # noqa: E402
from surgscene.keypoints import load_records  # noqa: E402

OUT = ROOT / "runs/v3_a0"
RUNS = {  # label -> run name prefix (newest match is used)
    "v2 s0": "kp_unet_r34_decentered_2",
    "v2 s1": "kp_unet_r34_decentered_s1_",
    "A0 s0": "kp_unet_r34_a0_2",
    "A0 s1": "kp_unet_r34_a0_s1_",
}


def finished(run: Path) -> bool:
    """best.pt is rewritten during training; only use runs whose log reached the last epoch."""
    try:
        log = json.loads((run / "log.json").read_text())
        cfg = json.loads((run / "runinfo.json").read_text())["config"]
    except (OSError, ValueError, KeyError):
        return False
    return bool(log) and log[-1]["epoch"] == cfg["epochs"] and (run / "best.pt").exists()


def find_run(prefix: str) -> Path | None:
    c = sorted(p for p in (ROOT / "runs").glob(prefix + "*") if finished(p))
    return c[-1] if c else None


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def metrics(O, traj, sel=None):
    err = np.linalg.norm(O["kp"] - O["gt"], axis=-1) * NATIVE
    ok = np.isfinite(err) if sel is None else np.isfinite(err) & sel
    e, conf, sig = err[ok], O["conf"][ok], O["sigma"][ok] * NATIVE
    tr = np.broadcast_to(traj[:, None], err.shape)[ok]
    bins = [0, 0.2, 0.6, 1.01]
    return {
        "n": int(ok.sum()), "mean": float(e.mean()), "median": float(np.median(e)),
        "pck5": float((e <= 5).mean()), "pck10": float((e <= 10).mean()),
        "per_traj_mean": {int(t): float(e[tr == t].mean()) for t in np.unique(tr)},
        "conf_median": float(np.median(conf)), "sigma_median_px": float(np.median(sig)),
        "spearman_sigma_err": spearman(sig, e), "spearman_conf_err": spearman(conf, e),
        "reliability": [{"conf": f"[{lo},{hi})", "n": int(((conf >= lo) & (conf < hi)).sum()),
                         "within10": float((e[(conf >= lo) & (conf < hi)] <= 10).mean())
                         if ((conf >= lo) & (conf < hi)).any() else None}
                        for lo, hi in zip(bins[:-1], bins[1:])],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [r for r in load_records("dev") + load_records("test") if r["traj"] in TUNE and r["eye"] == "left"]
    recs.sort(key=lambda r: (r["traj"], r["t"]))
    traj = np.array([r["traj"] for r in recs])
    plan = occlusion_plan(len(recs), 7)
    R = {"records": len(recs), "trajectories": TUNE, "runs": {}}
    for label, prefix in RUNS.items():
        run = find_run(prefix)
        if run is None:
            print(f"skip {label}: no finished run {prefix}*")
            continue
        net = load_kp_model(str(run / "best.pt"))
        clean = infer(net, recs, None, OUT / f"obs_{run.name}_clean.npz")
        occ = infer(net, recs, plan, OUT / f"obs_{run.name}_occ.npz", offset=0.6)
        del net
        torch.cuda.empty_cache()
        R["runs"][label] = {"run": run.name, "clean": metrics(clean, traj),
                            "occluded_kp": metrics(occ, traj, occ["occluded"])}
    (OUT / "results.json").write_text(json.dumps(R, indent=1))

    L = [f"# A0: focal-loss positives, v2 (threshold) vs fix (argmax) — tune trajectories {TUNE}, "
         f"left, every 5th frame ({len(recs)} frames)", "",
         "Errors in native px. Occluded = the keypoint covered by the v2 de-centered occluder protocol.", "",
         "| run | clean mean | median | PCK@5 | PCK@10 | occluded-kp mean | PCK@10 | conf median | σ median | ρ(σ, err) | ρ(conf, err) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in R["runs"].items():
        c, o = r["clean"], r["occluded_kp"]
        L.append(f"| {label} | {c['mean']:.2f} | {c['median']:.2f} | {c['pck5']:.3f} | {c['pck10']:.3f} | "
                 f"{o['mean']:.2f} | {o['pck10']:.3f} | {c['conf_median']:.2f} | {c['sigma_median_px']:.1f} | "
                 f"{c['spearman_sigma_err']:+.2f} | {c['spearman_conf_err']:+.2f} |")
    L += ["", "Clean mean error per trajectory:", "", "| run | " + " | ".join(str(t) for t in TUNE) + " |",
          "|---|" + "---|" * len(TUNE)]
    for label, r in R["runs"].items():
        L.append(f"| {label} | " + " | ".join(f"{r['clean']['per_traj_mean'].get(t, float('nan')):.2f}" for t in TUNE) + " |")
    L += ["", "Confidence reliability (clean, fraction within 10 px):", "", "| run | " +
          " | ".join(b["conf"] for b in next(iter(R["runs"].values()))["clean"]["reliability"]) + " |",
          "|---|---|---|---|"]
    for label, r in R["runs"].items():
        L.append(f"| {label} | " + " | ".join(f"{b['within10']:.3f} (n={b['n']})" if b["within10"] is not None else "—"
                                             for b in r["clean"]["reliability"]) + " |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
