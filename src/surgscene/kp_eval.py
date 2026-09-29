"""Keypoint evaluation helpers shared by the Phase 2-3, A0 and v3 Phase A scripts."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .keypoints import KeypointDataset
from .structured import observe

NATIVE = 2.0


OCC_RADIUS = 18.0  # cache px (36 px native)


def occlusion_plan(n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [[int(rng.integers(0, 5)), 5 + int(rng.integers(0, 5))] for _ in range(n)]


@torch.no_grad()
def infer(model, records, occlude, cache: Path, offset: float = 0.0):
    if cache.exists():
        return dict(np.load(cache))
    ds = KeypointDataset(records, occlude=occlude, occluder_radius=OCC_RADIUS, seed=1, occluder_offset=offset)
    out = {k: [] for k in ["kp", "conf", "sigma", "cand", "cand_conf", "gt"]}
    for x, _, kp, _ in tqdm(DataLoader(ds, 16, num_workers=8), desc=cache.stem):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm = model(x.cuda()).float().sigmoid()
        o = observe(hm)
        for k in ["kp", "conf", "sigma", "cand", "cand_conf"]:
            out[k].append(getattr(o, k))
        out["gt"].append(kp.numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["occluded"] = np.zeros(out["conf"].shape, bool)
    if occlude is not None:
        for i, js in enumerate(occlude):
            out["occluded"][i, js] = True
    np.savez_compressed(cache, **out)
    return out


def finished(run: Path) -> bool:
    """best.pt is rewritten during training; only use runs whose log reached the last epoch."""
    try:
        log = json.loads((run / "log.json").read_text())
        cfg = json.loads((run / "runinfo.json").read_text())["config"]
    except (OSError, ValueError, KeyError):
        return False
    return bool(log) and log[-1]["epoch"] == cfg["epochs"] and (run / "best.pt").exists()


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
