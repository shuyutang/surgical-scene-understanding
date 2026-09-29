"""v4 development on tune (18, 19, 20, 23): fusion variants with the instrument-type library.

  uv run python scripts/v4_dev_fusion.py [--variants ...]

Metric = the v3 C1 quantity on tune: hybrid 3D tip-midpoint error, every 5th frame, v2 filling the
warm-up, mean per trajectory (and per arm-trajectory). Development only: nothing here is a test.
Type sequences:
  oracle   the GT type (GT pivot->tip-midpoint distance > 15 mm), from frame 0 (upper bound)
  <name>   a causal classifier's per-frame output, from runs/v4_dev/type_seq_<name>.json
Writes runs/v4_dev/fusion_<variant>.json.
"""

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_fusion import hybrid, prepare  # noqa: E402

from surgscene.fusion import (ARMS, FusionParams, ToolGeometry, classify_type, fuse_arm,  # noqa: E402
                              length_constrained_depth, observations, pixel_std, triangulate_with_depth_prior)
from surgscene.sam2_track import load_masks  # noqa: E402

MASKS = ROOT / "data/cache/surgpose_masks"
GATE_PX = 10.0  # tip-midpoint detection farther than this outside its instrument mask: no length constraint
from surgscene.stage2 import TIP, load_selected, shape_models  # noqa: E402
from surgscene.temporal import KFParams  # noqa: E402

TUNE = [18, 19, 20, 23]
OUT = ROOT / "runs/v4_dev"
V3 = json.loads((ROOT / "configs/v3_fusion_selected.json").read_text())


def load_prepared(traj):
    path = OUT / f"prepared_{traj}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    sel = load_selected()
    models, _ = shape_models()
    D = prepare(traj, models, sel["gated"], KFParams(**sel["kf"]))
    path.write_bytes(pickle.dumps(D))
    return D


def oracle_type(D, arm):
    a, b = TIP[arm]
    G = D["G"]
    v = (G[:, a] + G[:, b]) / 2 - G[:, ARMS[arm]][:, 2]
    return "long" if np.nanmedian(np.linalg.norm(v, axis=1)) > 15.0 else "standard"


def run(D, p: FusionParams, types: dict | None):
    """types: {arm: (T,) type names or None} or None (v3 behaviour)."""
    sL = pixel_std(D["O"]["sigma"], D["O"]["conf"], p)
    sR = pixel_std(D["OR"]["sigma"], D["OR"]["conf"], p)
    tracks = {}
    for arm, sl in ARMS.items():
        Z, S = observations(D["gL"][:, sl], D["gR"][:, sl], sL[:, sl], sR[:, sl])
        lib = {t: ToolGeometry.from_library(arm, t) for t in ("standard", "long")} if types else None
        tracks[arm] = fuse_arm(D["rig"], D["kin"][arm], ToolGeometry.load(arm), Z, S, p,
                               type_priors=lib, type_seq=types[arm] if types else None)
    return tracks


LIB = json.loads((ROOT / "configs/v4_tool_library.json").read_text())


def causal_types(D, tracks):
    f = D["rig"].left.K[0, 0]
    out = {}
    for arm, (a, b) in TIP.items():
        piv = tracks[arm].X[:, 0]
        uP = D["kfL"][:, ARMS[arm]][:, 2]
        uM = (D["kfL"][:, a] + D["kfL"][:, b]) / 2
        out[arm] = classify_type(piv, uP, uM, f)
    return out


def mask_gate(traj, D):
    """{arm: (T,) bool} True where both left tip detections are within GATE_PX of their instrument's
    mask (not the midpoint: with open jaws it sits in the gap between them, outside the mask)."""
    import cv2
    d = load_masks(MASKS / f"{traj:06d}_left.npz")
    M, valid = d["masks"], d["valid"]
    out = {}
    for k, (arm, (a, b)) in enumerate(TIP.items()):
        u = D["kfL"][:, [a, b]]
        ok = np.ones(len(u), bool)
        for fr in D["f5"]:
            if not valid[fr] or not np.isfinite(u[fr]).all():
                continue
            dist = cv2.distanceTransform((~M[fr, k]).astype(np.uint8), cv2.DIST_L2, 5)
            xy = np.clip(np.round(u[fr]).astype(int), 0, [M.shape[-1] - 1, M.shape[-2] - 1])
            ok[fr] = bool((dist[xy[:, 1], xy[:, 0]] <= GATE_PX).all())
        out[arm] = ok
    return out


def hybrid_length(D, tracks, types, sd_kin, gate=None, kin_sd_by_type=False, apply_types=("standard", "long")):
    """v3 hybrid, but each tip's depth prior is shifted along the tip-midpoint ray by the
    length-constrained correction (fusion.length_constrained_depth), with the type's train length
    and length spread. types: {arm: (T,) type names or None}; None -> the v3 prior."""
    rig, f5 = D["rig"], D["f5"]
    out, cov = {}, {}
    Kinv = np.linalg.inv(rig.left.K)
    for arm, (a, b) in TIP.items():
        X = tracks[arm].X[f5].copy()
        C = tracks[arm].cov[f5].copy()
        for i, fr in enumerate(f5):
            if not np.isfinite(X[i]).all():
                continue
            uL, uR = D["kfL"][fr, [a, b]], D["kfR"][fr, [a, b]]
            sL, sR = D["std"][fr, [a, b]], D["stdR"][fr, [a, b]]
            if not (np.isfinite(uL).all() and np.isfinite(uR).all() and np.isfinite(sL).all() and np.isfinite(sR).all()):
                continue
            centres, sd = X[i, 2:4].copy(), sd_kin
            typ = types[arm][fr] if types else None
            if typ in apply_types and (gate is None or gate[arm][fr]):
                g = LIB["types"][typ]
                r = Kinv @ np.r_[uL.mean(0), 1.0]
                r /= np.linalg.norm(r)
                mid = X[i, 2:4].mean(0)
                sd_lat = float(r @ mid) * np.hypot(*sL) / 2 / rig.left.K[0, 0]  # tip-midpoint 2D std -> mm
                sk = g[arm]["scatter_mm_median"] if kin_sd_by_type else sd_kin
                c, sd = length_constrained_depth(r, X[i, 0], g[arm]["tip_offset_mm"], g["length_sd_mm"], mid, sk,
                                                 sd_lat)
                centres = centres + (c - mid)
            for j in range(2):
                X[i, 2 + j], C[i, 2 + j] = triangulate_with_depth_prior(rig, uL[j], uR[j], max(sL[j], 0.5), max(sR[j], 0.5),
                                                                        centres[j], sd)
        out[arm], cov[arm] = X, C
    return out, cov


def tip_error(D, tracks, types=None, length=False, gate=None, kin_sd_by_type=False, apply_types=("standard", "long")):
    if length:
        Xh, _ = hybrid_length(D, tracks, types, V3["hybrid_sd_depth_mm"], gate, kin_sd_by_type, apply_types)
    else:
        Xh, _ = hybrid(D, tracks, V3["hybrid_sd_depth_mm"])
    res = {}
    for arm, (a, b) in TIP.items():
        g = (D["G"][:, a] + D["G"][:, b]) / 2
        h = (Xh[arm][:, 2] + Xh[arm][:, 3]) / 2
        v = (D["V2"][:, a] + D["V2"][:, b]) / 2
        e = np.linalg.norm(np.where(np.isfinite(h), h, v) - g, axis=1)
        res[arm] = float(np.nanmean(e))
    return res


VARIANTS = {
    "v3": (dict(), None),
    "v3_tip3d": (dict(tip_mode="3d"), None),
    "oracle": (dict(), "oracle"),
    "oracle_sd1": (dict(c_prior_sd=1.0), "oracle"),
    "oracle_tip3d": (dict(tip_mode="3d"), "oracle"),
    "oracle_len": (dict(), "oracle"),  # v3 fusion + length-constrained hybrid (type only picks L)
    "oracle_len_gate": (dict(), "oracle"),  # + mask visibility gate
    "cls_len": (dict(), "causal"),  # causal type classifier (fusion.classify_type)
    "cls_len_gate": (dict(), "causal"),
    "cls_len_kinsd": (dict(), "causal"),  # kinematic depth sd = the type's train tip scatter (not v3's 5 mm)
    "cls_len_kinsd_longonly": (dict(), "causal"),  # ... applied to long-type arms only; standard = v3
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--classifier", help="type sequences from runs/v4_dev/type_seq_<name>.json (adds variants)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = {k: VARIANTS[k] for k in args.variants if k in VARIANTS}
    if args.classifier:
        for k in args.variants:
            if k not in VARIANTS:  # e.g. "cls_sd1" -> classifier types with c_prior_sd 1
                base = VARIANTS[k.replace("cls", "oracle")]
                variants[k] = (base[0], args.classifier)
    rows = {}
    for traj in TUNE:
        D = load_prepared(traj)
        T = len(D["O"]["gt"])
        for name, (over, tsrc) in variants.items():
            p = replace(FusionParams(**V3["fusion"]), **over)
            if tsrc is None:
                types = None
            elif tsrc == "causal":
                types = "causal"
            elif tsrc == "oracle":
                types = {arm: np.array([oracle_type(D, arm)] * T, dtype=object) for arm in ARMS}
            else:
                seq = json.loads((OUT / f"type_seq_{tsrc}.json").read_text())[str(traj)]
                types = {arm: np.array(seq[arm], dtype=object) for arm in ARMS}
            length = "_len" in name
            tracks = run(D, p, None if length else types)
            if types == "causal":
                types = causal_types(D, tracks)
            gate = mask_gate(traj, D) if name.endswith("_gate") else None
            if gate:
                print("  gated-out fraction", {k: round(1 - float(v[D["f5"]].mean()), 3) for k, v in gate.items()})
            e = tip_error(D, tracks, types, length, gate, kin_sd_by_type="_kinsd" in name,
                          apply_types=("long",) if name.endswith("_longonly") else ("standard", "long"))
            rows.setdefault(name, {})[traj] = e
            print(name, traj, {k: round(v, 2) for k, v in e.items()}, flush=True)
    for name, r in rows.items():
        per_traj = [np.mean(list(v.values())) for v in r.values()]
        print(f"{name:14s} mean {np.mean(per_traj):.2f} | " + " ".join(f"{t}:{x:.2f}" for t, x in zip(r, per_traj)))
        (OUT / f"fusion_{name}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
