"""v3 Phase C endpoints (C1-C6 + pre-specified subgroup), pre-registered in docs/validation_plan.md.

  uv run python scripts/eval_v3c.py --split tune     # sanity; numbers quoted in the pre-registration
  uv run python scripts/eval_v3c.py --split test2    # once, after the pre-registration is committed

Frozen configuration: configs/v3_fusion_selected.json (fusion parameters, depth-prior sd, kappa)
and configs/tool_geometry.json (train-fitted priors). The method under test is the *hybrid*: v2
Kalman 2D detections of each jaw tip, triangulated with a depth prior along the viewing ray from
the causal kinematics fusion (surgscene.fusion). Frames before the first causal calibration (1 s)
use the v2 estimate. Unit of analysis = trajectory; trajectory-clustered bootstrap, 2,000
replicates, percentile CIs (eval_stage2.boot).
Writes runs/v3c_<split>/{results.json, report.md}.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from surgscene.evaluation import boot
from surgscene.fusion import FusionParams
from surgscene.pipeline import SWAP_MARGIN_MM, prepare
from surgscene.pipeline import v3c_per_trajectory as per_trajectory
from surgscene.stage2 import SPLITS, load_selected, shape_models
from surgscene.temporal import KFParams

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs/v3_fusion_selected.json"
LONG_JAW_MM = 15.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test2"], required=True)
    ap.add_argument("--config", type=Path, default=CONFIG, help="alternative config (post-hoc analyses only)")
    ap.add_argument("--tag", default="", help="output suffix for post-hoc analyses")
    args = ap.parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    p = FusionParams(**cfg["fusion"])
    sel = load_selected()
    models, _ = shape_models()
    kfp = KFParams(**sel["kf"])
    out_dir = ROOT / "runs" / f"v3c_{args.split}{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per, sub = {}, {"long": {"c1": [], "v2": [], "c2": []}, "standard": {"c1": [], "v2": [], "c2": []}}
    offsets = {}
    for t in SPLITS[args.split]:
        D = prepare(t, models, sel["gated"], kfp)
        r, arms = per_trajectory(D, cfg, p)
        for k, v in r.items():
            per.setdefault(k, []).append(v)
        for arm, a in arms.items():
            grp = "long" if (a["tip_offset_mm"] or 0) > LONG_JAW_MM else "standard"
            offsets[f"{t}/{arm}"] = {"tip_offset_mm": a["tip_offset_mm"], "group": grp,
                                     "calib_first_frame": a["calib_first_frame"]}
            for k in ("c1", "v2", "c2"):
                sub[grp][k].append(a[k])  # cluster = trajectory-arm within a subgroup
        print(f"traj {t}: C1 {np.nanmean(r['c1']):.2f} mm vs v2 {np.nanmean(r['v2']):.2f}", flush=True)

    m = {k: boot(v) for k, v in per.items()}
    m["c5_diff"] = boot([a - b for a, b in zip(per["c5_h"], per["c5_v2"])])
    subgroups = {g: {k: boot(v) for k, v in d.items() if v} for g, d in sub.items()}
    ends = [
        ("C1 (primary, R1)", "Mean 3D tip error, hybrid (mm)", m["c1"], "UB ≤ 5.0", m["c1"]["hi"] <= 5.0),
        ("C2", "3D tip error, hybrid − v2 (paired, mm)", m["c2"], "UB < 0", m["c2"]["hi"] < 0),
        ("C3 (R7)", "Coverage of the κ-inflated 95% ellipsoid", m["c3"], "LB ≥ 0.85 and point ≤ 0.99",
         m["c3"]["lo"] >= 0.85 and m["c3"]["point"] <= 0.99),
        ("C4", "Jaw pivot 3D error, fused (mm)", m["c4"], "UB ≤ 5.0", m["c4"]["hi"] <= 5.0),
        ("C6", "2D left tip error, hybrid − v2 (px)", m["c6"], "UB ≤ +0.5", m["c6"]["hi"] <= 0.5),
    ]
    R = {"split": args.split, "config": cfg, "config_file": str(cfg_path.relative_to(ROOT)), "config_sha": sha(cfg_path),
         "tool_geometry_sha": sha(ROOT / "configs/tool_geometry.json"), "fusion_params": asdict(p),
         "metrics": m, "subgroups": subgroups, "arm_trajectories": offsets,
         "verdicts": {e[0]: bool(e[4]) for e in ends}}
    (out_dir / "results.json").write_text(json.dumps(R, indent=1))

    f = lambda d, n=2: f"{d['point']:.{n}f} [{d['lo']:.{n}f}, {d['hi']:.{n}f}]"
    L = [f"# v3 Phase C — {args.split}{' (POST HOC: ' + args.tag + ')' if args.tag else ''}", "",
         f"Config `{R['config_file']}` (sha256 {R['config_sha']}), tool geometry sha256 "
         f"{R['tool_geometry_sha']}. Trajectory-clustered bootstrap, 2,000 replicates.", "",
         "| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|"]
    for i, name, d, crit, ok in ends:
        L.append(f"| {i} | {name} | {f(d, 3 if i.startswith('C3') else 2)} | {crit} | {'PASS' if ok else 'FAIL'} |")
    L += ["", "Reported (no criterion):", "",
          f"- v2 3D tip error on the same frames: {f(m['v2'])} mm",
          f"- Hybrid after the warm-up only: {f(m['hyb_post'])} mm",
          f"- Along the viewing ray, hybrid / v2: {f(m['depth_h'])} / {f(m['depth_v2'])} mm",
          f"- Lateral, hybrid / v2: {f(m['lat_h'])} / {f(m['lat_v2'])} mm",
          f"- 2D left tips, hybrid / v2: {f(m['tip2d_h'])} / {f(m['tip2d_v2'])} px",
          f"- C5 tip-swap rate (> {SWAP_MARGIN_MM:g} mm), hybrid / v2: {f(m['c5_h'], 3)} / {f(m['c5_v2'], 3)}; "
          f"difference {f(m['c5_diff'], 3)}", "",
          f"Pre-specified subgroup: long-jaw instruments (online-estimated tip offset > {LONG_JAW_MM:g} mm at the "
          "last calibration) vs standard; cluster = trajectory-arm.", "",
          "| Subgroup | n arm-trajectories | Hybrid (mm) | v2 (mm) | Hybrid − v2 (mm) |", "|---|---|---|---|---|"]
    for g, d in subgroups.items():
        if d:
            L.append(f"| {g} | {len(sub[g]['c1'])} | {f(d['c1'])} | {f(d['v2'])} | {f(d['c2'])} |")
    L += ["", "Arm-trajectories (online tip offset, subgroup):", "",
          ", ".join(f"{k}: {v['tip_offset_mm']:.1f} mm ({v['group']})" for k, v in offsets.items() if v["tip_offset_mm"])]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
