"""v3 Phase A: DINOv2 keypoint network vs the v2 U-Net.

  uv run python scripts/eval_kp_vit.py --split tune     # development
  uv run python scripts/eval_kp_vit.py --split test2    # once, after the pre-registration is committed

Left eye, every 5th frame (the frame cache), clean and with the v2 de-centered occluder
protocol. Two seeds per architecture. Also: does the learned sigma track the actual error better
than the Laplace (peak-width) sigma? Endpoints on test2 use the seed-0 models (pre-specified);
seed 1 is reported for variability. Trajectory-clustered bootstrap for the paired differences.
Writes runs/v3_kp_vit_<split>/{results.json, report.md}.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_a0 import finished, metrics, spearman  # noqa: E402
from eval_kp import NATIVE, OCC_RADIUS, infer, occlusion_plan  # noqa: E402
from eval_stage2 import boot  # noqa: E402

from surgscene.frontend import TEST2, TUNE, load_kp_model  # noqa: E402
from surgscene.keypoints import KeypointDataset, load_records  # noqa: E402
from surgscene.kp_vit import DinoKeypointNet, read_at  # noqa: E402
from surgscene.structured import observe  # noqa: E402

UNET = {"v2 s0": "kp_unet_r34_decentered_2", "v2 s1": "kp_unet_r34_decentered_s1_"}
CONFIG = ROOT / "configs/v3_kp_vit_selected.json"
VIT = {"vit s0": "kp_vit_s_2", "vit s1": "kp_vit_s_s1_"}


def find(prefix):
    c = sorted(p for p in (ROOT / "runs").glob(prefix + "*") if finished(p))
    return c[-1] if c else None


@torch.no_grad()
def infer_vit(run: Path, records, occlude, cache: Path, offset=0.0):
    if cache.exists():
        return dict(np.load(cache))
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = DinoKeypointNet(cfg["num_keypoints"], layers=cfg["layers"], variant=cfg["variant"], pretrained=False)
    net.load_state_dict(ck["model"])
    net = net.cuda().eval()
    K = cfg["num_keypoints"]
    ds = KeypointDataset(records, occlude=occlude, occluder_radius=OCC_RADIUS, seed=1, occluder_offset=offset)
    out = {k: [] for k in ["kp", "conf", "sigma", "sigma_learned", "gt"]}
    for x, _, kp, _ in tqdm(DataLoader(ds, 16, num_workers=8), desc=cache.stem):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(x.cuda()).float()
        hm = y[:, :K].sigmoid()
        o = observe(hm)
        idx = hm.flatten(2).argmax(-1)
        s = read_at(y[:, K:], idx).clamp(-6, 12).mul(0.5).exp()  # cache px
        out["kp"].append(o.kp)
        out["conf"].append(o.conf)
        out["sigma"].append(o.sigma)
        out["sigma_learned"].append(s.cpu().numpy())
        out["gt"].append(kp.numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["occluded"] = np.zeros(out["conf"].shape, bool)
    if occlude is not None:
        for i, js in enumerate(occlude):
            out["occluded"][i, js] = True
    np.savez_compressed(cache, **out)
    del net
    torch.cuda.empty_cache()
    return out


def per_traj(O, traj, fn, sel=None):
    err = np.linalg.norm(O["kp"] - O["gt"], axis=-1) * NATIVE
    ok = np.isfinite(err) if sel is None else np.isfinite(err) & sel
    return [np.where(ok[traj == t], fn(err[traj == t]), np.nan).ravel() for t in np.unique(traj)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test2"], required=True)
    args = ap.parse_args()
    trajs = TUNE if args.split == "tune" else TEST2
    out_dir = ROOT / "runs" / f"v3_kp_vit_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = [r for r in load_records("dev") + load_records("test") if r["traj"] in trajs and r["eye"] == "left"]
    recs.sort(key=lambda r: (r["traj"], r["t"]))
    traj = np.array([r["traj"] for r in recs])
    plan = occlusion_plan(len(recs), 7)
    obs = {}
    for label, prefix in {**UNET, **VIT}.items():
        run = find(prefix)
        if run is None:
            print(f"skip {label}: no finished run {prefix}*")
            continue
        if label in UNET:
            net = load_kp_model(str(run / "best.pt"))
            obs[label] = {"clean": infer(net, recs, None, out_dir / f"obs_{run.name}_clean.npz"),
                          "occ": infer(net, recs, plan, out_dir / f"obs_{run.name}_occ.npz", offset=0.6)}
            del net
            torch.cuda.empty_cache()
        else:
            obs[label] = {"clean": infer_vit(run, recs, None, out_dir / f"obs_{run.name}_clean.npz"),
                          "occ": infer_vit(run, recs, plan, out_dir / f"obs_{run.name}_occ.npz", offset=0.6)}
        obs[label]["run"] = run.name

    R = {"split": args.split, "frames": len(recs), "runs": {}}
    for label, o in obs.items():
        c, oc = o["clean"], o["occ"]
        r = {"run": o["run"], "clean": metrics(c, traj), "occluded_kp": metrics(oc, traj, oc["occluded"])}
        if "sigma_learned" in c:
            err = np.linalg.norm(c["kp"] - c["gt"], axis=-1) * NATIVE
            ok = np.isfinite(err)
            r["spearman_learned_sigma_err"] = spearman(c["sigma_learned"][ok], err[ok])
            r["learned_sigma_median_px"] = float(np.median(c["sigma_learned"][ok]) * NATIVE)
            # calibration of the learned sigma: fraction of errors within the 2D 95% radius (2.45 s)
            r["learned_sigma_cover95"] = float((err[ok] <= 2.4477 * c["sigma_learned"][ok] * NATIVE).mean())
        R["runs"][label] = r
    # paired, trajectory-clustered: vit s0 - v2 s0
    if "vit s0" in obs and "v2 s0" in obs:
        pck = lambda e: (e <= 10).astype(float)
        a = per_traj(obs["vit s0"]["clean"], traj, pck)
        b = per_traj(obs["v2 s0"]["clean"], traj, pck)
        R["A1_pck10_vit_minus_v2"] = boot([x - y for x, y in zip(a, b)])
        a = per_traj(obs["vit s0"]["clean"], traj, lambda e: e)
        b = per_traj(obs["v2 s0"]["clean"], traj, lambda e: e)
        R["A2_mean_err_vit_minus_v2"] = boot([x - y for x, y in zip(a, b)])
        oc_v, oc_u = obs["vit s0"]["occ"], obs["v2 s0"]["occ"]
        a = per_traj(oc_v, traj, lambda e: e, oc_v["occluded"])
        b = per_traj(oc_u, traj, lambda e: e, oc_u["occluded"])
        R["occluded_mean_err_vit_minus_v2"] = boot([x - y for x, y in zip(a, b)])
    cfg = json.loads(CONFIG.read_text())
    ends = []
    if "vit s0" in obs and "v2 s0" in obs:
        assert obs["vit s0"]["run"] == cfg["model_run"] and obs["v2 s0"]["run"] == cfg["baseline_run"], "not the frozen runs"
        c = obs["vit s0"]["clean"]
        k = cfg["learned_sigma_scale"]
        cov = per_traj(c, traj, lambda e: e)  # errors per trajectory, then coverage with the frozen scale
        s = [np.where(traj[:, None] == t, c["sigma_learned"] * NATIVE, np.nan)[traj == t].ravel() for t in np.unique(traj)]
        R["A8_cover95_learned_sigma"] = boot([np.where(np.isfinite(e), (e <= 2.4477 * k * ss).astype(float), np.nan)
                                              for e, ss in zip(cov, s)])
        lap = R["runs"]["v2 s0"]["clean"]["spearman_sigma_err"]
        ends = [("A1 (primary)", "PCK@10, DINOv2 − v2 (clean)", R["A1_pck10_vit_minus_v2"], "LB > 0",
                 R["A1_pck10_vit_minus_v2"]["lo"] > 0),
                ("A2", "Mean error, DINOv2 − v2 (clean, px)", R["A2_mean_err_vit_minus_v2"], "UB < 0",
                 R["A2_mean_err_vit_minus_v2"]["hi"] < 0),
                ("A7", "Occluded keypoints mean error, DINOv2 − v2 (px)", R["occluded_mean_err_vit_minus_v2"], "UB < 0",
                 R["occluded_mean_err_vit_minus_v2"]["hi"] < 0),
                ("A5", "Spearman(σ, error): learned σ (DINOv2) vs Laplace σ (v2)",
                 {"point": R["runs"]["vit s0"]["spearman_learned_sigma_err"], "lo": float("nan"), "hi": float("nan")},
                 f"learned > Laplace ({lap:+.2f})", R["runs"]["vit s0"]["spearman_learned_sigma_err"] > lap),
                ("A8", f"Coverage of the 95% radius, learned σ × {k:g}", R["A8_cover95_learned_sigma"],
                 "LB ≥ 0.85 and point ≤ 0.99",
                 R["A8_cover95_learned_sigma"]["lo"] >= 0.85 and R["A8_cover95_learned_sigma"]["point"] <= 0.99)]
        R["verdicts"] = {e[0]: bool(e[4]) for e in ends}
    (out_dir / "results.json").write_text(json.dumps(R, indent=1))

    f = lambda d, n=3: f"{d['point']:.{n}f} [{d['lo']:.{n}f}, {d['hi']:.{n}f}]"
    L = [f"# Phase A — {args.split}: DINOv2 keypoints vs v2 U-Net ({len(recs)} left frames, every 5th)", ""]
    if ends:
        L += ["| ID | Endpoint | Result [95% CI] | Criterion | Verdict |", "|---|---|---|---|---|"]
        for i, name, d, crit, ok in ends:
            L.append(f"| {i} | {name} | {f(d)} | {crit} | {'PASS' if ok else 'FAIL'} |")
        L += [""]
    L += [
         "| Run | Clean mean (px) | Median | PCK@5 | PCK@10 | Occluded kp mean | ρ(Laplace σ, err) | ρ(learned σ, err) | learned-σ 95% cover |",
         "|---|---|---|---|---|---|---|---|---|"]
    for label, r in R["runs"].items():
        c, o = r["clean"], r["occluded_kp"]
        L.append(f"| {label} | {c['mean']:.2f} | {c['median']:.2f} | {c['pck5']:.3f} | {c['pck10']:.3f} | {o['mean']:.2f} | "
                 f"{c['spearman_sigma_err']:+.2f} | {r.get('spearman_learned_sigma_err', float('nan')):+.2f} | "
                 f"{r.get('learned_sigma_cover95', float('nan')):.3f} |")
    if "A1_pck10_vit_minus_v2" in R:
        L += ["", "Paired, trajectory-clustered (seed 0 vs seed 0):", "",
              f"- A1 PCK@10, vit − v2: {f(R['A1_pck10_vit_minus_v2'])}",
              f"- A2 mean error, vit − v2: {f(R['A2_mean_err_vit_minus_v2'], 2)} px",
              f"- Occluded keypoints, mean error vit − v2: {f(R['occluded_mean_err_vit_minus_v2'], 2)} px"]
    L += ["", "Clean mean error per trajectory:", "", "| Run | " + " | ".join(str(t) for t in trajs) + " |",
          "|---|" + "---|" * len(trajs)]
    for label, r in R["runs"].items():
        L.append(f"| {label} | " + " | ".join(f"{r['clean']['per_traj_mean'].get(t, float('nan')):.1f}" for t in trajs) + " |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
