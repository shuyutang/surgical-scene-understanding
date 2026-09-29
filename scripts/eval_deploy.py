"""Phase 6: TensorRT parity (D1), FP16 non-inferiority (D2), and goldens for the C++ runtime (D3).

  uv run --group deploy python scripts/eval_deploy.py

D1: TensorRT FP32 vs PyTorch FP32 deploy graph on 100 tune stereo pairs (200 images).
D2: TensorRT FP16 vs PyTorch FP32 argmax keypoint error on test2, every 5th frame, left eye.
Goldens (runs/deploy/goldens/*.npy) cover every C++ stage: engine I/O, shape models + gated MAP,
Kalman filter, triangulation.
"""

import json

import numpy as np
import torch
from tqdm import tqdm

from surgscene.deploy import DeployModel
from surgscene.evaluation import ci
from surgscene.frontend import TEST2, TUNE, load_kp_model
from surgscene.stage2 import ROOT, load_obs, load_rig, load_selected, shape_models, triangulate_seq
from surgscene.structured import fit_gated
from surgscene.temporal import KFParams, filter_keypoints
from surgscene.trt_runner import TrtRunner
from surgscene.video import frames  # noqa: E402

DEP = ROOT / "runs/deploy"
GOLD = DEP / "goldens"
CKPT = "runs/kp_unet_r34_decentered_20260925-004241/best.pt"


def boot_mean(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([np.nansum(v) for v in vals])
    c = np.array([np.isfinite(v).sum() for v in vals])
    reps = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(vals), len(vals)) for _ in range(n_boot))]
    lo, hi = ci(np.array(reps))
    return {"point": float(s.sum() / c.sum()), "lo": float(lo), "hi": float(hi)}


def main():
    GOLD.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = DeployModel(load_kp_model(CKPT)).cuda().eval()
    trt32, trt16 = TrtRunner(DEP / "kp_stereo_fp32.engine"), TrtRunner(DEP / "kp_stereo_fp16.engine")

    def run_ref(pair):
        with torch.no_grad():
            kp, conf, sig = ref(torch.from_numpy(pair).cuda())
        return {"kp": kp.cpu().numpy(), "conf": conf.cpu().numpy(), "sigma": sig.cpu().numpy()}

    R = {}
    # --- D1: parity on tune pairs
    d32, d16, dconf16 = [], [], []
    first = True
    for traj in TUNE:
        for t, pair in frames(traj, stride=40, limit=25):
            a, b, c = run_ref(pair), trt32(pair), trt16(pair)
            d32.append(np.linalg.norm(a["kp"] - b["kp"], axis=-1).ravel())
            d16.append(np.linalg.norm(a["kp"] - c["kp"], axis=-1).ravel())
            dconf16.append(np.abs(a["conf"] - c["conf"]).ravel())
            if first:  # engine goldens for the C++ parity test
                np.save(GOLD / "pair_frames.npy", pair)
                for k in ["kp", "conf", "sigma"]:
                    np.save(GOLD / f"pair_trt32_{k}.npy", b[k].astype(np.float32))
                    np.save(GOLD / f"pair_trt16_{k}.npy", c[k].astype(np.float32))
                first = False
    d32, d16, dconf16 = map(np.concatenate, (d32, d16, dconf16))
    R["D1"] = {"n_keypoints": int(d32.size), "trt32_vs_torch32_px": {"p50": float(np.median(d32)),
               "p99": float(np.percentile(d32, 99)), "max": float(d32.max())},
               "trt16_vs_torch32_px": {"p50": float(np.median(d16)), "p99": float(np.percentile(d16, 99)),
                                       "max": float(d16.max()), "frac_gt_2px": float(np.mean(d16 > 2))},
               "trt16_conf_absdiff_p99": float(np.percentile(dconf16, 99))}
    R["D1"]["pass"] = R["D1"]["trt32_vs_torch32_px"]["p99"] <= 0.1
    print(json.dumps(R["D1"], indent=1))

    # --- D2: FP16 non-inferiority on test2 (left eye, every 5th frame)
    e32, e16, dd = [], [], []
    for traj in tqdm(TEST2, desc="D2"):
        gt = load_obs(traj, "left")["gt"]
        a32, a16, dl = [], [], []
        for t, pair in frames(traj, stride=5):
            a, c = run_ref(pair), trt16(pair)
            ea = np.linalg.norm(a["kp"][0] - gt[t], axis=-1)
            ec = np.linalg.norm(c["kp"][0] - gt[t], axis=-1)
            a32.append(np.nanmean(ea) if np.isfinite(ea).any() else np.nan)
            a16.append(np.nanmean(ec) if np.isfinite(ec).any() else np.nan)
            dl.append(np.nanmean(ec - ea) if np.isfinite(ea).any() else np.nan)
        e32.append(np.array(a32)); e16.append(np.array(a16)); dd.append(np.array(dl))
    R["D2"] = {"torch_fp32_err": boot_mean(e32), "trt_fp16_err": boot_mean(e16), "delta": boot_mean(dd)}
    R["D2"]["pass"] = R["D2"]["delta"]["hi"] <= 0.25
    print(json.dumps(R["D2"], indent=1))

    # --- goldens for the C++ stages
    sel = load_selected()
    models, _ = shape_models()
    for arm, m in models.items():
        np.save(GOLD / f"shape_{arm}_mean.npy", m.mean)
        np.save(GOLD / f"shape_{arm}_P.npy", np.ascontiguousarray(m.P))
        np.save(GOLD / f"shape_{arm}_lam.npy", m.lam)
    O = load_obs(21, "left")
    idx = np.arange(0, 1001, 10)
    np.save(GOLD / "map_z.npy", O["kp"][idx])
    np.save(GOLD / "map_conf.npy", O["conf"][idx].astype(np.float64))
    np.save(GOLD / "map_sigma.npy", O["sigma"][idx].astype(np.float64))
    hp = sel["gated"]
    out = np.stack([np.concatenate([fit_gated(models[arm], O["kp"][i, sl], O["conf"][i, sl], O["sigma"][i, sl], **hp)
                                    for arm, sl in [("PSM1", slice(0, 5)), ("PSM3", slice(5, 10))]]) for i in idx])
    np.save(GOLD / "map_out.npy", out)
    kfp = KFParams(**sel["kf"])
    z, s, c = O["kp"][:300], O["sigma"][:300].astype(np.float64), O["conf"][:300].astype(np.float64)
    pos, std, unknown, _ = filter_keypoints(z, s, c, kfp)
    for k, v in {"kf_z": z, "kf_sigma": s, "kf_conf": c, "kf_pos": pos, "kf_std": std}.items():
        np.save(GOLD / f"{k}.npy", v)
    rig = load_rig(21)
    OR = load_obs(21, "right")
    uL, uR = O["kp"][:50], OR["kp"][:50]
    sL, sR = O["sigma"][:50].astype(np.float64), OR["sigma"][:50].astype(np.float64)
    X, C, rp = triangulate_seq(rig, uL, uR, sL, sR)
    for k, v in {"tri_uL": uL, "tri_uR": uR, "tri_sL": sL, "tri_sR": sR, "tri_X": X, "tri_cov": C}.items():
        np.save(GOLD / f"{k}.npy", v)
    np.save(GOLD / "rig_K.npy", np.stack([rig.left.K, rig.right.K]))
    np.save(GOLD / "rig_dist.npy", np.stack([rig.left.dist, rig.right.dist]))
    np.save(GOLD / "rig_R.npy", rig.R)
    np.save(GOLD / "rig_T.npy", rig.T)
    (GOLD / "params.json").write_text(json.dumps({"gated": hp, "kf": sel["kf"]}, indent=1))
    (DEP / "deploy_results.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
