"""v3 Phase E (development checks, tune only): the DINOv2 deploy graph under TensorRT.

  uv run --group deploy python scripts/eval_deploy_vit.py

  parity   TensorRT FP32 and FP16 vs the PyTorch FP32 deploy graph on 100 tune stereo pairs
           (trajectories 18, 19, 20, 23; 25 pairs each, every 10th frame): keypoint distance and
           relative difference of the learned sigma
  latency  TensorRT FP16 engine per stereo pair (uint8 in, keypoints out, host copies included),
           and the RAFT-Stereo checkpoints at reduced iteration counts in PyTorch FP16, half res
Writes runs/deploy_vit/{parity_latency.json, report.md}.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_deploy import frames  # noqa: E402

from surgscene.deploy import DeployModelVit  # noqa: E402
from surgscene.frontend import load_vit_model  # noqa: E402
from surgscene.trt_runner import TrtRunner  # noqa: E402

DEP = ROOT / "runs/deploy_vit"
TUNE = [18, 19, 20, 23]


def pct(x):
    x = np.asarray(x)
    return {"p50": float(np.median(x)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def main():
    info = json.loads((DEP / "export_info.json").read_text())
    ref = DeployModelVit(load_vit_model(info["checkpoint"])).cuda().eval()
    trt32, trt16 = TrtRunner(DEP / "kp_vit_stereo_fp32.engine"), TrtRunner(DEP / "kp_vit_stereo_fp16.engine")
    d32, d16, s32, s16 = [], [], [], []
    pairs = []
    for traj in TUNE:
        for _, pair in frames(traj, 10, 25):
            pairs.append(pair)
            with torch.no_grad():
                kp, conf, sig, sl = [t.cpu().numpy() for t in ref(torch.from_numpy(pair).cuda())]
            o32, o16 = trt32(pair), trt16(pair)
            d32.append(np.linalg.norm(o32["kp"] - kp, axis=-1).ravel())
            d16.append(np.linalg.norm(o16["kp"] - kp, axis=-1).ravel())
            s32.append((np.abs(o32["sigma_learned"] - sl) / sl).ravel())
            s16.append((np.abs(o16["sigma_learned"] - sl) / sl).ravel())
    d32, d16, s32, s16 = map(np.concatenate, (d32, d16, s32, s16))
    R = {"n_pairs": len(pairs), "trt32_vs_torch32_px": pct(d32), "trt16_vs_torch32_px": pct(d16),
         "trt16_frac_over_2px": float((d16 > 2).mean()),
         "trt32_sigma_learned_rel": pct(s32), "trt16_sigma_learned_rel": pct(s16)}

    # latency: FP16 engine, host uint8 -> outputs on host
    ms = []
    for i in range(300):
        t0 = time.perf_counter()
        trt16(pairs[i % len(pairs)])
        ms.append((time.perf_counter() - t0) * 1e3)
    R["trt16_ms_per_pair"] = pct(ms[50:])
    ms = []
    for i in range(100):
        t0 = time.perf_counter()
        trt32(pairs[i % len(pairs)])
        ms.append((time.perf_counter() - t0) * 1e3)
    R["trt32_ms_per_pair"] = pct(ms[20:])
    del ref, trt32, trt16
    torch.cuda.empty_cache()

    # stereo: RAFT-Stereo at reduced iterations (PyTorch, mixed precision), half-res rectified-size input
    from surgscene.learned_stereo import LearnedStereo
    L = np.ascontiguousarray(pairs[0][0][::2, ::2])
    Rr = np.ascontiguousarray(pairs[0][1][::2, ::2])
    R["stereo_ms"] = {}
    for ck, iters in [("middlebury", 32), ("middlebury", 8), ("realtime", 7), ("realtime", 4)]:
        net = LearnedStereo(ck, iters=iters)
        for _ in range(3):
            net.disparity(L, Rr)
        ms = []
        for _ in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net.disparity(L, Rr)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1e3)
        R["stereo_ms"][f"{ck}_it{iters}"] = pct(ms)
        del net
        torch.cuda.empty_cache()
    (DEP / "parity_latency.json").write_text(json.dumps(R, indent=1))
    Lr = ["# Phase E development checks (tune frames)", "",
          f"{R['n_pairs']} stereo pairs from tune 18/19/20/23.", "",
          "| Check | p50 | p99 | max |", "|---|---|---|---|",
          f"| TRT FP32 vs PyTorch FP32, keypoint px | {R['trt32_vs_torch32_px']['p50']:.4f} | {R['trt32_vs_torch32_px']['p99']:.4f} | {R['trt32_vs_torch32_px']['max']:.3f} |",
          f"| TRT FP16 vs PyTorch FP32, keypoint px | {R['trt16_vs_torch32_px']['p50']:.4f} | {R['trt16_vs_torch32_px']['p99']:.4f} | {R['trt16_vs_torch32_px']['max']:.3f} |",
          f"| TRT FP32 learned σ, relative | {R['trt32_sigma_learned_rel']['p50']:.4f} | {R['trt32_sigma_learned_rel']['p99']:.4f} | {R['trt32_sigma_learned_rel']['max']:.3f} |",
          f"| TRT FP16 learned σ, relative | {R['trt16_sigma_learned_rel']['p50']:.4f} | {R['trt16_sigma_learned_rel']['p99']:.4f} | {R['trt16_sigma_learned_rel']['max']:.3f} |",
          "", f"FP16 keypoints moving > 2 px: {R['trt16_frac_over_2px']:.4f}", "",
          "| Latency per stereo pair (ms) | p50 | p99 |", "|---|---|---|",
          f"| DINOv2 keypoints, TRT FP16 | {R['trt16_ms_per_pair']['p50']:.2f} | {R['trt16_ms_per_pair']['p99']:.2f} |",
          f"| DINOv2 keypoints, TRT FP32 | {R['trt32_ms_per_pair']['p50']:.2f} | {R['trt32_ms_per_pair']['p99']:.2f} |"]
    for k, v in R["stereo_ms"].items():
        Lr.append(f"| RAFT-Stereo {k}, PyTorch mixed precision | {v['p50']:.1f} | {v['p99']:.1f} |")
    (DEP / "report.md").write_text("\n".join(Lr) + "\n")
    print("\n".join(Lr))


if __name__ == "__main__":
    main()
