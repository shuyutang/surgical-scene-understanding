"""Serving benchmark summary: in-process TensorRT vs Triton vs Holoscan, same engine, called from C++.

  cpp/serving/run_serving_bench.sh 2000 3       # the measurement (Docker; see the script)
  uv run python scripts/serving_report.py       # this summary

Per arm: latency pooled over the repeats (mean, p50, p99, max), the spread of the per-repeat p50,
and the difference from the in-process reference with the same input copy (arm inproc_direct).
Parity: every arm's outputs for the 100 benchmark pairs must equal the reference's.
Writes runs/serving/{results.json, report.md}.
"""

import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "runs/serving/bench"
REF = "inproc_direct"
ARMS = {  # name -> (group, description)
    "inproc_pinned": ("In-process TensorRT", "repo runtime (TrtEngine): memcpy to pinned buffer, then async H2D"),
    "inproc_direct": ("In-process TensorRT", "one cudaMemcpy from pageable memory (reference)"),
    "holoscan_greedy": ("Holoscan 4.6", "Source -> InferenceOp -> Sink, greedy scheduler"),
    "holoscan_multithread": ("Holoscan 4.6", "same, MultiThreadScheduler (2 workers)"),
    "holoscan_greedy_graphs": ("Holoscan 4.6", "greedy, InferenceOp with CUDA graphs"),
    "triton_grpc": ("Triton 25.04", "gRPC, input in the request"),
    "triton_sysshm": ("Triton 25.04", "gRPC, input in system shared memory"),
    "triton_cudashm": ("Triton 25.04", "gRPC, input in CUDA shared memory (client H2D)"),
    "triton_cudashm_graphs": ("Triton 25.04", "CUDA shared memory, TensorRT backend with CUDA graphs"),
}


def load(arm):
    runs = sorted(glob.glob(str(BENCH / f"r*_{arm}.csv")))
    per_run = [np.loadtxt(p, delimiter=",", skiprows=1)[:, 1] for p in runs]
    outs = [np.fromfile(p.replace(".csv", ".out.f32"), np.float32).reshape(-1, 80) for p in runs]
    return per_run, outs


def main():
    data = {a: load(a) for a in ARMS}
    ref_runs, ref_outs = data[REF]
    ref_out = ref_outs[0]
    ref_all = np.concatenate(ref_runs)
    R = {}
    for arm, (per_run, outs) in data.items():
        if not per_run:
            continue
        x = np.concatenate(per_run)
        parity = max(float(np.abs(o - ref_out).max()) for o in outs)
        R[arm] = {"n": int(len(x)), "repeats": len(per_run), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
                  "p99": float(np.percentile(x, 99)), "max": float(x.max()),
                  "p50_per_repeat": [round(float(np.percentile(r, 50)), 3) for r in per_run],
                  "d_p50": float(np.percentile(x, 50) - np.percentile(ref_all, 50)),
                  "d_p99": float(np.percentile(x, 99) - np.percentile(ref_all, 99)),
                  "max_abs_output_diff_vs_ref": parity}
    log = (ROOT / "runs/serving/bench.log").read_text() if (ROOT / "runs/serving/bench.log").exists() else ""
    (ROOT / "runs/serving/results.json").write_text(json.dumps(R, indent=1))
    L = ["# Serving benchmark: in-process TensorRT vs Holoscan vs Triton (C++, same engine)", "",
         f"Per stereo pair (2 × 1400 × 986 × 3 uint8 in host memory → kp / conf / sigma in host memory), closed "
         f"loop, batch 1. {R[REF]['repeats']} repeats × {R[REF]['n'] // R[REF]['repeats']} pairs per arm (after 30 "
         "warm-up); Δ = difference from the in-process reference with the same input copy.", "",
         "| Stack | Arm | Mean | p50 | p99 | max | Δ p50 | Δ p99 | p50 per repeat | Outputs = reference |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, (group, desc) in ARMS.items():
        if arm not in R:
            continue
        m = R[arm]
        same = "yes" if m["max_abs_output_diff_vs_ref"] == 0 else f"max diff {m['max_abs_output_diff_vs_ref']:.2e}"
        L.append(f"| {group} | {desc} | {m['mean']:.3f} | {m['p50']:.3f} | {m['p99']:.3f} | {m['max']:.3f} | "
                 f"{m['d_p50']:+.3f} | {m['d_p99']:+.3f} | {', '.join(f'{v:.3f}' for v in m['p50_per_repeat'])} | {same} |")
    L += ["", "All times in ms."]
    if re.search(r"EXIT (\d+)", log) and not re.search(r"EXIT 0", log):
        L += ["", "**The benchmark run did not exit cleanly; see runs/serving/bench.log.**"]
    (ROOT / "runs/serving/report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
