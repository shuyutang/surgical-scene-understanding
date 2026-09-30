# Serving benchmark: in-process TensorRT vs Holoscan vs Triton (C++), 2026-09-30

**Question.** The real-time loop runs its TensorRT engine in-process (`cpp/src/trt_engine.cpp`, D4).
What would NVIDIA Holoscan or Triton Inference Server cost in latency for the same model, called
from C++?

**Answer.**
- **Holoscan adds about 0.07 ms** at the median (+0.25 ms at p99).
- **Triton adds 0.2 ms** with CUDA shared memory, **1.0 ms** with system shared memory, and
  **4.7 ms (p99 +8.4 ms)** when the 8.3 MB stereo pair travels inside the gRPC request.
- The largest single effect wasn't the framework but **how the input is copied to the GPU**: the
  repo's own runtime pays **0.48 ms** for copying into a pinned buffer first.
- Every arm returns outputs identical to the reference.

## Setup

- **Model:** the D4 keypoint graph (uint8 stereo pair 2 × 1400 × 986 × 3 → keypoints, confidence,
  σ), **one engine file** for every arm. It was built once with TensorRT 10.9.0 using the D4
  recipe (FP16 flag, 4 GB workspace), `runs/serving/kp_fp16_trt109.plan`, sha256 prefix
  `fc981f44c7fea679`.
  - Holoscan 4.6 (CUDA 12 image) and Triton 25.04 both ship TensorRT 10.9, so the identical file
    loads in both.
  - Triton 26.08 was tried first. It ships TensorRT 11.2, which dropped the `--fp16` flag
    (strongly typed networks only), and it needs CUDA forward compatibility on this driver.
- **Hardware:** RTX 4090, driver 580.159. Everything on one machine; Triton over localhost.
- **Metric:** per stereo pair, from the pair in ordinary (pageable) host memory to the three
  outputs in host memory.
  - Closed loop, batch 1, one pair in flight, as in the real-time pipeline.
  - 30 warm-up pairs, then 2,000 timed pairs cycling over 100 recorded SurgPose pairs.
  - 3 repeats per arm, with the order of the in-process/Holoscan and Triton groups rotated.
  - Δ is the difference from the in-process reference with the same input copy.
- **Parity:** each arm's outputs for the 100 pairs are compared with the reference's.

| Arm | What runs |
|---|---|
| In-process, pinned | `surgscene::TrtEngine`, the repo runtime: `memcpy` into a pinned buffer, then async H2D, `enqueueV3`, D2H, synchronize |
| In-process, direct (reference) | Same engine, raw TensorRT API: one `cudaMemcpyAsync` from pageable memory on the engine's stream |
| Holoscan | C++ app: Source (H2D into a `BlockMemoryPool` tensor) → `InferenceOp` (TensorRT backend, the engine file, input on GPU, outputs to host) → Sink. Closed loop by an ack cycle Sink → Source. Greedy or multi-thread scheduler; optional CUDA graphs |
| Triton | `tritonserver` (TensorRT backend, 1 GPU instance, no batching, metrics on) and a C++ gRPC client. Input in the request, in system shared memory, or in CUDA shared memory (the client copies H2D into a buffer the server opens by CUDA IPC) |

## Results (ms; 3 × 2,000 pairs per arm)

| Stack | Arm | Mean | p50 | p99 | max | Δ p50 | Δ p99 | Outputs = reference |
|---|---|---|---|---|---|---|---|---|
| In-process | direct copy (**reference**) | 1.897 | **1.891** | **1.961** | 2.328 | 0 | 0 | yes |
| In-process | repo runtime, pinned staging | 2.380 | 2.372 | 2.499 | 3.340 | +0.480 | +0.537 | yes |
| Holoscan 4.6 | greedy scheduler | 1.990 | **1.961** | 2.206 | 2.781 | **+0.069** | +0.245 | yes |
| Holoscan 4.6 | multi-thread scheduler (2 workers) | 2.014 | 1.997 | 2.559 | 3.193 | +0.106 | +0.598 | yes |
| Holoscan 4.6 | greedy + CUDA graphs | 2.114 | 2.112 | 2.321 | 2.853 | +0.221 | +0.360 | yes |
| Triton 25.04 | CUDA shared memory | 2.179 | **2.092** | 2.814 | 3.094 | **+0.200** | +0.853 | yes |
| Triton 25.04 | CUDA shared memory + CUDA graphs | 2.320 | 2.183 | 2.919 | 3.177 | +0.292 | +0.958 | yes |
| Triton 25.04 | system shared memory | 2.957 | 2.863 | 3.814 | 4.271 | +0.972 | +1.852 | yes |
| Triton 25.04 | input in the gRPC request | 6.480 | 6.556 | 10.378 | 13.950 | +4.665 | +8.416 | yes |

The per-repeat medians agree within 0.1 ms for every arm (`runs/serving/report.md`).

## Reading

- **Holoscan's framework cost is negligible here.** Scheduling three operators and passing
  tensors between them adds about 0.07 ms at the median. The greedy (single-thread) scheduler has
  the tighter tail; the multi-thread scheduler adds jitter (p99 +0.6 ms) without a benefit for a
  linear graph.
- **Triton's cost is the process boundary, mostly the input's trip across it.** Sending 8.3 MB
  inside the gRPC message (serialise, copy through the socket, deserialise, copy again) costs
  4.7 ms at the median and 8.4 ms at p99. System shared memory removes the socket copy (+1.0 ms);
  CUDA shared memory removes the host copy on the server entirely (+0.2 ms, p99 +0.85 ms). The
  outputs are 320 bytes, so the response costs little.
- **All options fit the 33 ms budget** at this size; the difference is the tail and what's left for
  the rest of the pipeline. For a closed control loop, in-process or Holoscan keeps the p99 within
  about 0.25 ms of the bare engine.
- **CUDA graphs didn't help** in either framework (+0.1 to +0.15 ms). The engine is one large
  network, not many small kernel launches, so there's little launch overhead to remove.
- **The repo's own runtime can be 0.48 ms faster.** `TrtEngine` copies the frame into a pinned
  buffer (`memcpy`) and then DMAs it: two serial copies. A single `cudaMemcpyAsync` from pageable
  memory lets the driver pipeline staging and DMA. Better still in a real system: have the
  capture write straight into the pinned buffer (no extra copy), or into GPU memory with GPUDirect
  RDMA, which Holoscan supports.

## What this benchmark doesn't show

- **Triton's strengths aren't exercised.** Dynamic batching, concurrent requests from many clients,
  several models on one GPU, and model management all matter for throughput-bound serving, not
  for one closed loop at batch 1.
- **Holoscan's strengths aren't exercised either.** GPUDirect RDMA capture (no H2D at all), a
  multi-operator GPU pipeline sharing streams, and Holoviz rendering. The Source here starts from
  host memory, like the other arms.
- **One machine, localhost, one GPU (RTX 4090).** Across a network, Triton's gRPC path would cost
  more, and CUDA shared memory would not be available. On Jetson/IGX the absolute numbers differ.
- **Development containers.** The engine was built in the Holoscan container and runs through the
  CUDA 12.8 (Holoscan) and 12.9 (Triton) runtimes, not the host's pip TensorRT 10.16 used in D4.
  The D4 numbers (2.5 ms p50) aren't directly comparable.

## Bugs found while building it (fixed before the reported run)

- **Race: pageable `cudaMemcpy` returns before its DMA finishes.** In the Triton CUDA-shared-memory
  client, the server sometimes read a frame still being overwritten. Outputs differed by up to
  1.3 (in keypoint pixels or confidence), and the timing looked 0.1 ms better than it was. The Holoscan source had the same
  unordered pattern (copy on the default stream, `InferenceOp` on its own stream), with no
  mismatch observed but no guarantee. Both now synchronise after the copy; the in-process arm
  copies on the engine's own stream and was never affected. The parity check caught it.
- **CUDA IPC across containers:** every fresh container's first process is PID 1, and Triton
  rejected a CUDA shared-memory registration that looked like it came from an earlier client
  ("invalid args"). Running the server and clients with `--pid=host` fixes it.
- **Closed loop in Holoscan:** gating the Source with a `BooleanCondition` that the Sink re-enables
  works with the greedy scheduler only. A disabled condition reports NEVER, and the multi-threaded
  schedulers drop the operator. An ack cycle (Sink → Source) works with every scheduler.

## Reproduce

```bash
cpp/serving/run_serving_bench.sh 2000 3     # Docker + NVIDIA runtime; images (tags in the script) are pulled on first use
uv run python scripts/serving_report.py     # runs/serving/{report.md, results.json}
```

Code: `cpp/serving/` (`bench_common.h`, `bench_inproc.cpp`, `bench_holoscan.cpp`, `bench_triton.cpp`,
`run_serving_bench.sh`).
