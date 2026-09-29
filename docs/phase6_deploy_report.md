# Phase 6: deployment (TensorRT + C++ runtime), 2026-09-25

Criteria D1–D4 were pre-specified in [validation_plan.md](validation_plan.md) (commit `d7bfdb5`)
before any of these numbers were computed. RTX 4090, TensorRT 10.16.1, CUDA 12.8 runtime,
g++ 13.3, Eigen 3.4.

## What is deployed

```text
uint8 BGR stereo pair (2 x 986 x 1400 x 3), pinned host buffer
  -> H2D (8.3 MB)
  -> TensorRT engine: avg-pool resize, pad, normalize -> U-Net/ResNet-34 -> sigmoid
                      -> argmax + sub-pixel + Laplace sigma          (all in the engine)
  -> D2H (240 bytes: kp, conf, sigma for 2 x 10 keypoints)
  -> C++ (Eigen): gated shape-prior MAP x 4 -> Kalman x 20 -> triangulation x 4 -> 3D tips + covariance
```

Preprocessing and decoding are part of the ONNX graph. The engine does all per-pixel work on the
GPU without custom CUDA kernels, and the host never touches a heatmap. That's also why there's no
nvcc dependency. The whole toolchain comes from pip wheels plus the TensorRT OSS headers
(`cpp/scripts/fetch_deps.sh`).

## Results

| ID | Endpoint | Result | Criterion | Verdict |
|---|---|---|---|---|
| D1 | TensorRT FP32 vs PyTorch FP32, keypoints (2,000 on 200 tune images) | p99 0.0034 px (max 0.008) | p99 ≤ 0.1 px | PASS |
| D2 (R6) | TensorRT FP16 − PyTorch FP32 mean argmax error, test2 (12 trajectories, every 5th frame, left) | +0.008 [−0.013, +0.036] px (13.23 → 13.24) | upper bound ≤ +0.25 px | PASS |
| D3 | C++ vs Python on goldens | gated MAP 1.3e-6 px; Kalman 2e-13 px; triangulation 1.7e-7 mm (covariance 2e-8 relative); engine outputs identical to the Python TensorRT runner (FP32 and FP16) | all 10 tests pass | PASS |
| D4 (R3) | C++ end-to-end latency per stereo pair, FP16, 2,000 iterations | p50 2.50 ms, **p99 2.59 ms** | p99 < 33 ms | PASS |

**FP16 detail (D1 side result).** FP16 moves the median keypoint by only 0.024 px, but 0.4% of keypoints
move by more than 2 px (max 25 px). When two heatmap peaks are nearly tied, FP16 rounding can flip
the argmax. The mean accuracy is unchanged (D2), but this is exactly the failure a
per-frame parity test catches and a mean metric hides. The Kalman gate absorbs these one-frame
jumps downstream.

## Latency by stage (FP16 engine, C++)

| Stage | p50 (ms) | p99 (ms) | Budget (ms) |
|---|---|---|---|
| Video decode, both eyes (OpenCV/FFmpeg, measured in Python, excluded from D4) | 0.93 | 3.70 | 3 |
| Copy into pinned input | 0.56 | 0.62 | |
| H2D (8.3 MB) | 0.44 | 0.49 | 2 |
| TensorRT: preprocess + network + decode, 2 images | 1.46 | 1.49 | 12 |
| D2H | 0.008 | 0.013 | |
| Gated MAP, 4 instrument-eyes | 0.019 | 0.027 | 3 |
| Kalman (20 filters) | 0.001 | 0.001 | 1 |
| Triangulation (4 points) | 0.005 | 0.006 | |
| **End to end (excluding decode)** | **2.50** | **2.59** | **< 33** |

FP32 engine: TensorRT 3.76 ms, end to end p99 5.0 ms.

The C++ MAP solver runs about 5 µs per instrument, versus 470 µs in NumPy (Phase 3). The algorithm
is the same; the gain comes from fixed-size Eigen types and no Python overhead.

## Not measured, and what would change on an embedded target

- **SGM tissue depth isn't in the C++ runtime.** It runs in Python/OpenCV at about 18 ms per
  half-resolution pair on the CPU. It's the stage that would dominate a real budget. On Jetson it
  would move to the VPI stereo hardware or a small learned stereo network.
- **No clock-locked or power-limited run.** That needs root for `nvidia-smi -lgc`, so it wasn't done.
  As a rough, unmeasured estimate, a Jetson AGX Orin is 5–10× slower than a 4090 for this kind of
  network (memory bandwidth 204 vs 1008 GB/s), so the engine would take roughly 7–15 ms per
  stereo pair. That still fits in 33 ms, but INT8 or a narrower input would be the first levers.
- **No pipelining across frames.** Stages run serially on one stream. Overlapping the H2D of
  frame t+1 with inference on frame t would cut about 0.4 ms, which isn't needed at this margin.
- **Failure handling implemented:** TensorRT version check at load (major.minor of library vs
  headers), engine I/O name/dtype/shape validation, NaN guard on outputs (confidence set to 0,
  so the Kalman filter predicts only), and MAP non-convergence fallback (use the observations).
