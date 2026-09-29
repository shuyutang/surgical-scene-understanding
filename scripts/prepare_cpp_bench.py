"""Pre-decode stereo pairs for the C++ latency benchmark (D4), and time video decoding separately.

  uv run python scripts/prepare_cpp_bench.py
Writes runs/deploy/bench_frames.u8 (100 consecutive pairs of trajectory 21, 2x986x1400x3 BGR each)
and runs/deploy/decode_timing.json.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/surgpose/raw/000021/regular"
N = 100


def main():
    cL, cR = cv2.VideoCapture(str(RAW / "left_video.mp4")), cv2.VideoCapture(str(RAW / "right_video.mp4"))
    ms = []
    with open(ROOT / "runs/deploy/bench_frames.u8", "wb") as f:
        for i in range(1000):
            t0 = time.perf_counter()
            okl, fl = cL.read()
            okr, fr = cR.read()
            ms.append((time.perf_counter() - t0) * 1e3)
            if not (okl and okr):
                break
            if i < N:
                f.write(np.ascontiguousarray(np.stack([fl, fr])).tobytes())
    ms = np.array(ms[30:])
    info = {"n_pairs": N, "decode_stereo_pair_ms": {"p50": float(np.median(ms)), "p99": float(np.percentile(ms, 99)),
                                                    "note": "OpenCV/FFmpeg software H.264 decode, both eyes, sequential"}}
    (ROOT / "runs/deploy/decode_timing.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
