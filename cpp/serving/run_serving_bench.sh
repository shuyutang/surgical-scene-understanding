#!/usr/bin/env bash
# Serving benchmark: the same TensorRT 10.9 FP16 engine called in-process, through Triton, and inside a
# Holoscan application, from C++. Run from the repo root on a GPU with nothing else running:
#   cpp/serving/run_serving_bench.sh [iters=2000] [repeats=3]
# Needs Docker with the NVIDIA runtime and runs/deploy/{kp_stereo.onnx,bench_frames.u8}. Writes
# runs/serving/bench/r<k>_<arm>.{csv,out.f32} and bench.log; summarise with scripts/serving_report.py.
set -euo pipefail
ITERS=${1:-2000}
REPEATS=${2:-3}
N_PAIRS=100
HOLO=nvcr.io/nvidia/clara-holoscan/holoscan:v4.6.0-cuda12-dgpu   # TensorRT 10.9.0, CUDA 12.8
TRITON=nvcr.io/nvidia/tritonserver:25.04-py3                      # TensorRT 10.9.0, CUDA 12.9
TRITON_SDK=nvcr.io/nvidia/tritonserver:25.04-py3-sdk
ENGINE=runs/serving/kp_fp16_trt109.plan
FRAMES=runs/deploy/bench_frames.u8
OUT=runs/serving/bench
MODELS=runs/serving/triton_models
mkdir -p "$OUT"
RUN=(docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=33554432
     -u "$(id -u):$(id -g)" -v "$PWD:/repo" -w /repo)

# 1. engine: the D4 recipe (FP16 flag, 4 GB workspace), built once with the TensorRT both runtimes share
if [ ! -f "$ENGINE" ]; then
  "${RUN[@]}" "$HOLO" trtexec --onnx=runs/deploy/kp_stereo.onnx --fp16 --memPoolSize=workspace:4096M \
    --saveEngine="$ENGINE" > /dev/null
fi
# 2. Triton model repository: the same engine file, with and without CUDA graphs
for m in kp_stereo kp_stereo_graphs; do
  mkdir -p "$MODELS/$m/1"
  cp "$ENGINE" "$MODELS/$m/1/model.plan"
  cat > "$MODELS/$m/config.pbtxt" <<EOF
name: "$m"
platform: "tensorrt_plan"
max_batch_size: 0
input [ { name: "frames" data_type: TYPE_UINT8 dims: [ 2, 986, 1400, 3 ] } ]
output [
  { name: "kp" data_type: TYPE_FP32 dims: [ 2, 10, 2 ] },
  { name: "conf" data_type: TYPE_FP32 dims: [ 2, 10 ] },
  { name: "sigma" data_type: TYPE_FP32 dims: [ 2, 10 ] }
]
instance_group [ { count: 1 kind: KIND_GPU gpus: [ 0 ] } ]
EOF
done
echo 'optimization { cuda { graphs: true } }' >> "$MODELS/kp_stereo_graphs/config.pbtxt"
# 3. binaries: in-process + Holoscan in the Holoscan image, the Triton client in the Triton SDK image
"${RUN[@]}" "$HOLO" bash -c "cmake -S cpp/serving -B cpp/serving/build-holoscan > /dev/null && \
  cmake --build cpp/serving/build-holoscan -j 16 > /dev/null"
"${RUN[@]}" "$TRITON_SDK" bash -c "mkdir -p cpp/serving/build-triton && g++ -O3 -std=c++17 -DTRITON_ENABLE_GPU \
  cpp/serving/bench_triton.cpp /workspace/client/src/c++/library/shm_utils.cc -I/workspace/install/include \
  -I/workspace/client/src/c++/library -I/usr/local/cuda/include -L/workspace/install/lib -L/usr/local/cuda/lib64 \
  -Wl,-rpath,/workspace/install/lib -lgrpcclient -lcudart -lrt -o cpp/serving/build-triton/bench_triton"

start_triton() {
  docker rm -f surgscene-triton > /dev/null 2>&1 || true
  # --metrics-port: 8002 may be taken on the host; metrics stay on, as in a default deployment
  # --pid=host (server and clients): CUDA IPC handles carry the exporter's PID; in fresh containers every
  # client is PID 1, and the server rejects a handle that looks like it comes from an earlier client
  docker run -d --name surgscene-triton --gpus all --ipc=host --pid=host --network host --ulimit memlock=-1 \
    -v "$PWD/$MODELS:/models" "$TRITON" tritonserver --model-repository=/models --metrics-port=8012 > /dev/null
  # wait for both models, then give the gRPC service a moment (HTTP health answers first)
  until curl -sf localhost:8000/v2/models/kp_stereo/ready > /dev/null &&
        curl -sf localhost:8000/v2/models/kp_stereo_graphs/ready > /dev/null; do
    # no pipe here: under pipefail, `docker ps | grep -q` fails when grep exits early (SIGPIPE)
    [ -n "$(docker ps -q --filter name=surgscene-triton)" ] || { docker logs surgscene-triton 2>&1 | tail; exit 1; }
    sleep 1
  done
  sleep 3
}

H="cpp/serving/build-holoscan"
T="cpp/serving/build-triton/bench_triton localhost:8001"
for r in $(seq 1 "$REPEATS"); do
  # rotate the order of the arm groups between repeats
  groups=(local triton)
  [ $((r % 2)) -eq 0 ] && groups=(triton local)
  for grp in "${groups[@]}"; do
    if [ "$grp" = local ]; then
      "${RUN[@]}" "$HOLO" bash -c "
        $H/bench_inproc $ENGINE $FRAMES $N_PAIRS $ITERS pinned $OUT/r${r}_inproc_pinned &&
        $H/bench_inproc $ENGINE $FRAMES $N_PAIRS $ITERS direct $OUT/r${r}_inproc_direct &&
        $H/bench_holoscan $ENGINE $FRAMES $N_PAIRS $ITERS greedy 0 $OUT/r${r}_holoscan_greedy &&
        $H/bench_holoscan $ENGINE $FRAMES $N_PAIRS $ITERS multithread 0 $OUT/r${r}_holoscan_multithread &&
        $H/bench_holoscan $ENGINE $FRAMES $N_PAIRS $ITERS greedy 1 $OUT/r${r}_holoscan_greedy_graphs" 2>&1 |
        grep -E "pairs/s|rror|what|only" | sed "s/^/r$r  /"
    else
      start_triton
      # one client container per arm
      for spec in "kp_stereo grpc triton_grpc" "kp_stereo sysshm triton_sysshm" "kp_stereo cudashm triton_cudashm" \
                  "kp_stereo_graphs cudashm triton_cudashm_graphs"; do
        set -- $spec
        docker run --rm --gpus all --ipc=host --pid=host --network host -u "$(id -u):$(id -g)" -v "$PWD:/repo" -w /repo \
          "$TRITON_SDK" $T "$1" $FRAMES $N_PAIRS $ITERS "$2" "$OUT/r${r}_$3" 2>&1 |
          grep -E "pairs/s|rror|what|only" | sed "s/^/r$r  /"
      done
      docker rm -f surgscene-triton > /dev/null
    fi
  done
done
