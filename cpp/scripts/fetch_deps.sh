#!/usr/bin/env bash
# TensorRT C++ headers are not in the pip wheels; take them from the matching OSS release branch.
# Libraries (libnvinfer, libcudart) come from the project's uv environment (tensorrt-cu12, torch cu128).
set -euo pipefail
cd "$(dirname "$0")/../third_party"
if [ ! -d tensorrt_oss/include ]; then
  git clone -q --depth 1 --filter=blob:none --sparse -b release/10.16 https://github.com/NVIDIA/TensorRT.git tensorrt_oss
  git -C tensorrt_oss sparse-checkout set include
fi
echo "TensorRT headers: $(pwd)/tensorrt_oss/include"
