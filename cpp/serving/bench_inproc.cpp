// Arm A: in-process TensorRT.
//   bench_inproc <engine.plan> <frames.u8> <n_pairs> <iters> <pinned|direct> <out_prefix>
//   pinned  the repo's own runtime (surgscene::TrtEngine, as in D4): copy the pair into a pinned host
//           buffer, then H2D + enqueueV3 + D2H on one stream, synchronize
//   direct  same engine, but the pair goes host -> device with one cudaMemcpy from pageable memory
//           (the driver stages and pipelines it), as the Triton CUDA-shm and Holoscan arms do; this
//           separates the input-copy strategy from the serving framework
#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <iostream>
#include <memory>

#include "bench_common.h"
#include "surgscene/trt_engine.h"

namespace {
void cuda_ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};

// The same engine through the raw TensorRT API with the input copied straight from pageable memory.
class DirectEngine {
 public:
  explicit DirectEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("engine deserialization failed");
    ctx_.reset(engine_->createExecutionContext());
    cuda_ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    cuda_ok(cudaMalloc(&d_in_, bench::kPairBytes), "d_in");
    cuda_ok(cudaMalloc(&d_out_, sizeof(bench::Outputs)), "d_out");
    cuda_ok(cudaMallocHost(reinterpret_cast<void**>(&h_out_), sizeof(bench::Outputs)), "h_out");
    auto* base = static_cast<float*>(d_out_);
    ctx_->setTensorAddress("frames", d_in_);
    ctx_->setTensorAddress("kp", base);
    ctx_->setTensorAddress("conf", base + bench::kImages * bench::kKeypoints * 2);
    ctx_->setTensorAddress("sigma", base + bench::kImages * bench::kKeypoints * 3);
  }
  void run(const uint8_t* pair, bench::Outputs& out) {
    cuda_ok(cudaMemcpyAsync(d_in_, pair, bench::kPairBytes, cudaMemcpyHostToDevice, stream_), "h2d");
    if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
    cuda_ok(cudaMemcpyAsync(h_out_, d_out_, sizeof(bench::Outputs), cudaMemcpyDeviceToHost, stream_), "d2h");
    cuda_ok(cudaStreamSynchronize(stream_), "sync");
    std::memcpy(out.v, h_out_->v, sizeof(out.v));
  }

 private:
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  cudaStream_t stream_{};
  void* d_in_ = nullptr;
  void* d_out_ = nullptr;
  bench::Outputs* h_out_ = nullptr;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: bench_inproc <engine> <frames.u8> <n_pairs> <iters> pinned|direct <out_prefix>\n";
    return 2;
  }
  const int n_pairs = std::stoi(argv[3]), iters = std::stoi(argv[4]);
  const std::string mode = argv[5];
  auto frames = bench::load_frames(argv[2], n_pairs);
  bench::Recorder rec("in-process TensorRT (" + mode + ")", n_pairs, iters);
  bench::Outputs o;
  if (mode == "pinned") {
    surgscene::TrtEngine engine(argv[1]);
    static_assert(sizeof(surgscene::EngineOutput) == sizeof(bench::Outputs), "output layouts differ");
    surgscene::EngineOutput out;
    for (int it = 0; it < rec.total_iters(); ++it) {
      const auto t0 = bench::Clock::now();
      std::memcpy(engine.input(), frames.data() + size_t(it % n_pairs) * bench::kPairBytes, bench::kPairBytes);
      engine.run(out);
      const double ms = bench::ms_since(t0);
      std::memcpy(o.v, &out, sizeof(out));
      rec.add(it, ms, o);
    }
  } else if (mode == "direct") {
    DirectEngine engine(argv[1]);
    for (int it = 0; it < rec.total_iters(); ++it) {
      const auto t0 = bench::Clock::now();
      engine.run(frames.data() + size_t(it % n_pairs) * bench::kPairBytes, o);
      rec.add(it, bench::ms_since(t0), o);
    }
  } else {
    throw std::runtime_error("unknown mode " + mode);
  }
  rec.finish(argv[6]);
  return 0;
}
