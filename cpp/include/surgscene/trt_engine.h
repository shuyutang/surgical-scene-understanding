// TensorRT engine wrapper for the deploy graph: uint8 stereo pair in, (kp, conf, sigma) out.
// All buffers are allocated once (pinned host + device); run() never allocates.
#pragma once
#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <array>
#include <memory>
#include <string>

namespace surgscene {

constexpr int kImages = 2, kH = 986, kW = 1400, kKeypoints = 10;

struct EngineOutput {
  float kp[kImages][kKeypoints][2];
  float conf[kImages][kKeypoints];
  float sigma[kImages][kKeypoints];
};

class TrtEngine {
 public:
  explicit TrtEngine(const std::string& engine_path);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // Pinned host input buffer (kImages x kH x kW x 3 BGR uint8); fill it, then call run().
  uint8_t* input() { return h_in_; }
  static constexpr size_t input_bytes() { return size_t(kImages) * kH * kW * 3; }

  // H2D, inference, D2H on the engine's stream; blocks until outputs are on the host.
  // Timings (ms) of the three GPU phases come from CUDA events.
  void run(EngineOutput& out, float* ms_h2d = nullptr, float* ms_infer = nullptr, float* ms_d2h = nullptr);

 private:
  struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override;
  } logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  cudaStream_t stream_{};
  cudaEvent_t ev_[4]{};
  uint8_t* h_in_ = nullptr;
  void* d_in_ = nullptr;
  void* d_kp_ = nullptr;
  void* d_conf_ = nullptr;
  void* d_sigma_ = nullptr;
  EngineOutput* h_out_ = nullptr;  // pinned
};

}  // namespace surgscene
