#include "surgscene/trt_engine.h"

#include <NvInferVersion.h>

#include <cmath>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace surgscene {

namespace {
void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void expect_tensor(nvinfer1::ICudaEngine& e, const char* name, nvinfer1::DataType dt, std::vector<int64_t> dims) {
  if (e.getTensorDataType(name) != dt) throw std::runtime_error(std::string("engine tensor dtype mismatch: ") + name);
  auto d = e.getTensorShape(name);
  if (d.nbDims != static_cast<int>(dims.size())) throw std::runtime_error(std::string("engine rank mismatch: ") + name);
  for (int i = 0; i < d.nbDims; ++i)
    if (d.d[i] != dims[i]) throw std::runtime_error(std::string("engine shape mismatch: ") + name);
}
}  // namespace

void TrtEngine::Logger::log(Severity s, const char* msg) noexcept {
  if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
}

TrtEngine::TrtEngine(const std::string& engine_path) {
  // Failure handling: the serialized engine is only valid for the TensorRT version that built it.
  const int32_t lib = getInferLibVersion();
  if (lib / 10000 != NV_TENSORRT_MAJOR || (lib / 100) % 100 != NV_TENSORRT_MINOR)
    throw std::runtime_error("TensorRT library version " + std::to_string(lib) + " != headers " +
                             std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR));
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("engine deserialization failed (built with another TensorRT/GPU?)");
  expect_tensor(*engine_, "frames", nvinfer1::DataType::kUINT8, {kImages, kH, kW, 3});
  expect_tensor(*engine_, "kp", nvinfer1::DataType::kFLOAT, {kImages, kKeypoints, 2});
  expect_tensor(*engine_, "conf", nvinfer1::DataType::kFLOAT, {kImages, kKeypoints});
  expect_tensor(*engine_, "sigma", nvinfer1::DataType::kFLOAT, {kImages, kKeypoints});
  ctx_.reset(engine_->createExecutionContext());

  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
  for (auto& e : ev_) check(cudaEventCreate(&e), "event");
  check(cudaMallocHost(reinterpret_cast<void**>(&h_in_), input_bytes()), "pinned input");
  check(cudaMallocHost(reinterpret_cast<void**>(&h_out_), sizeof(EngineOutput)), "pinned output");
  check(cudaMalloc(&d_in_, input_bytes()), "d_in");
  check(cudaMalloc(&d_kp_, sizeof(EngineOutput::kp)), "d_kp");
  check(cudaMalloc(&d_conf_, sizeof(EngineOutput::conf)), "d_conf");
  check(cudaMalloc(&d_sigma_, sizeof(EngineOutput::sigma)), "d_sigma");
  ctx_->setTensorAddress("frames", d_in_);
  ctx_->setTensorAddress("kp", d_kp_);
  ctx_->setTensorAddress("conf", d_conf_);
  ctx_->setTensorAddress("sigma", d_sigma_);
}

TrtEngine::~TrtEngine() {
  cudaStreamSynchronize(stream_);
  for (void* p : {d_in_, d_kp_, d_conf_, d_sigma_}) cudaFree(p);
  cudaFreeHost(h_in_);
  cudaFreeHost(h_out_);
  for (auto& e : ev_) cudaEventDestroy(e);
  cudaStreamDestroy(stream_);
}

void TrtEngine::run(EngineOutput& out, float* ms_h2d, float* ms_infer, float* ms_d2h) {
  check(cudaEventRecord(ev_[0], stream_), "ev0");
  check(cudaMemcpyAsync(d_in_, h_in_, input_bytes(), cudaMemcpyHostToDevice, stream_), "h2d");
  check(cudaEventRecord(ev_[1], stream_), "ev1");
  if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
  check(cudaEventRecord(ev_[2], stream_), "ev2");
  check(cudaMemcpyAsync(h_out_->kp, d_kp_, sizeof(out.kp), cudaMemcpyDeviceToHost, stream_), "d2h kp");
  check(cudaMemcpyAsync(h_out_->conf, d_conf_, sizeof(out.conf), cudaMemcpyDeviceToHost, stream_), "d2h conf");
  check(cudaMemcpyAsync(h_out_->sigma, d_sigma_, sizeof(out.sigma), cudaMemcpyDeviceToHost, stream_), "d2h sigma");
  check(cudaEventRecord(ev_[3], stream_), "ev3");
  check(cudaStreamSynchronize(stream_), "sync");
  std::memcpy(&out, h_out_, sizeof(EngineOutput));
  // NaN guard: a non-finite output is treated as "no detection" (confidence 0) downstream
  for (int i = 0; i < kImages; ++i)
    for (int k = 0; k < kKeypoints; ++k)
      if (!std::isfinite(out.kp[i][k][0]) || !std::isfinite(out.kp[i][k][1]) || !std::isfinite(out.conf[i][k]) ||
          !std::isfinite(out.sigma[i][k]))
        out.conf[i][k] = 0.0f;
  if (ms_h2d) cudaEventElapsedTime(ms_h2d, ev_[0], ev_[1]);
  if (ms_infer) cudaEventElapsedTime(ms_infer, ev_[1], ev_[2]);
  if (ms_d2h) cudaEventElapsedTime(ms_d2h, ev_[2], ev_[3]);
}

}  // namespace surgscene
