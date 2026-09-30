// Shared by the serving benchmarks (in-process TensorRT, Triton, Holoscan): the same frames, the same
// metric and the same output dump, so the arms differ only in how the engine is called.
//
// Metric: end-to-end latency per stereo pair, from the pair in ordinary (pageable) host memory to the
// three outputs (kp, conf, sigma) in host memory. Closed loop, one pair in flight (batch 1), as in
// the real-time pipeline. 30 warm-up iterations, then `iters` timed ones, cycling through the pairs.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

constexpr int kImages = 2, kH = 986, kW = 1400, kKeypoints = 10;
constexpr size_t kPairBytes = size_t(kImages) * kH * kW * 3;
constexpr int kWarmup = 30;
constexpr size_t kOutFloats = kImages * kKeypoints * 4;  // kp (x, y), conf, sigma

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Frames: n_pairs stereo pairs, uint8 (2, 986, 1400, 3) BGR each, back to back (runs/deploy/bench_frames.u8).
inline std::vector<uint8_t> load_frames(const std::string& path, int n_pairs) {
  std::vector<uint8_t> f(kPairBytes * n_pairs);
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(f.data()), f.size());
  if (!in) throw std::runtime_error("cannot read " + std::to_string(n_pairs) + " pairs from " + path);
  return f;
}

// One pair's outputs in a fixed layout: kp[2][10][2], conf[2][10], sigma[2][10].
struct Outputs {
  float v[kOutFloats];
  float* kp() { return v; }
  float* conf() { return v + kImages * kKeypoints * 2; }
  float* sigma() { return v + kImages * kKeypoints * 3; }
};

class Recorder {
 public:
  Recorder(std::string name, int n_pairs, int iters) : name_(std::move(name)), n_pairs_(n_pairs), iters_(iters) {
    dump_.resize(size_t(n_pairs) * kOutFloats);
  }
  // it counts from 0 including warm-up; outputs of the first pass over the pairs are kept for parity checks
  void add(int it, double ms, const Outputs& o) {
    if (it >= kWarmup) ms_.push_back(ms);
    const int k = it - kWarmup;
    if (k >= 0 && k < n_pairs_) std::memcpy(&dump_[size_t(k) * kOutFloats], o.v, sizeof(o.v));
  }
  int total_iters() const { return kWarmup + iters_; }
  static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5))];
  }
  // Writes <prefix>.csv (per-iteration ms) and <prefix>.out.f32 (outputs of pairs 0..n-1 at iterations
  // kWarmup..kWarmup+n-1, i.e. pair index (kWarmup + k) % n_pairs), prints a summary line.
  void finish(const std::string& prefix) const {
    std::ofstream c(prefix + ".csv");
    c << "iter,ms\n";
    for (size_t i = 0; i < ms_.size(); ++i) c << i << "," << ms_[i] << "\n";
    std::ofstream d(prefix + ".out.f32", std::ios::binary);
    d.write(reinterpret_cast<const char*>(dump_.data()), dump_.size() * sizeof(float));
    double mean = 0;
    for (double x : ms_) mean += x;
    mean /= ms_.size();
    std::printf("%-34s n=%zu  mean %7.3f  p50 %7.3f  p99 %7.3f  max %7.3f ms  (%.0f pairs/s)\n", name_.c_str(),
                ms_.size(), mean, pct(ms_, 50), pct(ms_, 99), pct(ms_, 100), 1000.0 / mean);
  }

 private:
  std::string name_;
  int n_pairs_, iters_;
  std::vector<double> ms_;
  std::vector<float> dump_;
};

}  // namespace bench
