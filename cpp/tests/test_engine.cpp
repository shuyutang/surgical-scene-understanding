// D3: the C++ TensorRT wrapper reproduces the Python TensorRT runner on the same engine and input.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "surgscene/npy.h"
#include "surgscene/trt_engine.h"

using namespace surgscene;

namespace {
void check_engine(const std::string& prec, double tol_px) {
  const std::string G = GOLDEN_DIR;
  TrtEngine eng(std::string(ENGINE_DIR) + "/kp_stereo_" + prec + ".engine");
  auto frames = load_npy(G + "/pair_frames.npy");
  ASSERT_EQ(frames.bytes.size(), TrtEngine::input_bytes());
  std::memcpy(eng.input(), frames.u8(), frames.bytes.size());
  EngineOutput out;
  eng.run(out);
  auto kp = load_npy(G + "/pair_" + (prec == "fp32" ? std::string("trt32") : std::string("trt16")) + "_kp.npy").as_double();
  auto conf = load_npy(G + "/pair_" + (prec == "fp32" ? std::string("trt32") : std::string("trt16")) + "_conf.npy").as_double();
  double worst = 0, worst_conf = 0;
  for (int i = 0; i < kImages; ++i)
    for (int k = 0; k < kKeypoints; ++k) {
      const int j = i * kKeypoints + k;
      worst = std::max({worst, std::abs(out.kp[i][k][0] - kp[2 * j]), std::abs(out.kp[i][k][1] - kp[2 * j + 1])});
      worst_conf = std::max(worst_conf, std::abs(out.conf[i][k] - conf[j]));
    }
  std::printf("[parity] engine %s max |C++ - Python TRT| = %.3g px, conf %.3g\n", prec.c_str(), worst, worst_conf);
  EXPECT_LT(worst, tol_px);
  EXPECT_LT(worst_conf, 1e-3);
}
}  // namespace

TEST(Engine, Fp32MatchesPythonRunner) { check_engine("fp32", 1e-3); }
TEST(Engine, Fp16MatchesPythonRunner) { check_engine("fp16", 1e-3); }

TEST(Engine, RejectsMissingEngine) { EXPECT_THROW(TrtEngine("/nonexistent.engine"), std::runtime_error); }
