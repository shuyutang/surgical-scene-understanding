// Per stereo pair: engine -> gated MAP (per eye, per instrument) -> Kalman (per eye, per keypoint)
// -> triangulated 3D jaw tips with covariance. Mirrors the Python stage-2 pipeline.
#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>

#include "surgscene/kalman.h"
#include "surgscene/map_solver.h"
#include "surgscene/stereo.h"
#include "surgscene/trt_engine.h"

namespace surgscene {

struct StageTimes {
  double h2d = 0, infer = 0, d2h = 0, map = 0, kalman = 0, triangulate = 0, total = 0;  // ms
};

struct TipEstimate {
  bool valid = false;
  Eigen::Vector3d X = Eigen::Vector3d::Constant(NAN);  // left camera frame, mm
  Eigen::Matrix3d cov = Eigen::Matrix3d::Constant(NAN);  // kappa-inflated
  bool unknown = true;
};

struct PipelineOutput {
  EngineOutput raw;
  std::array<std::array<Eigen::Vector2d, kKeypoints>, kImages> filtered;  // [eye][keypoint] px
  std::array<TipEstimate, 2> tips;                                         // PSM1, PSM3
  StageTimes times;
};

class Pipeline {
 public:
  Pipeline(const std::string& engine_path, const std::string& model_dir, const StereoRig& rig, const MapParams& mp,
           const KFParams& kp, double kappa);
  uint8_t* input() { return engine_.input(); }
  void process(PipelineOutput& out);

 private:
  TrtEngine engine_;
  std::array<ShapeModel, 2> shapes_;
  StereoRig rig_;
  MapParams mp_;
  double kappa_;
  std::array<std::array<KeypointKF, kKeypoints>, kImages> kf_;
};

}  // namespace surgscene
