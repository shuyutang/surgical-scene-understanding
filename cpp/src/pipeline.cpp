#include "surgscene/pipeline.h"

#include <chrono>

namespace surgscene {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
constexpr int kTip[2][2] = {{3, 4}, {8, 9}};
}  // namespace

Pipeline::Pipeline(const std::string& engine_path, const std::string& model_dir, const StereoRig& rig,
                   const MapParams& mp, const KFParams& kp, double kappa)
    : engine_(engine_path),
      shapes_{ShapeModel::load(model_dir, "PSM1"), ShapeModel::load(model_dir, "PSM3")},
      rig_(rig),
      mp_(mp),
      kappa_(kappa) {
  for (auto& eye : kf_)
    for (auto& f : eye) f = KeypointKF(kp);
}

void Pipeline::process(PipelineOutput& out) {
  const auto t0 = Clock::now();
  float h2d, infer, d2h;
  engine_.run(out.raw, &h2d, &infer, &d2h);
  out.times.h2d = h2d;
  out.times.infer = infer;
  out.times.d2h = d2h;

  // gated MAP per eye and instrument
  auto t1 = Clock::now();
  std::array<std::array<Eigen::Vector2d, kKeypoints>, kImages> est;
  for (int eye = 0; eye < kImages; ++eye)
    for (int arm = 0; arm < 2; ++arm) {
      Points z;
      Vec5 conf, sigma;
      for (int k = 0; k < kKp; ++k) {
        const int j = arm * kKp + k;
        z.row(k) << out.raw.kp[eye][j][0], out.raw.kp[eye][j][1];
        conf(k) = out.raw.conf[eye][j];
        sigma(k) = out.raw.sigma[eye][j];
      }
      const Points g = fit_gated(shapes_[arm], z, conf, sigma, mp_);
      for (int k = 0; k < kKp; ++k) est[eye][arm * kKp + k] = g.row(k).transpose();
    }
  out.times.map = ms_since(t1);

  // Kalman per eye and keypoint
  t1 = Clock::now();
  for (int eye = 0; eye < kImages; ++eye)
    for (int j = 0; j < kKeypoints; ++j) {
      kf_[eye][j].step(est[eye][j], out.raw.sigma[eye][j], out.raw.conf[eye][j]);
      out.filtered[eye][j] = kf_[eye][j].alive() ? kf_[eye][j].position() : Eigen::Vector2d::Constant(NAN);
    }
  out.times.kalman = ms_since(t1);

  // triangulate the two jaw tips per instrument; the tip is their midpoint
  t1 = Clock::now();
  for (int arm = 0; arm < 2; ++arm) {
    TipEstimate tip;
    Eigen::Vector3d X = Eigen::Vector3d::Zero();
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    bool ok = true, unknown = false;
    for (int j : kTip[arm]) {
      const auto& L = kf_[0][j];
      const auto& R = kf_[1][j];
      if (!L.alive() || !R.alive()) {
        ok = false;
        break;
      }
      auto tri = triangulate(rig_, L.position(), R.position(), std::max(L.position_std(), 0.5),
                             std::max(R.position_std(), 0.5));
      if (!tri) {
        ok = false;
        break;
      }
      X += tri->X / 2;
      C += tri->cov / 4;
      unknown |= L.unknown() || R.unknown();
    }
    if (ok) {
      tip.valid = true;
      tip.X = X;
      tip.cov = kappa_ * C;
      tip.unknown = unknown;
    }
    out.tips[arm] = tip;
  }
  out.times.triangulate = ms_since(t1);
  out.times.total = ms_since(t0);
}

}  // namespace surgscene
