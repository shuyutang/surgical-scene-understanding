// Phase 4: constant-velocity Kalman filter per keypoint (port of src/surgscene/temporal.py, causal part).
#pragma once
#include <Eigen/Dense>

namespace surgscene {

struct KFParams {
  double q = 2.0, r0 = 2.0, conf_min = 0.05, unknown_px = 20.0;
  int max_rejects = 3;
};

class KeypointKF {
 public:
  explicit KeypointKF(const KFParams& p = {}) : p_(p) {}
  // One frame. z = measurement (NaN if missing), sigma = Laplace std (px), conf = peak probability.
  void step(const Eigen::Vector2d& z, double sigma, double conf);
  bool alive() const { return alive_; }
  Eigen::Vector2d position() const { return x_.head<2>(); }
  double position_std() const { return std::sqrt(std::max(P_(0, 0), P_(1, 1))); }
  bool unknown() const { return !alive_ || position_std() > p_.unknown_px; }
  bool last_updated() const { return updated_; }
  double last_nis() const { return nis_; }

 private:
  void init(const Eigen::Vector2d& z, const Eigen::Matrix2d& R);
  KFParams p_;
  Eigen::Vector4d x_ = Eigen::Vector4d::Zero();
  Eigen::Matrix4d P_ = Eigen::Matrix4d::Identity();
  bool alive_ = false, updated_ = false;
  int rejects_ = 0;
  double nis_ = 0;
};

}  // namespace surgscene
