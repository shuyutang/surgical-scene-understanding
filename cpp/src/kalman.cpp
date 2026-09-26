#include "surgscene/kalman.h"

#include <cmath>

namespace surgscene {

namespace {
constexpr double kChi2_2_999 = 13.8155;

Eigen::Matrix4d F() {
  Eigen::Matrix4d f = Eigen::Matrix4d::Identity();
  f(0, 2) = f(1, 3) = 1.0;
  return f;
}

Eigen::Matrix4d Q(double q) {
  Eigen::Matrix4d m;
  m << 0.25, 0, 0.5, 0, 0, 0.25, 0, 0.5, 0.5, 0, 1, 0, 0, 0.5, 0, 1;
  return q * q * m;
}
}  // namespace

void KeypointKF::init(const Eigen::Vector2d& z, const Eigen::Matrix2d& R) {
  x_ << z(0), z(1), 0.0, 0.0;
  P_.setZero();
  P_.diagonal() << R(0, 0), R(1, 1), 100.0, 100.0;
}

void KeypointKF::step(const Eigen::Vector2d& z, double sigma, double conf) {
  updated_ = false;
  const bool valid = z.allFinite() && conf >= p_.conf_min;
  Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
  if (valid) R.diagonal().setConstant((p_.r0 * sigma) * (p_.r0 * sigma) / std::max(conf, 1e-3));
  if (!alive_) {
    if (valid) {
      init(z, R);
      alive_ = updated_ = true;
    }
    return;
  }
  static const Eigen::Matrix4d Fm = F();
  x_ = Fm * x_;
  P_ = Fm * P_ * Fm.transpose() + Q(p_.q);
  if (!valid) return;
  const Eigen::Vector2d nu = z - x_.head<2>();
  const Eigen::Matrix2d S = P_.topLeftCorner<2, 2>() + R;
  const double d2 = nu.dot(S.ldlt().solve(nu));
  if (d2 <= kChi2_2_999) {
    const Eigen::Matrix<double, 4, 2> K = P_.leftCols<2>() * S.inverse();
    x_ += K * nu;
    Eigen::Matrix4d IKH = Eigen::Matrix4d::Identity();
    IKH.leftCols<2>() -= K;
    P_ = IKH * P_;
    updated_ = true;
    nis_ = d2;
    rejects_ = 0;
  } else if (++rejects_ >= p_.max_rejects) {  // the target really moved: re-initialize
    init(z, R);
    updated_ = true;
    rejects_ = 0;
  }
}

}  // namespace surgscene
