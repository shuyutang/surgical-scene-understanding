#include "surgscene/stereo.h"

#include <cmath>

namespace surgscene {

Eigen::Vector2d Camera::distort(const Eigen::Vector2d& m) const {
  const double k1 = dist(0), k2 = dist(1), p1 = dist(2), p2 = dist(3), k3 = dist(4);
  const double x = m(0), y = m(1), r2 = x * x + y * y;
  const double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  return {x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y};
}

Eigen::Vector2d Camera::project(const Eigen::Vector3d& X) const {
  const Eigen::Vector2d md = distort(X.head<2>() / X(2));
  return K.topLeftCorner<2, 2>() * md + K.block<2, 1>(0, 2);
}

Eigen::Vector2d Camera::undistort(const Eigen::Vector2d& u, int iters) const {
  const Eigen::Vector2d md = K.topLeftCorner<2, 2>().inverse() * (u - K.block<2, 1>(0, 2));
  Eigen::Vector2d m = md;
  const double k1 = dist(0), k2 = dist(1), p1 = dist(2), p2 = dist(3), k3 = dist(4);
  for (int i = 0; i < iters; ++i) {
    const double x = m(0), y = m(1), r2 = x * x + y * y;
    const double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    const double dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x), dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
    m << (md(0) - dx) / radial, (md(1) - dy) / radial;
  }
  return m;
}

Eigen::Vector3d triangulate_dlt(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Matrix3d& R,
                                const Eigen::Vector3d& T) {
  Eigen::Matrix<double, 3, 4> P1 = Eigen::Matrix<double, 3, 4>::Zero(), P2;
  P1.leftCols<3>().setIdentity();
  P2 << R, T;
  Eigen::Matrix4d A;
  A.row(0) = a(0) * P1.row(2) - P1.row(0);
  A.row(1) = a(1) * P1.row(2) - P1.row(1);
  A.row(2) = b(0) * P2.row(2) - P2.row(0);
  A.row(3) = b(1) * P2.row(2) - P2.row(1);
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d X = svd.matrixV().col(3);
  return X.head<3>() / X(3);
}

namespace {
Eigen::Vector4d proj_both(const StereoRig& rig, const Eigen::Vector3d& X) {
  Eigen::Vector4d u;
  u << rig.left.project(X), rig.right.project(rig.R * X + rig.T);
  return u;
}

Eigen::Matrix<double, 4, 3> jac_both(const StereoRig& rig, const Eigen::Vector3d& X, double h = 1e-4) {
  Eigen::Matrix<double, 4, 3> J;
  for (int k = 0; k < 3; ++k) {
    Eigen::Vector3d e = Eigen::Vector3d::Zero();
    e(k) = h;
    J.col(k) = (proj_both(rig, X + e) - proj_both(rig, X - e)) / (2 * h);
  }
  return J;
}
}  // namespace

std::optional<Triangulated> triangulate(const StereoRig& rig, const Eigen::Vector2d& uL, const Eigen::Vector2d& uR,
                                        double sigL, double sigR, double z_min, double z_max) {
  if (!uL.allFinite() || !uR.allFinite()) return std::nullopt;
  Eigen::Vector3d X = triangulate_dlt(rig.left.undistort(uL), rig.right.undistort(uR), rig.R, rig.T);
  Eigen::Vector4d z;
  z << uL, uR;
  const Eigen::Vector4d W(1 / (sigL * sigL), 1 / (sigL * sigL), 1 / (sigR * sigR), 1 / (sigR * sigR));
  for (int i = 0; i < 5; ++i) {
    const auto J = jac_both(rig, X);
    const Eigen::Vector4d r = proj_both(rig, X) - z;
    const Eigen::Vector3d dX = (J.transpose() * W.asDiagonal() * J).ldlt().solve(-J.transpose() * W.cwiseProduct(r));
    X += dX;
    if (dX.cwiseAbs().maxCoeff() < 1e-6) break;
  }
  if (!X.allFinite() || X(2) < z_min || X(2) > z_max) return std::nullopt;
  const auto J = jac_both(rig, X);
  const Eigen::Vector4d r = proj_both(rig, X) - z;
  return Triangulated{X, (J.transpose() * W.asDiagonal() * J).inverse(), std::sqrt(r.squaredNorm() / 4)};
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
  const double th = w.norm();
  Eigen::Matrix3d W;
  W << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  if (th < 1e-8) return Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
  return Eigen::Matrix3d::Identity() + std::sin(th) / th * W + (1 - std::cos(th)) / (th * th) * W * W;
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
  const double c = std::clamp((R.trace() - 1) / 2, -1.0, 1.0);
  const double th = std::acos(c);
  const Eigen::Vector3d v(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  if (th < 1e-6) return v / 2;
  if (M_PI - th < 1e-2) {  // near pi: axis from the symmetric part, sign from the skew part
    const Eigen::Matrix3d aaT = ((R + R.transpose()) / 2 - c * Eigen::Matrix3d::Identity()) / (1 - c);
    int k;
    aaT.diagonal().maxCoeff(&k);
    Eigen::Vector3d a = aaT.col(k) / std::sqrt(aaT(k, k));
    if (a.dot(v) < 0) a = -a;
    return a * th;
  }
  return th / (2 * std::sin(th)) * v;
}

}  // namespace surgscene
