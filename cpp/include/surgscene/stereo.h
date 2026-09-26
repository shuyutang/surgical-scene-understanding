// Phase 5: camera model with Brown-Conrady distortion, stereo triangulation with first-order
// covariance, SO(3) utilities. Port of src/surgscene/geometry.py.
#pragma once
#include <Eigen/Dense>
#include <optional>

namespace surgscene {

struct Camera {
  Eigen::Matrix3d K;
  Eigen::Matrix<double, 5, 1> dist;  // k1, k2, p1, p2, k3
  Eigen::Vector2d distort(const Eigen::Vector2d& m) const;
  Eigen::Vector2d project(const Eigen::Vector3d& X) const;
  Eigen::Vector2d undistort(const Eigen::Vector2d& u, int iters = 20) const;
};

struct StereoRig {
  Camera left, right;
  Eigen::Matrix3d R;  // X_right = R X_left + T (mm)
  Eigen::Vector3d T;
};

struct Triangulated {
  Eigen::Vector3d X;
  Eigen::Matrix3d cov;
  double reproj_px;
};

Eigen::Vector3d triangulate_dlt(const Eigen::Vector2d& mL, const Eigen::Vector2d& mR, const Eigen::Matrix3d& R,
                                const Eigen::Vector3d& T);
// DLT init + Gauss-Newton on pixel reprojection error; nullopt outside the plausible depth range.
std::optional<Triangulated> triangulate(const StereoRig& rig, const Eigen::Vector2d& uL, const Eigen::Vector2d& uR,
                                        double sigL, double sigR, double z_min = 20.0, double z_max = 2000.0);

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w);
Eigen::Vector3d so3_log(const Eigen::Matrix3d& R);

}  // namespace surgscene
