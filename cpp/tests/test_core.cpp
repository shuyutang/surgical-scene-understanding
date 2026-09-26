// D3: C++ core vs the Python reference, on goldens written by scripts/eval_deploy.py.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "surgscene/kalman.h"
#include "surgscene/map_solver.h"
#include "surgscene/npy.h"
#include "surgscene/stereo.h"

using namespace surgscene;

namespace {
const std::string G = GOLDEN_DIR;
std::vector<double> ld(const std::string& name) { return load_npy(G + "/" + name + ".npy").as_double(); }

StereoRig load_rig() {
  auto K = ld("rig_K"), D = ld("rig_dist"), R = ld("rig_R"), T = ld("rig_T");
  StereoRig r;
  Camera* cams[2] = {&r.left, &r.right};
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < 9; ++i) cams[c]->K(i / 3, i % 3) = K[c * 9 + i];
    for (int i = 0; i < 5; ++i) cams[c]->dist(i) = D[c * 5 + i];
  }
  for (int i = 0; i < 9; ++i) r.R(i / 3, i % 3) = R[i];
  for (int i = 0; i < 3; ++i) r.T(i) = T[i];
  return r;
}
}  // namespace

TEST(MapSolver, GatedMatchesPython) {
  const ShapeModel models[2] = {ShapeModel::load(G, "PSM1"), ShapeModel::load(G, "PSM3")};
  auto z = ld("map_z"), conf = ld("map_conf"), sigma = ld("map_sigma"), ref = ld("map_out");
  const size_t n = conf.size() / 10;
  MapParams p;  // defaults are the tune-selected values; checked against params.json in the Python test
  double worst = 0;
  for (size_t i = 0; i < n; ++i)
    for (int arm = 0; arm < 2; ++arm) {
      Points zz;
      Vec5 c, s;
      for (int k = 0; k < kKp; ++k) {
        const size_t j = i * 10 + arm * kKp + k;
        zz.row(k) << z[2 * j], z[2 * j + 1];
        c(k) = conf[j];
        s(k) = sigma[j];
      }
      const Points out = fit_gated(models[arm], zz, c, s, p);
      for (int k = 0; k < kKp; ++k) {
        const size_t j = i * 10 + arm * kKp + k;
        if (std::isfinite(ref[2 * j]))
          worst = std::max({worst, std::abs(out(k, 0) - ref[2 * j]), std::abs(out(k, 1) - ref[2 * j + 1])});
      }
    }
  // Python initializes in partly float32 arithmetic (float32 observations); the LM optimum is the same
  std::printf("[parity] gated MAP max |C++ - Python| = %.3g px\n", worst);
  EXPECT_LT(worst, 1e-3) << "max |C++ - Python| px";
}

TEST(MapSolver, UmeyamaRecoversSimilarityAndRejectsReflection) {
  Eigen::MatrixX2d src(6, 2);
  src << 0, 0, 1, 0, 2, 1, 3, 3, 1, 4, -1, 2;
  const double s = 2.5, th = 0.7;
  Eigen::Matrix2d R;
  R << std::cos(th), -std::sin(th), std::sin(th), std::cos(th);
  Eigen::MatrixX2d dst = (s * src * R.transpose()).rowwise() + Eigen::RowVector2d(3, -4);
  auto r = umeyama2d(src, dst, Eigen::VectorXd::Ones(6));
  EXPECT_NEAR(r.s, s, 1e-12);
  EXPECT_TRUE(r.R.isApprox(R, 1e-12));
  Eigen::MatrixX2d mirrored = dst;
  mirrored.col(0) *= -1;
  EXPECT_GT(umeyama2d(src, mirrored, Eigen::VectorXd::Ones(6)).R.determinant(), 0);
}

TEST(Kalman, MatchesPython) {
  auto z = ld("kf_z"), s = ld("kf_sigma"), c = ld("kf_conf"), pos = ld("kf_pos"), sd = ld("kf_std");
  const size_t T = c.size() / 10;
  KFParams p;
  double worst = 0, worst_std = 0;
  for (int k = 0; k < 10; ++k) {
    KeypointKF f(p);
    for (size_t t = 0; t < T; ++t) {
      const size_t j = t * 10 + k;
      f.step(Eigen::Vector2d(z[2 * j], z[2 * j + 1]), s[j], c[j]);
      if (std::isfinite(pos[2 * j])) {
        ASSERT_TRUE(f.alive());
        worst = std::max({worst, std::abs(f.position()(0) - pos[2 * j]), std::abs(f.position()(1) - pos[2 * j + 1])});
        worst_std = std::max(worst_std, std::abs(f.position_std() - sd[j]));
      } else {
        ASSERT_FALSE(f.alive());
      }
    }
  }
  std::printf("[parity] Kalman max |C++ - Python| = %.3g px (std %.3g px)\n", worst, worst_std);
  EXPECT_LT(worst, 1e-9);
  EXPECT_LT(worst_std, 1e-9);
}

TEST(Kalman, PredictsThroughMissingMeasurements) {
  KeypointKF f(KFParams{0.5, 1.0, 0.05, 10.0, 3});
  for (int t = 0; t < 50; ++t) f.step(Eigen::Vector2d(100 + 2.0 * t, 50), 1.0, 1.0);
  const double s0 = f.position_std();
  for (int t = 50; t < 60; ++t) f.step(Eigen::Vector2d(NAN, NAN), 1.0, 0.0);
  EXPECT_GT(f.position_std(), s0);
  EXPECT_NEAR(f.position()(0), 100 + 2.0 * 59, 1.0);  // constant-velocity extrapolation
}

TEST(Stereo, TriangulationMatchesPython) {
  const StereoRig rig = load_rig();
  auto uL = ld("tri_uL"), uR = ld("tri_uR"), sL = ld("tri_sL"), sR = ld("tri_sR"), X = ld("tri_X"), C = ld("tri_cov");
  const size_t n = sL.size();
  int compared = 0;
  double worst = 0, worst_cov = 0;
  for (size_t j = 0; j < n; ++j) {
    auto t = triangulate(rig, {uL[2 * j], uL[2 * j + 1]}, {uR[2 * j], uR[2 * j + 1]}, std::max(sL[j], 0.5),
                         std::max(sR[j], 0.5));
    if (!std::isfinite(X[3 * j])) {
      EXPECT_FALSE(t.has_value());
      continue;
    }
    ASSERT_TRUE(t.has_value());
    ++compared;
    for (int a = 0; a < 3; ++a) worst = std::max(worst, std::abs(t->X(a) - X[3 * j + a]));
    for (int a = 0; a < 9; ++a)
      worst_cov = std::max(worst_cov, std::abs(t->cov(a / 3, a % 3) - C[9 * j + a]) / (std::abs(C[9 * j + a]) + 1e-9));
  }
  std::printf("[parity] triangulation max |C++ - Python| = %.3g mm, cov rel %.3g (%d points)\n", worst, worst_cov, compared);
  EXPECT_GT(compared, 400);
  EXPECT_LT(worst, 1e-6) << "mm";
  EXPECT_LT(worst_cov, 1e-4) << "relative";
}

TEST(Stereo, ProjectionRoundTrip) {
  const StereoRig rig = load_rig();
  const Eigen::Vector3d X(12.0, -8.0, 180.0);
  const auto t = triangulate(rig, rig.left.project(X), rig.right.project(rig.R * X + rig.T), 1, 1);
  ASSERT_TRUE(t.has_value());
  EXPECT_LT((t->X - X).norm(), 1e-6);
  EXPECT_LT(t->reproj_px, 1e-6);
  EXPECT_GT(t->cov(2, 2), 10 * t->cov(0, 0));  // depth is the weak direction
}

TEST(SO3, ExpLogRoundTripIncludingNearPi) {
  std::mt19937 g(0);
  std::normal_distribution<double> n(0, 1.5);
  for (int i = 0; i < 200; ++i) {
    const Eigen::Vector3d w(n(g), n(g), n(g));
    const Eigen::Matrix3d R = so3_exp(w);
    EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
    EXPECT_TRUE(so3_exp(so3_log(R)).isApprox(R, 1e-9));
  }
  for (double th : {1e-9, M_PI - 1e-6, M_PI}) {
    const Eigen::Vector3d a = Eigen::Vector3d(1, 2, -0.5).normalized();
    const Eigen::Matrix3d R = so3_exp(a * th);
    EXPECT_TRUE(so3_exp(so3_log(R)).isApprox(R, 1e-6));
  }
}
