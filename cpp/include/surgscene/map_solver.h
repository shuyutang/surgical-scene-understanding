// Phase 3 shape prior + robust MAP (Levenberg-Marquardt with Cauchy IRLS), ported line by line from
// src/surgscene/shape.py and structured.py. Double precision, no allocation in the solver loop
// beyond fixed-size Eigen temporaries.
#pragma once
#include <Eigen/Dense>
#include <optional>
#include <string>

namespace surgscene {

constexpr int kKp = 5;  // keypoints per instrument

using Points = Eigen::Matrix<double, kKp, 2>;
using Vec5 = Eigen::Matrix<double, kKp, 1>;

struct ShapeModel {
  Points mean;             // (K, 2)
  Eigen::MatrixXd P;       // (2K, m) PCA modes, rows x0, y0, x1, y1, ...
  Eigen::VectorXd lam;     // (m,) mode variances
  int modes() const { return static_cast<int>(lam.size()); }
  Points shape(const Eigen::VectorXd& b) const;

  static ShapeModel load(const std::string& dir, const std::string& arm);
};

struct Similarity2D {
  double s;
  Eigen::Matrix2d R;
  Eigen::Vector2d t;
};

// Weighted least-squares similarity dst ~= s R src + t (Umeyama), reflection-guarded.
Similarity2D umeyama2d(const Eigen::MatrixX2d& src, const Eigen::MatrixX2d& dst, const Eigen::VectorXd& w);

struct MapParams {
  double conf_min = 0.05, beta = 0.3, cauchy_c = 5.0, tau = 0.2;
  int iters = 30;
};

// Continuous robust MAP under the shape prior; nullopt if fewer than 2 keypoints carry weight.
std::optional<Points> map_fit(const ShapeModel& m, const Points& z, const Vec5& sigma, const Vec5& w,
                              double beta, double cauchy_c, int iters = 30);

// Gated estimate: argmax where conf >= tau, MAP elsewhere (falls back to the observations if MAP fails).
Points fit_gated(const ShapeModel& m, const Points& z, const Vec5& conf, const Vec5& sigma, const MapParams& p);

}  // namespace surgscene
