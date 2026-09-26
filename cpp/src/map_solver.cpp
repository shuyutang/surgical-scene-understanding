#include "surgscene/map_solver.h"

#include <cmath>
#include <stdexcept>

#include "surgscene/npy.h"

namespace surgscene {

Points ShapeModel::shape(const Eigen::VectorXd& b) const {
  Eigen::VectorXd d = P * b;
  Points Y = mean;
  for (int k = 0; k < kKp; ++k) {
    Y(k, 0) += d(2 * k);
    Y(k, 1) += d(2 * k + 1);
  }
  return Y;
}

ShapeModel ShapeModel::load(const std::string& dir, const std::string& arm) {
  auto mean = load_npy(dir + "/shape_" + arm + "_mean.npy");
  auto P = load_npy(dir + "/shape_" + arm + "_P.npy");
  auto lam = load_npy(dir + "/shape_" + arm + "_lam.npy");
  if (mean.shape != std::vector<size_t>{kKp, 2}) throw std::runtime_error("shape model: bad mean shape");
  ShapeModel m;
  auto mv = mean.as_double();
  for (int k = 0; k < kKp; ++k) m.mean.row(k) << mv[2 * k], mv[2 * k + 1];
  const size_t rows = P.shape[0], cols = P.shape[1];
  auto pv = P.as_double();
  m.P.resize(rows, cols);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j) m.P(i, j) = pv[i * cols + j];
  auto lv = lam.as_double();
  m.lam = Eigen::Map<Eigen::VectorXd>(lv.data(), lv.size());
  return m;
}

Similarity2D umeyama2d(const Eigen::MatrixX2d& src, const Eigen::MatrixX2d& dst, const Eigen::VectorXd& w_in) {
  Eigen::VectorXd w = w_in / w_in.sum();
  Eigen::RowVector2d mu_s = w.transpose() * src, mu_d = w.transpose() * dst;
  Eigen::MatrixX2d S = src.rowwise() - mu_s, D = dst.rowwise() - mu_d;
  Eigen::Matrix2d cov = (D.array().colwise() * w.array()).matrix().transpose() * S;
  Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d E = Eigen::Matrix2d::Identity();
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) E(1, 1) = -1;
  Similarity2D r;
  r.R = svd.matrixU() * E * svd.matrixV().transpose();
  double var_s = (w.array() * S.rowwise().squaredNorm().array()).sum();
  r.s = (svd.singularValues().array() * E.diagonal().array()).sum() / var_s;
  r.t = mu_d.transpose() - r.s * r.R * mu_s.transpose();
  return r;
}

namespace {

// params p = [a, c, tx, ty, b...]; x_k = [[a, -c], [c, a]] Y_k(b) + t
Points params_to_points(const ShapeModel& m, const Eigen::VectorXd& p) {
  Points Y = m.shape(p.tail(m.modes()));
  Points X;
  X.col(0) = p(0) * Y.col(0) - p(1) * Y.col(1) + Eigen::Matrix<double, kKp, 1>::Constant(p(2));
  X.col(1) = p(1) * Y.col(0) + p(0) * Y.col(1) + Eigen::Matrix<double, kKp, 1>::Constant(p(3));
  return X;
}

Eigen::MatrixXd jacobian(const ShapeModel& m, const Eigen::VectorXd& p) {
  const double a = p(0), c = p(1);
  Points Y = m.shape(p.tail(m.modes()));
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2 * kKp, 4 + m.modes());
  for (int k = 0; k < kKp; ++k) {
    J(2 * k, 0) = Y(k, 0);
    J(2 * k + 1, 0) = Y(k, 1);
    J(2 * k, 1) = -Y(k, 1);
    J(2 * k + 1, 1) = Y(k, 0);
    J(2 * k, 2) = 1;
    J(2 * k + 1, 3) = 1;
    for (int j = 0; j < m.modes(); ++j) {
      const double px = m.P(2 * k, j), py = m.P(2 * k + 1, j);
      J(2 * k, 4 + j) = a * px - c * py;
      J(2 * k + 1, 4 + j) = c * px + a * py;
    }
  }
  return J;
}

}  // namespace

std::optional<Points> map_fit(const ShapeModel& m, const Points& z, const Vec5& sigma, const Vec5& w,
                              double beta, double cauchy_c, int iters) {
  // init: weighted similarity from the mean shape to the weighted observations
  int n_use = 0;
  for (int k = 0; k < kKp; ++k) n_use += w(k) > 0;
  if (n_use < 2) return std::nullopt;
  Eigen::MatrixX2d src(n_use, 2), dst(n_use, 2);
  Eigen::VectorXd ww(n_use);
  for (int k = 0, i = 0; k < kKp; ++k)
    if (w(k) > 0) {
      src.row(i) = m.mean.row(k);
      dst.row(i) = z.row(k);
      ww(i++) = w(k);
    }
  const Similarity2D sim = umeyama2d(src, dst, ww);
  const int np = 4 + m.modes();
  Eigen::VectorXd p = Eigen::VectorXd::Zero(np);
  p(0) = sim.s * sim.R(0, 0);
  p(1) = sim.s * sim.R(1, 0);
  p(2) = sim.t(0);
  p(3) = sim.t(1);
  Eigen::VectorXd prior = Eigen::VectorXd::Zero(np);
  prior.tail(m.modes()) = beta * m.lam.cwiseInverse();
  const double c2 = cauchy_c * cauchy_c;

  auto energy = [&](const Eigen::VectorXd& q, Vec5& r2) {
    Points X = params_to_points(m, q);
    r2 = ((X - z).array().colwise() / sigma.array()).square().rowwise().sum();
    double data = (w.array() * c2 * (r2.array() / c2).log1p()).sum();
    return data + (prior.array() * q.array().square()).sum();
  };

  Vec5 r2;
  double E = energy(p, r2);
  double mu = 1e-3;
  for (int it = 1; it <= iters; ++it) {
    Vec5 irls = w.array() / (1 + r2.array() / c2);
    Eigen::VectorXd Wk(2 * kKp);
    for (int k = 0; k < kKp; ++k) Wk(2 * k) = Wk(2 * k + 1) = irls(k) / (sigma(k) * sigma(k));
    Eigen::MatrixXd J = jacobian(m, p);
    Points X = params_to_points(m, p);
    Eigen::VectorXd res(2 * kKp);
    for (int k = 0; k < kKp; ++k) {
      res(2 * k) = X(k, 0) - z(k, 0);
      res(2 * k + 1) = X(k, 1) - z(k, 1);
    }
    Eigen::MatrixXd H = J.transpose() * Wk.asDiagonal() * J;
    H.diagonal() += prior;
    Eigen::VectorXd g = J.transpose() * (Wk.array() * res.array()).matrix() + (prior.array() * p.array()).matrix();
    bool improved = false;
    Eigen::VectorXd step;
    for (int tries = 0; tries < 10; ++tries) {
      Eigen::MatrixXd A = H;
      A.diagonal() += mu * (H.diagonal().array() + 1e-9).matrix();
      step = A.partialPivLu().solve(-g);
      Vec5 r2n;
      double En = energy(p + step, r2n);
      if (En < E) {
        p += step;
        E = En;
        r2 = r2n;
        mu = std::max(mu / 3, 1e-7);
        improved = true;
        break;
      }
      mu *= 4;
    }
    if (!improved || step.cwiseAbs().maxCoeff() < 1e-6) break;
  }
  Points out = params_to_points(m, p);
  if (!out.allFinite()) return std::nullopt;  // non-convergence guard: caller falls back
  return out;
}

Points fit_gated(const ShapeModel& m, const Points& z, const Vec5& conf, const Vec5& sigma, const MapParams& p) {
  Vec5 w = (conf.array() >= p.conf_min).select(conf, 0.0);
  auto fit = map_fit(m, z, sigma, w, p.beta, p.cauchy_c, p.iters);
  Points out = fit ? *fit : z;
  for (int k = 0; k < kKp; ++k)
    if (conf(k) >= p.tau) out.row(k) = z.row(k);
  return out;
}

}  // namespace surgscene
