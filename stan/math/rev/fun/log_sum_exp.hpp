#ifndef STAN_MATH_REV_FUN_LOG_SUM_EXP_HPP
#define STAN_MATH_REV_FUN_LOG_SUM_EXP_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/calculate_chain.hpp>
#include <stan/math/prim/fun/log_sum_exp.hpp>


#include <stan/math/prim/fun/Eigen.hpp>
#include <vector>
#include <limits>














namespace stan {
namespace math {

namespace internal {

class log_sum_exp_vv_vari : public op_vv_vari {
 public:
  log_sum_exp_vv_vari(vari* avi, vari* bvi)
      : op_vv_vari(log_sum_exp(avi->val_, bvi->val_), avi, bvi) {}
  void chain() {
    avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
    bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_);
  }
};
class log_sum_exp_vd_vari : public op_vd_vari {
 public:
  log_sum_exp_vd_vari(vari* avi, double b)
      : op_vd_vari(log_sum_exp(avi->val_, b), avi, b) {}
  void chain() { avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_); }
};
class log_sum_exp_dv_vari : public op_dv_vari {
 public:
  log_sum_exp_dv_vari(double a, vari* bvi)
      : op_dv_vari(log_sum_exp(a, bvi->val_), a, bvi) {}
  void chain() { bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_); }
};

}  // namespace internal

/**
 * Returns the log sum of exponentials.
 */
inline var log_sum_exp(const var& a, const var& b) {
  return var(new internal::log_sum_exp_vv_vari(a.vi_, b.vi_));
}
/**
 * Returns the log sum of exponentials.
 */
inline var log_sum_exp(const var& a, double b) {
  return var(new internal::log_sum_exp_vd_vari(a.vi_, b));
}
/**
 * Returns the log sum of exponentials.
 */
inline var log_sum_exp(double a, const var& b) {
  return var(new internal::log_sum_exp_dv_vari(a, b.vi_));
}
















namespace internal {
inline double log_sum_exp_as_double(const std::vector<var>& x) {
  using std::exp;
  using std::log;
  using std::numeric_limits;
  double max = -numeric_limits<double>::infinity();
  for (size_t i = 0; i < x.size(); ++i)
    if (x[i] > max)
      max = x[i].val();
  double sum = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    if (x[i] != -numeric_limits<double>::infinity())
      sum += exp(x[i].val() - max);
  return max + log(sum);
}

class log_sum_exp_vector_vari : public op_vector_vari {
 public:
  explicit log_sum_exp_vector_vari(const std::vector<var>& x)
      : op_vector_vari(log_sum_exp_as_double(x), x) {}
  void chain() {
    for (size_t i = 0; i < size_; ++i) {
      vis_[i]->adj_ += adj_ * calculate_chain(vis_[i]->val_, val_);
    }
  }
};
}  // namespace internal

/**
 * Returns the log sum of exponentials.
 */
inline var log_sum_exp(const std::vector<var>& x) {
  return var(new internal::log_sum_exp_vector_vari(x));
}
















namespace internal {

// these function and the following class just translate
// log_sum_exp for std::vector for Eigen::Matrix

template <int R, int C>
inline double log_sum_exp_as_double(const Eigen::Matrix<var, R, C>& x) {
  using std::exp;
  using std::log;
  using std::numeric_limits;
  double max = -numeric_limits<double>::infinity();
  for (int i = 0; i < x.size(); ++i)
    if (x(i) > max)
      max = x(i).val();
  double sum = 0.0;
  for (int i = 0; i < x.size(); ++i)
    if (x(i) != -numeric_limits<double>::infinity())
      sum += exp(x(i).val() - max);
  return max + log(sum);
}

class log_sum_exp_matrix_vari : public op_matrix_vari {
 public:
  template <int R, int C>
  explicit log_sum_exp_matrix_vari(const Eigen::Matrix<var, R, C>& x)
      : op_matrix_vari(log_sum_exp_as_double(x), x) {}
  void chain() {
    for (size_t i = 0; i < size_; ++i) {
      vis_[i]->adj_ += adj_ * calculate_chain(vis_[i]->val_, val_);
    }
  }
};
}  // namespace internal

/**
 * Returns the log sum of exponentials.
 *
 * @param x matrix
 */
template <int R, int C>
inline var log_sum_exp(const Eigen::Matrix<var, R, C>& x) {
  return var(new internal::log_sum_exp_matrix_vari(x));
}

}  // namespace math
}  // namespace stan
#endif
