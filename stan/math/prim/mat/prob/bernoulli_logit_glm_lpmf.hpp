#ifndef STAN_MATH_PRIM_MAT_PROB_BERNOULLI_LOGIT_GLM_LPMF_HPP
#define STAN_MATH_PRIM_MAT_PROB_BERNOULLI_LOGIT_GLM_LPMF_HPP

#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/broadcast_array.hpp>
#include <stan/math/prim/mat/meta/assign_to_matrix_or_broadcast_array.hpp>
#include <stan/math/prim/scal/meta/operands_and_partials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_consistent_size.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/mat/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/scal/meta/scalar_seq_view.hpp>
#include <stan/math/prim/scal/fun/size_zero.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Returns the log PMF of the Generalized Linear Model (GLM)
 * with Bernoulli distribution and logit link function.
 * The idea is that bernoulli_logit_glm_lpmf(y, x, alpha, beta) should
 * compute a more efficient version of bernoulli_logit_lpmf(y, alpha + x * beta)
 * by using analytically simplified gradients.
 * If containers are supplied, returns the log sum of the probabilities.
 * @tparam T_y type of binary vector of dependent variables (labels);
 * this can also be a single binary value;
 * @tparam T_x type of the matrix of independent variables (features); this
 * should be an Eigen::Matrix type whose number of rows should match the
 * length of y and whose number of columns should match the length of beta
 * @tparam T_alpha type of the intercept(s);
 * this can be a vector (of the same length as y) of intercepts or a single
 * value (for models with constant intercept);
 * @tparam T_beta type of the weight vector;
 * this can also be a single value;
 * @param y binary vector parameter
 * @param x design matrix
 * @param alpha intercept (in log odds)
 * @param beta weight vector
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if x, beta or alpha is infinite.
 * @throw std::domain_error if y is not binary.
 * @throw std::invalid_argument if container sizes mismatch.
 */

template <bool propto, typename T_y, typename T_x, typename T_alpha,
          typename T_beta>
typename return_type<T_x, T_alpha, T_beta>::type bernoulli_logit_glm_lpmf(
    const T_y &y, const T_x &x, const T_alpha &alpha, const T_beta &beta) {
  static const char *function = "bernoulli_logit_glm_lpmf";
  typedef typename stan::partials_return_type<T_y, T_x, T_alpha, T_beta>::type
      T_partials_return;

  using Eigen::Dynamic;
  using Eigen::Matrix;
  using std::exp;

  if (size_zero(y, x, beta))
    return 0.0;

  T_partials_return logp(0.0);

  check_bounded(function, "Vector of dependent variables", y, 0, 1);
  check_finite(function, "Matrix of independent variables", x);
  check_finite(function, "Weight vector", beta);
  check_finite(function, "Intercept", alpha);
  check_consistent_size(function, "Vector of dependent variables", y,
                        x.col(0).size());
  check_consistent_size(function, "Weight vector", beta, x.row(0).size());
  if (is_vector<T_alpha>::value)
    check_consistent_sizes(function, "Vector of intercepts", alpha,
                           "Vector of dependent variables", y);

  if (!include_summand<propto, T_x, T_alpha, T_beta>::value)
    return 0.0;

  const size_t N = x.col(0).size();
  const size_t M = x.row(0).size();

  Matrix<T_partials_return, Dynamic, 1> signs(N, 1);
  {
    scalar_seq_view<T_y> y_vec(y);
    for (size_t n = 0; n < N; ++n) {
      signs[n] = 2 * y_vec[n] - 1;
    }
  }
  Matrix<T_partials_return, Dynamic, 1> beta_dbl(M, 1);
  {
    scalar_seq_view<T_beta> beta_vec(beta);
    for (size_t m = 0; m < M; ++m) {
      beta_dbl[m] = value_of(beta_vec[m]);
    }
  }
  Eigen::Array<T_partials_return, Dynamic, 1> ytheta
      = signs.array() * (value_of(x) * beta_dbl).array();
  scalar_seq_view<T_alpha> alpha_vec(alpha);
  for (size_t m = 0; m < N; ++m)
    ytheta[m] += signs[m] * value_of(alpha_vec[m]);

  Eigen::Array<T_partials_return, Dynamic, 1> exp_m_ytheta = (-ytheta).exp();

  static const double cutoff = 20.0;
  for (size_t n = 0; n < N; ++n) {
    // Compute the log-density and handle extreme values gracefully
    // using Taylor approximations.
    if (ytheta[n] > cutoff)
      logp -= exp_m_ytheta[n];
    else if (ytheta[n] < -cutoff)
      logp += ytheta[n];
    else
      logp -= log1p(exp_m_ytheta[n]);
  }

  // Compute the necessary derivatives.
  operands_and_partials<T_x, T_alpha, T_beta> ops_partials(x, alpha, beta);
  if (!(is_constant_struct<T_x>::value && is_constant_struct<T_beta>::value
        && is_constant_struct<T_alpha>::value)) {
    Matrix<T_partials_return, Dynamic, 1> theta_derivative(N, 1);
    for (size_t n = 0; n < N; ++n) {
      if (ytheta[n] > cutoff)
        theta_derivative[n] = -exp_m_ytheta[n];
      else if (ytheta[n] < -cutoff)
        theta_derivative[n] = signs[n];
      else
        theta_derivative[n]
            = signs[n] * exp_m_ytheta[n] / (exp_m_ytheta[n] + 1);
    }
    if (!is_constant_struct<T_beta>::value) {
      assign_to_matrix_or_broadcast_array(
          ops_partials.edge3_.partials_,
          value_of(x).transpose() * theta_derivative);
    }
    if (!is_constant_struct<T_x>::value) {
      ops_partials.edge1_.partials_ = theta_derivative * beta_dbl.transpose();
    }
    if (!is_constant_struct<T_alpha>::value) {
      if (is_vector<T_alpha>::value)
        assign_to_matrix_or_broadcast_array(ops_partials.edge2_.partials_,
                                            theta_derivative);
      else
        ops_partials.edge2_.partials_[0] = theta_derivative.sum();
    }
  }

  return ops_partials.build(logp);
}

template <typename T_y, typename T_x, typename T_alpha, typename T_beta>
inline typename return_type<T_x, T_beta, T_alpha>::type
bernoulli_logit_glm_lpmf(const T_y &y, const T_x &x, const T_alpha &alpha,
                         const T_beta &beta) {
  return bernoulli_logit_glm_lpmf<false>(y, x, alpha, beta);
}
}  // namespace math
}  // namespace stan
#endif
