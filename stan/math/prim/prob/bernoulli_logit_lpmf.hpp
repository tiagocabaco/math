#ifndef STAN_MATH_PRIM_PROB_BERNOULLI_LOGIT_LPMF_HPP
#define STAN_MATH_PRIM_PROB_BERNOULLI_LOGIT_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/log1p.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <cmath>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Returns the log PMF of the logit-parametrized Bernoulli distribution. If
 * containers are supplied, returns the log sum of the probabilities.
 *
 * @tparam T_n type of integer parameter
 * @tparam T_prob type of chance of success parameter
 * @param n integer parameter
 * @param theta logit-transformed chance of success parameter
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if theta is infinite.
 * @throw std::invalid_argument if container sizes mismatch.
 */
template <bool propto, typename T_n, typename T_prob>
return_type_t<T_prob> bernoulli_logit_lpmf(const T_n& n, const T_prob& theta) {
  using T_partials_return = partials_return_t<T_n, T_prob>;
  using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
  using std::exp;
  using T_n_ref = ref_type_t<T_n>;
  using T_theta_ref = ref_type_t<T_prob>;
  static const char* function = "bernoulli_logit_lpmf";
  check_consistent_sizes(function, "Random variable", n,
                         "Probability parameter", theta);
  T_n_ref n_ref = n;
  T_theta_ref theta_ref = theta;
  check_bounded(function, "n", n_ref, 0, 1);
  check_not_nan(function, "Logit transformed probability parameter", theta_ref);

  if (size_zero(n, theta)) {
    return 0.0;
  }
  if (!include_summand<propto, T_prob>::value) {
    return 0.0;
  }

  T_partials_return logp(0.0);
  operands_and_partials<T_theta_ref> ops_partials(theta_ref);

  const auto& theta_col = as_column_vector_or_scalar(theta_ref);
  const auto& theta_val = value_of(theta_col);
  const auto& theta_arr = as_array_or_scalar(theta_val);

  const auto& n_col = as_column_vector_or_scalar(n_ref);
  const auto& n_double = value_of_rec(n_col);

  auto signs = to_ref_if<!is_constant<T_prob>::value>(
      (2 * as_array_or_scalar(n_double) - 1));
  T_partials_array ntheta;
  if (is_vector<T_n>::value || is_vector<T_prob>::value) {
    ntheta = forward_as<T_partials_array>(signs * theta_arr);
  } else {
    T_partials_return ntheta_s
        = forward_as<T_partials_return>(signs * theta_arr);
    ntheta = T_partials_array::Constant(1, 1, ntheta_s);
  }
  T_partials_array exp_m_ntheta = exp(-ntheta);
  static const double cutoff = 20.0;
  logp += sum(
      (ntheta > cutoff)
          .select(-exp_m_ntheta,
                  (ntheta < -cutoff).select(ntheta, -log1p(exp_m_ntheta))));

  if (!is_constant_all<T_prob>::value) {
    if (is_vector<T_prob>::value) {
      ops_partials.edge1_.partials_ = forward_as<T_partials_array>(
          (ntheta > cutoff)
              .select(-exp_m_ntheta,
                      (ntheta >= -cutoff)
                          .select(signs * exp_m_ntheta / (exp_m_ntheta + 1),
                                  signs)));
    } else {
      ops_partials.edge1_.partials_[0]
          = sum((ntheta > cutoff)
                    .select(-exp_m_ntheta, (ntheta >= -cutoff)
                                               .select(signs * exp_m_ntheta
                                                           / (exp_m_ntheta + 1),
                                                       signs)));
    }
  }
  return ops_partials.build(logp);
}

template <typename T_n, typename T_prob>
inline return_type_t<T_prob> bernoulli_logit_lpmf(const T_n& n,
                                                  const T_prob& theta) {
  return bernoulli_logit_lpmf<false>(n, theta);
}

}  // namespace math
}  // namespace stan
#endif
