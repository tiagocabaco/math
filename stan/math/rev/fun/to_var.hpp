#ifndef STAN_MATH_REV_FUN_TO_VAR_HPP
#define STAN_MATH_REV_FUN_TO_VAR_HPP

#include <stan/math/rev/core.hpp>

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/rev/fun/typedefs.hpp>

#include <vector>







namespace stan {
namespace math {

/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] x A scalar value
 * @return An automatic differentiation variable with the input value.
 */
inline var to_var(double x) { return var(x); }

/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] x An automatic differentiation variable.
 * @return An automatic differentiation variable with the input value.
 */
inline var to_var(const var& x) { return x; }














/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] v A std::vector<double>
 * @return A std::vector<var> with the values set
 */
inline std::vector<var> to_var(const std::vector<double>& v) {
  std::vector<var> var_vector(v.size());
  for (size_t n = 0; n < v.size(); n++)
    var_vector[n] = v[n];
  return var_vector;
}

/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] v A std::vector<var>
 * @return A std::vector<var>
 */
inline std::vector<var> to_var(const std::vector<var>& v) { return v; }
















/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] m A Matrix with scalars
 * @return A Matrix with automatic differentiation variables
 */
inline matrix_v to_var(const matrix_d& m) {
  matrix_v m_v(m.rows(), m.cols());
  for (int j = 0; j < m.cols(); ++j)
    for (int i = 0; i < m.rows(); ++i)
      m_v(i, j) = m(i, j);
  return m_v;
}
/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] m A Matrix with automatic differentiation variables.
 * @return A Matrix with automatic differentiation variables.
 */
inline matrix_v to_var(const matrix_v& m) { return m; }
/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] v A Vector of scalars
 * @return A Vector of automatic differentiation variables with
 *   values of v
 */
inline vector_v to_var(const vector_d& v) {
  vector_v v_v(v.size());
  for (int i = 0; i < v.size(); ++i)
    v_v[i] = v[i];
  return v_v;
}
/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] v A Vector of automatic differentiation variables
 * @return A Vector of automatic differentiation variables with
 *   values of v
 */
inline vector_v to_var(const vector_v& v) { return v; }
/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] rv A row vector of scalars
 * @return A row vector of automatic differentation variables with
 *   values of rv.
 */
inline row_vector_v to_var(const row_vector_d& rv) {
  row_vector_v rv_v(rv.size());
  for (int i = 0; i < rv.size(); ++i)
    rv_v[i] = rv[i];
  return rv_v;
}
/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] rv A row vector with automatic differentiation variables
 * @return A row vector with automatic differentiation variables
 *    with values of rv.
 */
inline row_vector_v to_var(const row_vector_v& rv) { return rv; }

}  // namespace math
}  // namespace stan
#endif
