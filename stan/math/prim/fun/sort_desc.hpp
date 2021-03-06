#ifndef STAN_MATH_PRIM_FUN_SORT_DESC_HPP
#define STAN_MATH_PRIM_FUN_SORT_DESC_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <algorithm>
#include <functional>
#include <vector>

namespace stan {
namespace math {

/**
 * Return the specified standard vector in descending order.
 *
 * @tparam T Type of elements contained in vector.
 * @param xs Vector to order.
 * @return Vector in descending order.
 * @throw std::domain_error If any of the values are NaN.
 */
template <typename T>
inline std::vector<T> sort_desc(std::vector<T> xs) {
  check_not_nan("sort_asc", "container argument", xs);
  std::sort(xs.begin(), xs.end(), std::greater<T>());
  return xs;
}

/**
 * Return the specified vector in descending order.
 *
 * @tparam EigVec type of the vector
 *
 * @param xs Vector to order.
 * @return Vector in descending order.
 * @throw std::domain_error If any of the values are NaN.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline plain_type_t<EigVec> sort_desc(EigVec&& xs) {
  plain_type_t<EigVec> x = std::forward<EigVec>(xs);
  check_not_nan("sort_asc", "container argument", x);
  std::sort(x.data(), x.data() + x.size(),
            std::greater<value_type_t<EigVec>>());
  return x;
}

}  // namespace math
}  // namespace stan

#endif
