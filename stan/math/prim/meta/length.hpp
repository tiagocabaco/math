#ifndef STAN_MATH_PRIM_META_LENGTH_HPP
#define STAN_MATH_PRIM_META_LENGTH_HPP



#include <stan/math/prim/fun/Eigen.hpp>
#include <cstdlib>
#include <vector>








namespace stan {

template <typename T>
size_t length(const T& /*x*/) {
  return 1U;
}










template <typename T>
size_t length(const std::vector<T>& x) {
  return x.size();
}









template <typename T, int R, int C>
size_t length(const Eigen::Matrix<T, R, C>& m) {
  return m.size();
}
}  // namespace stan
#endif
