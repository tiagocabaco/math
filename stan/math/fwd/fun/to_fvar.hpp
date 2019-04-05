#ifndef STAN_MATH_FWD_FUN_TO_FVAR_HPP
#define STAN_MATH_FWD_FUN_TO_FVAR_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/fun/Eigen.hpp>

#include <stan/math/prim/err/check_matching_dims.hpp>
#include <vector>







namespace stan {
namespace math {

template <typename T>
inline fvar<T> to_fvar(const T& x) {
  return fvar<T>(x);
}

template <typename T>
inline fvar<T> to_fvar(const fvar<T>& x) {
  return x;
}














template <typename T>
inline std::vector<fvar<T> > to_fvar(const std::vector<T>& v) {
  std::vector<fvar<T> > x(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    x[i] = T(v[i]);
  return x;
}

template <typename T>
inline std::vector<fvar<T> > to_fvar(const std::vector<T>& v,
                                     const std::vector<T>& d) {
  std::vector<fvar<T> > x(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    x[i] = fvar<T>(v[i], d[i]);
  return x;
}

template <typename T>
inline std::vector<fvar<T> > to_fvar(const std::vector<fvar<T> >& v) {
  return v;
}















template <int R, int C, typename T>
inline Eigen::Matrix<T, R, C> to_fvar(const Eigen::Matrix<T, R, C>& m) {
  return m;
}

template <int R, int C>
inline Eigen::Matrix<fvar<double>, R, C> to_fvar(
    const Eigen::Matrix<double, R, C>& m) {
  Eigen::Matrix<fvar<double>, R, C> m_fd(m.rows(), m.cols());
  for (int i = 0; i < m.size(); ++i)
    m_fd(i) = m(i);
  return m_fd;
}

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> to_fvar(
    const Eigen::Matrix<T, R, C>& val, const Eigen::Matrix<T, R, C>& deriv) {
  check_matching_dims("to_fvar", "value", val, "deriv", deriv);
  Eigen::Matrix<fvar<T>, R, C> ret(val.rows(), val.cols());
  for (int i = 0; i < val.rows(); i++) {
    for (int j = 0; j < val.cols(); j++) {
      ret(i, j).val_ = val(i, j);
      ret(i, j).d_ = deriv(i, j);
    }
  }
  return ret;
}

}  // namespace math
}  // namespace stan
#endif
