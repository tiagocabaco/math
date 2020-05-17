#ifndef TEST_UNIT_MATH_UTIL_HPP
#define TEST_UNIT_MATH_UTIL_HPP

#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <type_traits>
#include <string>
#include <vector>


#define EXPECT_THROW_MSG(expr, T_e, msg) \
  EXPECT_THROW_MSG_WITH_COUNT(expr, T_e, msg, 1)

#define EXPECT_MATRIX_NEAR(A, B, DELTA) \
  EXPECT_EQ(A.rows(), B.rows());        \
  EXPECT_EQ(A.cols(), B.cols());        \
  for (int i = 0; i < A.size(); i++)    \
    EXPECT_NEAR(A(i), B(i), DELTA);

#define EXPECT_THROW_MSG_WITH_COUNT(expr, T_e, msg, count) \
  EXPECT_THROW(expr, T_e);                                 \
  try {                                                    \
    expr;                                                  \
  } catch (const T_e& e) {                                 \
    EXPECT_EQ(count, stan::test::count_matches(msg, e.what()))         \
        << "expected message: " << msg << std::endl        \
        << "found message:    " << e.what();               \
  }

#define EXPECT_SUM(A, N) \
  EXPECT_TRUE((N * (N + 1)) / 2 == A.sum())

namespace stan {
namespace test {

/**
 * Return the Eigen vector with the same size and elements as the
 * specified standard vector.  Elements are copied from the specified
 * input vector.
 *
 * @tparam T type of scalars in containers
 * @param x standard vector to copy
 * @return Eigen vector corresponding to specified standard vector
 */
template <typename T>
Eigen::Matrix<T, -1, 1> to_eigen_vector(const std::vector<T>& x) {
  return Eigen::Map<const Eigen::Matrix<T, -1, 1>>(x.data(), x.size());
}

/**
 * Return the standard vector with the same size and elements as the
 * specified Eigen matrix, vector, or row vector.
 *
 * @tparam T type of scalars in containers
 * @tparam R row specification of input matrix
 * @tparam C column specification of input matrix
 * @param x Eigen matrix, vector, or row vector to copy
 * @return standard vector corresponding to input
 */
template <typename T, int R, int C>
std::vector<T> to_std_vector(const Eigen::Matrix<T, R, C>& x) {
  std::vector<T> y;
  y.reserve(x.size());
  for (int i = 0; i < x.size(); ++i)
    y.push_back(x(i));
  return y;
}

int count_matches(const std::string& target, const std::string& s) {
  if (target.size() == 0)
    return -1;  // error
  int count = 0;
  for (size_t pos = 0; (pos = s.find(target, pos)) != std::string::npos;
       pos += target.size())
    ++count;
  return count;
}

template <typename T1, typename T2>
void expect_same_type() {
  bool b = std::is_same<T1, T2>::value;
  EXPECT_TRUE(b);
}

}  // namespace test
}  // namespace stan
#endif
