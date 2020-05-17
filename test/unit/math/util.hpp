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

#define EXPECT_MATRIX_EQ(A, B)       \
  for (int i = 0; i < A.size(); i++) \
    EXPECT_EQ(A(i), B(i));

namespace stan {
namespace test {

template <int R, int C>
void correct_type_row_vector(const Eigen::Matrix<double, R, C>& x) {
  EXPECT_EQ(Eigen::Dynamic, C);
  EXPECT_EQ(1, R);
}

template <int R, int C>
void correct_type_matrix(const Eigen::Matrix<double, R, C>& x) {
  EXPECT_EQ(Eigen::Dynamic, C);
  EXPECT_EQ(Eigen::Dynamic, R);
}

template <int R, int C>
void correct_type_vector(const Eigen::Matrix<double, R, C>& x) {
  EXPECT_EQ(Eigen::Dynamic, R);
  EXPECT_EQ(1, C);
}

void expect_matrix_eq(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& a,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& b) {
  EXPECT_EQ(a.rows(), b.rows());
  EXPECT_EQ(a.cols(), b.cols());
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j)
      EXPECT_FLOAT_EQ(a(i, j), b(i, j));
}

template <typename T, typename = stan::require_arithmetic_t<T>>
void expect_std_vector_eq(const std::vector<T>& a, const std::vector<T>& b) {
  EXPECT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); ++i)
    EXPECT_FLOAT_EQ(a[i], b[i]);
}

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
