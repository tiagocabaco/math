#include <stan/math/prim.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace prim_choose_test {
int round_to_int(double x) {
  return static_cast<int>(x < 0 ? x - 0.5 : x + 0.5);
}

int finite_choose_test(int N, int n) {
  using std::exp;
  return round_to_int(exp(lgamma(N + 1) - lgamma(n + 1) - lgamma(N - n + 1)));
}
}

TEST(MathFunctions, choose) {
  for (int N = 0; N <= 32; ++N)
    for (int n = 0; n <= 32; ++n) {
      if (n > N)
        EXPECT_EQ(0, stan::math::choose(N, n));
      else
        EXPECT_EQ(prim_choose_test::finite_choose_test(N, n), stan::math::choose(N, n));
    }
}

TEST(MathFunctions, chooseThrow) {
  using stan::math::choose;
  EXPECT_THROW(choose(36, 18), std::domain_error);
  EXPECT_THROW(choose(-2, 1), std::domain_error);
  EXPECT_THROW(choose(2, -1), std::domain_error);
}

TEST(MathFunctions, choose_nan) {
  using stan::math::choose;
  int nan = std::numeric_limits<int>::quiet_NaN() - 1;
  // quiet_NaN() returns 0 which would otherwise be valid
  EXPECT_THROW(choose(2, nan), std::domain_error);
  EXPECT_THROW(choose(nan, 2), std::domain_error);
  EXPECT_THROW(choose(nan, nan), std::domain_error);
}
