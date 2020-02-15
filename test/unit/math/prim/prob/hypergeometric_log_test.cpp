#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(ProbHypergeometric, log_matches_lpmf) {
  int n = 1;
  int N = 4;
  int a = 1;
  int b = 3;

  EXPECT_FLOAT_EQ((stan::math::hypergeometric_lpmf(n, N, a, b)),
                  (stan::math::hypergeometric_log(n, N, a, b)));
  EXPECT_FLOAT_EQ((stan::math::hypergeometric_lpmf<true>(n, N, a, b)),
                  (stan::math::hypergeometric_log<true>(n, N, a, b)));
  EXPECT_FLOAT_EQ((stan::math::hypergeometric_lpmf<false>(n, N, a, b)),
                  (stan::math::hypergeometric_log<false>(n, N, a, b)));
  EXPECT_FLOAT_EQ(
      (stan::math::hypergeometric_lpmf<true, int, int, int, int>(n, N, a, b)),
      (stan::math::hypergeometric_log<true, int, int, int, int>(n, N, a, b)));
  EXPECT_FLOAT_EQ(
      (stan::math::hypergeometric_lpmf<false, int, int, int, int>(n, N, a, b)),
      (stan::math::hypergeometric_log<false, int, int, int, int>(n, N, a, b)));
  EXPECT_FLOAT_EQ(
      (stan::math::hypergeometric_lpmf<int, int, int, int>(n, N, a, b)),
      (stan::math::hypergeometric_log<int, int, int, int>(n, N, a, b)));
}

TEST(ProbHypergeometric, n_equal_N) {
  int n = 2;
  int N = 2;
  int a = 2;
  int b = 4;

  EXPECT_NO_THROW(stan::math::hypergeometric_lpmf(n, N, a, b));
}
