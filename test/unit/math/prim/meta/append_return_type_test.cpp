#include <stan/math/prim/meta.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/util.hpp>
#include <vector>

using stan::math::append_return_type;

TEST(MathMetaPrim, test_append_return_type) {
  stan::test::expect_same_type<int, append_return_type<int, int>::type>();
  stan::test::expect_same_type<double, append_return_type<double, double>::type>();
  stan::test::expect_same_type<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
      append_return_type<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>>::type>();
}
