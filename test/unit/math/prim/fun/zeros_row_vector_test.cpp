#include <stan/math/prim.hpp>
#include <test/unit/math/util.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, zeros_row_vector) {
  for (int K = 0; K < 5; K++) {
    Eigen::RowVectorXd v = Eigen::RowVectorXd::Zero(K);
    stan::test::expect_matrix_eq(v, stan::math::zeros_row_vector(K));
  }
}

TEST(MathFunctions, zeros_row_vector_throw) {
  EXPECT_THROW(stan::math::zeros_row_vector(-1), std::domain_error);
}
