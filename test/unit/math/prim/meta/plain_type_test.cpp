#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>

using stan::plain_type_t;

TEST(MetaTraitsPrimMat, plain_type_non_eigen) {
  stan::test::expect_same_type<double, plain_type_t<double>>();
  stan::test::expect_same_type<std::vector<double>,
                         plain_type_t<std::vector<double>>>();
  stan::test::expect_same_type<stan::math::var, plain_type_t<stan::math::var>>();
  stan::test::expect_same_type<std::vector<stan::math::var>,
                         plain_type_t<std::vector<stan::math::var>>>();
}

TEST(MetaTraitsPrimMat, plain_type_eigen) {
  stan::test::expect_same_type<Eigen::MatrixXd, plain_type_t<Eigen::MatrixXd>>();
  stan::test::expect_same_type<Eigen::VectorXd, plain_type_t<Eigen::VectorXd>>();
  stan::test::expect_same_type<Eigen::ArrayXXd, plain_type_t<Eigen::ArrayXXd>>();
  stan::test::expect_same_type<stan::math::matrix_v,
                         plain_type_t<stan::math::matrix_v>>();

  Eigen::MatrixXd m1, m2;
  Eigen::VectorXd v1, v2;
  Eigen::ArrayXXd a1, a2;
  stan::math::matrix_v mv1, mv2;
  stan::test::expect_same_type<Eigen::MatrixXd, plain_type_t<decltype(m1 - m2)>>();
  stan::test::expect_same_type<Eigen::VectorXd, plain_type_t<decltype(v1 + v2)>>();
  stan::test::expect_same_type<Eigen::ArrayXXd,
                         plain_type_t<decltype(a1 * a2.sin())>>();
  stan::test::expect_same_type<stan::math::matrix_v,
                         plain_type_t<decltype(mv1 * mv2)>>();
}
