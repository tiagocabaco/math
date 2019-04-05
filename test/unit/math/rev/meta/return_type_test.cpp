
#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

#include <vector>











using stan::math::var;
using stan::return_type;

TEST(MetaTraits, ReturnTypeVar) {
  test::expect_same_type<var, return_type<var>::type>();
}

TEST(MetaTraits, ReturnTypeVarTenParams) {
  test::expect_same_type<var,
                         return_type<double, var, double, int, double, float,
                                     float, float, var, int>::type>();
}





using stan::math::var;
using stan::return_type;
using std::vector;

TEST(MetaTraits_arr, ReturnTypeVarArray) {
  test::expect_same_type<var, return_type<vector<var> >::type>();
  test::expect_same_type<var, return_type<vector<var>, double>::type>();
  test::expect_same_type<var, return_type<vector<var>, double>::type>();
}

TEST(MetaTraits_arr, ReturnTypeDoubleArray) {
  test::expect_same_type<double, return_type<vector<double> >::type>();
  test::expect_same_type<double, return_type<vector<double>, double>::type>();
  test::expect_same_type<double, return_type<vector<double>, double>::type>();
}






using stan::math::matrix_d;
using stan::math::matrix_v;
using stan::math::var;
using stan::math::vector_d;
using stan::math::vector_v;
using stan::return_type;
using std::vector;

TEST(MetaTraits_mat, ReturnTypeVarMat) {
  test::expect_same_type<var, return_type<vector_v>::type>();
  test::expect_same_type<var, return_type<matrix_v>::type>();
  test::expect_same_type<var, return_type<matrix_v, double>::type>();
  test::expect_same_type<var, return_type<matrix_v, var>::type>();
  test::expect_same_type<var, return_type<matrix_d, matrix_v>::type>();
}

TEST(MetaTraits_mat, ReturnTypeMatMultivar) {
  // test::expect_same_type<var, return_type<vector<vector_v> >::type>();
  test::expect_same_type<var, return_type<vector<matrix_v> >::type>();
  test::expect_same_type<var, return_type<vector<matrix_v>, double>::type>();
  test::expect_same_type<var, return_type<vector<matrix_v>, var>::type>();
  test::expect_same_type<var, return_type<vector<matrix_d>, matrix_v>::type>();
}

TEST(MetaTraits_mat, ReturnTypeDoubleMat) {
  test::expect_same_type<double, return_type<vector_d>::type>();
  test::expect_same_type<double, return_type<matrix_d, double>::type>();
  test::expect_same_type<var, return_type<matrix_d, var>::type>();
}
