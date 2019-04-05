
#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

#include <vector>







using stan::math::var;
using stan::partials_return_type;

TEST(MetaTraits, PartialsReturnTypeVar) {
  test::expect_same_type<double, partials_return_type<var>::type>();
}

TEST(MetaTraits, PartialsReturnTypeVarTenParams) {
  test::expect_same_type<
      double, partials_return_type<double, var, double, int, double, float,
                                   float, float, var, int>::type>();
}




TEST(MetaTraits_arr, partials_return_type) {
  using stan::math::var;
  using stan::partials_return_type;

  partials_return_type<double, stan::math::var,
                       std::vector<stan::math::var> >::type g(5.0);
  EXPECT_EQ(5.0, g);
}
