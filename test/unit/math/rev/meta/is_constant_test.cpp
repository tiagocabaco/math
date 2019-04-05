


#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
TEST(MetaTraits, isConstant) {
  using stan::is_constant;
  using stan::math::var;

  EXPECT_FALSE(is_constant<var>::value);
}
