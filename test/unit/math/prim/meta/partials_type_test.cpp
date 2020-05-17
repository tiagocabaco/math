#include <stan/math/prim/meta.hpp>
#include <test/unit/math/util.hpp>
#include <gtest/gtest.h>

using stan::partials_type;

TEST(MathMetaPrim, PartialsTypeDouble) {
  stan::test::expect_same_type<double, partials_type<double>::type>();
}

TEST(MathMetaPrim, PartialsTypeFloat) {
  stan::test::expect_same_type<float, partials_type<float>::type>();
}

TEST(MathMetaPrim, PartialsTypeInt) {
  stan::test::expect_same_type<int, partials_type<int>::type>();
}
