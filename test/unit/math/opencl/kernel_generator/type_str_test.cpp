#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixCL, type_str) {
  EXPECT_EQ(type_str<double>(), "double");
  EXPECT_EQ(type_str<int>(), "int");
  EXPECT_EQ(type_str<bool>(), "bool");
}

#endif
