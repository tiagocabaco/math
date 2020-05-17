#ifndef TEST_UTIL_EXPECT_MACROS
#define TEST_UTIL_EXPECT_MACROS

#define EXPECT_MATRIX_NEAR(A, B, DELTA) \
  EXPECT_EQ(A.rows(), B.rows());        \
  EXPECT_EQ(A.cols(), B.cols());        \
  for (int i = 0; i < A.size(); i++)    \
    EXPECT_NEAR(A(i), B(i), DELTA);

#endif
