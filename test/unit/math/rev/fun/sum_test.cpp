
#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/util.hpp>

#include <test/unit/math/rev/fun/util.hpp>
#include <vector>









TEST(AgradRev, sum_std_vector) {
  using stan::math::sum;
  using stan::math::var;
  using std::vector;

  vector<var> x;
  for (size_t i = 0; i < 6; ++i)
    x.push_back(i + 1);

  var fx = 3.7 * sum(x);
  EXPECT_FLOAT_EQ(3.7 * 21.0, fx.val());

  vector<double> gx;
  fx.grad(x, gx);
  EXPECT_EQ(6, gx.size());
  for (size_t i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(3.7, gx[i]);

  x = vector<var>();
  EXPECT_FLOAT_EQ(0.0, sum(x).val());
}

TEST(AgradRev, check_varis_on_stack) {
  std::vector<stan::math::var> x;
  for (size_t i = 0; i < 6; ++i)
    x.push_back(i + 1);
  test::check_varis_on_stack(stan::math::sum(x));
}






TEST(AgradRevMatrix_mat, sum_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d d(6);
  vector_v v(6);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;

  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  std::vector<double> grad;
  std::vector<AVAR> x(v.size());
  for (int i = 0; i < v.size(); ++i)
    x[i] = v(i);
  output.grad(x, grad);
  EXPECT_EQ(6, grad.size());
  for (int i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(1.0, grad[i]);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}
TEST(AgradRevMatrix_mat, sum_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;
  using stan::math::sum;

  row_vector_d d(6);
  row_vector_v v(6);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;

  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}
TEST(AgradRevMatrix_mat, sum_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::sum;

  matrix_d d(2, 3);
  matrix_v v(2, 3);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;

  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}

TEST(AgradRevMatrix_mat, check_varis_on_stack) {
  stan::math::matrix_v m(2, 2);
  m << 1, 2, 3, 4;

  stan::math::vector_v v(3);
  v << 1, 2, 3;

  stan::math::row_vector_v rv(2);
  rv << 1, 2;

  test::check_varis_on_stack(stan::math::sum(m));
  test::check_varis_on_stack(stan::math::sum(v));
  test::check_varis_on_stack(stan::math::sum(rv));
}
