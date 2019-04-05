
#include <stan/math/fwd.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/fwd/fun/nan_util.hpp>






TEST(AgradFwdLogSumExp, Fvar) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<double> x(0.5, 1.0);
  fvar<double> y(1.2, 2.0);
  double z = 1.4;

  fvar<double> a = log_sum_exp(x, y);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ((1.0 * exp(0.5) + 2.0 * exp(1.2)) / (exp(0.5) + exp(1.2)),
                  a.d_);

  fvar<double> b = log_sum_exp(x, z);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.4), b.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), b.d_);

  fvar<double> c = log_sum_exp(z, x);
  EXPECT_FLOAT_EQ(log_sum_exp(1.4, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), c.d_);
}

TEST(AgradFwdLogSumExp, FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(x, y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0, 6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.045176659, a.d_.d_);
}

struct log_sum_exp_fun {
  template <typename T0, typename T1>
  inline typename boost::math::tools::promote_args<T0, T1>::type operator()(
      const T0 arg1, const T1 arg2) const {
    return log_sum_exp(arg1, arg2);
  }
};

TEST(AgradFwdLogSumExp, nan) {
  log_sum_exp_fun log_sum_exp_;
  test_nan_fwd(log_sum_exp_, 3.0, 5.0, false);
}



using stan::math::fvar;
using stan::math::log_sum_exp;
using stan::math::log_sum_exp;

TEST(AgradFwdMatrixLogSumExp_mat, vector_fd) {
  using stan::math::vector_fd;

  vector_fd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_);
  EXPECT_FLOAT_EQ(1, a.d_);
}
TEST(AgradFwdMatrixLogSumExp_mat, row_vector_fd) {
  using stan::math::row_vector_fd;

  row_vector_fd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_);
  EXPECT_FLOAT_EQ(1, a.d_);
}

TEST(AgradFwdMatrixLogSumExp_mat, matrix_fd) {
  using stan::math::matrix_fd;

  matrix_fd b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 1.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_);
  EXPECT_FLOAT_EQ(1, a.d_);
}

TEST(AgradFwdMatrixLogSumExp_mat, vector_ffd) {
  using stan::math::vector_ffd;

  vector_ffd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
}
TEST(AgradFwdMatrixLogSumExp_mat, row_vector_ffd) {
  using stan::math::row_vector_ffd;

  row_vector_ffd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
}

TEST(AgradFwdMatrixLogSumExp_mat, matrix_ffd) {
  using stan::math::matrix_ffd;

  matrix_ffd b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 1.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
}
