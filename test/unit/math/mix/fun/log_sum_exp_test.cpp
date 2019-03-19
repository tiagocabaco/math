
#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/fun/util.hpp>
#include <test/unit/math/mix/fun/nan_util.hpp>

#include <vector>








TEST(AgradFwdLogSumExp, FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<var> x(3.0, 1.3);
  fvar<var> z(6.0, 1.0);
  fvar<var> a = log_sum_exp(x, z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0, 6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(3.0) + 1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)),
                  a.d_.val());

  AVEC y = createAVEC(x.val_, z.val_);
  VEC g;
  a.val_.grad(y, g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[1]);
}
TEST(AgradFwdLogSumExp, FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<var> x(3.0, 1.3);
  double z(6.0);
  fvar<var> a = log_sum_exp(x, z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0, 6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(3.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y, g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(AgradFwdLogSumExp, Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  double x(3.0);
  fvar<var> z(6.0, 1.0);
  fvar<var> a = log_sum_exp(x, z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0, 6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y, g);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(AgradFwdLogSumExp, FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<var> x(3.0, 1.3);
  fvar<var> z(6.0, 1.0);
  fvar<var> a = log_sum_exp(x, z);

  EXPECT_FLOAT_EQ((1.3 * exp(3.0) + 1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)),
                  a.d_.val());

  AVEC y = createAVEC(x.val_, z.val_);
  VEC g;
  a.d_.grad(y, g);
  EXPECT_FLOAT_EQ((1.3 * exp(3.0) * (exp(3.0) + exp(6.0))
                   - exp(3.0) * (1.3 * exp(3.0) + exp(6.0)))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[0]);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0))
                   - exp(6.0) * (1.3 * exp(3.0) + exp(6.0)))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[1]);
}
TEST(AgradFwdLogSumExp, FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<var> x(3.0, 1.3);
  double z(6.0);
  fvar<var> a = log_sum_exp(x, z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y, g);
  EXPECT_FLOAT_EQ(1.3 * (exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) * exp(3.0))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[0]);
}
TEST(AgradFwdLogSumExp, Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  double x(3.0);
  fvar<var> z(6.0, 1.0);
  fvar<var> a = log_sum_exp(x, z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y, g);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) * exp(6.0))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[0]);
}
TEST(AgradFwdLogSumExp, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x, y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0, 6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.045176659, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_, y.val_.val_);
  VEC g;
  a.val_.val_.grad(p, g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[1]);
}
TEST(AgradFwdLogSumExp, FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x, y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0, 6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p, g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(AgradFwdLogSumExp, Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x, y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0, 6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p, g);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(AgradFwdLogSumExp, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x, y);

  AVEC p = createAVEC(x.val_.val_, y.val_.val_);
  VEC g;
  a.val_.d_.grad(p, g);
  EXPECT_FLOAT_EQ((exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) * (exp(3.0)))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[0]);
  EXPECT_FLOAT_EQ(-0.045176659, g[1]);
}
TEST(AgradFwdLogSumExp, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x, y);

  AVEC p = createAVEC(x.val_.val_, y.val_.val_);
  VEC g;
  a.d_.val_.grad(p, g);
  EXPECT_FLOAT_EQ(-0.045176659, g[0]);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) * (exp(6.0)))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[1]);
}
TEST(AgradFwdLogSumExp, FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x, y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p, g);
  EXPECT_FLOAT_EQ((exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) * (exp(3.0)))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[0]);
}
TEST(AgradFwdLogSumExp, Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x, y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p, g);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) * (exp(6.0)))
                      / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),
                  g[0]);
}
TEST(AgradFwdLogSumExp, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x, y);

  AVEC p = createAVEC(x.val_.val_, y.val_.val_);
  VEC g;
  a.d_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(-0.040891573, g[0]);
  EXPECT_FLOAT_EQ(0.040891573, g[1]);
}
TEST(AgradFwdLogSumExp, FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x, y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(0.040891574660943478616430308425, g[0]);
}
TEST(AgradFwdLogSumExp, Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::log_sum_exp;
  using stan::math::var;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x, y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(-0.040891574660943478616430308, g[0]);
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
  test_nan_mix(log_sum_exp_, 3.0, 5.0, false);
}





using stan::math::fvar;
using stan::math::log_sum_exp;
using stan::math::log_sum_exp;
using stan::math::var;

TEST(AgradMixMatrixLogSumExp_mat, vector_fv_1st_deriv) {
  using stan::math::vector_fv;

  vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.val());

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.032058604, h[0]);
  EXPECT_FLOAT_EQ(0.087144315, h[1]);
  EXPECT_FLOAT_EQ(0.23688282, h[2]);
  EXPECT_FLOAT_EQ(0.64391428, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, row_vector_fv_1st_deriv) {
  using stan::math::row_vector_fv;

  row_vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.val());

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.032058604, h[0]);
  EXPECT_FLOAT_EQ(0.087144315, h[1]);
  EXPECT_FLOAT_EQ(0.23688282, h[2]);
  EXPECT_FLOAT_EQ(0.64391428, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, matrix_fv_1st_deriv) {
  using stan::math::matrix_fv;

  matrix_fv b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 1.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.val());

  std::vector<var> z;
  z.push_back(b(0, 0).val_);
  z.push_back(b(0, 1).val_);
  z.push_back(b(1, 0).val_);
  z.push_back(b(1, 1).val_);

  VEC h;
  a.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.032058604, h[0]);
  EXPECT_FLOAT_EQ(0.087144315, h[1]);
  EXPECT_FLOAT_EQ(0.23688282, h[2]);
  EXPECT_FLOAT_EQ(0.64391428, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, vector_fv_2nd_deriv) {
  using stan::math::vector_fv;

  vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val());
  EXPECT_FLOAT_EQ(1.0320586, a.d_.val());

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.d_.grad(z, h);
  EXPECT_FLOAT_EQ(0.031030849, h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251, h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323, h[2]);
  EXPECT_FLOAT_EQ(-0.020642992, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, row_vector_fv_2nd_deriv) {
  using stan::math::row_vector_fv;

  row_vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.d_.grad(z, h);
  EXPECT_FLOAT_EQ(0.031030849, h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251, h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323, h[2]);
  EXPECT_FLOAT_EQ(-0.020642992, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, matrix_fv_2nd_deriv) {
  using stan::math::matrix_fv;

  matrix_fv b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 2.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0, 0).val_);
  z.push_back(b(0, 1).val_);
  z.push_back(b(1, 0).val_);
  z.push_back(b(1, 1).val_);

  VEC h;
  a.d_.grad(z, h);
  EXPECT_FLOAT_EQ(0.031030849, h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251, h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323, h[2]);
  EXPECT_FLOAT_EQ(-0.020642992, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, vector_ffv_1st_deriv) {
  using stan::math::vector_ffv;

  vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.val_.val());

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.val_.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.032058604, h[0]);
  EXPECT_FLOAT_EQ(0.087144315, h[1]);
  EXPECT_FLOAT_EQ(0.23688282, h[2]);
  EXPECT_FLOAT_EQ(0.64391428, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, row_vector_ffv_1st_deriv) {
  using stan::math::row_vector_ffv;

  row_vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.val_.val());

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.val_.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.032058604, h[0]);
  EXPECT_FLOAT_EQ(0.087144315, h[1]);
  EXPECT_FLOAT_EQ(0.23688282, h[2]);
  EXPECT_FLOAT_EQ(0.64391428, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, matrix_ffv_1st_deriv) {
  using stan::math::matrix_ffv;

  matrix_ffv b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 1.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898, a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.d_.val_.val());

  std::vector<var> z;
  z.push_back(b(0, 0).val_.val_);
  z.push_back(b(0, 1).val_.val_);
  z.push_back(b(1, 0).val_.val_);
  z.push_back(b(1, 1).val_.val_);

  VEC h;
  a.val_.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.032058604, h[0]);
  EXPECT_FLOAT_EQ(0.087144315, h[1]);
  EXPECT_FLOAT_EQ(0.23688282, h[2]);
  EXPECT_FLOAT_EQ(0.64391428, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, vector_ffv_2nd_deriv) {
  using stan::math::vector_ffv;

  vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.031030849, h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251, h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323, h[2]);
  EXPECT_FLOAT_EQ(-0.020642992, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, row_vector_ffv_2nd_deriv) {
  using stan::math::row_vector_ffv;

  row_vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.031030849, h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251, h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323, h[2]);
  EXPECT_FLOAT_EQ(-0.020642992, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, matrix_ffv_2nd_deriv) {
  using stan::math::matrix_ffv;

  matrix_ffv b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 2.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0, 0).val_.val_);
  z.push_back(b(0, 1).val_.val_);
  z.push_back(b(1, 0).val_.val_);
  z.push_back(b(1, 1).val_.val_);

  VEC h;
  a.d_.val_.grad(z, h);
  EXPECT_FLOAT_EQ(0.031030849, h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251, h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323, h[2]);
  EXPECT_FLOAT_EQ(-0.020642992, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, vector_ffv_3rd_deriv) {
  using stan::math::vector_ffv;

  vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;
  b(0).val_.d_ = 2.0;
  b(1).val_.d_ = 1.0;
  b(2).val_.d_ = 1.0;
  b(3).val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.d_.grad(z, h);
  EXPECT_FLOAT_EQ(0.029041238, h[0]);
  EXPECT_FLOAT_EQ(-0.0026145992, h[1]);
  EXPECT_FLOAT_EQ(-0.0071072178, h[2]);
  EXPECT_FLOAT_EQ(-0.019319421, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, row_vector_ffv_3rd_deriv) {
  using stan::math::row_vector_ffv;

  row_vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;
  b(0).val_.d_ = 2.0;
  b(1).val_.d_ = 1.0;
  b(2).val_.d_ = 1.0;
  b(3).val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.d_.grad(z, h);
  EXPECT_FLOAT_EQ(0.029041238, h[0]);
  EXPECT_FLOAT_EQ(-0.0026145992, h[1]);
  EXPECT_FLOAT_EQ(-0.0071072178, h[2]);
  EXPECT_FLOAT_EQ(-0.019319421, h[3]);
}

TEST(AgradMixMatrixLogSumExp_mat, matrix_ffv_3rd_deriv) {
  using stan::math::matrix_ffv;

  matrix_ffv b(2, 2);
  b << 1, 2, 3, 4;
  b(0, 0).d_ = 2.0;
  b(0, 1).d_ = 1.0;
  b(1, 0).d_ = 1.0;
  b(1, 1).d_ = 1.0;
  b(0).val_.d_ = 2.0;
  b(1).val_.d_ = 1.0;
  b(2).val_.d_ = 1.0;
  b(3).val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0, 0).val_.val_);
  z.push_back(b(0, 1).val_.val_);
  z.push_back(b(1, 0).val_.val_);
  z.push_back(b(1, 1).val_.val_);

  VEC h;
  a.d_.d_.grad(z, h);
  EXPECT_FLOAT_EQ(0.029041238, h[0]);
  EXPECT_FLOAT_EQ(-0.0026145992, h[1]);
  EXPECT_FLOAT_EQ(-0.0071072178, h[2]);
  EXPECT_FLOAT_EQ(-0.019319421, h[3]);
}
