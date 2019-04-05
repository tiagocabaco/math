
#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/fun/util.hpp>
#include <test/unit/math/mix/fun/nan_util.hpp>

#include <cmath>
#include <typeinfo>
#include <vector>













void test_log_mix_3xfvar_var_D1(double theta, double lambda1, double lambda2,
                                double theta_d, double lambda1_d,
                                double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<var> theta_fv(theta, theta_d);
  fvar<var> lambda1_fv(lambda1, lambda1_d);
  fvar<var> lambda2_fv(lambda2, lambda2_d);

  var theta_v(theta);
  var lambda1_v(lambda1);
  var lambda2_v(lambda2);

  var theta_d_v(theta_d);
  var lambda1_d_v(lambda1_d);
  var lambda2_d_v(lambda2_d);

  fvar<var> res = log_mix(theta_fv, lambda1_fv, lambda2_fv);
  double result = log_mix(theta_fv.val_.val(), lambda1_fv.val_.val(),
                          lambda2_fv.val_.val());
  double deriv_denom = exp(lambda1_fv.val_.val()) * theta_fv.val_.val()
                       + exp(lambda2_fv.val_.val()) * (1 - theta_fv.val_.val());
  double theta_deriv
      = 1 / deriv_denom
        * (exp(lambda1_fv.val_.val()) - exp(lambda2_fv.val_.val()));
  double lambda1_deriv
      = 1 / deriv_denom * exp(lambda1_fv.val_.val()) * theta_fv.val_.val();
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val())
                         * (1 - theta_fv.val_.val());
  double deriv = theta_deriv * theta_fv.d_.val()
                 + lambda2_deriv * lambda2_fv.d_.val()
                 + lambda1_deriv * lambda1_fv.d_.val();

  VEC g = cgrad(res.val_, theta_fv.val_, lambda1_fv.val_, lambda2_fv.val_);

  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv, g[0]);
  EXPECT_FLOAT_EQ(lambda1_deriv, g[1]);
  EXPECT_FLOAT_EQ(lambda2_deriv, g[2]);

  fvar<var> theta_fv_invalid(-1.0, theta_d);
  EXPECT_THROW(log_mix(theta_fv_invalid, lambda1_fv, lambda2_fv),
               std::domain_error);
  stan::math::recover_memory();
}

VEC log_mix_D3(double theta, double lambda1, double lambda2, double theta_d,
               double lambda1_d, double lambda2_d, double theta_d2,
               double lambda1_d2, double lambda2_d2) {
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  var theta_v(theta);
  var lambda1_v(lambda1);
  var lambda2_v(lambda2);
  double d_theta(0.0);
  double d_lambda1(0.0);
  double d_lambda2(0.0);
  double d2_theta(0.0);
  double d2_lambda1(0.0);
  double d2_lambda2(0.0);
  var d_theta_v;
  var d_lambda1_v;
  var d_lambda2_v;
  var d2_theta_v;
  var d2_lambda1_v;
  var d2_lambda2_v;
  if (lambda1 > lambda2) {
    double lam2_m_lam1 = lambda2 - lambda1;
    double exp_lam2_m_lam1 = exp(lam2_m_lam1);
    double one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
    double one_m_t = 1 - theta;
    double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
    double t_plus_one_m_t_prod_exp_lam2_m_lam1
        = theta + one_m_t_prod_exp_lam2_m_lam1;
    var lam2_m_lam1_v = lambda2_v - lambda1_v;
    var exp_lam2_m_lam1_v = exp(lam2_m_lam1_v);
    var one_m_exp_lam2_m_lam1_v = 1 - exp_lam2_m_lam1_v;
    var one_m_t_v = 1 - theta_v;
    var one_m_t_prod_exp_lam2_m_lam1_v = one_m_t_v * exp_lam2_m_lam1_v;
    var t_plus_one_m_t_prod_exp_lam2_m_lam1_v
        = theta_v + one_m_t_prod_exp_lam2_m_lam1_v;
    d_theta = one_m_exp_lam2_m_lam1 / t_plus_one_m_t_prod_exp_lam2_m_lam1;
    d_lambda1 = theta / t_plus_one_m_t_prod_exp_lam2_m_lam1;
    d_lambda2
        = one_m_t_prod_exp_lam2_m_lam1 / t_plus_one_m_t_prod_exp_lam2_m_lam1;
    d_theta_v = one_m_exp_lam2_m_lam1_v / t_plus_one_m_t_prod_exp_lam2_m_lam1_v;
    d_lambda1_v = theta_v / t_plus_one_m_t_prod_exp_lam2_m_lam1_v;
    d_lambda2_v = one_m_t_prod_exp_lam2_m_lam1_v
                  / t_plus_one_m_t_prod_exp_lam2_m_lam1_v;
    d2_theta = lambda1_d
                   * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1
                      - d_lambda1 * d_theta)
               - lambda2_d
                     * (exp_lam2_m_lam1 / t_plus_one_m_t_prod_exp_lam2_m_lam1
                        + d_lambda2 * d_theta)
               - theta_d * pow(d_theta, 2.0);
    d2_lambda1 = lambda1_d * (d_lambda1 - pow(d_lambda1, 2.0))
                 - lambda2_d * d_lambda1 * d_lambda2
                 + theta_d
                       * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1
                          - d_lambda1 * d_theta);
    d2_lambda2
        = lambda2_d * (d_lambda2 - pow(d_lambda2, 2.0))
          - lambda1_d * d_lambda1 * d_lambda2
          - theta_d
                * (d_lambda2 * d_theta
                   + exp_lam2_m_lam1 / t_plus_one_m_t_prod_exp_lam2_m_lam1);
    d2_theta_v
        = lambda1_d
              * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1_v
                 - d_lambda1_v * d_theta_v)
          - lambda2_d
                * (exp_lam2_m_lam1_v / t_plus_one_m_t_prod_exp_lam2_m_lam1_v
                   + d_lambda2_v * d_theta_v)
          - theta_d * pow(d_theta_v, 2.0);
    d2_lambda1_v = lambda1_d * (d_lambda1_v - pow(d_lambda1_v, 2.0))
                   - lambda2_d * d_lambda1_v * d_lambda2_v
                   + theta_d
                         * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1_v
                            - d_lambda1_v * d_theta_v);
    d2_lambda2_v
        = lambda2_d * (d_lambda2_v - pow(d_lambda2_v, 2.0))
          - lambda1_d * d_lambda1_v * d_lambda2_v
          - theta_d
                * (d_lambda2_v * d_theta_v
                   + exp_lam2_m_lam1_v / t_plus_one_m_t_prod_exp_lam2_m_lam1_v);
  } else {
    double lam1_m_lam2 = lambda1 - lambda2;
    double exp_lam1_m_lam2 = exp(lam1_m_lam2);
    double exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
    double one_m_t = 1 - theta;
    double t_prod_exp_lam1_m_lam2 = theta * exp_lam1_m_lam2;
    double one_m_t_plus_t_prod_exp_lam1_m_lam2
        = one_m_t + t_prod_exp_lam1_m_lam2;
    var lam1_m_lam2_v = lambda1_v - lambda2_v;
    var exp_lam1_m_lam2_v = exp(lam1_m_lam2_v);
    var exp_lam1_m_lam2_m_1_v = exp_lam1_m_lam2_v - 1;
    var one_m_t_v = 1 - theta_v;
    var t_prod_exp_lam1_m_lam2_v = theta_v * exp_lam1_m_lam2_v;
    var one_m_t_plus_t_prod_exp_lam1_m_lam2_v
        = one_m_t_v + t_prod_exp_lam1_m_lam2_v;
    d_theta = exp_lam1_m_lam2_m_1 / one_m_t_plus_t_prod_exp_lam1_m_lam2;
    d_lambda1 = t_prod_exp_lam1_m_lam2 / one_m_t_plus_t_prod_exp_lam1_m_lam2;
    d_lambda2 = one_m_t / one_m_t_plus_t_prod_exp_lam1_m_lam2;
    d_theta_v = exp_lam1_m_lam2_m_1_v / one_m_t_plus_t_prod_exp_lam1_m_lam2_v;
    d_lambda1_v
        = t_prod_exp_lam1_m_lam2_v / one_m_t_plus_t_prod_exp_lam1_m_lam2_v;
    d_lambda2_v = one_m_t_v / one_m_t_plus_t_prod_exp_lam1_m_lam2_v;
    d2_theta = lambda1_d
                   * (exp_lam1_m_lam2 / one_m_t_plus_t_prod_exp_lam1_m_lam2
                      - d_lambda1 * d_theta)
               - lambda2_d
                     * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2
                        + d_lambda2 * d_theta)
               - theta_d * pow(d_theta, 2.0);
    d2_lambda1 = lambda1_d * (d_lambda1 - pow(d_lambda1, 2.0))
                 - lambda2_d * d_lambda1 * d_lambda2
                 + theta_d
                       * (exp_lam1_m_lam2 / one_m_t_plus_t_prod_exp_lam1_m_lam2
                          - d_lambda1 * d_theta);
    d2_lambda2 = lambda2_d * (d_lambda2 - pow(d_lambda2, 2.0))
                 - lambda1_d * d_lambda1 * d_lambda2
                 - theta_d
                       * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2
                          + d_theta * d_lambda2);
    d2_theta_v
        = lambda1_d
              * (exp_lam1_m_lam2_v / one_m_t_plus_t_prod_exp_lam1_m_lam2_v
                 - d_lambda1_v * d_theta_v)
          - lambda2_d
                * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2_v
                   + d_lambda2_v * d_theta_v)
          - theta_d * pow(d_theta_v, 2.0);
    d2_lambda1_v
        = lambda1_d * (d_lambda1_v - pow(d_lambda1_v, 2.0))
          - lambda2_d * d_lambda1_v * d_lambda2_v
          + theta_d
                * (exp_lam1_m_lam2_v / one_m_t_plus_t_prod_exp_lam1_m_lam2_v
                   - d_lambda1_v * d_theta_v);
    d2_lambda2_v = lambda2_d * (d_lambda2_v - pow(d_lambda2_v, 2.0))
                   - lambda1_d * d_lambda1_v * d_lambda2_v
                   - theta_d
                         * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2_v
                            + d_theta_v * d_lambda2_v);
  }

  double deriv
      = d_theta * theta_d + d_lambda2 * lambda2_d + d_lambda1 * lambda1_d;
  double deriv_2
      = d2_theta * theta_d2 + d2_lambda1 * lambda1_d2 + d2_lambda2 * lambda2_d2;

  var deriv_2_v = d2_theta_v * theta_d2 + d2_lambda1_v * lambda1_d2
                  + d2_lambda2_v * lambda2_d2;

  VEC d1_d2_d3;
  VEC d3 = cgrad(deriv_2_v, theta_v, lambda1_v, lambda2_v);

  d1_d2_d3.push_back(deriv);
  d1_d2_d3.push_back(deriv_2);
  d1_d2_d3.push_back(d_theta);
  d1_d2_d3.push_back(d_lambda1);
  d1_d2_d3.push_back(d_lambda2);
  d1_d2_d3.push_back(d2_theta);
  d1_d2_d3.push_back(d2_lambda1);
  d1_d2_d3.push_back(d2_lambda2);
  d1_d2_d3.push_back(d3[0]);
  d1_d2_d3.push_back(d3[1]);
  d1_d2_d3.push_back(d3[2]);

  stan::math::recover_memory();
  return d1_d2_d3;
}

void test_log_mix_3xfvar_var_D2(double theta, double lambda1, double lambda2,
                                double theta_d, double lambda1_d,
                                double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<var> theta_fv(theta, theta_d);
  fvar<var> lambda1_fv(lambda1, lambda1_d);
  fvar<var> lambda2_fv(lambda2, lambda2_d);

  var theta_v(theta);
  var lambda1_v(lambda1);
  var lambda2_v(lambda2);

  var theta_d_v(theta_d);
  var lambda1_d_v(lambda1_d);
  var lambda2_d_v(lambda2_d);

  fvar<var> res = log_mix(theta_fv, lambda1_fv, lambda2_fv);
  double result = log_mix(theta_fv.val_.val(), lambda1_fv.val_.val(),
                          lambda2_fv.val_.val());

  double deriv_denom = exp(lambda1_fv.val_.val()) * theta_fv.val_.val()
                       + exp(lambda2_fv.val_.val()) * (1 - theta_fv.val_.val());
  double theta_deriv
      = 1 / deriv_denom
        * (exp(lambda1_fv.val_.val()) - exp(lambda2_fv.val_.val()));
  double lambda1_deriv
      = 1 / deriv_denom * exp(lambda1_fv.val_.val()) * theta_fv.val_.val();
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val())
                         * (1 - theta_fv.val_.val());
  double deriv = theta_deriv * theta_fv.d_.val()
                 + lambda2_deriv * lambda2_fv.d_.val()
                 + lambda1_deriv * lambda1_fv.d_.val();

  VEC g2_func = cgrad(res.d_, theta_fv.val_, lambda1_fv.val_, lambda2_fv.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, theta_d, lambda1_d,
                             lambda2_d, 0, 0, 0);

  size_t k = 5;
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(auto_calc[k], g2_func[i], 8e-13)
        << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_theta_D3(double theta, double lambda1,
                                                  double lambda2,
                                                  double theta_d,
                                                  double theta_d2) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  theta_ffv.val_.val_ = theta;
  theta_ffv.val_.d_ = theta_d2;
  theta_ffv.d_.val_ = theta_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(), lambda1, lambda2);

  VEC g2_func = cgrad(res.d_.d_, theta_ffv.val_.val_);

  VEC auto_calc
      = log_mix_D3(theta, lambda1, lambda2, theta_d, 0, 0, theta_d2, 0, 0);

  EXPECT_NEAR(auto_calc[8], g2_func[0], 8e-13);

  EXPECT_NEAR(auto_calc[1], res.d_.d_.val(), 8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(), 8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid, lambda1, lambda2), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(double theta, double lambda1,
                                                  double lambda2,
                                                  double lambda1_d,
                                                  double lambda1_d2) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > lambda1_ffv;
  lambda1_ffv.val_.val_ = lambda1;
  lambda1_ffv.val_.d_ = lambda1_d2;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2);
  double result = log_mix(theta, lambda1_ffv.val_.val_.val(), lambda2);

  VEC g2_func = cgrad(res.d_.d_, lambda1_ffv.val_.val_);

  VEC auto_calc
      = log_mix_D3(theta, lambda1, lambda2, 0, lambda1_d, 0, 0, lambda1_d2, 0);

  EXPECT_NEAR(auto_calc[9], g2_func[0], 8e-13);

  EXPECT_NEAR(auto_calc[1], res.d_.d_.val(), 8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(), 8e-13);

  EXPECT_THROW(log_mix(-1.0, lambda1_ffv, lambda2), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(double theta, double lambda1,
                                                  double lambda2,
                                                  double lambda2_d,
                                                  double lambda2_d2) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > lambda2_ffv;
  lambda2_ffv.val_.val_ = lambda2;
  lambda2_ffv.val_.d_ = lambda2_d2;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1, lambda2_ffv);
  double result = log_mix(theta, lambda1, lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_, lambda2_ffv.val_.val_);

  VEC auto_calc
      = log_mix_D3(theta, lambda1, lambda2, 0, 0, lambda2_d, 0, 0, lambda2_d2);

  EXPECT_NEAR(auto_calc[10], g2_func[0], 8e-13);

  EXPECT_NEAR(auto_calc[1], res.d_.d_.val(), 8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(), 8e-13);

  EXPECT_THROW(log_mix(-1.0, lambda1, lambda2_ffv), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_theta_D3(double theta, double lambda1,
                                              double lambda2, double lambda1_d,
                                              double lambda2_d,
                                              double lambda1_d2,
                                              double lambda2_d2) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  lambda1_ffv.val_.d_ = lambda1_d2;
  lambda2_ffv.val_.d_ = lambda2_d2;

  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2_ffv);
  double result = log_mix(theta, lambda1_ffv.val_.val_.val(),
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_, lambda1_ffv.val_.val_, lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, 0, lambda1_d, lambda2_d,
                             0, lambda1_d2, lambda2_d2);

  size_t k = 9;
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(auto_calc[k], g2_func[i], 8e-13)
        << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_NEAR(auto_calc[1], res.d_.d_.val(), 8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(), 8e-13);

  EXPECT_THROW(log_mix(-1.0, lambda1_ffv, lambda2_ffv), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(double theta, double lambda1,
                                              double lambda2, double theta_d,
                                              double lambda2_d,
                                              double lambda2_d2,
                                              double theta_d2) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d2;
  lambda2_ffv.val_.d_ = lambda2_d2;

  theta_ffv.d_.val_ = theta_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2_ffv);
  double result = log_mix(theta_ffv.val_.val_.val(), lambda1,
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_, theta_ffv.val_.val_, lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, theta_d, 0, lambda2_d,
                             theta_d2, 0, lambda2_d2);

  EXPECT_NEAR(auto_calc[8], g2_func[0], 8e-13);
  EXPECT_NEAR(auto_calc[10], g2_func[1], 8e-13);

  EXPECT_NEAR(auto_calc[1], res.d_.d_.val(), 8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(), 8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid, lambda1, lambda2_ffv),
               std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(double theta, double lambda1,
                                              double lambda2, double theta_d,
                                              double lambda1_d,
                                              double lambda1_d2,
                                              double theta_d2) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;

  theta_ffv.val_.d_ = theta_d2;
  lambda1_ffv.val_.d_ = lambda1_d2;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(),
                          lambda1_ffv.val_.val_.val(), lambda2);

  VEC g2_func = cgrad(res.d_.d_, theta_ffv.val_.val_, lambda1_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, theta_d, lambda1_d, 0,
                             theta_d2, lambda1_d2, 0);

  size_t k = 8;
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(auto_calc[k], g2_func[i], 8e-13)
        << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_NEAR(auto_calc[1], res.d_.d_.val(), 8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(), 8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid, lambda1_ffv, lambda2),
               std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_theta_D2(double theta, double lambda1,
                                                  double lambda2,
                                                  double theta_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  theta_ffv.val_.val_ = theta;
  theta_ffv.val_.d_ = theta_d;
  theta_ffv.d_.val_ = theta_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(), lambda1, lambda2);

  VEC g2_func = cgrad(res.d_.val_, theta_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, theta_d, 0, 0, 0, 0, 0);

  EXPECT_NEAR(auto_calc[5], g2_func[0], 8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid, lambda1, lambda2), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(double theta, double lambda1,
                                                  double lambda2,
                                                  double lambda1_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > lambda1_ffv;
  lambda1_ffv.val_.val_ = lambda1;
  lambda1_ffv.val_.d_ = lambda1_d;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2);
  double result = log_mix(theta, lambda1_ffv.val_.val_.val(), lambda2);

  VEC g2_func = cgrad(res.d_.val_, lambda1_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, 0, lambda1_d, 0, 0, 0, 0);

  EXPECT_NEAR(auto_calc[6], g2_func[0], 8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  EXPECT_THROW(log_mix(-1.0, lambda1_ffv, lambda2), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(double theta, double lambda1,
                                                  double lambda2,
                                                  double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > lambda2_ffv;
  lambda2_ffv.val_.val_ = lambda2;
  lambda2_ffv.val_.d_ = lambda2_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1, lambda2_ffv);
  double result = log_mix(theta, lambda1, lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.val_, lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, 0, 0, lambda2_d, 0, 0, 0);

  EXPECT_NEAR(auto_calc[7], g2_func[0], 8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  EXPECT_THROW(log_mix(-1.0, lambda1, lambda2_ffv), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(double theta, double lambda1,
                                              double lambda2, double theta_d,
                                              double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d;
  lambda2_ffv.val_.d_ = lambda2_d;

  theta_ffv.d_.val_ = theta_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2_ffv);
  double result = log_mix(theta_ffv.val_.val_.val(), lambda1,
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.val_, theta_ffv.val_.val_, lambda2_ffv.val_.val_);

  VEC auto_calc
      = log_mix_D3(theta, lambda1, lambda2, theta_d, 0, lambda2_d, 0, 0, 0);

  EXPECT_NEAR(auto_calc[5], g2_func[0], 8e-13);
  EXPECT_NEAR(auto_calc[7], g2_func[1], 8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid, lambda1, lambda2_ffv),
               std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(double theta, double lambda1,
                                              double lambda2, double theta_d,
                                              double lambda1_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;

  theta_ffv.val_.d_ = theta_d;
  lambda1_ffv.val_.d_ = lambda1_d;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(),
                          lambda1_ffv.val_.val_.val(), lambda2);

  VEC g2_func = cgrad(res.d_.val_, theta_ffv.val_.val_, lambda1_ffv.val_.val_);

  VEC auto_calc
      = log_mix_D3(theta, lambda1, lambda2, theta_d, lambda1_d, 0, 0, 0, 0);

  size_t k = 5;
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(auto_calc[k], g2_func[i], 8e-13)
        << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid, lambda1_ffv, lambda2),
               std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_theta_D2(double theta, double lambda1,
                                              double lambda2, double lambda1_d,
                                              double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  lambda1_ffv.val_.d_ = lambda1_d;
  lambda2_ffv.val_.d_ = lambda2_d;

  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2_ffv);
  double result = log_mix(theta, lambda1_ffv.val_.val_.val(),
                          lambda2_ffv.val_.val_.val());

  VEC g2_func
      = cgrad(res.d_.val_, lambda1_ffv.val_.val_, lambda2_ffv.val_.val_);

  VEC auto_calc
      = log_mix_D3(theta, lambda1, lambda2, 0, lambda1_d, lambda2_d, 0, 0, 0);

  size_t k = 6;
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(auto_calc[k], g2_func[i], 8e-13)
        << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  EXPECT_THROW(log_mix(-1.0, lambda1_ffv, lambda2_ffv), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_3xfvar_fvar_var_D3(double theta, double lambda1,
                                     double lambda2, double theta_d,
                                     double lambda1_d, double lambda2_d,
                                     double lambda1_d2, double lambda2_d2,
                                     double theta_d2) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d2;
  lambda1_ffv.val_.d_ = lambda1_d2;
  lambda2_ffv.val_.d_ = lambda2_d2;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2_ffv);
  double result
      = log_mix(theta_ffv.val_.val_.val(), lambda1_ffv.val_.val_.val(),
                lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_, theta_ffv.val_.val_, lambda1_ffv.val_.val_,
                      lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, theta_d, lambda1_d,
                             lambda2_d, theta_d2, lambda1_d2, lambda2_d2);

  size_t k = 8;
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(auto_calc[k], g2_func[i], 8e-13)
        << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_NEAR(auto_calc[1], res.d_.d_.val(), 8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(), 8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid, lambda1_ffv, lambda2_ffv),
               std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_3xfvar_fvar_var_D2(double theta, double lambda1,
                                     double lambda2, double theta_d,
                                     double lambda1_d, double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;
  using std::pow;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d;
  lambda1_ffv.val_.d_ = lambda1_d;
  lambda2_ffv.val_.d_ = lambda2_d;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2_ffv);
  double result
      = log_mix(theta_ffv.val_.val_.val(), lambda1_ffv.val_.val_.val(),
                lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.val_, theta_ffv.val_.val_, lambda1_ffv.val_.val_,
                      lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2, theta_d, lambda1_d,
                             lambda2_d, 0, 0, 0);

  size_t k = 5;
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(auto_calc[k], g2_func[i], 8e-13)
        << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val_.val(), auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_var_lam_2_double(double theta, double lambda1,
                                          double lambda2, double theta_d,
                                          double lambda1_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;

  fvar<var> theta_fv(theta, theta_d);
  fvar<var> lambda1_fv(lambda1, lambda1_d);

  fvar<var> res = log_mix(theta_fv, lambda1_fv, lambda2);
  double result = log_mix(theta_fv.val_.val(), lambda1_fv.val_.val(), lambda2);
  double deriv_denom = exp(lambda1_fv.val_.val()) * theta_fv.val_.val()
                       + exp(lambda2) * (1 - theta_fv.val_.val());
  double theta_deriv
      = 1 / deriv_denom * (exp(lambda1_fv.val_.val()) - exp(lambda2));
  double lambda1_deriv
      = 1 / deriv_denom * exp(lambda1_fv.val_.val()) * theta_fv.val_.val();
  double deriv
      = theta_deriv * theta_fv.d_.val() + lambda1_deriv * lambda1_fv.d_.val();

  AVEC y = createAVEC(theta_fv.val_, lambda1_fv.val_);
  VEC g;
  res.val_.grad(y, g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv, g[0]);
  EXPECT_FLOAT_EQ(lambda1_deriv, g[1]);

  fvar<var> theta_fv_invalid(-1.0, theta_d);

  EXPECT_THROW(log_mix(theta_fv_invalid, lambda1_fv, lambda2),
               std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_var_lam_1_double(double theta, double lambda1,
                                          double lambda2, double theta_d,
                                          double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;

  fvar<var> theta_fv(theta, theta_d);
  fvar<var> lambda2_fv(lambda2, lambda2_d);

  fvar<var> res = log_mix(theta_fv, lambda1, lambda2_fv);
  double result = log_mix(theta_fv.val_.val(), lambda1, lambda2_fv.val_.val());
  double deriv_denom = exp(lambda1) * theta_fv.val_.val()
                       + exp(lambda2_fv.val_.val()) * (1 - theta_fv.val_.val());
  double theta_deriv
      = 1 / deriv_denom * (exp(lambda1) - exp(lambda2_fv.val_.val()));
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val())
                         * (1 - theta_fv.val_.val());
  double deriv
      = theta_deriv * theta_fv.d_.val() + lambda2_deriv * lambda2_fv.d_.val();

  AVEC y = createAVEC(theta_fv.val_, lambda2_fv.val_);
  VEC g;
  res.val_.grad(y, g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv, g[0]);
  EXPECT_FLOAT_EQ(lambda2_deriv, g[1]);

  fvar<var> theta_fv_invalid(-1.0, theta_d);

  EXPECT_THROW(log_mix(theta_fv_invalid, lambda1, lambda2_fv),
               std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_var_theta_double(double theta, double lambda1,
                                          double lambda2, double lambda1_d,
                                          double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;

  fvar<var> lambda1_fv(lambda1, lambda1_d);
  fvar<var> lambda2_fv(lambda2, lambda2_d);

  fvar<var> res = log_mix(theta, lambda1_fv, lambda2_fv);
  double result = log_mix(theta, lambda1_fv.val_.val(), lambda2_fv.val_.val());
  double deriv_denom = exp(lambda1_fv.val_.val()) * theta
                       + exp(lambda2_fv.val_.val()) * (1 - theta);
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_fv.val_.val()) * theta;
  double lambda2_deriv
      = 1 / deriv_denom * exp(lambda2_fv.val_.val()) * (1 - theta);
  double deriv = lambda2_deriv * lambda2_fv.d_.val()
                 + lambda1_deriv * lambda1_fv.d_.val();

  AVEC y = createAVEC(lambda1_fv.val_, lambda2_fv.val_);
  VEC g;
  res.val_.grad(y, g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(lambda1_deriv, g[0]);
  EXPECT_FLOAT_EQ(lambda2_deriv, g[1]);

  EXPECT_THROW(log_mix(-1.0, lambda1_fv, lambda2_fv), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_theta_fvar_var(double theta, double lambda1,
                                          double lambda2, double theta_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;

  fvar<var> theta_fv(theta, theta_d);

  fvar<var> res = log_mix(theta_fv, lambda1, lambda2);
  double result = log_mix(theta_fv.val_.val(), lambda1, lambda2);
  double deriv_denom = exp(lambda1) * theta_fv.val_.val()
                       + exp(lambda2) * (1 - theta_fv.val_.val());
  double theta_deriv = 1 / deriv_denom * (exp(lambda1) - exp(lambda2));
  double deriv = theta_deriv * theta_fv.d_.val();

  AVEC y = createAVEC(theta_fv.val_);
  VEC g;
  res.val_.grad(y, g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv, g[0]);

  fvar<var> theta_fv_invalid(-1.0, theta_d);

  EXPECT_THROW(log_mix(theta_fv_invalid, lambda1, lambda2), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_lam_1_fvar_var(double theta, double lambda1,
                                          double lambda2, double lambda1_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;

  fvar<var> lambda1_fv(lambda1, lambda1_d);

  fvar<var> res = log_mix(theta, lambda1_fv, lambda2);
  double result = log_mix(theta, lambda1_fv.val_.val(), lambda2);
  double deriv_denom
      = exp(lambda1_fv.val_.val()) * theta + exp(lambda2) * (1 - theta);
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_fv.val_.val()) * theta;
  double deriv = lambda1_deriv * lambda1_fv.d_.val();

  AVEC y = createAVEC(lambda1_fv.val_);
  VEC g;
  res.val_.grad(y, g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(lambda1_deriv, g[0]);

  EXPECT_THROW(log_mix(-1.0, lambda1_fv, lambda2), std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_lam_2_fvar_var(double theta, double lambda1,
                                          double lambda2, double lambda2_d) {
  using stan::math::fvar;
  using stan::math::log_mix;
  using stan::math::var;
  using std::exp;

  fvar<var> lambda2_fv(lambda2, lambda2_d);

  fvar<var> res = log_mix(theta, lambda1, lambda2_fv);
  double result = log_mix(theta, lambda1, lambda2_fv.val_.val());
  double deriv_denom
      = exp(lambda1) * theta + exp(lambda2_fv.val_.val()) * (1 - theta);
  double lambda2_deriv
      = 1 / deriv_denom * exp(lambda2_fv.val_.val()) * (1 - theta);
  double deriv = lambda2_deriv * lambda2_fv.d_.val();

  AVEC y = createAVEC(lambda2_fv.val_);
  VEC g;
  res.val_.grad(y, g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(lambda2_deriv, g[0]);

  EXPECT_THROW(log_mix(-1.0, lambda1, lambda2_fv), std::domain_error);
  stan::math::recover_memory();
}

TEST(AgradFwdLogMix, FvarFvarVar_Double_Double_D3) {
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, 6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 2.0, 6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 1.0, 2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, -6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, -6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 2.0, -6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 1.0, -2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, 6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 2.0, 6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 1.0, 2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, -6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, -6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 2.0, -6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 1.0, -2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, 6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 2.0, 6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 1.0, 2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, -6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, -6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 2.0, -6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 1.0, -2.0, 0, 1);
}

TEST(AgradFwdLogMix, FvarFvarVar_Double_Double_D2) {
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.3, 2.0, -6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.3, 2.0, -6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.3, 2.0, -6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.3, 2.0, -6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.3, 2.0, -6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.3, 2.0, -6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.3, 2.0, -6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.3, 2.0, -6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.3, 2.0, -6.0, 0);
}

TEST(AgradFwdLogMix, FvarVar_FvarVar_FvarVar_D1) {
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1.3, 5.0, 3.0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 0, 0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 0, 1, 0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 0, 0, 1);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 0, 1);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 0, 1, 1);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 1, 0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 1, 1);

  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1.3, 2.0, 3.0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 0, 0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 0, 1, 0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 0, 0, 1);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 0, 1);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 0, 1, 1);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 1, 0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 1, 1);
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_Double_D3) {
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 0, 1, 1);
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_Double_D2) {
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 0.0, 1.0);
}

TEST(AgradFwdLogMix, FvarVar_FvarVar_FvarVar_D2) {
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.3, 5.0, 3.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.7, 4.0, 5.0, 0.0, 1.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 1.0);

  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.3, 2.0, 3.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.1);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.1);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 1.0);
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_FvarFvarVar_D2) {
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.3, 5.0, 3.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 4.0, 5.0, 0.0, 1.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 1.0);

  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.3, 2.0, 3.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.1);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.1);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 1.0);
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_FvarFvarVar_D3) {
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.3, 5.0, 3.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 0.0, 0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 0.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 0.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 0.0, 1.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 4.0, 5.0, 0.0, 1.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 1.0, 1.0, 3.0, 4.0, 1.0);

  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.3, 2.0, 3.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 0.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.1, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.1, 1.1, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.0, 1.1, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 1.0, 1.0, 3.0, 4.0, 1.0);
}

TEST(AgradFwdLogMix, FvarVar_FvarVar_Double) {
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 1.3, 2.0);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 1.3, 2.0);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 1.3, 2.0);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 0, 0);
}

TEST(AgradFwdLogMix, FvarVar_Double_Double) {
  test_log_mix_2xdouble_theta_fvar_var(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_theta_fvar_var(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_theta_fvar_var(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_theta_fvar_var(0.2, -2.0, 6.0, 1);
  test_log_mix_2xdouble_theta_fvar_var(0.2, -2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_1_fvar_var(0.7, 2.0, 6.0, 0.3);
  test_log_mix_2xdouble_lam_1_fvar_var(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_1_fvar_var(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_1_fvar_var(0.2, -2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_1_fvar_var(0.2, -2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_2_fvar_var(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_lam_2_fvar_var(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_2_fvar_var(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_2_fvar_var(0.2, -2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_2_fvar_var(0.2, -2.0, 6.0, 0);
}

struct log_mix_fun {
  template <typename T0, typename T1, typename T2>
  inline typename boost::math::tools::promote_args<T0, T1, T2>::type operator()(
      const T0 arg1, const T1 arg2, const T2 arg3) const {
    return log_mix(arg1, arg2, arg3);
  }
};

TEST(AgradFwdLogMix, log_mix_NaN) {
  log_mix_fun log_mix_;
  test_nan_mix(log_mix_, 0.7, 3.0, 5.0, true);
}





using stan::math::fvar;
using stan::math::log_mix;
using stan::math::row_vector_d;
using stan::math::row_vector_ffv;
using stan::math::row_vector_fv;
using stan::math::var;
using stan::math::vector_d;
using stan::math::vector_ffv;
using stan::math::vector_fv;

template <typename T_a, typename T_b>
void fv_fv_test(T_a a, T_b b) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;

  b[0].val_ = -3.581;
  b[1].val_ = -8.114;
  b[2].val_ = -11.215;
  b[3].val_ = -5.658;
  b[0].d_ = 1.0;
  b[1].d_ = 1.0;
  b[2].d_ = 1.0;
  b[3].d_ = 1.0;

  fvar<var> out = log_mix(a, b);

  EXPECT_FLOAT_EQ(out.val_.val(), -4.218931574);
  EXPECT_FLOAT_EQ(out.d_.val(), 3.150968236);
  out.d_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.892562202198, 0.020341982471, 0.000915474153, 0.237148577149;
  vector_d dens_deriv(4);
  dens_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.adj(), prob_deriv[i]);
    EXPECT_FLOAT_EQ(b[i].d_.adj(), dens_deriv[i]);
  }
}

template <typename T_a, typename T_b>
void fv_fv_vec_test(T_a a, T_b b1) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;

  b1[0].val_ = -3.581;
  b1[1].val_ = -8.114;
  b1[2].val_ = -11.215;
  b1[3].val_ = -5.658;
  b1[0].d_ = 1.0;
  b1[1].d_ = 1.0;
  b1[2].d_ = 1.0;
  b1[3].d_ = 1.0;

  T_b b2(4);

  b2[0].val_ = -8.594;
  b2[1].val_ = -3.251;
  b2[2].val_ = -7.281;
  b2[3].val_ = -3.556;
  b2[0].d_ = 1.0;
  b2[1].d_ = 1.0;
  b2[2].d_ = 1.0;
  b2[3].d_ = 1.0;

  T_b b3(4);

  b3[0].val_ = -11.554;
  b3[1].val_ = -6.628;
  b3[2].val_ = -15.229;
  b3[3].val_ = -9.561;
  b3[0].d_ = 1.0;
  b3[1].d_ = 1.0;
  b3[2].d_ = 1.0;
  b3[3].d_ = 1.0;

  std::vector<T_b> c{b1, b2, b3};

  fvar<var> out = log_mix(a, c);

  EXPECT_FLOAT_EQ(out.val_.val(), -16.36331174);
  EXPECT_FLOAT_EQ(out.d_.val(), 13.73648479);
  out.d_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.930840738468, 6.257235890773, 0.051642416014, 2.496765742829;
  vector_d dens1_deriv(4);
  dens1_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;
  vector_d dens2_deriv(4);
  dens2_deriv << 0.00692718708, 0.80047459791, 0.00561100267, 0.18698721234;
  vector_d dens3_deriv(4);
  dens3_deriv << 0.01274798056, 0.97080327205, 0.00007041482, 0.01637833257;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.adj(), prob_deriv[i]);
    EXPECT_FLOAT_EQ(c[0][i].d_.adj(), dens1_deriv[i]);
    EXPECT_FLOAT_EQ(c[1][i].d_.adj(), dens2_deriv[i]);
    EXPECT_FLOAT_EQ(c[2][i].d_.adj(), dens3_deriv[i]);
  }
}

TEST(AgradMixMatrixLogMix_mat, fv_fv) {
  vector_fv vecfv_prob(4);
  vector_fv vecfv_dens(4);
  row_vector_fv row_vecfv_prob(4);
  row_vector_fv row_vecfv_dens(4);
  std::vector<fvar<var> > std_vecfv_prob(4);
  std::vector<fvar<var> > std_vecfv_dens(4);

  fv_fv_test(vecfv_prob, vecfv_dens);
  fv_fv_test(vecfv_prob, row_vecfv_dens);
  fv_fv_test(vecfv_prob, std_vecfv_dens);

  fv_fv_vec_test(vecfv_prob, vecfv_dens);
  fv_fv_vec_test(vecfv_prob, row_vecfv_dens);
  fv_fv_vec_test(vecfv_prob, std_vecfv_dens);

  fv_fv_test(row_vecfv_prob, vecfv_dens);
  fv_fv_test(row_vecfv_prob, row_vecfv_dens);
  fv_fv_test(row_vecfv_prob, std_vecfv_dens);

  fv_fv_vec_test(row_vecfv_prob, vecfv_dens);
  fv_fv_vec_test(row_vecfv_prob, row_vecfv_dens);
  fv_fv_vec_test(row_vecfv_prob, std_vecfv_dens);

  fv_fv_test(std_vecfv_prob, vecfv_dens);
  fv_fv_test(std_vecfv_prob, row_vecfv_dens);
  fv_fv_test(std_vecfv_prob, std_vecfv_dens);

  fv_fv_vec_test(std_vecfv_prob, vecfv_dens);
  fv_fv_vec_test(std_vecfv_prob, row_vecfv_dens);
  fv_fv_vec_test(std_vecfv_prob, std_vecfv_dens);
}

template <typename T_a, typename T_b>
void fv_d_test(T_a a, T_b b) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;

  b[0] = -3.581;
  b[1] = -8.114;
  b[2] = -11.215;
  b[3] = -5.658;

  fvar<var> out = log_mix(a, b);

  EXPECT_FLOAT_EQ(out.val_.val(), -4.2189315474);
  EXPECT_FLOAT_EQ(out.d_.val(), 2.150968235971);
  out.d_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.892562202198, 0.020341982471, 0.000915474153, 0.237148577149;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.adj(), prob_deriv[i]);
  }
}

template <typename T_a, typename T_b>
void fv_d_vec_test(T_a a, T_b b1) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;

  b1[0] = -3.581;
  b1[1] = -8.114;
  b1[2] = -11.215;
  b1[3] = -5.658;

  T_b b2(4);

  b2[0] = -8.594;
  b2[1] = -3.251;
  b2[2] = -7.281;
  b2[3] = -3.556;

  T_b b3(4);

  b3[0] = -11.554;
  b3[1] = -6.628;
  b3[2] = -15.229;
  b3[3] = -9.561;

  std::vector<T_b> c{b1, b2, b3};

  fvar<var> out = log_mix(a, c);

  EXPECT_FLOAT_EQ(out.val_.val(), -16.36331174);
  EXPECT_FLOAT_EQ(out.d_.val(), 10.73648479);
  out.d_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.930840738468, 6.257235890773, 0.051642416014, 2.496765742829;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.adj(), prob_deriv[i]);
  }
}

TEST(AgradMixMatrixLogMix_mat, fv_d) {
  vector_fv vecfv_prob(4);
  vector_d vecd_dens(4);
  row_vector_fv row_vecfv_prob(4);
  row_vector_d row_vecd_dens(4);
  std::vector<fvar<var> > std_vecfv_prob(4);
  std::vector<double> std_vecd_dens(4);

  fv_d_test(vecfv_prob, vecd_dens);
  fv_d_test(vecfv_prob, row_vecd_dens);
  fv_d_test(vecfv_prob, std_vecd_dens);

  fv_d_vec_test(vecfv_prob, vecd_dens);
  fv_d_vec_test(vecfv_prob, row_vecd_dens);
  fv_d_vec_test(vecfv_prob, std_vecd_dens);

  fv_d_test(row_vecfv_prob, vecd_dens);
  fv_d_test(row_vecfv_prob, row_vecd_dens);
  fv_d_test(row_vecfv_prob, std_vecd_dens);

  fv_d_vec_test(row_vecfv_prob, vecd_dens);
  fv_d_vec_test(row_vecfv_prob, row_vecd_dens);
  fv_d_vec_test(row_vecfv_prob, std_vecd_dens);

  fv_d_test(std_vecfv_prob, vecd_dens);
  fv_d_test(std_vecfv_prob, row_vecd_dens);
  fv_d_test(std_vecfv_prob, std_vecd_dens);

  fv_d_vec_test(std_vecfv_prob, vecd_dens);
  fv_d_vec_test(std_vecfv_prob, row_vecd_dens);
  fv_d_vec_test(std_vecfv_prob, std_vecd_dens);
}

template <typename T_a, typename T_b>
void d_fv_test(T_a a, T_b b) {
  a[0] = 0.514;
  a[1] = 0.284;
  a[2] = 0.112;
  a[3] = 0.090;

  b[0].val_ = -3.581;
  b[1].val_ = -8.114;
  b[2].val_ = -11.215;
  b[3].val_ = -5.658;
  b[0].d_ = 1.0;
  b[1].d_ = 1.0;
  b[2].d_ = 1.0;
  b[3].d_ = 1.0;

  fvar<var> out = log_mix(a, b);

  EXPECT_FLOAT_EQ(out.val_.val(), -4.218931574);
  EXPECT_FLOAT_EQ(out.d_.val(), 1.0);
  out.d_.grad();

  vector_d dens_deriv(4);
  dens_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(b[i].d_.adj(), dens_deriv[i]);
  }
}

template <typename T_a, typename T_b>
void d_fv_vec_test(T_a a, T_b b1) {
  a[0] = 0.514;
  a[1] = 0.284;
  a[2] = 0.112;
  a[3] = 0.090;

  b1[0].val_ = -3.581;
  b1[1].val_ = -8.114;
  b1[2].val_ = -11.215;
  b1[3].val_ = -5.658;
  b1[0].d_ = 1.0;
  b1[1].d_ = 1.0;
  b1[2].d_ = 1.0;
  b1[3].d_ = 1.0;

  T_b b2(4);

  b2[0].val_ = -8.594;
  b2[1].val_ = -3.251;
  b2[2].val_ = -7.281;
  b2[3].val_ = -3.556;
  b2[0].d_ = 1.0;
  b2[1].d_ = 1.0;
  b2[2].d_ = 1.0;
  b2[3].d_ = 1.0;

  T_b b3(4);

  b3[0].val_ = -11.554;
  b3[1].val_ = -6.628;
  b3[2].val_ = -15.229;
  b3[3].val_ = -9.561;
  b3[0].d_ = 1.0;
  b3[1].d_ = 1.0;
  b3[2].d_ = 1.0;
  b3[3].d_ = 1.0;

  std::vector<T_b> c{b1, b2, b3};

  fvar<var> out = log_mix(a, c);

  EXPECT_FLOAT_EQ(out.val_.val(), -16.36331174);
  EXPECT_FLOAT_EQ(out.d_.val(), 3.0);
  out.d_.grad();

  vector_d dens1_deriv(4);
  dens1_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;
  vector_d dens2_deriv(4);
  dens2_deriv << 0.00692718708, 0.80047459791, 0.00561100267, 0.18698721234;
  vector_d dens3_deriv(4);
  dens3_deriv << 0.01274798056, 0.97080327205, 0.00007041482, 0.01637833257;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(c[0][i].d_.adj(), dens1_deriv[i]);
    EXPECT_FLOAT_EQ(c[1][i].d_.adj(), dens2_deriv[i]);
    EXPECT_FLOAT_EQ(c[2][i].d_.adj(), dens3_deriv[i]);
  }
}

TEST(AgradMixMatrixLogMix_mat, d_fv) {
  vector_d vecd_prob(4);
  vector_fv vecfv_dens(4);
  row_vector_d row_vecd_prob(4);
  row_vector_fv row_vecfv_dens(4);
  std::vector<double> std_vecd_prob(4);
  std::vector<fvar<var> > std_vecfv_dens(4);

  d_fv_test(vecd_prob, vecfv_dens);
  d_fv_test(vecd_prob, row_vecfv_dens);
  d_fv_test(vecd_prob, std_vecfv_dens);

  d_fv_vec_test(vecd_prob, vecfv_dens);
  d_fv_vec_test(vecd_prob, row_vecfv_dens);
  d_fv_vec_test(vecd_prob, std_vecfv_dens);

  d_fv_test(row_vecd_prob, vecfv_dens);
  d_fv_test(row_vecd_prob, row_vecfv_dens);
  d_fv_test(row_vecd_prob, std_vecfv_dens);

  d_fv_vec_test(row_vecd_prob, vecfv_dens);
  d_fv_vec_test(row_vecd_prob, row_vecfv_dens);
  d_fv_vec_test(row_vecd_prob, std_vecfv_dens);

  d_fv_test(std_vecd_prob, vecfv_dens);
  d_fv_test(std_vecd_prob, row_vecfv_dens);
  d_fv_test(std_vecd_prob, std_vecfv_dens);

  d_fv_vec_test(std_vecd_prob, vecfv_dens);
  d_fv_vec_test(std_vecd_prob, row_vecfv_dens);
  d_fv_vec_test(std_vecd_prob, std_vecfv_dens);
}

template <typename T_a, typename T_b>
void ffv_ffv_test(T_a a, T_b b) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;
  a[0].val_.d_ = 1.0;
  a[1].val_.d_ = 1.0;
  a[2].val_.d_ = 1.0;
  a[3].val_.d_ = 1.0;

  b[0].val_ = -3.581;
  b[1].val_ = -8.114;
  b[2].val_ = -11.215;
  b[3].val_ = -5.658;
  b[0].d_ = 1.0;
  b[1].d_ = 1.0;
  b[2].d_ = 1.0;
  b[3].d_ = 1.0;
  b[0].val_.d_ = 1.0;
  b[1].val_.d_ = 1.0;
  b[2].val_.d_ = 1.0;
  b[3].val_.d_ = 1.0;

  fvar<fvar<var> > out = log_mix(a, b);

  EXPECT_FLOAT_EQ(out.val_.val_.val(), -4.218931574);
  EXPECT_FLOAT_EQ(out.d_.val_.val(), 3.150968236);
  out.d_.val_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.892562202198, 0.020341982471, 0.000915474153, 0.237148577149;
  vector_d dens_deriv(4);
  dens_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.val_.adj(), prob_deriv[i]);
    EXPECT_FLOAT_EQ(b[i].d_.val_.adj(), dens_deriv[i]);
  }
}

template <typename T_a, typename T_b>
void ffv_ffv_vec_test(T_a a, T_b b1) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;
  a[0].val_.d_ = 1.0;
  a[1].val_.d_ = 1.0;
  a[2].val_.d_ = 1.0;
  a[3].val_.d_ = 1.0;

  b1[0].val_ = -3.581;
  b1[1].val_ = -8.114;
  b1[2].val_ = -11.215;
  b1[3].val_ = -5.658;

  b1[0].d_ = 1.0;
  b1[1].d_ = 1.0;
  b1[2].d_ = 1.0;
  b1[3].d_ = 1.0;
  b1[0].val_.d_ = 1.0;
  b1[1].val_.d_ = 1.0;
  b1[2].val_.d_ = 1.0;
  b1[3].val_.d_ = 1.0;

  T_b b2(4), b3(4);

  b2[0].val_ = -8.594;
  b2[1].val_ = -3.251;
  b2[2].val_ = -7.281;
  b2[3].val_ = -3.556;

  b2[0].d_ = 1.0;
  b2[1].d_ = 1.0;
  b2[2].d_ = 1.0;
  b2[3].d_ = 1.0;
  b2[0].val_.d_ = 1.0;
  b2[1].val_.d_ = 1.0;
  b2[2].val_.d_ = 1.0;
  b2[3].val_.d_ = 1.0;

  b3[0].val_ = -11.554;
  b3[1].val_ = -6.628;
  b3[2].val_ = -15.229;
  b3[3].val_ = -9.561;

  b3[0].d_ = 1.0;
  b3[1].d_ = 1.0;
  b3[2].d_ = 1.0;
  b3[3].d_ = 1.0;
  b3[0].val_.d_ = 1.0;
  b3[1].val_.d_ = 1.0;
  b3[2].val_.d_ = 1.0;
  b3[3].val_.d_ = 1.0;

  std::vector<T_b> c{b1, b2, b3};

  fvar<fvar<var> > out = log_mix(a, c);

  EXPECT_FLOAT_EQ(out.val_.val_.val(), -16.36331174);
  EXPECT_FLOAT_EQ(out.d_.val_.val(), 13.73648479);
  out.d_.val_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.930840738468, 6.257235890773, 0.051642416014, 2.496765742829;
  vector_d dens1_deriv(4);
  dens1_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;
  vector_d dens2_deriv(4);
  dens2_deriv << 0.00692718708, 0.80047459791, 0.00561100267, 0.18698721234;
  vector_d dens3_deriv(4);
  dens3_deriv << 0.01274798056, 0.97080327205, 0.00007041482, 0.01637833257;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.val_.adj(), prob_deriv[i]);
    EXPECT_FLOAT_EQ(c[0][i].d_.val_.adj(), dens1_deriv[i]);
    EXPECT_FLOAT_EQ(c[1][i].d_.val_.adj(), dens2_deriv[i]);
    EXPECT_FLOAT_EQ(c[2][i].d_.val_.adj(), dens3_deriv[i]);
  }
}

TEST(AgradMixMatrixLogMix_mat, ffv_ffv) {
  vector_ffv vecffv_prob(4);
  vector_ffv vecffv_dens(4);
  row_vector_ffv row_vecffv_prob(4);
  row_vector_ffv row_vecffv_dens(4);
  std::vector<fvar<fvar<var> > > std_vecffv_prob(4);
  std::vector<fvar<fvar<var> > > std_vecffv_dens(4);

  ffv_ffv_test(vecffv_prob, vecffv_dens);
  ffv_ffv_test(vecffv_prob, row_vecffv_dens);
  ffv_ffv_test(vecffv_prob, std_vecffv_dens);

  ffv_ffv_vec_test(vecffv_prob, vecffv_dens);
  ffv_ffv_vec_test(vecffv_prob, row_vecffv_dens);
  ffv_ffv_vec_test(vecffv_prob, std_vecffv_dens);

  ffv_ffv_test(row_vecffv_prob, vecffv_dens);
  ffv_ffv_test(row_vecffv_prob, row_vecffv_dens);
  ffv_ffv_test(row_vecffv_prob, std_vecffv_dens);

  ffv_ffv_test(std_vecffv_prob, vecffv_dens);
  ffv_ffv_test(std_vecffv_prob, row_vecffv_dens);
  ffv_ffv_test(std_vecffv_prob, std_vecffv_dens);
}

template <typename T_a, typename T_b>
void ffv_d_test(T_a a, T_b b) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;
  a[0].val_.d_ = 1.0;
  a[1].val_.d_ = 1.0;
  a[2].val_.d_ = 1.0;
  a[3].val_.d_ = 1.0;

  b[0] = -3.581;
  b[1] = -8.114;
  b[2] = -11.215;
  b[3] = -5.658;

  fvar<fvar<var> > out = log_mix(a, b);

  EXPECT_FLOAT_EQ(out.val_.val_.val(), -4.218931574);
  EXPECT_FLOAT_EQ(out.d_.val_.val(), 2.150968236);
  out.d_.val_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.892562202198, 0.020341982471, 0.000915474153, 0.237148577149;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.val_.adj(), prob_deriv[i]);
  }
}

template <typename T_a, typename T_b>
void ffv_d_vec_test(T_a a, T_b b1) {
  a[0].val_ = 0.514;
  a[1].val_ = 0.284;
  a[2].val_ = 0.112;
  a[3].val_ = 0.090;
  a[0].d_ = 1.0;
  a[1].d_ = 1.0;
  a[2].d_ = 1.0;
  a[3].d_ = 1.0;
  a[0].val_.d_ = 1.0;
  a[1].val_.d_ = 1.0;
  a[2].val_.d_ = 1.0;
  a[3].val_.d_ = 1.0;

  b1[0] = -3.581;
  b1[1] = -8.114;
  b1[2] = -11.215;
  b1[3] = -5.658;

  T_b b2(4), b3(4);

  b2[0] = -8.594;
  b2[1] = -3.251;
  b2[2] = -7.281;
  b2[3] = -3.556;

  b3[0] = -11.554;
  b3[1] = -6.628;
  b3[2] = -15.229;
  b3[3] = -9.561;

  std::vector<T_b> c{b1, b2, b3};

  fvar<fvar<var> > out = log_mix(a, c);

  EXPECT_FLOAT_EQ(out.val_.val_.val(), -16.36331174);
  EXPECT_FLOAT_EQ(out.d_.val_.val(), 10.73648479);
  out.d_.val_.grad();

  vector_d prob_deriv(4);
  prob_deriv << 1.930840738468, 6.257235890773, 0.051642416014, 2.496765742829;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(a[i].d_.val_.adj(), prob_deriv[i]);
  }
}

TEST(AgradMixMatrixLogMix_mat, ffv_d) {
  vector_ffv vecffv_prob(4);
  vector_d vecd_dens(4);
  row_vector_ffv row_vecffv_prob(4);
  row_vector_d row_vecd_dens(4);
  std::vector<fvar<fvar<var> > > std_vecffv_prob(4);
  std::vector<double> std_vecd_dens(4);

  ffv_d_test(vecffv_prob, vecd_dens);
  ffv_d_test(vecffv_prob, row_vecd_dens);
  ffv_d_test(vecffv_prob, std_vecd_dens);

  ffv_d_vec_test(vecffv_prob, vecd_dens);
  ffv_d_vec_test(vecffv_prob, row_vecd_dens);
  ffv_d_vec_test(vecffv_prob, std_vecd_dens);

  ffv_d_test(row_vecffv_prob, vecd_dens);
  ffv_d_test(row_vecffv_prob, row_vecd_dens);
  ffv_d_test(row_vecffv_prob, std_vecd_dens);

  ffv_d_vec_test(row_vecffv_prob, vecd_dens);
  ffv_d_vec_test(row_vecffv_prob, row_vecd_dens);
  ffv_d_vec_test(row_vecffv_prob, std_vecd_dens);

  ffv_d_test(std_vecffv_prob, vecd_dens);
  ffv_d_test(std_vecffv_prob, row_vecd_dens);
  ffv_d_test(std_vecffv_prob, std_vecd_dens);

  ffv_d_vec_test(std_vecffv_prob, vecd_dens);
  ffv_d_vec_test(std_vecffv_prob, row_vecd_dens);
  ffv_d_vec_test(std_vecffv_prob, std_vecd_dens);
}

template <typename T_a, typename T_b>
void d_ffv_test(T_a a, T_b b) {
  a[0] = 0.514;
  a[1] = 0.284;
  a[2] = 0.112;
  a[3] = 0.090;

  b[0].val_ = -3.581;
  b[1].val_ = -8.114;
  b[2].val_ = -11.215;
  b[3].val_ = -5.658;
  b[0].d_ = 1.0;
  b[1].d_ = 1.0;
  b[2].d_ = 1.0;
  b[3].d_ = 1.0;
  b[0].val_.d_ = 1.0;
  b[1].val_.d_ = 1.0;
  b[2].val_.d_ = 1.0;
  b[3].val_.d_ = 1.0;

  fvar<fvar<var> > out = log_mix(a, b);

  EXPECT_FLOAT_EQ(out.val_.val_.val(), -4.218931574);
  EXPECT_FLOAT_EQ(out.d_.val_.val(), 1.0);
  out.d_.val_.grad();

  vector_d dens_deriv(4);
  dens_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(b[i].d_.val_.adj(), dens_deriv[i]);
  }
}

template <typename T_a, typename T_b>
void d_ffv_vec_test(T_a a, T_b b1) {
  a[0] = 0.514;
  a[1] = 0.284;
  a[2] = 0.112;
  a[3] = 0.090;

  b1[0].val_ = -3.581;
  b1[1].val_ = -8.114;
  b1[2].val_ = -11.215;
  b1[3].val_ = -5.658;

  b1[0].d_ = 1.0;
  b1[1].d_ = 1.0;
  b1[2].d_ = 1.0;
  b1[3].d_ = 1.0;
  b1[0].val_.d_ = 1.0;
  b1[1].val_.d_ = 1.0;
  b1[2].val_.d_ = 1.0;
  b1[3].val_.d_ = 1.0;

  T_b b2(4), b3(4);

  b2[0].val_ = -8.594;
  b2[1].val_ = -3.251;
  b2[2].val_ = -7.281;
  b2[3].val_ = -3.556;

  b2[0].d_ = 1.0;
  b2[1].d_ = 1.0;
  b2[2].d_ = 1.0;
  b2[3].d_ = 1.0;
  b2[0].val_.d_ = 1.0;
  b2[1].val_.d_ = 1.0;
  b2[2].val_.d_ = 1.0;
  b2[3].val_.d_ = 1.0;

  b3[0].val_ = -11.554;
  b3[1].val_ = -6.628;
  b3[2].val_ = -15.229;
  b3[3].val_ = -9.561;

  b3[0].d_ = 1.0;
  b3[1].d_ = 1.0;
  b3[2].d_ = 1.0;
  b3[3].d_ = 1.0;
  b3[0].val_.d_ = 1.0;
  b3[1].val_.d_ = 1.0;
  b3[2].val_.d_ = 1.0;
  b3[3].val_.d_ = 1.0;

  std::vector<T_b> c{b1, b2, b3};

  fvar<fvar<var> > out = log_mix(a, c);

  EXPECT_FLOAT_EQ(out.val_.val_.val(), -16.36331174);
  EXPECT_FLOAT_EQ(out.d_.val_.val(), 3.0);
  out.d_.val_.grad();

  vector_d dens1_deriv(4);
  dens1_deriv << 0.97277697193, 0.00577712302, 0.00010253311, 0.02134337194;
  vector_d dens2_deriv(4);
  dens2_deriv << 0.00692718708, 0.80047459791, 0.00561100267, 0.18698721234;
  vector_d dens3_deriv(4);
  dens3_deriv << 0.01274798056, 0.97080327205, 0.00007041482, 0.01637833257;

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(c[0][i].d_.val_.adj(), dens1_deriv[i]);
    EXPECT_FLOAT_EQ(c[1][i].d_.val_.adj(), dens2_deriv[i]);
    EXPECT_FLOAT_EQ(c[2][i].d_.val_.adj(), dens3_deriv[i]);
  }
}

TEST(AgradMixMatrixLogMix_mat, d_ffv) {
  vector_d vecd_prob(4);
  vector_ffv vecffv_dens(4);
  row_vector_d row_vecd_prob(4);
  row_vector_ffv row_vecffv_dens(4);
  std::vector<double> std_vecd_prob(4);
  std::vector<fvar<fvar<var> > > std_vecffv_dens(4);

  d_ffv_test(vecd_prob, vecffv_dens);
  d_ffv_test(vecd_prob, row_vecffv_dens);
  d_ffv_test(vecd_prob, std_vecffv_dens);

  d_ffv_vec_test(vecd_prob, vecffv_dens);
  d_ffv_vec_test(vecd_prob, row_vecffv_dens);
  d_ffv_vec_test(vecd_prob, std_vecffv_dens);

  d_ffv_test(row_vecd_prob, vecffv_dens);
  d_ffv_test(row_vecd_prob, row_vecffv_dens);
  d_ffv_test(row_vecd_prob, std_vecffv_dens);

  d_ffv_vec_test(row_vecd_prob, vecffv_dens);
  d_ffv_vec_test(row_vecd_prob, row_vecffv_dens);
  d_ffv_vec_test(row_vecd_prob, std_vecffv_dens);

  d_ffv_test(std_vecd_prob, vecffv_dens);
  d_ffv_test(std_vecd_prob, row_vecffv_dens);
  d_ffv_test(std_vecd_prob, std_vecffv_dens);

  d_ffv_vec_test(std_vecd_prob, vecffv_dens);
  d_ffv_vec_test(std_vecd_prob, row_vecffv_dens);
  d_ffv_vec_test(std_vecd_prob, std_vecffv_dens);
}

TEST(AgradMixMatrixLogMix_mat, fv_fv_old) {
  auto mix_fv_fv = [](auto a, auto b) {
    a[0].val_ = 0.15;
    a[1].val_ = 0.70;
    a[2].val_ = 0.10;
    a[3].val_ = 0.05;
    a[0].d_ = 1.0;
    a[1].d_ = 1.0;
    a[2].d_ = 1.0;
    a[3].d_ = 1.0;

    b[0].val_ = -1.0;
    b[1].val_ = -2.0;
    b[2].val_ = -3.0;
    b[3].val_ = -4.0;
    b[0].d_ = 1.0;
    b[1].d_ = 1.0;
    b[2].d_ = 1.0;
    b[3].d_ = 1.0;

    fvar<var> out = log_mix(a, b);

    EXPECT_FLOAT_EQ(out.val_.val(), -1.85911088);
    EXPECT_FLOAT_EQ(out.d_.val(), 4.66673118);
    out.d_.grad();

    vector_d prob_deriv(4);
    prob_deriv << 2.3610604993, 0.8685856170, 0.3195347914, 0.1175502804;
    vector_d dens_deriv(4);
    dens_deriv << 0.3541590748, 0.6080099319, 0.0319534791, 0.0058775140;

    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(a[i].d_.adj(), prob_deriv[i]);
      EXPECT_FLOAT_EQ(b[i].d_.adj(), dens_deriv[i]);
    }
  };

  vector_fv vecfv_prob(4);
  vector_fv vecfv_dens(4);
  row_vector_fv row_vecfv_prob(4);
  row_vector_fv row_vecfv_dens(4);
  std::vector<fvar<var> > std_vecfv_prob(4);
  std::vector<fvar<var> > std_vecfv_dens(4);

  mix_fv_fv(vecfv_prob, vecfv_dens);
  mix_fv_fv(vecfv_prob, row_vecfv_dens);
  mix_fv_fv(vecfv_prob, std_vecfv_dens);

  mix_fv_fv(row_vecfv_prob, vecfv_dens);
  mix_fv_fv(row_vecfv_prob, row_vecfv_dens);
  mix_fv_fv(row_vecfv_prob, std_vecfv_dens);

  mix_fv_fv(std_vecfv_prob, vecfv_dens);
  mix_fv_fv(std_vecfv_prob, row_vecfv_dens);
  mix_fv_fv(std_vecfv_prob, std_vecfv_dens);
}

TEST(AgradMixMatrixLogMix_mat, fv_d_old) {
  auto mix_fv_d = [](auto a, auto b) {
    a[0].val_ = 0.15;
    a[1].val_ = 0.70;
    a[2].val_ = 0.10;
    a[3].val_ = 0.05;
    a[0].d_ = 1.0;
    a[1].d_ = 1.0;
    a[2].d_ = 1.0;
    a[3].d_ = 1.0;

    b[0] = -1.0;
    b[1] = -2.0;
    b[2] = -3.0;
    b[3] = -4.0;

    fvar<var> out = log_mix(a, b);

    EXPECT_FLOAT_EQ(out.val_.val(), -1.85911088);
    EXPECT_FLOAT_EQ(out.d_.val(), 3.66673118);
    out.d_.grad();

    vector_d prob_deriv(4);
    prob_deriv << 2.3610604993, 0.8685856170, 0.3195347914, 0.1175502804;

    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(a[i].d_.adj(), prob_deriv[i]);
    }
  };

  vector_fv vecfv_prob(4);
  vector_d vecd_dens(4);
  row_vector_fv row_vecfv_prob(4);
  row_vector_d row_vecd_dens(4);
  std::vector<fvar<var> > std_vecfv_prob(4);
  std::vector<double> std_vecd_dens(4);

  mix_fv_d(vecfv_prob, vecd_dens);
  mix_fv_d(vecfv_prob, row_vecd_dens);
  mix_fv_d(vecfv_prob, std_vecd_dens);

  mix_fv_d(row_vecfv_prob, vecd_dens);
  mix_fv_d(row_vecfv_prob, row_vecd_dens);
  mix_fv_d(row_vecfv_prob, std_vecd_dens);

  mix_fv_d(std_vecfv_prob, vecd_dens);
  mix_fv_d(std_vecfv_prob, row_vecd_dens);
  mix_fv_d(std_vecfv_prob, std_vecd_dens);
}

TEST(AgradMixMatrixLogMix_mat, d_fv_old) {
  auto mix_d_fv = [](auto a, auto b) {
    a[0] = 0.15;
    a[1] = 0.70;
    a[2] = 0.10;
    a[3] = 0.05;

    b[0].val_ = -1.0;
    b[1].val_ = -2.0;
    b[2].val_ = -3.0;
    b[3].val_ = -4.0;
    b[0].d_ = 1.0;
    b[1].d_ = 1.0;
    b[2].d_ = 1.0;
    b[3].d_ = 1.0;

    fvar<var> out = log_mix(a, b);

    EXPECT_FLOAT_EQ(out.val_.val(), -1.85911088);
    EXPECT_FLOAT_EQ(out.d_.val(), 1.0);
    out.d_.grad();

    vector_d dens_deriv(4);
    dens_deriv << 0.3541590748, 0.6080099319, 0.0319534791, 0.0058775140;

    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(b[i].d_.adj(), dens_deriv[i]);
    }
  };

  vector_d vecd_prob(4);
  vector_fv vecfv_dens(4);
  row_vector_d row_vecd_prob(4);
  row_vector_fv row_vecfv_dens(4);
  std::vector<double> std_vecd_prob(4);
  std::vector<fvar<var> > std_vecfv_dens(4);

  mix_d_fv(vecd_prob, vecfv_dens);
  mix_d_fv(vecd_prob, row_vecfv_dens);
  mix_d_fv(vecd_prob, std_vecfv_dens);

  mix_d_fv(row_vecd_prob, vecfv_dens);
  mix_d_fv(row_vecd_prob, row_vecfv_dens);
  mix_d_fv(row_vecd_prob, std_vecfv_dens);

  mix_d_fv(std_vecd_prob, vecfv_dens);
  mix_d_fv(std_vecd_prob, row_vecfv_dens);
  mix_d_fv(std_vecd_prob, std_vecfv_dens);
}

TEST(AgradMixMatrixLogMix_mat, ffv_ffv_old) {
  auto mix_ffv_ffv = [](auto a, auto b) {
    a[0].val_ = 0.15;
    a[1].val_ = 0.70;
    a[2].val_ = 0.10;
    a[3].val_ = 0.05;
    a[0].d_ = 1.0;
    a[1].d_ = 1.0;
    a[2].d_ = 1.0;
    a[3].d_ = 1.0;
    a[0].val_.d_ = 1.0;
    a[1].val_.d_ = 1.0;
    a[2].val_.d_ = 1.0;
    a[3].val_.d_ = 1.0;

    b[0].val_ = -1.0;
    b[1].val_ = -2.0;
    b[2].val_ = -3.0;
    b[3].val_ = -4.0;
    b[0].d_ = 1.0;
    b[1].d_ = 1.0;
    b[2].d_ = 1.0;
    b[3].d_ = 1.0;
    b[0].val_.d_ = 1.0;
    b[1].val_.d_ = 1.0;
    b[2].val_.d_ = 1.0;
    b[3].val_.d_ = 1.0;

    fvar<fvar<var> > out = log_mix(a, b);

    EXPECT_FLOAT_EQ(out.val_.val_.val(), -1.85911088);
    EXPECT_FLOAT_EQ(out.d_.val_.val(), 4.66673118);
    out.d_.val_.grad();

    stan::math::vector_d prob_deriv(4);
    prob_deriv << 2.3610604993, 0.8685856170, 0.3195347914, 0.1175502804;
    stan::math::vector_d dens_deriv(4);
    dens_deriv << 0.3541590748, 0.6080099319, 0.0319534791, 0.0058775140;

    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(a[i].d_.val_.adj(), prob_deriv[i]);
      EXPECT_FLOAT_EQ(b[i].d_.val_.adj(), dens_deriv[i]);
    }
  };

  vector_ffv vecffv_prob(4);
  vector_ffv vecffv_dens(4);
  row_vector_ffv row_vecffv_prob(4);
  row_vector_ffv row_vecffv_dens(4);
  std::vector<fvar<fvar<var> > > std_vecffv_prob(4);
  std::vector<fvar<fvar<var> > > std_vecffv_dens(4);

  mix_ffv_ffv(vecffv_prob, vecffv_dens);
  mix_ffv_ffv(vecffv_prob, row_vecffv_dens);
  mix_ffv_ffv(vecffv_prob, std_vecffv_dens);

  mix_ffv_ffv(row_vecffv_prob, vecffv_dens);
  mix_ffv_ffv(row_vecffv_prob, row_vecffv_dens);
  mix_ffv_ffv(row_vecffv_prob, std_vecffv_dens);

  mix_ffv_ffv(std_vecffv_prob, vecffv_dens);
  mix_ffv_ffv(std_vecffv_prob, row_vecffv_dens);
  mix_ffv_ffv(std_vecffv_prob, std_vecffv_dens);
}

TEST(AgradMixMatrixLogMix_mat, ffv_d_old) {
  auto mix_ffv_d = [](auto a, auto b) {
    a[0].val_ = 0.15;
    a[1].val_ = 0.70;
    a[2].val_ = 0.10;
    a[3].val_ = 0.05;
    a[0].d_ = 1.0;
    a[1].d_ = 1.0;
    a[2].d_ = 1.0;
    a[3].d_ = 1.0;
    a[0].val_.d_ = 1.0;
    a[1].val_.d_ = 1.0;
    a[2].val_.d_ = 1.0;
    a[3].val_.d_ = 1.0;

    b[0] = -1.0;
    b[1] = -2.0;
    b[2] = -3.0;
    b[3] = -4.0;

    fvar<fvar<var> > out = log_mix(a, b);

    EXPECT_FLOAT_EQ(out.val_.val_.val(), -1.85911088);
    EXPECT_FLOAT_EQ(out.d_.val_.val(), 3.66673118);
    out.d_.val_.grad();

    stan::math::vector_d prob_deriv(4);
    prob_deriv << 2.3610604993, 0.8685856170, 0.3195347914, 0.1175502804;

    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(a[i].d_.val_.adj(), prob_deriv[i]);
    }
  };

  vector_ffv vecffv_prob(4);
  vector_d vecd_dens(4);
  row_vector_ffv row_vecffv_prob(4);
  row_vector_d row_vecd_dens(4);
  std::vector<fvar<fvar<var> > > std_vecffv_prob(4);
  std::vector<double> std_vecd_dens(4);

  mix_ffv_d(vecffv_prob, vecd_dens);
  mix_ffv_d(vecffv_prob, row_vecd_dens);
  mix_ffv_d(vecffv_prob, std_vecd_dens);

  mix_ffv_d(row_vecffv_prob, vecd_dens);
  mix_ffv_d(row_vecffv_prob, row_vecd_dens);
  mix_ffv_d(row_vecffv_prob, std_vecd_dens);

  mix_ffv_d(std_vecffv_prob, vecd_dens);
  mix_ffv_d(std_vecffv_prob, row_vecd_dens);
  mix_ffv_d(std_vecffv_prob, std_vecd_dens);
}

TEST(AgradMixMatrixLogMix_mat, d_ffv_old) {
  auto mix_d_ffv = [](auto a, auto b) {
    a[0] = 0.15;
    a[1] = 0.70;
    a[2] = 0.10;
    a[3] = 0.05;

    b[0].val_ = -1.0;
    b[1].val_ = -2.0;
    b[2].val_ = -3.0;
    b[3].val_ = -4.0;
    b[0].d_ = 1.0;
    b[1].d_ = 1.0;
    b[2].d_ = 1.0;
    b[3].d_ = 1.0;
    b[0].val_.d_ = 1.0;
    b[1].val_.d_ = 1.0;
    b[2].val_.d_ = 1.0;
    b[3].val_.d_ = 1.0;

    fvar<fvar<var> > out = log_mix(a, b);

    EXPECT_FLOAT_EQ(out.val_.val_.val(), -1.85911088);
    EXPECT_FLOAT_EQ(out.d_.val_.val(), 1.0);
    out.d_.val_.grad();

    stan::math::vector_d dens_deriv(4);
    dens_deriv << 0.3541590748, 0.6080099319, 0.0319534791, 0.0058775140;

    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(b[i].d_.val_.adj(), dens_deriv[i]);
    }
  };

  vector_d vecd_prob(4);
  vector_ffv vecffv_dens(4);
  row_vector_d row_vecd_prob(4);
  row_vector_ffv row_vecffv_dens(4);
  std::vector<double> std_vecd_prob(4);
  std::vector<fvar<fvar<var> > > std_vecffv_dens(4);

  mix_d_ffv(vecd_prob, vecffv_dens);
  mix_d_ffv(vecd_prob, row_vecffv_dens);
  mix_d_ffv(vecd_prob, std_vecffv_dens);

  mix_d_ffv(row_vecd_prob, vecffv_dens);
  mix_d_ffv(row_vecd_prob, row_vecffv_dens);
  mix_d_ffv(row_vecd_prob, std_vecffv_dens);

  mix_d_ffv(std_vecd_prob, vecffv_dens);
  mix_d_ffv(std_vecd_prob, row_vecffv_dens);
  mix_d_ffv(std_vecd_prob, std_vecffv_dens);
}
