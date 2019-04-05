
#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/fun/util.hpp>
#include <test/unit/math/mix/fun/nan_util.hpp>
#include <test/unit/math/prim/vectorize/prim_scalar_unary_test.hpp>
#include <test/unit/math/rev/vectorize/rev_scalar_unary_test.hpp>
#include <test/unit/math/fwd/vectorize/fwd_scalar_unary_test.hpp>
#include <test/unit/math/mix/vectorize/mix_scalar_unary_test.hpp>
#include <test/unit/math/prim/vectorize/vector_builder.hpp>

#include <vector>

// Phi_approx needs inv_logit in order for this to work







TEST(AgradFwdPhi_approx, FvarVar_1stDeriv) {
  using stan::math::Phi_approx;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(1.0, 1.3);
  fvar<var> a = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val());
  EXPECT_FLOAT_EQ(0.31398547, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y, g);
  EXPECT_FLOAT_EQ(0.24152729, g[0]);
}
TEST(AgradFwdPhi_approx, FvarVar_2ndDeriv) {
  using stan::math::Phi_approx;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(1.0, 1.3);
  fvar<var> a = Phi_approx(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y, g);
  EXPECT_FLOAT_EQ(-0.3143405, g[0]);
}

TEST(AgradFwdPhi_approx, FvarFvarVar_1stDeriv) {
  using stan::math::Phi_approx;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.24152729, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p, g);
  EXPECT_FLOAT_EQ(0.24152729, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = Phi_approx(y);
  EXPECT_FLOAT_EQ(0.84133035, b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0.24152729, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q, r);
  EXPECT_FLOAT_EQ(0.24152729, r[0]);
}

TEST(AgradFwdPhi_approx, FvarFvarVar_2ndDeriv) {
  using stan::math::Phi_approx;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = Phi_approx(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(-0.24180038, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = Phi_approx(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q, r);
  EXPECT_FLOAT_EQ(-0.24180038, r[0]);
}
TEST(AgradFwdPhi_approx, FvarFvarVar_3rdDeriv) {
  using stan::math::Phi_approx;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = Phi_approx(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(0.1590839, g[0]);
}

struct Phi_approx_fun {
  template <typename T0>
  inline T0 operator()(const T0& arg1) const {
    return stan::math::Phi_approx(arg1);
  }
};

TEST(AgradFwdPhi_approx, Phi_approx_NaN) {
  Phi_approx_fun Phi_approx_;
  test_nan_mix(Phi_approx_, false);
}









/**
 * This is the structure for testing vectorized Phi_approx (defined in the
 * testing framework).
 */
struct Phi_approx_test {
  /**
   * Redefinition of function brought in from stan::math.  The reason
   * to do this is that it wraps it up in this static template class.
   *
   * This is the version that's being tested.
   *
   * WARNING:  assumes that the scalar values for all instantiations
   * (prim, rev, fwd, mix) ***have already been tested***.
   *
   * @tparam R Return type.
   * @tparam T Argument type.
   */
  template <typename R, typename T>
  static R apply(const T& x) {
    using stan::math::Phi_approx;
    return Phi_approx(x);
  }

  /**
   * This defines the truth against which we're testing.
   *
   * Because this is *not an independent test*, this function just
   * delegates to the actual function defined in stan::math.
   *
   * Redundant definition of function from stan::math to apply to an
   * integer and return a double.
   *
   * This function delegates to apply(), defined above, directly.
   *
   * WARNING:  this is *not an independent test*.
   */
  static double apply_base(int x) { return apply<double>(x); }

  /**
   * This is the generic version of the integer version defined
   * above.  For every other type, the return type is the same as the
   * reference type.
   *
   * WARNING:  this is *not an independent test of the underlying function*.
   */
  template <typename T>
  static T apply_base(const T& x) {
    return apply<T>(x);
  }

  /**
   * Return sequence of valid double-valued inputs.
   */
  static std::vector<double> valid_inputs() {
    return test::math::vector_builder<double>()
        .add(-15.2)
        .add(0.0)
        .add(1.3)
        .build();
  }

  /**
   * Return sequence of invalid double-valued inputs.
   */
  static std::vector<double> invalid_inputs() {
    return test::math::vector_builder<double>().build();
  }

  /**
   * Return sequence of valid integer inputs.
   */
  static std::vector<int> int_valid_inputs() {
    return test::math::vector_builder<int>()
        .add(-10)
        .add(0)
        .add(1)
        .add(5)
        .add(10)
        .build();
  }

  /**
   * Return sequence of invalid integer inputs.
   */
  static std::vector<int> int_invalid_inputs() {
    return test::math::vector_builder<int>().build();
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(, prim_scalar_unary_test, Phi_approx_test);
INSTANTIATE_TYPED_TEST_CASE_P(, rev_scalar_unary_test, Phi_approx_test);
INSTANTIATE_TYPED_TEST_CASE_P(, fwd_scalar_unary_test, Phi_approx_test);
INSTANTIATE_TYPED_TEST_CASE_P(, mix_scalar_unary_test, Phi_approx_test);
