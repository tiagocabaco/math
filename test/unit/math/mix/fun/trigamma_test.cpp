
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








TEST(AgradFwdTrigamma, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::trigamma;
  using stan::math::var;

  fvar<var> x(0.5, 1.3);
  fvar<var> a = trigamma(x);

  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * -16.8288, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y, g);
  EXPECT_FLOAT_EQ(-16.8288, g[0]);
}
TEST(AgradFwdTrigamma, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::trigamma;
  using stan::math::var;

  fvar<var> x(0.5, 1.3);
  fvar<var> a = trigamma(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y, g);
  EXPECT_FLOAT_EQ(126.63182, g[0]);
}
TEST(AgradFwdTrigamma, FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::trigamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = trigamma(x);

  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val_);
  EXPECT_FLOAT_EQ(-16.8288, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = trigamma(y);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(-16.8288, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdTrigamma, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::trigamma;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = trigamma(x);

  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val_.val());
  EXPECT_FLOAT_EQ(-16.8288, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p, g);
  EXPECT_FLOAT_EQ(-16.8288, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = trigamma(y);
  EXPECT_FLOAT_EQ(4.9348022005446793094, b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(-16.8288, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q, r);
  EXPECT_FLOAT_EQ(-16.8288, r[0]);
}
TEST(AgradFwdTrigamma, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::trigamma;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = trigamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(97.409088, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = trigamma(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q, r);
  EXPECT_FLOAT_EQ(97.409088, r[0]);
}
TEST(AgradFwdTrigamma, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::trigamma;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = trigamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(-771.47424982666722519053592192, g[0]);
}

struct trigamma_fun {
  template <typename T0>
  inline T0 operator()(const T0& arg1) const {
    return stan::math::trigamma(arg1);
  }
};

TEST(AgradFwdTrigamma, trigamma_NaN) {
  trigamma_fun trigamma_;
  test_nan_mix(trigamma_, false);
}









/**
 * This is the structure for testing vectorized trigamma (defined in the
 * testing framework).
 */
struct trigamma_test {
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
    using stan::math::trigamma;
    return trigamma(x);
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
        .add(-0.9)
        .add(0)
        .add(1.3)
        .add(19.2)
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

INSTANTIATE_TYPED_TEST_CASE_P(, prim_scalar_unary_test, trigamma_test);
INSTANTIATE_TYPED_TEST_CASE_P(, rev_scalar_unary_test, trigamma_test);
INSTANTIATE_TYPED_TEST_CASE_P(, fwd_scalar_unary_test, trigamma_test);
INSTANTIATE_TYPED_TEST_CASE_P(, mix_scalar_unary_test, trigamma_test);
