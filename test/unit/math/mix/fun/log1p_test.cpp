
#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/fun/util.hpp>
#include <test/unit/math/mix/fun/nan_util.hpp>
#include <test/unit/math/fwd/vectorize/fwd_scalar_unary_test.hpp>
#include <test/unit/math/mix/vectorize/mix_scalar_unary_test.hpp>
#include <test/unit/math/prim/vectorize/prim_scalar_unary_test.hpp>
#include <test/unit/math/prim/vectorize/vector_builder.hpp>
#include <test/unit/math/rev/vectorize/rev_scalar_unary_test.hpp>


#include <vector>
#include <exception>








TEST(AgradFwdLog1p, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log1p;
  using stan::math::var;

  fvar<var> x(0.5, 1.3);
  fvar<var> a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (1 + 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y, g);
  EXPECT_FLOAT_EQ(1 / (1.5), g[0]);
}
TEST(AgradFwdLog1p, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::log1p;
  using stan::math::var;

  fvar<var> x(0.5, 1.3);
  fvar<var> a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (1 + 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y, g);
  EXPECT_FLOAT_EQ(-1.3 / (1.5 * 1.5), g[0]);
}
TEST(AgradFwdLog1p, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::log1p;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1 / (1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p, g);
  EXPECT_FLOAT_EQ(1.0 / 1.5, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1p(y);
  EXPECT_FLOAT_EQ(log1p(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(1 / (1.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q, r);
  EXPECT_FLOAT_EQ(1.0 / 1.5, r[0]);
}
TEST(AgradFwdLog1p, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::log1p;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1p(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(-1.0 / 2.25, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1p(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q, r);
  EXPECT_FLOAT_EQ(-1.0 / 2.25, r[0]);
}
TEST(AgradFwdLog1p, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log1p(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p, g);
  EXPECT_FLOAT_EQ(2 / (1.5 * 1.5 * 1.5), g[0]);
}

struct log1p_fun {
  template <typename T0>
  inline T0 operator()(const T0& arg1) const {
    return log1p(arg1);
  }
};

TEST(AgradFwdLog1p, log1p_NaN) {
  log1p_fun log1p_;
  test_nan_mix(log1p_, false);
}










/**
 * This is the structure for testing vectorized log1p (defined in the
 * testing framework).
 */
struct log1p_test {
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
    return stan::math::log1p(x);
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
    // no throw in base implementation (may change in future)
    return test::math::vector_builder<double>()
        .add(-0.5)
        .add(0)
        .add(7.2)
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
    // no throw in base implementation (may change in future)
    return test::math::vector_builder<int>().add(0).add(5).add(1000).build();
  }

  /**
   * Return sequence of invalid integer inputs.
   */
  static std::vector<int> int_invalid_inputs() {
    return test::math::vector_builder<int>().add(-2).build();
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(, prim_scalar_unary_test, log1p_test);
INSTANTIATE_TYPED_TEST_CASE_P(, rev_scalar_unary_test, log1p_test);
INSTANTIATE_TYPED_TEST_CASE_P(, fwd_scalar_unary_test, log1p_test);
INSTANTIATE_TYPED_TEST_CASE_P(, mix_scalar_unary_test, log1p_test);
