
#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/fun/util.hpp>
#include <test/unit/math/mix/fun/nan_util.hpp>
#include <test/unit/math/prim/vectorize/prim_scalar_unary_test.hpp>
#include <test/unit/math/rev/vectorize/rev_scalar_unary_test.hpp>
#include <test/unit/math/fwd/vectorize/fwd_scalar_unary_test.hpp>
#include <test/unit/math/mix/vectorize/mix_scalar_unary_test.hpp>
#include <stan/math/prim/fun/acos.hpp>
#include <test/unit/math/prim/vectorize/vector_builder.hpp>

#include <vector>








TEST(AgradFwdAcos, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<var> x(0.5, 0.3);
  fvar<var> a = acos(x);

  EXPECT_FLOAT_EQ(acos(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(-0.3 / sqrt(1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y, g);
  EXPECT_FLOAT_EQ(-1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);
}
TEST(AgradFwdAcos, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<var> x(0.5, 0.3);
  fvar<var> a = acos(x);

  AVEC z = createAVEC(x.val_);
  VEC h;
  a.d_.grad(z, h);
  EXPECT_FLOAT_EQ(-0.5 * 0.3 / (sqrt(1.0 - 0.5 * 0.5) * 0.75), h[0]);
}

TEST(AgradFwdAcos, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 2.0;

  fvar<fvar<var> > b = acos(z);

  EXPECT_FLOAT_EQ(acos(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(-2.0 / sqrt(1.0 - 0.5 * 0.5), b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC y = createAVEC(z.val_.val_);
  VEC g;
  b.val_.val_.grad(y, g);
  EXPECT_FLOAT_EQ(-1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);

  fvar<fvar<var> > w;
  w.val_.val_ = 0.5;
  w.d_.val_ = 2.0;

  b = acos(w);
  EXPECT_FLOAT_EQ(acos(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(-2.0 / sqrt(1.0 - 0.5 * 0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC p = createAVEC(w.val_.val_);
  VEC q;
  b.val_.val_.grad(p, q);
  EXPECT_FLOAT_EQ(-1.0 / sqrt(1.0 - 0.5 * 0.5), q[0]);
}
TEST(AgradFwdAcos, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 2.0;

  fvar<fvar<var> > b = acos(z);

  AVEC y = createAVEC(z.val_.val_);
  VEC g;
  b.val_.d_.grad(y, g);
  EXPECT_FLOAT_EQ(-0.5 * 2.0 / (sqrt(1.0 - 0.5 * 0.5) * 0.75), g[0]);

  fvar<fvar<var> > w;
  w.val_.val_ = 0.5;
  w.d_.val_ = 2.0;

  fvar<fvar<var> > c = acos(w);

  AVEC p = createAVEC(w.val_.val_);
  VEC q;
  c.d_.val_.grad(p, q);
  EXPECT_FLOAT_EQ(-0.5 * 2.0 / (sqrt(1.0 - 0.5 * 0.5) * 0.75), q[0]);
}

TEST(AgradFwdAcos, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 1.0;
  z.d_.val_ = 1.0;

  fvar<fvar<var> > b = acos(z);

  AVEC y = createAVEC(z.val_.val_);
  VEC g;
  b.d_.d_.grad(y, g);
  EXPECT_FLOAT_EQ(-3.07920143567800, g[0]);
}

struct acos_fun {
  template <typename T0>
  inline T0 operator()(const T0& arg1) const {
    return acos(arg1);
  }
};

TEST(AgradFwdAcos, acos_NaN) {
  acos_fun acos_;
  test_nan_mix(acos_, false);
}










/**
 * This is the structure for testing vectorized acos (defined in the
 * testing framework).
 */
struct acos_test {
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
    using stan::math::acos;
    return acos(x);
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
        .add(0.5)
        .add(-0.8)
        .add(0)
        .add(-2.2)
        .add(1.5)
        .build();
  }

  /**
   * Return sequence of invalid double-valued inputs.
   */
  static std::vector<double> invalid_inputs() { return std::vector<double>(); }

  /**
   * Return sequence of valid integer inputs.
   */
  static std::vector<int> int_valid_inputs() {
    return test::math::vector_builder<int>()
        .add(1)
        .add(-1)
        .add(0)
        .add(3)
        .add(-4)
        .build();
  }

  /**
   * Return sequence of invalid integer inputs.
   */
  static std::vector<int> int_invalid_inputs() { return std::vector<int>(); }
};

INSTANTIATE_TYPED_TEST_CASE_P(, prim_scalar_unary_test, acos_test);
INSTANTIATE_TYPED_TEST_CASE_P(, rev_scalar_unary_test, acos_test);
INSTANTIATE_TYPED_TEST_CASE_P(, fwd_scalar_unary_test, acos_test);
INSTANTIATE_TYPED_TEST_CASE_P(, mix_scalar_unary_test, acos_test);
