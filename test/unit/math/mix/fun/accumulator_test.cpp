#include <stan/math/mix.hpp>
#include <test/unit/math/util.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(AgradMixMatrixAccumulate, fvar_var) {
  using stan::math::accumulator;
  using stan::math::fvar;
  using stan::math::var;

  accumulator<fvar<var> > a;
  EXPECT_SUM(a, 0);

  a.add(1.0);
  EXPECT_SUM(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  EXPECT_SUM(a, 1000);
}

TEST(AgradMixMatrixAccumulate, collection_fvar_var) {
  using stan::math::accumulator;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using std::vector;
  using stan::math::fvar;
  using stan::math::var;

  accumulator<fvar<var> > a;

  int pos = 0;
  EXPECT_SUM(a, 0);

  vector<fvar<var> > v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = pos++;
  a.add(v);
  EXPECT_SUM(a, (pos - 1));

  a.add(pos++);
  EXPECT_SUM(a, (pos - 1));

  double x = pos++;
  a.add(x);
  EXPECT_SUM(a, (pos - 1));

  vector<vector<fvar<var> > > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<fvar<var> > w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  EXPECT_SUM(a, (pos - 1));

  matrix_fv m(5, 6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i, j) = pos++;
  a.add(m);
  EXPECT_SUM(a, (pos - 1));

  vector_fv mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  EXPECT_SUM(a, (pos - 1));

  vector<vector_fv> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    vector_fv vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  EXPECT_SUM(a, (pos - 1));
}

TEST(AgradMixMatrixAccumulate, fvar_fvar_var) {
  using stan::math::accumulator;
  using stan::math::fvar;
  using stan::math::var;

  accumulator<fvar<fvar<var> > > a;
  EXPECT_SUM(a, 0);

  a.add(1.0);
  EXPECT_SUM(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  EXPECT_SUM(a, 1000);
}

TEST(AgradMixMatrixAccumulate, collection_fvar_fvar_var) {
  using stan::math::accumulator;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using std::vector;
  using stan::math::fvar;
  using stan::math::var;

  accumulator<fvar<fvar<var> > > a;

  int pos = 0;
  EXPECT_SUM(a, 0);

  vector<fvar<fvar<var> > > v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = pos++;
  a.add(v);
  EXPECT_SUM(a, (pos - 1));

  a.add(pos++);
  EXPECT_SUM(a, (pos - 1));

  int x = pos++;
  a.add(x);
  EXPECT_SUM(a, (pos - 1));

  vector<vector<fvar<fvar<var> > > > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<fvar<fvar<var> > > w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  EXPECT_SUM(a, (pos - 1));

  matrix_ffv m(5, 6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i, j) = pos++;
  a.add(m);
  EXPECT_SUM(a, (pos - 1));

  vector_ffv mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  EXPECT_SUM(a, (pos - 1));

  vector<vector_ffv> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    vector_ffv vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  EXPECT_SUM(a, (pos - 1));
}
