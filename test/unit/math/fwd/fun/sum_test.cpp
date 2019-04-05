
#include <stan/math/fwd.hpp>
#include <gtest/gtest.h>

#include <vector>







TEST(AgradFwdMatrixSum, vectorFvar) {
  using stan::math::fvar;
  using stan::math::sum;
  using std::vector;

  vector<fvar<double> > v(6);

  for (int i = 0; i < 6; ++i) {
    v[i] = i + 1;
    v[i].d_ = 1.0;
  }

  fvar<double> output;
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ(6.0, output.d_);

  vector<fvar<double> > ve;
  EXPECT_FLOAT_EQ(0.0, sum(ve).val_);
  EXPECT_FLOAT_EQ(0.0, sum(ve).d_);
}

TEST(AgradFwdMatrixSum, ffd_vector) {
  using stan::math::fvar;
  using stan::math::sum;
  using std::vector;

  vector<fvar<fvar<double> > > v(6);

  for (int i = 0; i < 6; ++i) {
    v[i] = i + 1;
    v[i].d_ = 1.0;
  }

  fvar<fvar<double> > output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ(6.0, output.d_.val());

  vector<fvar<fvar<double> > > ve;
  EXPECT_FLOAT_EQ(0.0, sum(ve).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(ve).d_.val());
}



using stan::math::fvar;
TEST(AgradFwdMatrixSum_mat, fd_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d(6);
  vector_fd v(6);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  v(0).d_ = 1.0;
  v(1).d_ = 1.0;
  v(2).d_ = 1.0;
  v(3).d_ = 1.0;
  v(4).d_ = 1.0;
  v(5).d_ = 1.0;

  fvar<double> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ(6.0, output.d_);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_);
  EXPECT_FLOAT_EQ(0.0, sum(v).d_);
}
TEST(AgradFwdMatrixSum_mat, fd_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;
  using stan::math::sum;

  row_vector_d d(6);
  row_vector_fd v(6);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  v(0).d_ = 1.0;
  v(1).d_ = 1.0;
  v(2).d_ = 1.0;
  v(3).d_ = 1.0;
  v(4).d_ = 1.0;
  v(5).d_ = 1.0;

  fvar<double> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ(6.0, output.d_);

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_);
  EXPECT_FLOAT_EQ(0.0, sum(v).d_);
}
TEST(AgradFwdMatrixSum_mat, fd_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::sum;

  matrix_d d(2, 3);
  matrix_fd v(2, 3);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  v(0, 0).d_ = 1.0;
  v(0, 1).d_ = 1.0;
  v(0, 2).d_ = 1.0;
  v(1, 0).d_ = 1.0;
  v(1, 1).d_ = 1.0;
  v(1, 2).d_ = 1.0;

  fvar<double> output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);
  EXPECT_FLOAT_EQ(6.0, output.d_);

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_);
  EXPECT_FLOAT_EQ(0.0, sum(v).d_);
}
TEST(AgradFwdMatrixSum_mat, ffd_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d(6);
  vector_ffd v(6);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  v(0).d_ = 1.0;
  v(1).d_ = 1.0;
  v(2).d_ = 1.0;
  v(3).d_ = 1.0;
  v(4).d_ = 1.0;
  v(5).d_ = 1.0;

  fvar<fvar<double> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ(6.0, output.d_.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradFwdMatrixSum_mat, ffd_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::sum;

  row_vector_d d(6);
  row_vector_ffd v(6);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  v(0).d_ = 1.0;
  v(1).d_ = 1.0;
  v(2).d_ = 1.0;
  v(3).d_ = 1.0;
  v(4).d_ = 1.0;
  v(5).d_ = 1.0;

  fvar<fvar<double> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ(6.0, output.d_.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
TEST(AgradFwdMatrixSum_mat, ffd_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::sum;

  matrix_d d(2, 3);
  matrix_ffd v(2, 3);

  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  v(0, 0).d_ = 1.0;
  v(0, 1).d_ = 1.0;
  v(0, 2).d_ = 1.0;
  v(1, 0).d_ = 1.0;
  v(1, 1).d_ = 1.0;
  v(1, 2).d_ = 1.0;

  fvar<fvar<double> > output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());

  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());
  EXPECT_FLOAT_EQ(6.0, output.d_.val());

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(v).d_.val());
}
