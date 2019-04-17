
#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <boost/typeof/typeof.hpp>

#include <type_traits>
#include <vector>

using stan::math::promote_elements;
//using stan::math::var;
using std::vector;

TEST(MathFunctionsScalPromote_Elements, int2double) {
  int from;
  promote_elements<double, int> p;
  typedef BOOST_TYPEOF(p.promote(from)) result_t;
  bool same = std::is_same<double, result_t>::value;
  EXPECT_TRUE(same);
}

TEST(MathFunctionsScalPromote_Elements, double2double) {
  double from;
  promote_elements<double, double> p;
  typedef BOOST_TYPEOF(p.promote(from)) result_t;
  bool same = std::is_same<double, result_t>::value;
  EXPECT_TRUE(same);
}


TEST(MathFunctionsArrPromote_Elements_arr, intVec2doubleVec) {
  vector<int> from;
  from.push_back(1);
  from.push_back(2);
  from.push_back(3);
  promote_elements<vector<double>, vector<int> > p;
  typedef BOOST_TYPEOF(p.promote(from)) result_t;
  bool same = std::is_same<vector<double>, result_t>::value;
  EXPECT_TRUE(same);
}

TEST(MathFunctionsArrPromote_Elements_arr, doubleVec2doubleVec) {
  vector<double> from;
  from.push_back(1);
  from.push_back(2);
  from.push_back(3);
  promote_elements<vector<double>, vector<double> > p;
  typedef BOOST_TYPEOF(p.promote(from)) result_t;
  bool same = std::is_same<vector<double>, result_t>::value;
  EXPECT_TRUE(same);
}

TEST(MathFunctionsMatPromote_Elements_mat, doubleMat2doubleMat) {
  stan::math::matrix_d m1(2, 3);
  m1 << 1, 2, 3, 4, 5, 6;
  promote_elements<Eigen::Matrix<double, 2, 3>, Eigen::Matrix<double, 2, 3> > p;
  typedef BOOST_TYPEOF(p.promote(m1)) result_t;
  bool same = std::is_same<Eigen::Matrix<double, 2, 3>, result_t>::value;
  EXPECT_TRUE(same);
}
