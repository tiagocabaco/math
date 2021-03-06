#ifndef STAN_MATH_OPENCL_REV_SUB_BLOCK_HPP
#define STAN_MATH_OPENCL_REV_SUB_BLOCK_HPP
#ifdef STAN_OPENCL

#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/sub_block.hpp>
#include <CL/cl2.hpp>
#include <vector>

namespace stan {
namespace math {
/** \ingroup matrix_cl_group
 * Write the contents of A into
 * \c this starting at the top left of \c this
 * @param A input matrix
 * @param A_i the offset row in A
 * @param A_j the offset column in A
 * @param this_i the offset row for the matrix to be subset into
 * @param this_j the offset col for the matrix to be subset into
 * @param nrows the number of rows in the submatrix
 * @param ncols the number of columns in the submatrix
 */
template <typename T>
inline void matrix_cl<T, require_var_t<T>>::sub_block(
    const matrix_cl<T, require_var_t<T>>& A, size_t A_i, size_t A_j,
    size_t this_i, size_t this_j, size_t nrows, size_t ncols) try {
  this->val().sub_block(A.val(), A_i, A_j, this_i, this_j, nrows, ncols);
  this->adj().sub_block(A.adj(), A_i, A_j, this_i, this_j, nrows, ncols);
} catch (const cl::Error& e) {
  check_opencl_error("copy_submatrix", e);
}

}  // namespace math
}  // namespace stan

#endif
#endif
