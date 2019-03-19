#ifndef STAN_MATH_MIX_HPP
#define STAN_MATH_MIX_HPP

#include <stan/math/prim/meta/ad_promotable.hpp>
#include <stan/math/rev/meta/ad_promotable.hpp>
#include <stan/math/fwd/meta/ad_promotable.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/meta/is_fvar.hpp>
#include <stan/math/fwd/meta/partials_type.hpp>
#include <stan/math/fwd/meta/operands_and_partials.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta/is_var.hpp>
#include <stan/math/rev/meta/partials_type.hpp>
#include <stan/math/rev/meta/operands_and_partials.hpp>
#include <stan/math/prim.hpp>
#include <stan/math/fwd.hpp>
#include <stan/math/rev.hpp>
#include <stan/math/mix/fun/typedefs.hpp>
#include <stan/math/mix/functor/derivative.hpp>
#include <stan/math/mix/functor/finite_diff_grad_hessian.hpp>
#include <stan/math/mix/functor/grad_hessian.hpp>
#include <stan/math/mix/functor/grad_tr_mat_times_hessian.hpp>
#include <stan/math/mix/functor/gradient_dot_vector.hpp>
#include <stan/math/mix/functor/hessian.hpp>
#include <stan/math/mix/functor/hessian_times_vector.hpp>
#include <stan/math/mix/functor/partial_derivative.hpp>
#endif
