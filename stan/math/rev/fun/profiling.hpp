#ifndef STAN_MATH_REV_FUN_PROFILING_HPP
#define STAN_MATH_REV_FUN_PROFILING_HPP

#include <stan/math/prim/core.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/acosh.hpp>
#include <stan/math/prim/fun/isnan.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/fun/value_of_rec.hpp>
#include <stan/math/rev/fun/abs.hpp>
#include <stan/math/rev/fun/arg.hpp>
#include <stan/math/rev/fun/cosh.hpp>
#include <stan/math/rev/fun/is_nan.hpp>
#include <stan/math/rev/fun/log.hpp>
#include <stan/math/rev/fun/polar.hpp>
#include <stan/math/rev/fun/sqrt.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <cmath>
#include <complex>
#include <utility>

namespace stan {
namespace math {

class profiler_state {
  //using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  private:
    std::chrono::time_point<std::chrono::steady_clock> forward_pass_time_start_;
    std::chrono::time_point<std::chrono::steady_clock> forward_pass_time_end_;
    std::chrono::time_point<std::chrono::steady_clock> backward_pass_time_start_;
    std::chrono::time_point<std::chrono::steady_clock> backward_pass_time_end_;
    bool forward_pass_running_;
    bool backward_pass_running_;
    size_t start_var_stack_size_;
    size_t stop_var_stack_size_;
    std::string id_;
  public:
    profiler_state(std::string id) : forward_pass_running_(false), backward_pass_running_(false),
                       start_var_stack_size_(0), forward_pass_time_(0), backward_pass_time_(0),
                       var_stack_used_(0), id_(id) { }

    double forward_pass_time() {
      return (forward_pass_time_end_ - forward_pass_time_start_).count();
    }

    double backward_pass_time() {
      return (backward_pass_time_end_ - backward_pass_time_start_).count();
    }

    double total_time() {
      return backward_pass_time() + forward_pass_time();
    };

    size_t var_stack_used_() {
      return stop_var_stack_size_- start_var_stack_size_;
    }

    void forward_pass_start() {
      forward_pass_running_ = true;
      forward_pass_time_start_ = std::chrono::steady_clock::now();
      start_var_stack_size_ = ChainableStack::instance_->var_stack_.size();
    }

    void forward_pass_end() {
      forward_pass_running_ = false;
      forward_pass_time_end_ = std::chrono::steady_clock::now();
      stop_var_stack_size_ = ChainableStack::instance_->var_stack_.size();
    }

    void backward_pass_start() {
      backward_pass_running_ = true;
      backward_pass_time_start_ = std::chrono::steady_clock::now();
    }

    void backward_pass_end() {
      backward_pass_running_ = false;
      backward_pass_time_end_ = std::chrono::steady_clock::now();
    }
}

using profilers_states = std::map<std::string, profiler_state>;

namespace internal {

class profiler_start_vari : public vari {
  std::string id_;
  profilers& pp;

 public:
  profiler_start_vari(std::string& id, profilers& p) : id_(id), vari(0), pp(p) {
    profilers::iterator it;
    it = p.find(id);
    if (it == p.end()) {
      p[id] = profiler_state(id);
    }
    p[id].forward_pass_start();
  }
  void chain() {
    pp[id_].backward_pass_end();
  }
};

class profiler_end_vari : public vari {
  std::string id_;
  profilers& pp;

 public:
  profiler_end_vari(std::string& id, profilers& p) : id_(id), vari(0), pp(p) {
    p[id].forward_pass_end();
  }
  void chain() {
    pp[id_].backward_pass_start();
  }
};

/**
 * Places the vari that marks the start of a profiling section
 * on the stack.
 * @param id id of the profiling section
 * @param p map of profilers
 * @return created var
 */
inline var profiler_start(std::string id, profiles& p) {
  return var(new internal::profiler_start_vari(id, p));
}

/**
 * Places the vari that marks the end of a profiling section
 * on the stack.
 * @param id id of the profiling section
 * @param p map of profilers 
 * @return created var
 */
inline var profiler_end(std::string id, profilers& p) {
  return var(new internal::profiler_end_vari(id, p));
}

/**
 * Stops the running profilers. Should be used in case of an exception.
 * @param id id of the profiling section
 * @param p map of profilers
 */
inline void profiler_profiler_stop(std::string id, profilers& p) {
  if (p[id].fwd_pass_running) {
    std::chrono::duration<double> diff
        = std::chrono::steady_clock::now() - p[id].fwd_pass_time_start;
    p[id].fwd_pass_time += diff.count();
    p[id].fwd_pass_running = false;
  }
  if (p[id].bckwd_pass_running) {
    std::chrono::duration<double> diff
        = std::chrono::steady_clock::now() - p[id].bkcwd_pass_time_start;
    p[id].bckwd_pass_time += diff.count();
    p[id].bckwd_pass_running = false;
  }
}

}  // namespace internal

/**
 * Class used for profiling sections.
 */
class profiler {
  std::string id_;
  profilers& pp;

  public:
  profiler(std::string id, profilers& p) : id_(id), pp(p) {
    internal::profiler_start(id_, pp);
  }

  ~profiler() {
    internal::profiler_stop(id_, pp);
  }

  void end() {
    internal::profiler_end(id_, pp);
  }
};



}  // namespace math
}  // namespace stan
#endif
