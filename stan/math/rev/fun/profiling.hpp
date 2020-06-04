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

namespace internal {
class start_profiling_vari : public vari {
  std::string id_;
  profilers& pp;

 public:
  start_profiling_vari(std::string& id, profilers& p) : id_(id), vari(0), pp(p) {
    profilers::iterator it;
    it = p.find(id);
    if (it == p.end()) {
      p[id].fwd_pass_time = 0.0;
      p[id].bckwd_pass_time = 0.0;
    }
    p[id].fwd_pass_time_start = std::chrono::steady_clock::now();
    if (p[id].fwd_pass_running) {
      // std::stringstream s;
      // s << "profiling with id = " << id << " was already started!" << std::endl;
      // throw std::runtime_error(s.str());
    } else {
      p[id].fwd_pass_running = true;
    }

    std::cout << "Forward pass start: " << id << std::endl;
  }
  void chain() {
    std::chrono::duration<double> diff
        = std::chrono::steady_clock::now() - pp[id_].fwd_pass_time_start;
    pp[id_].bckwd_pass_time += diff.count();
    pp[id_].bckwd_pass_running = false;
    std::cout << "Reverse pass end: " << id_ << std::endl;
  }
};

class stop_profiling_vari : public vari {
  std::string id_;
  profilers& pp;

 public:
  stop_profiling_vari(std::string& id, profilers& p) : id_(id), vari(0), pp(p) {
    std::chrono::duration<double> diff
        = std::chrono::steady_clock::now() - p[id].fwd_pass_time_start;
    p[id].fwd_pass_time += diff.count();
    if (!p[id].fwd_pass_running) {
      // std::stringstream s;
      // s << "profiling with id = " << id << " was already stopped!" << std::endl;
      // throw std::runtime_error(s.str());
    } else {
      p[id].fwd_pass_running = false;
    }
    std::cout << "Forward pass stop: " << id << std::endl;
  }
  void chain() {
    pp[id_].bkcwd_pass_time_start = std::chrono::steady_clock::now();
    pp[id_].bckwd_pass_running = true;
    std::cout << "Reverse pass start: " << id_ << std::endl;
  }
};
}  // namespace internal

template <typename... Types>
inline var start_profiling(std::string id, profilers& p) {
  return var(new internal::start_profiling_vari(id, p));
}

template <typename... Types>
inline var stop_profiling(std::string id, profilers& p) {
  return var(new internal::stop_profiling_vari(id, p));
}

class profiler {
  std::string id_;
  profilers& pp;
  public:
  profiler(std::string id, profilers& p) : id_(id), pp(p)  {
    start_profiling(id_, pp);
  }
  void stop() {
    stop_profiling(id_, pp);
  }
  ~profiler() {
    //stop_profiling(id_, pp);
  }
};

inline void print_profiling(profilers& p) {
  std::cout << "section,fun_eval,gradient" << std::endl;
  for (auto const& x : p) {
    std::cout << x.first << "," << x.second.fwd_pass_time << "," << x.second.bckwd_pass_time << std::endl;
  }
}

}  // namespace math
}  // namespace stan
#endif
