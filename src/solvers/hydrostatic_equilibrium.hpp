#pragma once

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

/**
 * @class HydrostaticEquilibrium
 * @brief Integrator to construct an initial state in hydrostatic equilibrium.
 */
class HydrostaticEquilibrium {
 public:
  HydrostaticEquilibrium(double rho_c, double p_threshold, const EOS *eos,
                         double k, double n)
      : rho_c_(rho_c), p_threshold_(p_threshold), eos_(eos), k_(k), n_(n) {}

  void solve(State *state, GridStructure *grid, ProblemIn *pin);

 private:
  double rho_c_; // central density
  double p_threshold_; // surface pressure threshold

  const EOS *eos_;
  // pulling in polytropic constants manually..
  double k_;
  double n_;

  static constexpr int iP_ = 0;

  [[nodiscard]] auto rhs(double mass_enc, double p, double r) const -> double;
};
