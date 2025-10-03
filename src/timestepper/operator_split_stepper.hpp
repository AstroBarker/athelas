#pragma once

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "interface/packages_base.hpp"
#include "state/state.hpp"

/**
 * @class OperatorSplitStepper
 * @brief Updates operator split packages: U -> U + dU dt
 * @note We make some asusmptions on the packages that go here.
 *   1. Hydro will never be in here. The grid advection will never be in here.
 *   2. We are not limiting fields in these packages. This should be remedied
 *      when limiters are moved into packages.
 */
class OperatorSplitStepper {
 public:
  OperatorSplitStepper(const GridStructure &grid, EOS *eos, int nvars);

  void step(PackageManager *pkgs, State *state, const GridStructure &grid,
            double t, double dt);

 private:
  int nvars_evolved_;
  View3D<double> dU_;

  // hold EOS ptr for convenience
  EOS *eos_;
};
