/**
 * @file gravity_package.hpp
 * --------------
 *
 * @brief Gravitational source package
 **/

#pragma once

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"
#include "utils/abstractions.hpp"

namespace athelas::gravity {

using bc::BoundaryConditions;

class GravityPackage {
 public:
  GravityPackage(const ProblemIn * /*pin*/, GravityModel model, double gval,
                 basis::ModalBasis *basis, double cfl, bool active = true);

  void update_explicit(const State *const state, View3D<double> dU,
                       const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;

  template <GravityModel Model>
  void gravity_update(const View3D<double> state, View3D<double> dU,
                      const GridStructure &grid) const;

  [[nodiscard]] KOKKOS_FUNCTION auto
  min_timestep(const State *const /*state*/, const GridStructure & /*grid*/,
               const TimeStepInfo & /*dt_info*/) const -> double;

  [[nodiscard]] KOKKOS_FUNCTION auto name() const noexcept -> std::string_view;

  [[nodiscard]] KOKKOS_FUNCTION auto is_active() const noexcept -> bool;

  void fill_derived(State *state, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  KOKKOS_FUNCTION
  void set_active(bool active);

 private:
  bool active_;
  GravityModel model_;

  double gval_; // constant gravity

  basis::ModalBasis *basis_;

  double cfl_;
};

} // namespace athelas::gravity
