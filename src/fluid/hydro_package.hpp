#pragma once

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace fluid {

using bc::BoundaryConditions;

class HydroPackage {
 public:
  HydroPackage(const ProblemIn* /*pin*/, int n_stages, EOS* eos,
               ModalBasis* basis, BoundaryConditions* bcs, double cfl, int nx,
               bool active = true);

  KOKKOS_FUNCTION
  void update_explicit(View3D<double> state, View3D<double> dU,
                       const GridStructure& grid, const TimeStepInfo& dt_info);

  KOKKOS_FUNCTION
  void fluid_divergence(const View3D<double> state, View3D<double> dU,
                        const GridStructure& grid, int stage);

  KOKKOS_FUNCTION
  void fluid_geometry(const View3D<double> state, View3D<double> dU,
                      const GridStructure& grid);

  [[nodiscard]] KOKKOS_FUNCTION auto
  min_timestep(const View3D<double> state, const GridStructure& grid,
               const TimeStepInfo& /*dt_info*/) const -> double;

  [[nodiscard]] KOKKOS_FUNCTION auto name() const noexcept -> std::string_view;

  [[nodiscard]] KOKKOS_FUNCTION auto is_active() const noexcept -> bool;

  KOKKOS_FUNCTION
  void set_active(bool active);

  [[nodiscard]] auto get_flux_u(int stage, int ix) const -> double;

  [[nodiscard]] static constexpr auto num_vars() noexcept -> int {
      return NUM_VARS_;
  }

 private:
  bool active_;

  int nx_;
  double cfl_;

  EOS* eos_;
  ModalBasis* basis_;
  BoundaryConditions* bcs_;

  // package storage
  View2D<double> dFlux_num_; // stores Riemann solutions
  View2D<double> u_f_l_; // left faces
  View2D<double> u_f_r_; // right faces
  View2D<double> flux_u_; // Riemann velocities

  // constants
  static constexpr int NUM_VARS_ = 3;
};

} // namespace fluid
