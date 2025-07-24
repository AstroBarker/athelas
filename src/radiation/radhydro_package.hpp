#pragma once

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace radiation {

using bc::BoundaryConditions;

class RadHydroPackage {
 public:
  RadHydroPackage(const ProblemIn* /*pin*/, int n_stages, EOS* eos,
                  Opacity* opac, ModalBasis* fluid_basis, ModalBasis* rad_basis,
                  BoundaryConditions* bcs, double cfl, int nx,
                  bool active = true);

  // TODO(astrobarker): mark const
  KOKKOS_FUNCTION
  void update_explicit(View3D<double> state, View3D<double> dU,
                       const GridStructure& grid, const TimeStepInfo& dt_info);
  KOKKOS_FUNCTION
  void update_implicit(View3D<double> state, View3D<double> dU,
                       const GridStructure& grid, const TimeStepInfo& dt_info);
  KOKKOS_FUNCTION
  void update_implicit_iterative(View3D<double> state, View3D<double> dU,
                                 const GridStructure& grid,
                                 const TimeStepInfo& dt_info);
  KOKKOS_FUNCTION
  auto radhydro_source(const View2D<double> uCRH, const GridStructure& grid,
                       int iX, int k) const
      -> std::tuple<double, double, double, double>;

  KOKKOS_FUNCTION
  void radhydro_divergence(const View3D<double> state, View3D<double> dU,
                           const GridStructure& grid, int stage);

  [[nodiscard]] KOKKOS_FUNCTION auto
  min_timestep(const View3D<double> state, const GridStructure& grid,
               const TimeStepInfo& /*dt_info*/) const -> double;

  [[nodiscard]] KOKKOS_FUNCTION auto name() const noexcept -> std::string_view;

  [[nodiscard]] KOKKOS_FUNCTION auto is_active() const noexcept -> bool;

  KOKKOS_FUNCTION
  void set_active(bool active);

  [[nodiscard]] KOKKOS_FUNCTION auto get_flux_u(int stage, int ix) const
      -> double;
  [[nodiscard]] KOKKOS_FUNCTION auto get_fluid_basis() const
      -> const ModalBasis*;
  [[nodiscard]] KOKKOS_FUNCTION auto get_rad_basis() const -> const ModalBasis*;

  [[nodiscard]] static constexpr auto num_vars() noexcept -> int {
    return NUM_VARS_;
  }

 private:
  bool active_;

  int nx_;
  double cfl_;

  EOS* eos_;
  Opacity* opac_;
  ModalBasis* fluid_basis_;
  ModalBasis* rad_basis_;
  BoundaryConditions* bcs_;

  // package storage
  View2D<double> dFlux_num_; // stores Riemann solutions
  View2D<double> u_f_l_; // left faces
  View2D<double> u_f_r_; // right faces
  View2D<double> flux_u_; // Riemann velocities

  // iterative solver storage
  View3D<double> scratch_k_;
  View3D<double> scratch_km1_;
  View3D<double> scratch_sol_;

  // constants
  static constexpr int NUM_VARS_ = 5;
};

} // namespace radiation
