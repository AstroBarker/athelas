#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "radiation/radiation_discretization.hpp"
#include "radiation/hydro_package.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace radiation {

RadHydroPackage::RadHydroPackage(const ProblemIn* /*pin*/, int n_stages, EOS* eos,
                           ModalBasis* fluid_basis, ModalBasis* rad_basis, BoundaryConditions* bcs,
                           double cfl, int nx, bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), fluid_basis_(fluid_basis), rad_basis_(rad_basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", 3, nx + 2 + 1),
      u_f_l_("hydro::u_f_l_", 3, nx + 2), u_f_r_("hydro::u_f_r_", 3, nx + 2),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1) {
} // Need long term solution for flux_u_

KOKKOS_FUNCTION
void RadHydroPackage::update_explicit(const View3D<double> state,
                                   View3D<double> dU, const GridStructure& grid,
                                   const TimeStepInfo& dt_info) {
  const auto& order = basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  const auto stage = dt_info.stage;

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(state, &grid, rad_basis_, bcs_);
  bc::fill_ghost_zones<3>(state, &grid, fluid_basis_, bcs_);

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "RadHydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_LAMBDA(const int icf, const int ix, const int k) {
        dU(icf, ix, k) = 0.0;
      });

  // --- radiation Increment : Divergence ---
  radiation_divergence(state, dU, grid, stage);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "RadHydro :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int icf, const int ix, const int k) {
        dU(icf, ix, k) /= (basis_->get_mass_matrix(ix, k));
      });
}

/**
 * @brief explicit hydrodynamic timestep restriction
 **/
KOKKOS_FUNCTION
auto RadHydroPackage::min_timestep(const View3D<double> state,
                                const GridStructure& grid,
                                const TimeStepInfo& /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  const int& ilo = grid.get_ilo();
  const int& ihi = grid.get_ihi();

  double dt_out = 0.0;
  Kokkos::parallel_reduce(
      "RadHydro::min_timestep", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix, double& lmin) {
        const double dr = grid.get_widths(ix);
        static constexpr double eigval = constants::c_cgs;
        const double dt_old = std::abs(dr) / eigval;

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::name() const noexcept
    -> std::string_view {
  return "Hydro";
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void RadHydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::get_flux_u(const int stage,
                                                            const int ix) const
    -> double {
  return flux_u_(stage, ix);
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::get_basis() const
    -> const ModalBasis* {
  return basis_;
}

} // namespace radiation
