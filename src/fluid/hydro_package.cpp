#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_discretization.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace fluid {

HydroPackage::HydroPackage(const ProblemIn* /*pin*/, int n_stages, EOS* eos,
                           ModalBasis* basis, BoundaryConditions* bcs,
                           double cfl, int nx, bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), basis_(basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", 3, nx + 2 + 1), // fix?
      u_f_l_("hydro::u_f_l_", 3, nx + 2), u_f_r_("hydro::u_f_r_", 3, nx + 2),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1) {}

KOKKOS_FUNCTION
void HydroPackage::update_explicit(const View3D<double> state,
                                   View3D<double> dU, const GridStructure& grid,
                                   const TimeStepInfo& dt_info) {
  const auto& order = basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  const auto stage = dt_info.stage;

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(state, &grid, basis_, bcs_);

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "Hydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_LAMBDA(const int icf, const int ix, const int k) {
        dU(icf, ix, k) = 0.0;
      });

  Kokkos::parallel_for(
      ihi + 2, KOKKOS_CLASS_LAMBDA(const int ix) { flux_u_(stage, ix) = 0.0; });

  // --- Fluid Increment : Divergence ---
  fluid_divergence(state, dU, grid, stage);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Hydro :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int icf, const int ix, const int k) {
        dU(icf, ix, k) /= (basis_->get_mass_matrix(ix, k));
      });

  // --- Increment from Geometry ---
  if (grid.do_geometry()) {
    fluid_geometry(state, dU, grid);
  }
}

/**
 * @brief explicit hydrodynamic timestep restriction
 *
 * NOTE: Is it worthwhile to use nodal values instead of averages?
 **/
KOKKOS_FUNCTION
auto HydroPackage::min_timestep(const View3D<double> state,
                                const GridStructure& grid,
                                const TimeStepInfo& /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  const int& ilo = grid.get_ilo();
  const int& ihi = grid.get_ihi();

  double dt_out = 0.0;
  Kokkos::parallel_reduce(
      "Hydro::min_timestep", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int ix, double& lmin) {
        // --- Using Cell Averages ---
        const double tau_x  = state(0, ix, 0);
        const double vel_x  = state(1, ix, 0);
        const double eint_x = state(2, ix, 0);

        const double dr = grid.get_widths(ix);

        auto lambda = nullptr;
        const double Cs =
            sound_speed_from_conserved(eos_, tau_x, vel_x, eint_x, lambda);
        const double eigval = Cs + std::abs(vel_x);

        const double dt_old = std::abs(dr) / std::abs(eigval);

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::name() const noexcept
    -> std::string_view {
  return "Hydro";
}

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void HydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::get_flux_u(const int stage,
                                                            const int ix) const
    -> double {
  return flux_u_(stage, ix);
}

} // namespace fluid
