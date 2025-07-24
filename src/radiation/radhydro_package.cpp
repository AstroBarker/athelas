#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_discretization.hpp"
#include "radiation/radhydro_package.hpp"
#include "solvers/root_finders.hpp"
#include "utils/abstractions.hpp"

namespace radiation {

RadHydroPackage::RadHydroPackage(const ProblemIn* /*pin*/, int n_stages,
                                 EOS* eos, Opacity* opac,
                                 ModalBasis* fluid_basis, ModalBasis* rad_basis,
                                 BoundaryConditions* bcs, double cfl, int nx,
                                 bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), opac_(opac),
      fluid_basis_(fluid_basis), rad_basis_(rad_basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", 5, nx + 2 + 1),
      u_f_l_("hydro::u_f_l_", 5, nx + 2), u_f_r_("hydro::u_f_r_", 5, nx + 2),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1),
      scratch_k_("scratch_k_", nx + 2, 5, fluid_basis_->get_order()),
      scratch_km1_("scratch_km1_", nx + 2, 5, fluid_basis_->get_order()),
      scratch_sol_("scratch_k_", nx + 2, 5, fluid_basis_->get_order()) {
} // Need long term solution for flux_u_

KOKKOS_FUNCTION
void RadHydroPackage::update_explicit(const View3D<double> state,
                                      View3D<double> dU,
                                      const GridStructure& grid,
                                      const TimeStepInfo& dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  const auto stage = dt_info.stage;

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(state, &grid, rad_basis_, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(state, &grid, fluid_basis_, bcs_, {0, 2});

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "RadHydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_LAMBDA(const int q, const int ix, const int k) {
        dU(q, ix, k) = 0.0;
      });

  // --- radiation Increment : Divergence ---
  radhydro_divergence(state, dU, grid, stage);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "RadHydro :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int ix, const int k) {
        for (int q = 0; q < 3; ++q) {
          dU(q, ix, k) /= (fluid_basis_->get_mass_matrix(ix, k));
        }
        for (int q = 3; q < NUM_VARS_; ++q) {
          dU(q, ix, k) /= (rad_basis_->get_mass_matrix(ix, k));
        }
      });
} // update_explicit

/**
 * @brief radiation hydrodynamic implicit term
 * Computes dU from source terms
 **/
KOKKOS_FUNCTION
void RadHydroPackage::update_implicit(const View3D<double> state,
                                      View3D<double> dU,
                                      const GridStructure& grid,
                                      const TimeStepInfo& dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "RadHydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_LAMBDA(const int q, const int ix, const int k) {
        dU(q, ix, k) = 0.0;
      });

  Kokkos::parallel_for(
      "RadHydro :: Implicit",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        auto state_i = Kokkos::subview(state, Kokkos::ALL, i, Kokkos::ALL);
        const auto [du1, du2, du3, du4] = radhydro_source(state_i, grid, i, k);
        dU(1, i, k)                     = du1;
        dU(2, i, k)                     = du2;
        dU(3, i, k)                     = du3;
        dU(4, i, k)                     = du4;
      });
} // update_implicit

KOKKOS_FUNCTION
void RadHydroPackage::update_implicit_iterative(const View3D<double> state,
                                                View3D<double> dU,
                                                const GridStructure& grid,
                                                const TimeStepInfo& dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto& order = fluid_basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  Kokkos::parallel_for(
      "RadHydro :: implicit iterative", Kokkos::RangePolicy<>(ilo, ihi + 1),
      KOKKOS_CLASS_LAMBDA(const int iX) {
        auto state_i = Kokkos::subview(state, Kokkos::ALL, iX, Kokkos::ALL);
        auto scratch_sol_ix =
            Kokkos::subview(scratch_sol_, iX, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_k =
            Kokkos::subview(scratch_k_, iX, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_ix_km1 =
            Kokkos::subview(scratch_km1_, iX, Kokkos::ALL, Kokkos::ALL);
        auto R_ix = Kokkos::subview(dU, Kokkos::ALL, iX, Kokkos::ALL);

        // TODO(astrobarker): invert loops
        for (int k = 0; k < order; ++k) {
          // set radhydro vars
          for (int i = 0; i < NUM_VARS_; ++i) {
            scratch_sol_ix_k(i, k)   = state_i(i, k);
            scratch_sol_ix_km1(i, k) = state_i(i, k);
            scratch_sol_ix(i, k)     = state_i(i, k);
          }
        }

        root_finders::fixed_point_radhydro(
            R_ix, dt_info.dt_a, scratch_sol_ix_k, scratch_sol_ix_km1,
            scratch_sol_ix, grid, fluid_basis_, rad_basis_, eos_, opac_, iX);

        // TODO(astrobarker): invert loops
        for (int k = 0; k < order; ++k) {
          for (int q = 1; q < NUM_VARS_; ++q) {
            state(q, iX, k) = scratch_sol_ix(q, k);
          }
        }
      });

} // update_implicit_iterative

/**
 * @brief explicit radiation hydrodynamic timestep restriction
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
        const double dr                = grid.get_widths(ix);
        static constexpr double eigval = constants::c_cgs;
        const double dt_old            = dr / eigval;

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::name() const noexcept
    -> std::string_view {
  return "RadHydro";
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void RadHydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] KOKKOS_FUNCTION auto
RadHydroPackage::get_flux_u(const int stage, const int ix) const -> double {
  return flux_u_(stage, ix);
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::get_fluid_basis() const
    -> const ModalBasis* {
  return fluid_basis_;
}

[[nodiscard]] KOKKOS_FUNCTION auto RadHydroPackage::get_rad_basis() const
    -> const ModalBasis* {
  return rad_basis_;
}

} // namespace radiation
