/**
 * @file hydro_package.cpp
 * --------------
 *
 * @brief Pure hydrodynamics package
 */
#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "utils/abstractions.hpp"

namespace fluid {

HydroPackage::HydroPackage(const ProblemIn* /*pin*/, int n_stages, EOS* eos,
                           ModalBasis* basis, BoundaryConditions* bcs,
                           double cfl, int nx, bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), basis_(basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", 3, nx + 2 + 1),
      u_f_l_("hydro::u_f_l_", 3, nx + 2), u_f_r_("hydro::u_f_r_", 3, nx + 2),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1) {
} // Need long term solution for flux_u_

KOKKOS_FUNCTION
void HydroPackage::update_explicit(const View3D<double> state,
                                   View3D<double> dU, const GridStructure& grid,
                                   const TimeStepInfo& dt_info) const {
  const auto& order = basis_->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();

  const auto stage = dt_info.stage;

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(state, &grid, basis_, bcs_, {0, 2});

  // --- Zero out dU  ---
  Kokkos::parallel_for(
      "Hydro :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_LAMBDA(const int icf, const int ix, const int k) {
        dU(icf, ix, k) = 0.0;
      });

  // --- Fluid Increment : Divergence ---
  fluid_divergence(state, dU, grid, stage);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Hydro :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int icf, const int ix, const int k) {
        dU(icf, ix, k) /= basis_->get_mass_matrix(ix, k);
      });

  // --- Increment from Geometry ---
  if (grid.do_geometry()) {
    fluid_geometry(state, dU, grid);
  }
}

// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
KOKKOS_FUNCTION
void HydroPackage::fluid_divergence(const View3D<double> state,
                                    View3D<double> dU,
                                    const GridStructure& grid,
                                    const int stage) const {
  const auto& nNodes = grid.get_n_nodes();
  const auto& order  = basis_->get_order();
  const auto& ilo    = grid.get_ilo();
  const auto& ihi    = grid.get_ihi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Hydro :: Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, ilo}, {NUM_VARS_, ihi + 2}),
      KOKKOS_CLASS_LAMBDA(const int iCF, const int iX) {
        u_f_l_(iCF, iX) = basis_->basis_eval(state, iX - 1, iCF, nNodes + 1);
        u_f_r_(iCF, iX) = basis_->basis_eval(state, iX, iCF, 0);
      });

  // --- Calc numerical flux at all faces ---
  Kokkos::parallel_for(
      "Hydro :: Numerical Fluxes", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_CLASS_LAMBDA(int iX) {
        auto uCF_L = Kokkos::subview(u_f_l_, Kokkos::ALL, iX);
        auto uCF_R = Kokkos::subview(u_f_r_, Kokkos::ALL, iX);

        // Debug mode assertions.
        assert(uCF_L(2) > 0.0 && !std::isnan(uCF_L(2)) &&
               "fluid_discretization :: Numerical Fluxes bad energy.");
        assert(uCF_R(2) > 0.0 && !std::isnan(uCF_R(2)) &&
               "fluid_discretization :: Numerical Fluxes bad energy.");

        auto lambda = nullptr;
        const double P_L =
            pressure_from_conserved(eos_, uCF_L(0), uCF_L(1), uCF_L(2), lambda);
        const double Cs_L = sound_speed_from_conserved(eos_, uCF_L(0), uCF_L(1),
                                                       uCF_L(2), lambda);

        const double P_R =
            pressure_from_conserved(eos_, uCF_R(0), uCF_R(1), uCF_R(2), lambda);
        const double Cs_R = sound_speed_from_conserved(eos_, uCF_R(0), uCF_R(1),
                                                       uCF_R(2), lambda);

        // --- Numerical Fluxes ---

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( uCF_L( 1 ), uCF_R( 1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            uCF_L(0), uCF_R(0), uCF_L(1), uCF_R(1), P_L, P_R, Cs_L, Cs_R);
        flux_u_(stage, iX) = flux_u;

        dFlux_num_(0, iX) = -flux_u_(stage, iX);
        dFlux_num_(1, iX) = flux_p;
        dFlux_num_(2, iX) = +flux_u_(stage, iX) * flux_p;
      });

  flux_u_(stage, ilo - 1) = flux_u_(stage, ilo);
  flux_u_(stage, ihi + 2) = flux_u_(stage, ihi + 1);

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Hydro :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {NUM_VARS_, ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int iCF, const int iX, const int k) {
        const auto& Poly_L   = basis_->get_phi(iX, 0, k);
        const auto& Poly_R   = basis_->get_phi(iX, nNodes + 1, k);
        const auto& X_L      = grid.get_left_interface(iX);
        const auto& X_R      = grid.get_left_interface(iX + 1);
        const auto& SqrtGm_L = grid.get_sqrt_gm(X_L);
        const auto& SqrtGm_R = grid.get_sqrt_gm(X_R);

        dU(iCF, iX, k) -= (+dFlux_num_(iCF, iX + 1) * Poly_R * SqrtGm_R -
                           dFlux_num_(iCF, iX + 0) * Poly_L * SqrtGm_L);
      });

  if (order > 1) [[likely]] {
    // --- Volume Term ---
    Kokkos::parallel_for(
        "Hydro :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
        KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          for (int iN = 0; iN < nNodes; ++iN) {
            const double weight  = grid.get_weights(iN);
            const double dphi    = basis_->get_d_phi(iX, iN + 1, k);
            const double X       = grid.node_coordinate(iX, iN);
            const double sqrt_gm = grid.get_sqrt_gm(X);

            auto lambda      = nullptr;
            const double vel = basis_->basis_eval(state, iX, 1, iN + 1);
            const double P   = pressure_from_conserved(
                eos_, basis_->basis_eval(state, iX, 0, iN + 1), vel,
                basis_->basis_eval(state, iX, 2, iN + 1), lambda);
            const auto [flux1, flux2, flux3] = flux_fluid(vel, P);

            local_sum1 += weight * flux1 * dphi * sqrt_gm;
            local_sum2 += weight * flux2 * dphi * sqrt_gm;
            local_sum3 += weight * flux3 * dphi * sqrt_gm;
          }

          dU(0, iX, k) += local_sum1;
          dU(1, iX, k) += local_sum2;
          dU(2, iX, k) += local_sum3;
        });
  }
}

KOKKOS_FUNCTION
void HydroPackage::fluid_geometry(const View3D<double> state, View3D<double> dU,
                                  const GridStructure& grid) const {
  const int& nNodes = grid.get_n_nodes();
  const int& order  = basis_->get_order();
  const int& ilo    = grid.get_ilo();
  const int& ihi    = grid.get_ihi();

  Kokkos::parallel_for(
      "Hydro :: Geometry Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_CLASS_LAMBDA(const int iX, const int k) {
        double local_sum = 0.0;
        auto lambda      = nullptr;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double P = pressure_from_conserved(
              eos_, basis_->basis_eval(state, iX, 0, iN + 1),
              basis_->basis_eval(state, iX, 1, iN + 1),
              basis_->basis_eval(state, iX, 2, iN + 1), lambda);

          const double X = grid.node_coordinate(iX, iN);

          local_sum +=
              grid.get_weights(iN) * P * basis_->get_phi(iX, iN + 1, k) * X;
        }

        dU(1, iX, k) += (2.0 * local_sum * grid.get_widths(iX)) /
                        basis_->get_mass_matrix(iX, k);
      });
}
/**
 * @brief explicit hydrodynamic timestep restriction
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

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::get_basis() const
    -> const ModalBasis* {
  return basis_;
}

} // namespace fluid
