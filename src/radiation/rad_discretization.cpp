/**
 * @file fluid_discretization.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Contains the main discretization routines for radiation.
 *
 * @details We implement the core DG updates for radiation here, including
 *          - ComputerIncrement_Rad_Divergence (hyperbolic term)
 *          - compute_increment_rad_source (coupling source term)
 */

#include "rad_discretization.hpp"
#include "boundary_conditions.hpp"
#include "constants.hpp"
#include "eos_variant.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"

namespace radiation {

// Compute the divergence of the flux term for the update
void compute_increment_rad_divergence(
    const View3D<double> uCR, const View3D<double> uCF,
    const GridStructure& grid, const ModalBasis* basis,
    const ModalBasis* fluid_basis, const EOS* /*eos*/, View3D<double> dU,
    View2D<double> dFlux_num, View2D<double> uCR_F_L, View2D<double> uCR_F_R,
    View1D<double> Flux_U) {
  const auto& nNodes = grid.get_n_nodes();
  const auto& order  = basis->get_order();
  const auto& ilo    = grid.get_ilo();
  const auto& ihi    = grid.get_ihi();
  const int nvars    = 2;

  static constexpr double c2 = constants::c_cgs * constants::c_cgs;

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Radiation :: Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, ilo}, {nvars, ihi + 2}),
      KOKKOS_LAMBDA(const int iCR, const int iX) {
        uCR_F_L(iCR, iX) = basis->basis_eval(uCR, iX - 1, iCR, nNodes + 1);
        uCR_F_R(iCR, iX) = basis->basis_eval(uCR, iX, iCR, 0);
      });

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Radiation :: Numerical Fluxes", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_LAMBDA(const int iX) {
        auto uCR_L = Kokkos::subview(uCR_F_L, Kokkos::ALL, iX);
        auto uCR_R = Kokkos::subview(uCR_F_R, Kokkos::ALL, iX);

        assert(uCR_L(0) > 0.0 && !std::isnan(uCR_L(0)) &&
               "rad_Discretization :: Numerical Fluxes bad energy.");
        assert(uCR_R(0) > 0.0 && !std::isnan(uCR_R(0)) &&
               "rad_Discretization :: Numerical Fluxes bad energy.");
        assert(!std::isnan(uCR_L(1)) &&
               "rad_Discretization :: Numerical Fluxes bad flux.");
        assert(!std::isnan(uCR_R(1)) &&
               "rad_Discretization :: Numerical Fluxes bad flux.");

        const double E_L = uCR_L(0);
        const double F_L = uCR_L(1);
        const double E_R = uCR_R(0);
        const double F_R = uCR_R(1);

        const double P_L = compute_closure(E_L, F_L);
        const double P_R = compute_closure(E_R, F_R);

        // --- Numerical Fluxes ---

        // Riemann Problem

        // TODO(astrobarker): make this flexible.
        const double vstar = Flux_U(iX);
        // auto [flux_e, flux_f] =
        //    numerical_flux_hll_rad( E_L, E_R, F_L, F_R, P_L, P_R, vstar );
        const double eddington_factor = P_L / E_L;
        const double alpha =
            (constants::c_cgs - vstar) * std::sqrt(eddington_factor);
        auto flux_e             = llf_flux(F_R, F_L, E_R, E_L, alpha);
        auto flux_f             = llf_flux(c2 * P_R, c2 * P_L, F_R, F_L, alpha);
        double advective_flux_e = 0.0;
        double advective_flux_f = 0.0;

        advective_flux_e = (vstar >= 0) ? vstar * E_L : vstar * E_R;
        advective_flux_f = (vstar >= 0) ? vstar * F_L : vstar * F_R;

        dFlux_num(0, iX) = flux_e - advective_flux_e;
        dFlux_num(1, iX) = flux_f - advective_flux_f;
      });

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Radiation :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {nvars, ihi + 1, order}),
      KOKKOS_LAMBDA(const int iCR, const int iX, const int k) {
        const auto& Poly_L   = basis->get_phi(iX, 0, k);
        const auto& Poly_R   = basis->get_phi(iX, nNodes + 1, k);
        const auto& X_L      = grid.get_left_interface(iX);
        const auto& X_R      = grid.get_left_interface(iX + 1);
        const auto& SqrtGm_L = grid.get_sqrt_gm(X_L);
        const auto& SqrtGm_R = grid.get_sqrt_gm(X_R);

        dU(iCR, iX, k) -= (+dFlux_num(iCR, iX + 1) * Poly_R * SqrtGm_R -
                           dFlux_num(iCR, iX + 0) * Poly_L * SqrtGm_L);
      });

  if (order > 1) {
    // --- Volume Term ---
    Kokkos::parallel_for(
        "Radiation :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                               {nvars, ihi + 1, order}),
        KOKKOS_LAMBDA(const int iCR, const int iX, const int k) {
          double local_sum = 0.0;
          for (int iN = 0; iN < nNodes; ++iN) {
            const auto P =
                compute_closure(basis->basis_eval(uCR, iX, 0, iN + 1),
                                basis->basis_eval(uCR, iX, 1, iN + 1));
            const double flux =
                flux_rad(basis->basis_eval(uCR, iX, 0, iN + 1),
                         basis->basis_eval(uCR, iX, 1, iN + 1), P,
                         fluid_basis->basis_eval(uCF, iX, 1, iN + 1), iCR);
            const auto X = grid.node_coordinate(iX, iN);
            local_sum += grid.get_weights(iN) * flux *
                         basis->get_d_phi(iX, iN + 1, k) * grid.get_sqrt_gm(X);
          }

          dU(iCR, iX, k) += local_sum;
        });
  }
}

/**
 * Compute rad increment from source terms
 **/
auto compute_increment_rad_source(const View2D<double> uCR, const int k,
                                  const int iCR, const View2D<double> uCF,
                                  const GridStructure& grid,
                                  const ModalBasis* fluid_basis,
                                  const ModalBasis* rad_basis, const EOS* eos,
                                  const Opacity* opac, const int iX) -> double {
  constexpr static double c = constants::c_cgs;
  const int nNodes          = grid.get_n_nodes();

  double local_sum = 0.0;
  for (int iN = 0; iN < nNodes; ++iN) {
    const double tau  = fluid_basis->basis_eval(uCF, iX, 0, iN + 1);
    const double rho  = 1.0 / tau;
    const double vel  = fluid_basis->basis_eval(uCF, iX, 1, iN + 1);
    const double em_t = fluid_basis->basis_eval(uCF, iX, 2, iN + 1);

    auto lambda      = nullptr;
    const double t_g = temperature_from_conserved(eos, tau, vel, em_t, lambda);

    // TODO(astrobarker): composition
    const double X = 1.0;
    const double Y = 1.0;
    const double Z = 1.0;

    const double kappa_r = rosseland_mean(opac, rho, t_g, X, Y, Z, lambda);
    const double kappa_p = planck_mean(opac, rho, t_g, X, Y, Z, lambda);

    const double E_r = rad_basis->basis_eval(uCR, iX, 0, iN + 1);
    const double F_r = rad_basis->basis_eval(uCR, iX, 1, iN + 1);
    const double P_r = compute_closure(E_r, F_r);
    const auto [G0, G] =
        radiation_four_force(rho, vel, t_g, kappa_r, kappa_p, E_r, F_r, P_r);

    const double this_source = (iCR == 0) ? -c * G0 : -c * c * G;

    local_sum +=
        grid.get_weights(iN) * rad_basis->get_phi(iX, iN + 1, k) * this_source;
  }

  return (local_sum * grid.get_widths(iX)) / rad_basis->get_mass_matrix(iX, k);
}

/**
 * @brief Compute source terms for radiation hydrodynamics system
 * @note Returns tuple<S_egas, S_vgas, S_erad, S_frad>
 *
 *   Note that here we take in a single 2D view uCRH representing the radhydro
 *   state on a given cell. The indices are:
 *     0: fluid specific volume
 *     1: fluid velocity
 *     2: fluid total specific energy
 *     3: radiation energy density
 *     4: radiation flux F
 **/
auto compute_increment_radhydro_source(const View2D<double> uCRH, const int k,
                                       const GridStructure& grid,
                                       const ModalBasis* fluid_basis,
                                       const ModalBasis* rad_basis,
                                       const EOS* eos, const Opacity* opac,
                                       const int iX)
    -> std::tuple<double, double, double, double> {
  constexpr static double c  = constants::c_cgs;
  constexpr static double c2 = c * c;

  const int nNodes = grid.get_n_nodes();

  double local_sum_e_r = 0.0; // radiation energy source
  double local_sum_m_r = 0.0; // radiation momentum (flux) source
  double local_sum_e_g = 0.0; // gas energy source
  double local_sum_m_g = 0.0; // gas momentum (velocity) source
  for (int iN = 0; iN < nNodes; ++iN) {
    // Note: basis evaluations are awkward here.
    // must be sure to use the correct basis functions.
    const double tau  = fluid_basis->basis_eval(uCRH, iX, 0, iN + 1);
    const double rho  = 1.0 / tau;
    const double vel  = fluid_basis->basis_eval(uCRH, iX, 1, iN + 1);
    const double em_t = fluid_basis->basis_eval(uCRH, iX, 2, iN + 1);

    auto lambda      = nullptr;
    const double t_g = temperature_from_conserved(eos, tau, vel, em_t, lambda);

    // TODO(astrobarker): composition
    const double X = 1.0;
    const double Y = 1.0;
    const double Z = 1.0;

    const double kappa_r = rosseland_mean(opac, rho, t_g, X, Y, Z, lambda);
    const double kappa_p = planck_mean(opac, rho, t_g, X, Y, Z, lambda);

    const double E_r = rad_basis->basis_eval(uCRH, iX, 3, iN + 1);
    const double F_r = rad_basis->basis_eval(uCRH, iX, 4, iN + 1);
    const double P_r = compute_closure(E_r, F_r);

    // 4 force
    const auto [G0, G] =
        radiation_four_force(rho, vel, t_g, kappa_r, kappa_p, E_r, F_r, P_r);

    const double source_e_r = -c * G0;
    const double source_m_r = -c2 * G;
    const double source_e_g = c * G0;
    const double source_m_g = G;

    local_sum_e_r +=
        grid.get_weights(iN) * rad_basis->get_phi(iX, iN + 1, k) * source_e_r;
    local_sum_m_r +=
        grid.get_weights(iN) * rad_basis->get_phi(iX, iN + 1, k) * source_m_r;
    local_sum_e_g +=
        grid.get_weights(iN) * fluid_basis->get_phi(iX, iN + 1, k) * source_e_g;
    local_sum_m_g +=
        grid.get_weights(iN) * fluid_basis->get_phi(iX, iN + 1, k) * source_m_g;
  }
  // \Delta x / M_kk
  const double dx_o_mkk_fluid =
      grid.get_widths(iX) / fluid_basis->get_mass_matrix(iX, k);
  const double dx_o_mkk_rad =
      grid.get_widths(iX) / rad_basis->get_mass_matrix(iX, k);

  return {local_sum_m_g * dx_o_mkk_fluid, local_sum_e_g * dx_o_mkk_fluid,
          local_sum_e_r * dx_o_mkk_rad, local_sum_m_r * dx_o_mkk_rad};
}

/** Compute dU for timestep update. e.g., U = U + dU * dt
 *
 * Parameters:
 * -----------
 * U                : Conserved variables
 * grid             : grid object
 * basis            : basis object
 * dU               : Update vector
 * dFLux_num        : numerical surface flux
 * uCR_F_L, uCR_F_R : left/right face states
 * Flux_U           : Fluxes (from Riemann problem)
 * uCR_L, uCR_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void compute_increment_rad_explicit(
    const View3D<double> uCR, const View3D<double> uCF,
    const GridStructure& grid, const ModalBasis* basis,
    const ModalBasis* fluid_basis, const EOS* eos, View3D<double> dU,
    View2D<double> dFlux_num, View2D<double> uCR_F_L, View2D<double> uCR_F_R,
    View1D<double> Flux_U, const Options* opts, BoundaryConditions* bcs) {

  const auto& order          = basis->get_order();
  const auto& ilo            = grid.get_ilo();
  const auto& ihi            = grid.get_ihi();
  static constexpr int nvars = 2;

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(uCR, &grid, basis, bcs);
  bc::fill_ghost_zones<3>(uCF, &grid, fluid_basis, bcs);

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Rad :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {nvars, ihi + 1, order}),
      KOKKOS_LAMBDA(const int iCR, const int iX, const int k) {
        dU(iCR, iX, k) = 0.0;
      });

  // --- Increment : Divergence ---
  compute_increment_rad_divergence(uCR, uCF, grid, basis, fluid_basis, eos, dU,
                                   dFlux_num, uCR_F_L, uCR_F_R, Flux_U);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Rad :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {nvars, ihi + 1, order}),
      KOKKOS_LAMBDA(const int iCR, const int iX, const int k) {
        dU(iCR, iX, k) /= (basis->get_mass_matrix(iX, k));
      });
}

} // namespace radiation
