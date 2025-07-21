/**
 * @file fluid_discretization.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Contains the main discretization routines for the fluid
 *
 * @details We implement the core DG updates for the fluid here, including
 *          - compute_increment_fluid_divergence (hyperbolic term)
 *          - compute_increment_fluid_geometry (geometric source)
 *          - compute_increment_fluid_source (radiation source term)
 */

#include "fluid_discretization.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "fluid/hydro_package.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"

namespace fluid {
// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
KOKKOS_FUNCTION
void HydroPackage::fluid_divergence(const View3D<double> state,
                                    View3D<double> dU,
                                    const GridStructure& grid,
                                    const int stage) {
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
        auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
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

  if (order > 1) {
    // --- Volume Term ---
    Kokkos::parallel_for(
        "Hydro :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                               {NUM_VARS_, ihi + 1, order}),
        KOKKOS_CLASS_LAMBDA(const int iCF, const int iX, const int k) {
          double local_sum = 0.0;
          for (int iN = 0; iN < nNodes; ++iN) {
            auto lambda    = nullptr;
            const double P = pressure_from_conserved(
                eos_, basis_->basis_eval(state, iX, 0, iN + 1),
                basis_->basis_eval(state, iX, 1, iN + 1),
                basis_->basis_eval(state, iX, 2, iN + 1), lambda);
            const double flux =
                flux_fluid(basis_->basis_eval(state, iX, 1, iN + 1), P, iCF);
            auto X = grid.node_coordinate(iX, iN);
            local_sum += grid.get_weights(iN) * flux *
                         basis_->get_d_phi(iX, iN + 1, k) * grid.get_sqrt_gm(X);
          }

          dU(iCF, iX, k) += local_sum;
        });
  }
}
// Compute the divergence of the flux term for the update
void compute_increment_fluid_divergence(
    const View3D<double> U, const GridStructure& grid, const ModalBasis* basis,
    const EOS* eos, View3D<double> dU, View2D<double> dFlux_num,
    View2D<double> u_f_l_, View2D<double> u_f_r_, View1D<double> Flux_U) {
  const auto& nNodes = grid.get_n_nodes();
  const auto& order  = basis->get_order();
  const auto& ilo    = grid.get_ilo();
  const auto& ihi    = grid.get_ihi();
  const int nvars    = U.extent(0);

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Fluid :: Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, ilo}, {nvars, ihi + 2}),
      KOKKOS_LAMBDA(const int iCF, const int iX) {
        u_f_l_(iCF, iX) = basis->basis_eval(U, iX - 1, iCF, nNodes + 1);
        u_f_r_(iCF, iX) = basis->basis_eval(U, iX, iCF, 0);
      });

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Fluid :: Numerical Fluxes", Kokkos::RangePolicy<>(ilo, ihi + 2),
      KOKKOS_LAMBDA(int iX) {
        auto uCF_L = Kokkos::subview(u_f_l_, Kokkos::ALL, iX);
        auto uCF_R = Kokkos::subview(u_f_r_, Kokkos::ALL, iX);

        // Debug mode assertions.
        assert(uCF_L(2) > 0.0 && !std::isnan(uCF_L(2)) &&
               "fluid_discretization :: Numerical Fluxes bad energy.");
        assert(uCF_R(2) > 0.0 && !std::isnan(uCF_R(2)) &&
               "fluid_discretization :: Numerical Fluxes bad energy.");

        auto lambda = nullptr;
        const double P_L =
            pressure_from_conserved(eos, uCF_L(0), uCF_L(1), uCF_L(2), lambda);
        const double Cs_L = sound_speed_from_conserved(eos, uCF_L(0), uCF_L(1),
                                                       uCF_L(2), lambda);

        const double P_R =
            pressure_from_conserved(eos, uCF_R(0), uCF_R(1), uCF_R(2), lambda);
        const double Cs_R = sound_speed_from_conserved(eos, uCF_R(0), uCF_R(1),
                                                       uCF_R(2), lambda);

        // --- Numerical Fluxes ---

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( uCF_L( 1 ), uCF_R( 1
        // ), P_L, P_R, lam_L, lam_R);
        auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            uCF_L(0), uCF_R(0), uCF_L(1), uCF_R(1), P_L, P_R, Cs_L, Cs_R);
        Flux_U[iX] = flux_u;

        // TODO(astrobarker): Clean This Up
        dFlux_num(0, iX) = -Flux_U(iX);
        dFlux_num(1, iX) = flux_p;
        dFlux_num(2, iX) = +Flux_U(iX) * flux_p;
      });

  Flux_U(ilo - 1) = Flux_U(ilo);
  Flux_U(ihi + 2) = Flux_U(ihi + 1);

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Fluid :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {nvars, ihi + 1, order}),
      KOKKOS_LAMBDA(const int iCF, const int iX, const int k) {
        const auto& Poly_L   = basis->get_phi(iX, 0, k);
        const auto& Poly_R   = basis->get_phi(iX, nNodes + 1, k);
        const auto& X_L      = grid.get_left_interface(iX);
        const auto& X_R      = grid.get_left_interface(iX + 1);
        const auto& SqrtGm_L = grid.get_sqrt_gm(X_L);
        const auto& SqrtGm_R = grid.get_sqrt_gm(X_R);

        dU(iCF, iX, k) -= (+dFlux_num(iCF, iX + 1) * Poly_R * SqrtGm_R -
                           dFlux_num(iCF, iX + 0) * Poly_L * SqrtGm_L);
      });

  if (order > 1) {
    // --- Volume Term ---
    Kokkos::parallel_for(
        "Fluid :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                               {nvars, ihi + 1, order}),
        KOKKOS_LAMBDA(const int iCF, const int iX, const int k) {
          double local_sum = 0.0;
          for (int iN = 0; iN < nNodes; ++iN) {
            auto lambda    = nullptr;
            const double P = pressure_from_conserved(
                eos, basis->basis_eval(U, iX, 0, iN + 1),
                basis->basis_eval(U, iX, 1, iN + 1),
                basis->basis_eval(U, iX, 2, iN + 1), lambda);
            const double flux =
                flux_fluid(basis->basis_eval(U, iX, 1, iN + 1), P, iCF);
            auto X = grid.node_coordinate(iX, iN);
            local_sum += grid.get_weights(iN) * flux *
                         basis->get_d_phi(iX, iN + 1, k) * grid.get_sqrt_gm(X);
          }

          dU(iCF, iX, k) += local_sum;
        });
  }
}

/**
 * Compute fluid increment from geometry in spherical symmetry
 * TODO: ? missing sqrt(det gamma) ?
 **/
void compute_increment_fluid_geometry(const View3D<double> U,
                                      const GridStructure& grid,
                                      const ModalBasis* basis, const EOS* eos,
                                      View3D<double> dU) {
  const int nNodes = grid.get_n_nodes();
  const int order  = basis->get_order();
  const int ilo    = grid.get_ilo();
  const int ihi    = grid.get_ihi();

  Kokkos::parallel_for(
      "Fluid :: Geometry Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ilo, 0}, {ihi + 1, order}),
      KOKKOS_LAMBDA(const int iX, const int k) {
        double local_sum = 0.0;
        auto lambda      = nullptr;
        for (int iN = 0; iN < nNodes; ++iN) {
          const double P = pressure_from_conserved(
              eos, basis->basis_eval(U, iX, 0, iN + 1),
              basis->basis_eval(U, iX, 1, iN + 1),
              basis->basis_eval(U, iX, 2, iN + 1), lambda);

          double X = grid.node_coordinate(iX, iN);

          local_sum +=
              grid.get_weights(iN) * P * basis->get_phi(iX, iN + 1, k) * X;
        }

        dU(1, iX, k) += (2.0 * local_sum * grid.get_widths(iX)) /
                        basis->get_mass_matrix(iX, k);
      });
}

KOKKOS_FUNCTION
void HydroPackage::fluid_geometry(const View3D<double> state, View3D<double> dU,
                                  const GridStructure& grid) {
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
 * Compute fluid increment from radiation sources
 * TODO: Modify inputs?
 **/
auto compute_increment_fluid_source(const View2D<double> uCF, const int k,
                                    const int iCF, const View2D<double> uCR,
                                    const GridStructure& grid,
                                    const ModalBasis* fluid_basis,
                                    const ModalBasis* rad_basis, const EOS* eos,
                                    const Opacity* opac, const int iX)
    -> double {
  const int nNodes = grid.get_n_nodes();

  double local_sum = 0.0;
  for (int iN = 0; iN < nNodes; ++iN) {
    const double tau = fluid_basis->basis_eval(uCF, iX, 0, iN + 1);
    const double D   = 1.0 / tau;
    const double Vel = fluid_basis->basis_eval(uCF, iX, 1, iN + 1);
    const double EmT = fluid_basis->basis_eval(uCF, iX, 2, iN + 1);

    const double Er = rad_basis->basis_eval(uCR, iX, 0, iN + 1);
    const double Fr = rad_basis->basis_eval(uCR, iX, 1, iN + 1);
    const double Pr = radiation::compute_closure(Er, Fr);

    auto lambda    = nullptr;
    const double T = temperature_from_conserved(eos, tau, Vel, EmT, lambda);

    // TODO(astrobarker): composition
    const double X = 1.0;
    const double Y = 1.0;
    const double Z = 1.0;

    const double kappa_r = rosseland_mean(opac, D, T, X, Y, Z, lambda);
    const double kappa_p = planck_mean(opac, D, T, X, Y, Z, lambda);

    local_sum += grid.get_weights(iN) * fluid_basis->get_phi(iX, iN + 1, k) *
                 source_fluid_rad(D, Vel, T, kappa_r, kappa_p, Er, Fr, Pr, iCF);
  }

  return (local_sum * grid.get_widths(iX)) /
         fluid_basis->get_mass_matrix(iX, k);
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
 * u_f_l_, u_f_r_ : left/right face states
 * Flux_U           : Fluxes (from Riemann problem)
 * uCF_L, uCF_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void compute_increment_fluid_explicit(
    const View3D<double> U, const GridStructure& grid, const ModalBasis* basis,
    const EOS* eos, View3D<double> dU, View2D<double> dFlux_num,
    View2D<double> u_f_l_, View2D<double> u_f_r_, View1D<double> Flux_U,
    const Options* opts, BoundaryConditions* bcs) {

  const auto& order = basis->get_order();
  const auto& ilo   = grid.get_ilo();
  const auto& ihi   = grid.get_ihi();
  const int nvars   = U.extent(0);

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(U, &grid, basis, bcs);

  // --- Detect Shocks ---
  // TODO(astrobarker): Code up a shock detector...

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                             {nvars, ihi + 1, order}),
      KOKKOS_LAMBDA(const int iCF, const int iX, const int k) {
        dU(iCF, iX, k) = 0.0;
      });

  Kokkos::parallel_for(
      ihi + 2, KOKKOS_LAMBDA(const int iX) { Flux_U(iX) = 0.0; });

  // --- Fluid Increment : Divergence ---
  compute_increment_fluid_divergence(U, grid, basis, eos, dU, dFlux_num, u_f_l_,
                                     u_f_r_, Flux_U);

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Fluid::Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ilo, 0},
                                             {nvars, ihi + 1, order}),
      KOKKOS_LAMBDA(const int iCF, const int iX, const int k) {
        dU(iCF, iX, k) /= (basis->get_mass_matrix(iX, k));
      });

  /* --- Increment from Geometry --- */
  if (grid.do_geometry()) {
    compute_increment_fluid_geometry(U, grid, basis, eos, dU);
  }

  /* --- Increment Additional Explicit Sources --- */
}

} // namespace fluid
