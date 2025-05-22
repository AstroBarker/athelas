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

#include <iostream>

#include "Kokkos_Core.hpp"

#include "boundary_conditions.hpp"
#include "constants.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "grid.hpp"
#include "opacity/opac.hpp"
#include "polynomial_basis.hpp"
#include "rad_discretization.hpp"
#include "rad_utilities.hpp"

namespace radiation {

// Compute the divergence of the flux term for the update
void compute_increment_rad_divergence(
    const View3D<Real> uCR, const View3D<Real> uCF, const GridStructure& grid,
    const ModalBasis* basis, const EOS* /*eos*/, View3D<Real> dU,
    View3D<Real> Flux_q, View2D<Real> dFlux_num, View2D<Real> uCR_F_L,
    View2D<Real> uCR_F_R, View1D<Real> Flux_U, View1D<Real> Flux_P ) {
  const auto& nNodes = grid.get_n_nodes( );
  const auto& order  = basis->get_order( );
  const auto& ilo    = grid.get_ilo( );
  const auto& ihi    = grid.get_ihi( );
  const int nvars    = 2;

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Radiation :: Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, ilo }, { nvars, ihi + 2 } ),
      KOKKOS_LAMBDA( const int iCR, const int iX ) {
        uCR_F_L( iCR, iX ) = basis->basis_eval( uCR, iX - 1, iCR, nNodes + 1 );
        uCR_F_R( iCR, iX ) = basis->basis_eval( uCR, iX, iCR, 0 );
      } );

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Radiation :: Numerical Fluxes", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_LAMBDA( const int iX ) {
        auto uCR_L = Kokkos::subview( uCR_F_L, Kokkos::ALL, iX );
        auto uCR_R = Kokkos::subview( uCR_F_R, Kokkos::ALL, iX );

        const Real tauR = basis->basis_eval( uCF, iX, 0, 0 );
        const Real tauL = basis->basis_eval( uCF, iX - 1, 0, nNodes + 1 );

        // Debug mode assertions.
        assert( tauL > 0.0 && !std::isnan( tauL ) &&
                "rad_discretization :: Numerical Fluxes bad specific volume." );
        assert( tauR > 0.0 && !std::isnan( tauR ) &&
                "rad_discretization :: Numerical Fluxes bad specific volume." );
        assert( uCR_L( 0 ) > 0.0 && !std::isnan( uCR_L( 0 ) ) &&
                "rad_Discretization :: Numerical Fluxes bad energy." );
        assert( uCR_R( 0 ) > 0.0 && !std::isnan( uCR_R( 0 ) ) &&
                "rad_Discretization :: Numerical Fluxes bad energy." );
        assert( !std::isnan( uCR_L( 1 ) ) &&
                "rad_Discretization :: Numerical Fluxes bad flux." );
        assert( !std::isnan( uCR_R( 1 ) ) &&
                "rad_Discretization :: Numerical Fluxes bad flux." );

        const Real Em_L = uCR_L( 0 ) / tauL;
        const Real Fm_L = uCR_L( 1 ) / tauL;
        const Real Em_R = uCR_R( 0 ) / tauR;
        const Real Fm_R = uCR_R( 1 ) / tauR;

        const Real P_L = compute_closure( Em_L, Fm_L );
        const Real P_R = compute_closure( Em_R, Fm_R );

        // --- Numerical Fluxes ---

        // Riemann Problem

        /*
        auto c_cgs = constants::c_cgs;
        Real Fp = flux_rad( Em_R, Fm_R, vR, P_R, 0 );
        Real Fm = flux_rad( Em_L, Fm_L, vL, P_L, 0 );
        auto flux_e = llf_flux( Fp, Fm, Em_R, Em_L, c_cgs );

        Fp = flux_rad( Em_R, Fm_R, P_R, vR, 1 );
        Fm = flux_rad( Em_L, Fm_L, P_L, vL, 1 );
        auto flux_f = llf_flux( Fp, Fm, Fm_R, Fm_L, c_cgs );
        */

        const Real vstar = Flux_U( iX );
        auto [flux_e, flux_f] =
            numerical_flux_hll_rad( Em_L, Em_R, Fm_L, Fm_R, P_L, P_R, vstar );

        // upwind advective fluxes
        const Real advective_flux_e =
            ( vstar >= 0.0 ) ? vstar * Em_L : vstar * Em_R;
        const Real advective_flux_f =
            ( vstar >= 0.0 ) ? vstar * Fm_L : vstar * Fm_R;

        flux_e -= advective_flux_e;
        flux_f -= advective_flux_f;

        dFlux_num( 0, iX ) = flux_e;
        dFlux_num( 1, iX ) = flux_f;
      } );

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Radiation :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { nvars, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCR, const int iX, const int k ) {
        const auto& Poly_L   = basis->get_phi( iX, 0, k );
        const auto& Poly_R   = basis->get_phi( iX, nNodes + 1, k );
        const auto& X_L      = grid.get_left_interface( iX );
        const auto& X_R      = grid.get_left_interface( iX + 1 );
        const auto& SqrtGm_L = grid.get_sqrt_gm( X_L );
        const auto& SqrtGm_R = grid.get_sqrt_gm( X_R );

        dU( iCR, iX, k ) -= ( +dFlux_num( iCR, iX + 1 ) * Poly_R * SqrtGm_R -
                              dFlux_num( iCR, iX + 0 ) * Poly_L * SqrtGm_L );
      } );

  if ( order > 1 ) {
    // --- Compute Flux_q everywhere for the Volume term ---
    Kokkos::parallel_for(
        "Radiation :: Flux_q",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { nvars, ihi + 1, nNodes } ),
        KOKKOS_LAMBDA( const int iCR, const int iX, const int iN ) {
          const Real rho = 1.0 / basis->basis_eval( uCF, iX, 0, iN + 1 );
          const auto P =
              compute_closure( basis->basis_eval( uCR, iX, 0, iN + 1 ) * rho,
                               basis->basis_eval( uCR, iX, 1, iN + 1 ) * rho );
          Flux_q( iCR, iX, iN ) =
              flux_rad( basis->basis_eval( uCR, iX, 0, iN + 1 ) * rho,
                        basis->basis_eval( uCR, iX, 1, iN + 1 ) * rho, P,
                        basis->basis_eval( uCF, iX, 1, iN + 1 ), iCR );
        } );

    // --- Volume Term ---
    // TODO(astrobarker): Make Flux_q a function?
    Kokkos::parallel_for(
        "Radiation :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { nvars, ihi + 1, order } ),
        KOKKOS_LAMBDA( const int iCR, const int iX, const int k ) {
          Real local_sum = 0.0;
          for ( int iN = 0; iN < nNodes; iN++ ) {
            auto X = grid.node_coordinate( iX, iN );
            local_sum += grid.get_weights( iN ) * Flux_q( iCR, iX, iN ) *
                         basis->get_d_phi( iX, iN + 1, k ) *
                         grid.get_sqrt_gm( X );
          }

          dU( iCR, iX, k ) += local_sum;
        } );
  }
}

/**
 * Compute rad increment from source terms
 **/
auto compute_increment_rad_source( View2D<Real> uCR, const int k, const int iCR,
                                   const View2D<Real> uCF,
                                   const GridStructure& grid,
                                   const ModalBasis* basis, const EOS* eos,
                                   const Opacity* opac, const int iX ) -> Real {
  const int nNodes = grid.get_n_nodes( );

  Real local_sum = 0.0;
  for ( int iN = 0; iN < nNodes; iN++ ) {
    const Real D    = 1.0 / basis->basis_eval( uCF, iX, 0, iN + 1 );
    const Real V    = basis->basis_eval( uCF, iX, 1, iN + 1 );
    const Real Em_T = basis->basis_eval( uCF, iX, 2, iN + 1 );

    auto lambda  = nullptr;
    const Real T = eos->temperature_from_conserved( 1.0 / D, V, Em_T, lambda );

    // TODO(astrobarker): composition
    const Real X = 1.0;
    const Real Y = 1.0;
    const Real Z = 1.0;

    const Real kappa_r = rosseland_mean( opac, D, T, X, Y, Z, lambda );
    const Real kappa_p = planck_mean( opac, D, T, X, Y, Z, lambda );

    const Real E_r = basis->basis_eval( uCR, iX, 0, iN + 1 ) * D;
    const Real F_r = basis->basis_eval( uCR, iX, 1, iN + 1 ) * D;
    const Real P_r = compute_closure( E_r, F_r );

    const Real this_source =
        source_rad( D, V, T, kappa_r, kappa_p, E_r, F_r, P_r, iCR );

    local_sum +=
        grid.get_weights( iN ) * basis->get_phi( iX, iN + 1, k ) * this_source;
  }

  return ( local_sum * grid.get_widths( iX ) ) /
         basis->get_mass_matrix( iX, k );
}

/** Compute dU for timestep update. e.g., U = U + dU * dt
 *
 * Parameters:
 * -----------
 * U                : Conserved variables
 * grid             : grid object
 * basis            : basis object
 * dU               : Update vector
 * Flux_q           : Nodal fluxes, for volume term
 * dFLux_num        : numerical surface flux
 * uCR_F_L, uCR_F_R : left/right face states
 * Flux_U, Flux_P   : Fluxes (from Riemann problem)
 * uCR_L, uCR_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void compute_increment_rad_explicit(
    const View3D<Real> uCR, const View3D<Real> uCF, const GridStructure& grid,
    const ModalBasis* basis, const EOS* eos, View3D<Real> dU,
    View3D<Real> Flux_q, View2D<Real> dFlux_num, View2D<Real> uCR_F_L,
    View2D<Real> uCR_F_R, View1D<Real> Flux_U, View1D<Real> Flux_P,
    const Options* opts, BoundaryConditions* bcs ) {

  const auto& order = basis->get_order( );
  const auto& ilo   = grid.get_ilo( );
  const auto& ihi   = grid.get_ihi( );
  const int nvars   = 2;

  // --- Apply BC ---
  bc::fill_ghost_zones<2>( uCR, &grid, order, bcs );
  bc::fill_ghost_zones<3>( uCF, &grid, order, bcs );

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Rad :: Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                              { nvars, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCR, const int iX, const int k ) {
        dU( iCR, iX, k ) = 0.0;
      } );

  // --- Increment : Divergence ---
  compute_increment_rad_divergence( uCR, uCF, grid, basis, eos, dU, Flux_q,
                                    dFlux_num, uCR_F_L, uCR_F_R, Flux_U,
                                    Flux_P );

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Rad :: Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { nvars, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCR, const int iX, const int k ) {
        dU( iCR, iX, k ) /= ( basis->get_mass_matrix( iX, k ) );
      } );

  /* --- Increment Source Terms --- */
}

} // namespace radiation
