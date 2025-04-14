/**
 * @file fluid_discretization.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Contains the main discretization routines for the fluid
 *
 * @details We implement the core DG updates for the fluid here, including
 *          - ComputerIncrement_Fluid_Divergence (hyperbolic term)
 *          - compute_increment_fluid_geometry (geometric source)
 *          - compute_increment_fluid_rad (radiation source term)
 */

#include <iostream>

#include "Kokkos_Core.hpp"

#include "boundary_conditions.hpp"
#include "error.hpp"
#include "fluid_discretization.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"

namespace fluid {
// Compute the divergence of the flux term for the update
void compute_increment_fluid_divergence(
    const View3D<Real> U, const GridStructure& Grid, const ModalBasis* Basis,
    const EOS* eos, View3D<Real> dU, View3D<Real> Flux_q,
    View2D<Real> dFlux_num, View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
    View1D<Real> Flux_U, View1D<Real> Flux_P ) {
  const auto& nNodes = Grid.get_n_nodes( );
  const auto& order  = Basis->get_order( );
  const auto& ilo    = Grid.get_ilo( );
  const auto& ihi    = Grid.get_ihi( );
  const int nvars    = U.extent( 0 );

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Fluid :: Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, ilo }, { nvars, ihi + 2 } ),
      KOKKOS_LAMBDA( const int iCF, const int iX ) {
        uCF_F_L( iCF, iX ) = Basis->basis_eval( U, iX - 1, iCF, nNodes + 1 );
        uCF_F_R( iCF, iX ) = Basis->basis_eval( U, iX, iCF, 0 );
      } );

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Fluid :: Numerical Fluxes", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_LAMBDA( int iX ) {
        auto uCF_L = Kokkos::subview( uCF_F_L, Kokkos::ALL, iX );
        auto uCF_R = Kokkos::subview( uCF_F_R, Kokkos::ALL, iX );

        const Real rho_L = 1.0 / uCF_L( 0 );
        const Real rho_R = 1.0 / uCF_R( 0 );

        // Debug mode assertions.
        assert( rho_L > 0.0 && !std::isnan( rho_L ) &&
                "fluid_discretization :: Numerical Fluxes bad rho." );
        assert( rho_R > 0.0 && !std::isnan( rho_R ) &&
                "fluid_discretization :: Numerical Fluxes bad rho." );
        assert( uCF_L( 2 ) > 0.0 && !std::isnan( uCF_L( 2 ) ) &&
                "fluid_discretization :: Numerical Fluxes bad energy." );
        assert( uCF_R( 2 ) > 0.0 && !std::isnan( uCF_R( 2 ) ) &&
                "fluid_discretization :: Numerical Fluxes bad energy." );

        auto lambda     = nullptr;
        const Real P_L  = eos->pressure_from_conserved( uCF_L( 0 ), uCF_L( 1 ),
                                                        uCF_L( 2 ), lambda );
        const Real Cs_L = eos->sound_speed_from_conserved(
            uCF_L( 0 ), uCF_L( 1 ), uCF_L( 2 ), lambda );
        const Real lam_L = Cs_L * rho_L;

        const Real P_R  = eos->pressure_from_conserved( uCF_R( 0 ), uCF_R( 1 ),
                                                        uCF_R( 2 ), lambda );
        const Real Cs_R = eos->sound_speed_from_conserved(
            uCF_R( 0 ), uCF_R( 1 ), uCF_R( 2 ), lambda );
        const Real lam_R = Cs_R * rho_R;

        // --- Numerical Fluxes ---

        // Riemann Problem
        numerical_flux_gudonov( uCF_L( 1 ), uCF_R( 1 ), P_L, P_R, lam_L, lam_R,
                                Flux_U( iX ), Flux_P( iX ) );
        // numerical_flux_hllc( uCF_L( 1 ), uCF_R( 1 ), P_L, P_R, Cs_L, Cs_R,
        //  rho_L, rho_R, Flux_U( iX ), Flux_P( iX ) );

        // TODO(astrobarker): Clean This Up
        dFlux_num( 0, iX ) = -Flux_U( iX );
        dFlux_num( 1, iX ) = +Flux_P( iX );
        dFlux_num( 2, iX ) = +Flux_U( iX ) * Flux_P( iX );
      } );

  Flux_U( ilo - 1 ) = Flux_U( ilo );
  Flux_U( ihi + 2 ) = Flux_U( ihi + 1 );

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Fluid :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { nvars, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
        const auto& Poly_L   = Basis->get_phi( iX, 0, k );
        const auto& Poly_R   = Basis->get_phi( iX, nNodes + 1, k );
        const auto& X_L      = Grid.get_left_interface( iX );
        const auto& X_R      = Grid.get_left_interface( iX + 1 );
        const auto& SqrtGm_L = Grid.get_sqrt_gm( X_L );
        const auto& SqrtGm_R = Grid.get_sqrt_gm( X_R );

        dU( iCF, iX, k ) -= ( +dFlux_num( iCF, iX + 1 ) * Poly_R * SqrtGm_R -
                              dFlux_num( iCF, iX + 0 ) * Poly_L * SqrtGm_L );
      } );

  if ( order > 1 ) {
    // --- Compute Flux_q everywhere for the Volume term ---
    Kokkos::parallel_for(
        "Fluid :: Flux_q",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { nvars, ihi + 1, nNodes } ),
        KOKKOS_LAMBDA( const int iCF, const int iX, const int iN ) {
          auto lambda  = nullptr;
          const Real P = eos->pressure_from_conserved(
              Basis->basis_eval( U, iX, 0, iN + 1 ),
              Basis->basis_eval( U, iX, 1, iN + 1 ),
              Basis->basis_eval( U, iX, 2, iN + 1 ), lambda );
          Flux_q( iCF, iX, iN ) =
              flux_fluid( Basis->basis_eval( U, iX, 1, iN + 1 ), P, iCF );
        } );

    // --- Volume Term ---
    // TODO(astrobarker): Make Flux_q a function?
    Kokkos::parallel_for(
        "Fluid :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { nvars, ihi + 1, order } ),
        KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
          Real local_sum = 0.0;
          for ( int iN = 0; iN < nNodes; iN++ ) {
            auto X = Grid.node_coordinate( iX, iN );
            local_sum += Grid.get_weights( iN ) * Flux_q( iCF, iX, iN ) *
                         Basis->get_d_phi( iX, iN + 1, k ) *
                         Grid.get_sqrt_gm( X );
          }

          dU( iCF, iX, k ) += local_sum;
        } );
  }
}

/**
 * Compute fluid increment from geometry in spherical symmetry
 * TODO: ? missing sqrt(det gamma) ?
 **/
void compute_increment_fluid_geometry( const View3D<Real> U,
                                       const GridStructure& Grid,
                                       const ModalBasis* Basis, const EOS* eos,
                                       View3D<Real> dU ) {
  const int nNodes = Grid.get_n_nodes( );
  const int order  = Basis->get_order( );
  const int ilo    = Grid.get_ilo( );
  const int ihi    = Grid.get_ihi( );

  Kokkos::parallel_for(
      "Fluid :: Geometry Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { ilo, 0 }, { ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iX, const int k ) {
        Real local_sum = 0.0;
        auto lambda    = nullptr;
        for ( int iN = 0; iN < nNodes; iN++ ) {
          const Real P = eos->pressure_from_conserved(
              Basis->basis_eval( U, iX, 0, iN + 1 ),
              Basis->basis_eval( U, iX, 1, iN + 1 ),
              Basis->basis_eval( U, iX, 2, iN + 1 ), lambda );

          Real X = Grid.node_coordinate( iX, iN );

          local_sum +=
              Grid.get_weights( iN ) * P * Basis->get_phi( iX, iN + 1, k ) * X;
        }

        dU( 1, iX, k ) += ( 2.0 * local_sum * Grid.get_widths( iX ) ) /
                          Basis->get_mass_matrix( iX, k );
      } );
}

/**
 * Compute fluid increment from radiation sources
 * TODO: Modify inputs?
 **/
auto compute_increment_fluid_rad( View2D<Real> uCF, const int k, const int iCF,
                                  const View2D<Real> uCR,
                                  const GridStructure& Grid,
                                  const ModalBasis* Basis, const EOS* eos,
                                  const Opacity* opac, const int iX ) -> Real {
  const int nNodes = Grid.get_n_nodes( );

  Real local_sum = 0.0;
  for ( int iN = 0; iN < nNodes; iN++ ) {
    const Real D   = 1.0 / Basis->basis_eval( uCF, iX, 0, iN + 1 );
    const Real Vel = Basis->basis_eval( uCF, iX, 1, iN + 1 );
    const Real EmT = Basis->basis_eval( uCF, iX, 2, iN + 1 );

    const Real Er = Basis->basis_eval( uCR, iX, 0, iN + 1 ) * D;
    const Real Fr = Basis->basis_eval( uCR, iX, 1, iN + 1 ) * D;
    const Real Pr = radiation::compute_closure( Er, Fr );

    auto lambda  = nullptr;
    const Real P = eos->pressure_from_conserved( 1.0 / D, Vel, EmT, lambda );
    const Real T = eos->temperature_from_tau_pressure( 1.0 / D, P, lambda );

    // TODO(astrobarker): composition
    const Real X = 1.0;
    const Real Y = 1.0;
    const Real Z = 1.0;

    const Real kappa_r = rosseland_mean( opac, D, T, X, Y, Z, lambda );
    const Real kappa_p = planck_mean( opac, D, T, X, Y, Z, lambda );

    local_sum +=
        Grid.get_weights( iN ) * Basis->get_phi( iX, iN + 1, k ) *
        source_fluid_rad( D, Vel, T, kappa_r, kappa_p, Er, Fr, Pr, iCF );
  }

  return ( local_sum * Grid.get_widths( iX ) ) /
         Basis->get_mass_matrix( iX, k );
}

/** Compute dU for timestep update. e.g., U = U + dU * dt
 *
 * Parameters:
 * -----------
 * U                : Conserved variables
 * Grid             : Grid object
 * Basis            : Basis object
 * dU               : Update vector
 * Flux_q           : Nodal fluxes, for volume term
 * dFLux_num        : numerical surface flux
 * uCF_F_L, uCF_F_R : left/right face states
 * Flux_U, Flux_P   : Fluxes (from Riemann problem)
 * uCF_L, uCF_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void compute_increment_explicit( const View3D<Real> U, const View3D<Real> uCR,
                                 const GridStructure& Grid,
                                 const ModalBasis* Basis, const EOS* eos,
                                 View3D<Real> dU, View3D<Real> Flux_q,
                                 View2D<Real> dFlux_num, View2D<Real> uCF_F_L,
                                 View2D<Real> uCF_F_R, View1D<Real> Flux_U,
                                 View1D<Real> Flux_P, const Options* opts ) {

  const auto& order = Basis->get_order( );
  const auto& ilo   = Grid.get_ilo( );
  const auto& ihi   = Grid.get_ihi( );
  const int nvars   = U.extent( 0 );

  // --- Apply BC ---
  bc::apply_bc( U, &Grid, order, opts->BC );
  if ( opts->do_rad ) {
    bc::apply_bc( uCR, &Grid, order, opts->BC );
  }

  // --- Detect Shocks ---
  // TODO(astrobarker): Code up a shock detector...

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                              { nvars, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
        dU( iCF, iX, k ) = 0.0;
      } );

  Kokkos::parallel_for(
      ihi + 2, KOKKOS_LAMBDA( const int iX ) { Flux_U( iX ) = 0.0; } );

  // --- Fluid Increment : Divergence ---
  compute_increment_fluid_divergence( U, Grid, Basis, eos, dU, Flux_q,
                                      dFlux_num, uCF_F_L, uCF_F_R, Flux_U,
                                      Flux_P );

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Fluid::Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { nvars, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
        dU( iCF, iX, k ) /= ( Basis->get_mass_matrix( iX, k ) );
      } );

  /* --- Increment from Geometry --- */
  if ( Grid.do_geometry( ) ) {
    compute_increment_fluid_geometry( U, Grid, Basis, eos, dU );
  }

  /* --- Increment Additional Explicit Sources --- */
}

} // namespace fluid
