/**
 * File     :  Rad_Discretization.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : The main radiation spatial update routines go here.
 *  Compute divergence term.
 **/

#include <iostream>

#include "Kokkos_Core.hpp"

#include "Error.hpp"
#include "Grid.hpp"
#include "PolynomialBasis.hpp"
#include "Rad_Discretization.hpp"
#include "BoundaryConditionsLibrary.hpp"
#include "EquationOfStateLibrary_IDEAL.hpp"
#include "RadUtilities.hpp"

// Compute the divergence of the flux term for the update
void ComputeIncrement_Rad_Divergence(
    const Kokkos::View<Real ***> U, Kokkos::View<Real ***> uCF, 
    GridStructure *Grid, ModalBasis *Basis, Kokkos::View<Real ***> dU, 
    Kokkos::View<Real ***> Flux_q, Kokkos::View<Real **> dFlux_num, 
    Kokkos::View<Real **> uCR_F_L, Kokkos::View<Real **> uCR_F_R, 
    Kokkos::View<Real *> Flux_U, Kokkos::View<Real *> Flux_P )
{
  const auto &nNodes = Grid->Get_nNodes( );
  const auto &order  = Basis->Get_Order( );
  const auto &ilo    = Grid->Get_ilo( );
  const auto &ihi    = Grid->Get_ihi( );

  // Real rho_L, rho_R, P_L, P_R, Cs_L, Cs_R, lam_L, lam_R, P;

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Interface States; Rad",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { ilo, 0 }, { ihi + 2, 3 } ),
      KOKKOS_LAMBDA( const int iX, const int iCF ) {
        uCR_F_L( iCF, iX ) =
            Basis->BasisEval( U, iX - 1, iCF, nNodes + 1, false );
        uCR_F_R( iCF, iX ) = Basis->BasisEval( U, iX, iCF, 0, false );
      } );

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Numerical Fluxes; Rad", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_LAMBDA( UInt iX ) {
        auto uCR_L = Kokkos::subview( uCR_F_L, Kokkos::ALL, iX );
        auto uCR_R = Kokkos::subview( uCR_F_R, Kokkos::ALL, iX );

        const Real Em_L = uCR_L( 0 );
        const Real Fm_L = uCR_L( 1 );
        const Real Em_R = uCR_R( 0 );
        const Real Fm_R = uCR_R( 1 );

        const Real P_L = ComputeClosure( Em_L, Fm_L );
        const Real P_R = ComputeClosure( Em_R, Fm_R );


        // --- Numerical Fluxes ---

        // Riemann Problem
        // TODO: radiation riemann solver
        NumericalFlux_Gudonov( uCR_L( 1 ), uCR_R( 1 ), P_L, P_R, lam_L, lam_R,
                               Flux_U( iX ), Flux_P( iX ) );
        // NumericalFlux_HLLC( uCR_L( 1 ), uCR_R( 1 ), P_L, P_R, Cs_L, Cs_R,
        //  rho_L, rho_R, Flux_U( iX ), Flux_P( iX ) );

        // TODO: Clean This Up
        dFlux_num( 0, iX ) = -Flux_U( iX );
        dFlux_num( 1, iX ) = +Flux_P( iX );
        dFlux_num( 2, iX ) = +Flux_U( iX ) * Flux_P( iX );
      } );

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Surface Term; Rad",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { order, ihi + 1, 3 } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
        const auto &Poly_L   = Basis->Get_Phi( iX, 0, k );
        const auto &Poly_R   = Basis->Get_Phi( iX, nNodes + 1, k );
        const auto &X_L      = Grid->Get_LeftInterface( iX );
        const auto &X_R      = Grid->Get_LeftInterface( iX + 1 );
        const auto &SqrtGm_L = Grid->Get_SqrtGm( X_L );
        const auto &SqrtGm_R = Grid->Get_SqrtGm( X_R );

        dU( iCF, iX, k ) -= ( +dFlux_num( iCF, iX + 1 ) * Poly_R * SqrtGm_R -
                              dFlux_num( iCF, iX + 0 ) * Poly_L * SqrtGm_L );
      } );

  if ( order > 1 )
  {
    // --- Compute Flux_q everywhere for the Volume term ---
    Kokkos::parallel_for(
        "Flux_q; Rad",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { nNodes, ihi + 1, 3 } ),
        KOKKOS_LAMBDA( const int iN, const int iX, const int iCF ) {
          const auto P = ComputePressureFromConserved_IDEAL(
              Basis->BasisEval( U, iX, 0, iN + 1, false ),
              Basis->BasisEval( U, iX, 1, iN + 1, false ),
              Basis->BasisEval( U, iX, 2, iN + 1, false ) );
          Flux_q( iCF, iX, iN ) =
              Flux_Rad( Basis->BasisEval( U, iX, 1, iN + 1, false ), P, iCF );
        } );

    // --- Volume Term ---
    // TODO: Make Flux_q a function?
    Kokkos::parallel_for(
        "Volume Term; Rad",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { order, ihi + 1, 3 } ),
        KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
          Real local_sum = 0.0;
          for ( UInt iN = 0; iN < nNodes; iN++ )
          {
            auto X = Grid->NodeCoordinate( iX, iN );
            local_sum += Grid->Get_Weights( iN ) * Flux_q( iCF, iX, iN ) *
                         Basis->Get_dPhi( iX, iN + 1, k ) *
                         Grid->Get_SqrtGm( X );
          }

          dU( iCF, iX, k ) += local_sum;
        } );
  }
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
 * uCR_F_L, uCR_F_R : left/right face states
 * Flux_U, Flux_P   : Fluxes (from Riemann problem)
 * uCR_L, uCR_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void Compute_Increment_Explicit_Rad(
    const Kokkos::View<Real ***> U, GridStructure *Grid, ModalBasis *Basis,
    Kokkos::View<Real ***> dU, Kokkos::View<Real ***> Flux_q,
    Kokkos::View<Real **> dFlux_num, Kokkos::View<Real **> uCR_F_L,
    Kokkos::View<Real **> uCR_F_R, Kokkos::View<Real *> Flux_U,
    Kokkos::View<Real *> Flux_P, const std::string BC )
{

  const auto &order = Basis->Get_Order( );
  const auto &ilo   = Grid->Get_ilo( );
  const auto &ihi   = Grid->Get_ihi( );

  // --- Apply BC ---
  ApplyBC( U, Grid, order, BC );

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Zero dU; Rad",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                              { order, ihi + 1, 3 } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
        dU( iCF, iX, k ) = 0.0;
      } );

  Kokkos::parallel_for(
      ihi + 2, KOKKOS_LAMBDA( UInt iX ) { Flux_U( iX ) = 0.0; } );

  // --- Increment : Divergence ---
  ComputeIncrement_Rad_Divergence( U, Grid, Basis, dU, Flux_q, dFlux_num,
                                     uCR_F_L, uCR_F_R, Flux_U, Flux_P );

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Divide Update / Mass Matrix; Rad",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { order, ihi + 1, 3 } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
        dU( iCF, iX, k ) /= ( Basis->Get_MassMatrix( iX, k ) );
      } );


  /* --- Increment Source Terms --- */
}
