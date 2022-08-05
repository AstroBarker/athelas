/**
 * File     :  Fluid_Discretization.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : The main fluid spatial update routines go here.
 *  Compute divergence term.
 **/

#include <iostream>

#include "Kokkos_Core.hpp"

#include "Error.h"
#include "Grid.h"
#include "PolynomialBasis.h"
#include "Fluid_Discretization.h"
#include "BoundaryConditionsLibrary.h"
#include "EquationOfStateLibrary_IDEAL.h"
#include "FluidUtilities.h"

// Compute the divergence of the flux term for the update
void ComputeIncrement_Fluid_Divergence(
    const Kokkos::View<double***> U, const GridStructure& Grid,
    const ModalBasis& Basis, Kokkos::View<double***> dU,
    Kokkos::View<double***> Flux_q, Kokkos::View<double**> dFlux_num,
    Kokkos::View<double**> uCF_F_L, Kokkos::View<double**> uCF_F_R,
    Kokkos::View<double*> Flux_U, Kokkos::View<double*> Flux_P )
{
  const unsigned int& nNodes = Grid.Get_nNodes( );
  const unsigned int& order  = Basis.Get_Order( );
  const unsigned int& ilo    = Grid.Get_ilo( );
  const unsigned int& ihi    = Grid.Get_ihi( );

  // double rho_L, rho_R, P_L, P_R, Cs_L, Cs_R, lam_L, lam_R, P;

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, ilo }, { 3, ihi + 2 } ),
      KOKKOS_LAMBDA( const int iCF, const int iX ) {
        uCF_F_L( iX, iCF ) =
            Basis.BasisEval( U, iX - 1, iCF, nNodes + 1, false );
        uCF_F_R( iX, iCF ) = Basis.BasisEval( U, iX, iCF, 0, false );
      } );

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Numerical Fluxes", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_LAMBDA( unsigned int iX ) {
        auto uCF_L = Kokkos::subview( uCF_F_L, iX, Kokkos::ALL );
        auto uCF_R = Kokkos::subview( uCF_F_R, iX, Kokkos::ALL );

        double rho_L = 1.0 / uCF_L( 0 );
        double rho_R = 1.0 / uCF_R( 0 );

        double P_L = ComputePressureFromConserved_IDEAL( uCF_L( 0 ), uCF_L( 1 ),
                                                         uCF_L( 2 ) );
        double Cs_L = ComputeSoundSpeedFromConserved_IDEAL(
            uCF_L( 0 ), uCF_L( 1 ), uCF_L( 2 ) );
        double lam_L = Cs_L * rho_L;

        double P_R = ComputePressureFromConserved_IDEAL( uCF_R( 0 ), uCF_R( 1 ),
                                                         uCF_R( 2 ) );
        double Cs_R = ComputeSoundSpeedFromConserved_IDEAL(
            uCF_R( 0 ), uCF_R( 1 ), uCF_R( 2 ) );
        double lam_R = Cs_R * rho_R;

        // --- Numerical Fluxes ---

        // Riemann Problem
        NumericalFlux_Gudonov( uCF_L( 1 ), uCF_R( 1 ), P_L, P_R, lam_L, lam_R,
                               Flux_U( iX ), Flux_P( iX ) );
        // NumericalFlux_HLLC( uCF_L( 1 ), uCF_R( 1 ), P_L, P_R, Cs_L, Cs_R,
        //  rho_L, rho_R, Flux_U( iX ), Flux_P( iX ) );

        // TODO: Clean This Up
        dFlux_num( 0, iX ) = -Flux_U( iX );
        dFlux_num( 1, iX ) = +Flux_P( iX );
        dFlux_num( 2, iX ) = +Flux_U( iX ) * Flux_P( iX );
      } );

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { 3, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
        double Poly_L   = Basis.Get_Phi( iX, 0, k );
        double Poly_R   = Basis.Get_Phi( iX, nNodes + 1, k );
        double X_L      = Grid.Get_LeftInterface( iX );
        double X_R      = Grid.Get_LeftInterface( iX + 1 );
        double SqrtGm_L = Grid.Get_SqrtGm( X_L );
        double SqrtGm_R = Grid.Get_SqrtGm( X_R );

        dU( iX, k, iCF ) -= ( +dFlux_num( iCF, iX + 1 ) * Poly_R * SqrtGm_R -
                              dFlux_num( iCF, iX + 0 ) * Poly_L * SqrtGm_L );
      } );

  // --- Compute Flux_q everywhere for the Volume term ---
  Kokkos::parallel_for(
      "Flux_q",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { 3, ihi + 1, nNodes } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int iN ) {
        double P = ComputePressureFromConserved_IDEAL(
            Basis.BasisEval( U, iX, 0, iN + 1, false ),
            Basis.BasisEval( U, iX, 1, iN + 1, false ),
            Basis.BasisEval( U, iX, 2, iN + 1, false ) );
        Flux_q( iX, iN, iCF ) =
            Flux_Fluid( Basis.BasisEval( U, iX, 1, iN + 1, false ), P, iCF );
      } );

  // --- Volume Term ---
  Kokkos::parallel_for(
      "Volume Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { 3, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
        double local_sum = 0.0;
        double X         = 0.0;
        for ( unsigned int iN = 0; iN < nNodes; iN++ )
        {
          X = Grid.NodeCoordinate( iX, iN );
          local_sum += Grid.Get_Weights( iN ) * Flux_q( iX, iN, iCF ) *
                       Basis.Get_dPhi( iX, iN + 1, k ) * Grid.Get_SqrtGm( X );
        }

        dU( iX, k, iCF ) += local_sum;
      } );
}

/**
 * Compute fluid increment from geometry in spherical symmetry
 **/
void ComputeIncrement_Fluid_Geometry( Kokkos::View<double***> U,
                                      const GridStructure& Grid,
                                      const ModalBasis& Basis,
                                      Kokkos::View<double***> dU )
{
  const unsigned int nNodes = Grid.Get_nNodes( );
  const unsigned int order  = Basis.Get_Order( );
  const unsigned int ilo    = Grid.Get_ilo( );
  const unsigned int ihi    = Grid.Get_ihi( );

  Kokkos::parallel_for(
      "Geometry Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { ilo, 0 }, { ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iX, const int k ) {
        double local_sum = 0.0;
        for ( unsigned int iN = 0; iN < nNodes; iN++ )
        {
          double P = ComputePressureFromConserved_IDEAL(
              Basis.BasisEval( U, iX, 0, iN + 1, false ),
              Basis.BasisEval( U, iX, 1, iN + 1, false ),
              Basis.BasisEval( U, iX, 2, iN + 1, false ) );

          double X = Grid.NodeCoordinate( iX, iN );

          local_sum +=
              Grid.Get_Weights( iN ) * P * Basis.Get_Phi( iX, iN + 1, k ) * X;
        }

        dU( iX, k, 1 ) += ( 2.0 * local_sum * Grid.Get_Widths( iX ) ) /
                          Basis.Get_MassMatrix( iX, k );
      } );
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
void Compute_Increment_Explicit(
    const Kokkos::View<double***> U, const GridStructure& Grid,
    const ModalBasis& Basis, Kokkos::View<double***> dU,
    Kokkos::View<double***> Flux_q, Kokkos::View<double**> dFlux_num,
    Kokkos::View<double**> uCF_F_L, Kokkos::View<double**> uCF_F_R,
    Kokkos::View<double*> Flux_U, Kokkos::View<double*> Flux_P,
    const std::string BC )
{

  const unsigned int& order = Basis.Get_Order( );
  const unsigned int& ilo   = Grid.Get_ilo( );
  const unsigned int& ihi   = Grid.Get_ihi( );

  // --- Apply BC ---
  ApplyBC_Fluid( U, Grid, order, BC );

  // --- Detect Shocks ---
  // TODO: Code up a shock detector...

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                              { 3, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
        dU( iX, k, iCF ) = 0.0;
      } );

  Kokkos::parallel_for(
      ihi + 2, KOKKOS_LAMBDA( unsigned int iX ) { Flux_U( iX ) = 0.0; } );

  // --- Fluid Increment : Divergence ---
  ComputeIncrement_Fluid_Divergence( U, Grid, Basis, dU, Flux_q, dFlux_num,
                                     uCF_F_L, uCF_F_R, Flux_U, Flux_P );

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { 3, ihi + 1, order } ),
      KOKKOS_LAMBDA( const int iCF, const int iX, const int k ) {
        dU( iX, k, iCF ) /= ( Basis.Get_MassMatrix( iX, k ) );
      } );

  // --- Increment from Geometry ---
  if ( Grid.DoGeometry( ) )
  {
    ComputeIncrement_Fluid_Geometry( U, Grid, Basis, dU );
  }

  // --- Increment Gravity ---
}
