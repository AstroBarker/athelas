/**
 * File     :  Fluid_Discretization.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : The main fluid spatial update routines go here.
 *  Compute divergence term.
 **/

#include <iostream>
#include <algorithm>
#include <math.h> /* atan */

// #include "omp.h"
#include "Kokkos_Core.hpp"

#include "Error.h"
#include "Grid.h"
#include "PolynomialBasis.h"
#include "Fluid_Discretization.h"
#include "BoundaryConditionsLibrary.h"
#include "EquationOfStateLibrary_IDEAL.h"
#include "FluidUtilities.h"
#include "Constants.h"

// Compute the divergence of the flux term for the update
void ComputeIncrement_Fluid_Divergence(
    Kokkos::View<double***> U, GridStructure& Grid, ModalBasis& Basis,
    Kokkos::View<double***> dU, Kokkos::View<double***> Flux_q, Kokkos::View<double**> dFlux_num,
    Kokkos::View<double**> uCF_F_L, Kokkos::View<double**> uCF_F_R,
    Kokkos::View<double*> Flux_U, Kokkos::View<double*> Flux_P )
{
  const unsigned int nNodes = Grid.Get_nNodes( );
  const unsigned int order  = Basis.Get_Order( );
  const unsigned int ilo    = Grid.Get_ilo( );
  const unsigned int ihi    = Grid.Get_ihi( );

  double rho_L, rho_R, P_L, P_R, Cs_L, Cs_R, lam_L, lam_R, P;

  double Poly_L, Poly_R;
  double X_L, X_R;
  double SqrtGm_L, SqrtGm_R;

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    for ( unsigned int iX = ilo; iX <= ihi + 1; iX++ )
    {
      uCF_F_L( iCF, iX ) = Basis.BasisEval( U, iX - 1, iCF, nNodes + 1, false );
      uCF_F_R( iCF, iX ) = Basis.BasisEval( U, iX, iCF, 0, false );
    }

  // --- Calc numerical flux at all faces
  // Kokkos::parallel_for( Kokkos::RangePolicy<>(ilo, ihi+1), KOKKOS_LAMBDA ( unsigned int iX ) {
  for ( unsigned int iX = ilo; iX <= ihi + 1; iX++ )
  {

    auto uCF_L = Kokkos::subview( uCF_F_L, Kokkos::ALL, iX );
    auto uCF_R = Kokkos::subview( uCF_F_R, Kokkos::ALL, iX );

    double rho_L = 1.0 / uCF_L(0);
    double rho_R = 1.0 / uCF_R(0);

    double P_L  = ComputePressureFromConserved_IDEAL( uCF_L(0), uCF_L(1), uCF_L(2) );
    double Cs_L = ComputeSoundSpeedFromConserved_IDEAL( uCF_L(0), uCF_L(1), uCF_L(2) );
    double lam_L = Cs_L * rho_L;

    double P_R  = ComputePressureFromConserved_IDEAL( uCF_R(0), uCF_R(1), uCF_R(2) );
    double Cs_R = ComputeSoundSpeedFromConserved_IDEAL( uCF_R(0), uCF_R(1), uCF_R(2) );
    double lam_R = Cs_R * rho_R;

    // --- Numerical Fluxes ---

    // Riemann Problem
    NumericalFlux_Gudonov( uCF_L(1), uCF_R(1), P_L, P_R, lam_L, lam_R,
                           Flux_U(iX), Flux_P(iX) );

    // TODO: Clean This Up
    dFlux_num( 0, iX ) = -Flux_U(iX);
    dFlux_num( 1, iX ) = +Flux_P(iX);
    dFlux_num( 2, iX ) = +Flux_U(iX) * Flux_P(iX);
  }
  // });

  // --- Surface Term ---
  Kokkos::parallel_for( "Surface Term", 3, KOKKOS_LAMBDA ( unsigned int iCF ) {
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < order; k++ )
      {

        double Poly_L   = Basis.Get_Phi( iX, 0, k );
        double Poly_R   = Basis.Get_Phi( iX, nNodes + 1, k );
        double X_L      = Grid.Get_LeftInterface( iX );
        double X_R      = Grid.Get_LeftInterface( iX + 1 );
        double SqrtGm_L = Grid.Get_SqrtGm( X_L );
        double SqrtGm_R = Grid.Get_SqrtGm( X_R );

        dU( iCF, iX, k ) -= ( +dFlux_num( iCF, iX + 1 ) * Poly_R * SqrtGm_R -
                              dFlux_num( iCF, iX + 0 ) * Poly_L * SqrtGm_L );
      }
  });

  // --- Compute Flux_q everywhere for the Volume term ---
  Kokkos::parallel_for( 3, KOKKOS_LAMBDA ( unsigned int iCF ) {
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        double P = ComputePressureFromConserved_IDEAL(
            Basis.BasisEval( U, iX, 0, iN + 1, false ),
            Basis.BasisEval( U, iX, 1, iN + 1, false ),
            Basis.BasisEval( U, iX, 2, iN + 1, false ) );
        Flux_q( iCF, iX, iN ) =
            Flux_Fluid( Basis.BasisEval( U, iX, 1, iN + 1, false ), P, iCF );
      }
  });
      

  // --- Volume Term ---

  Kokkos::parallel_for( 3, KOKKOS_LAMBDA ( unsigned int iCF ) {
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < order; k++ )
      {
        double local_sum = 0.0;
        double X         = 0.0;
        // local_sum = 0.0;
        for ( unsigned int iN = 0; iN < nNodes; iN++ )
        {
          X = Grid.NodeCoordinate( iX, iN );
          local_sum += Grid.Get_Weights( iN ) * Flux_q( iCF, iX, iN ) *
                       Basis.Get_dPhi( iX, iN + 1, k ) * Grid.Get_SqrtGm( X );
        }

        dU( iCF, iX, k ) += local_sum;
      }
  });
}

/**
 * Compute fluid increment from geometry in spherical symmetry
 **/
void ComputeIncrement_Fluid_Geometry( Kokkos::View<double***> U, GridStructure& Grid,
                                      ModalBasis& Basis, Kokkos::View<double***> dU )
{
  const unsigned int nNodes = Grid.Get_nNodes( );
  const unsigned int order  = Basis.Get_Order( );
  const unsigned int ilo    = Grid.Get_ilo( );
  const unsigned int ihi    = Grid.Get_ihi( );

  double local_sum = 0.0;
  double P         = 0.0;
  double X         = 0.0;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int k = 0; k < order; k++ )
    {
      local_sum = 0.0;
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        P = ComputePressureFromConserved_IDEAL(
            Basis.BasisEval( U, iX, 0, iN + 1, false ),
            Basis.BasisEval( U, iX, 1, iN + 1, false ),
            Basis.BasisEval( U, iX, 2, iN + 1, false ) );

        X = Grid.NodeCoordinate( iX, iN );

        local_sum +=
            Grid.Get_Weights( iN ) * P * Basis.Get_Phi( iX, iN + 1, k ) * X;
      }

      dU( 1, iX, k ) += ( 2.0 * local_sum * Grid.Get_Widths( iX ) ) /
                        Basis.Get_MassMatrix( iX, k );
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
 * uCF_F_L, uCF_F_R : left/right face states
 * Flux_U, Flux_P   : Fluxes (from Riemann problem)
 * uCF_L, uCF_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void Compute_Increment_Explicit(
    Kokkos::View<double***> U, GridStructure& Grid, ModalBasis& Basis,
    Kokkos::View<double***> dU, Kokkos::View<double***> Flux_q, Kokkos::View<double**> dFlux_num,
    Kokkos::View<double**> uCF_F_L, Kokkos::View<double**> uCF_F_R,
    Kokkos::View<double*> Flux_U, Kokkos::View<double*> Flux_P, 
    const std::string BC )
{

  const unsigned int order = Basis.Get_Order( );
  const unsigned int ilo   = Grid.Get_ilo( );
  const unsigned int ihi   = Grid.Get_ihi( );

  // --- Apply BC ---
  ApplyBC_Fluid( U, Grid, order, BC );

  // --- Detect Shocks ---
  // TODO: Code up a shock detector...

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for( 3, KOKKOS_LAMBDA ( unsigned int iCF ) {
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    {
      for ( unsigned int k = 0; k < order; k++ )
      {
        dU( iCF, iX, k ) = 0.0;
      }
    }
  });

  // #pragma omp parallel for
  Kokkos::parallel_for( ihi + 1, KOKKOS_LAMBDA ( unsigned int iX ) {
    Flux_U(iX) = 0.0;
  });

  // --- Fluid Increment : Divergence ---
  ComputeIncrement_Fluid_Divergence( U, Grid, Basis, dU, Flux_q, dFlux_num,
                                     uCF_F_L, uCF_F_R, Flux_U, Flux_P );

  Kokkos::parallel_for( 3, KOKKOS_LAMBDA ( unsigned int iCF ) {
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < order; k++ )
      {
        dU( iCF, iX, k ) /= ( Basis.Get_MassMatrix( iX, k ) );
      }
  });

  // --- Increment from Geometry ---
  if ( Grid.DoGeometry( ) )
  {
    ComputeIncrement_Fluid_Geometry( U, Grid, Basis, dU );
  }

  // --- Increment Gravity ---
}
