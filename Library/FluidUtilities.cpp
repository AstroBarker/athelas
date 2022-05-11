/**
 * File     :  FluidUtilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for fluid fields. Includes Riemann solvers.
 **/

#include <iostream>
#include <vector>
#include <cstdlib>   /* abs */
#include <algorithm> // std::min, std::max

#include "Error.h"
#include "Grid.h"
#include "PolynomialBasis.h"
#include "EquationOfStateLibrary_IDEAL.h"
#include "FluidUtilities.h"

/**
 * Compute the primitive quantities (density, momemtum, energy density)
 * from conserved quantities. Primitive quantities are stored at Gauss-Legendre
 * nodes.
 **/
void ComputePrimitiveFromConserved( Kokkos::View<double***> uCF,
                                    Kokkos::View<double***> uPF,
                                    ModalBasis& Basis, GridStructure& Grid )
{
  const unsigned int nNodes = Grid.Get_nNodes( );
  const unsigned int ilo    = Grid.Get_ilo( );
  const unsigned int ihi    = Grid.Get_ihi( );

  double Tau = 0.0;
  double Vel = 0.0;
  double EmT = 0.0;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      // Density
      Tau              = Basis.BasisEval( uCF, 0, iX, iN + 1, false );
      uPF( 0, iX, iN ) = 1.0 / Tau;

      // Momentum
      Vel              = Basis.BasisEval( uCF, 1, iX, iN + 1, false );
      uPF( 1, iX, iN ) = uPF( 0, iX, iN ) * Vel;

      // Specific Total Energy
      EmT              = Basis.BasisEval( uCF, 2, iX, iN + 1, false );
      uPF( 2, iX, iN ) = EmT / Tau;
    }
}

// Fluid vector.
// ! Flag For Removal: Unused !
double Fluid( double Tau, double V, double Em_T, int iCF )
{
  if ( iCF == 0 )
  {
    return Tau;
  }
  else if ( iCF == 1 )
  {
    return V;
  }
  else if ( iCF == 2 )
  {
    return Em_T;
  }
  else
  { // Error case. Shouldn't ever trigger.
    throw Error( "Please input a valid iCF! (0,1,2). " );
    return -1; // just a formality.
  }
}

/**
 * Return a component iCF of the flux vector.
 * TODO: Flux_Fluid needs streamlining
 **/
double Flux_Fluid( double V, double P, unsigned int iCF )
{
  if ( iCF == 0 )
  {
    return -V;
  }
  else if ( iCF == 1 )
  {
    return +P;
  }
  else if ( iCF == 2 )
  {
    return +P * V;
  }
  else
  { // Error case. Shouldn't ever trigger.
    throw Error( "Please input a valid iCF! (0,1,2). " );
    return -1.0; // just a formality.
  }
}

/**
 * Gudonov style numerical flux. Constucts v* and p* states.
 **/
void NumericalFlux_Gudonov( double vL, double vR, double pL, double pR,
                            double zL, double zR, double& Flux_U,
                            double& Flux_P )
{
  Flux_U = ( pL - pR + zR * vR + zL * vL ) / ( zR + zL );
  Flux_P = ( zR * pL + zL * pR + zL * zR * ( vL - vR ) ) / ( zR + zL );
}

/**
 * Gudonov style numerical flux. Constucts v* and p* states.
 **/
void NumericalFlux_HLLC( double vL, double vR, double pL, double pR, double cL,
                         double cR, double rhoL, double rhoR, double& Flux_U,
                         double& Flux_P )
{
  double aL = vL - cL; // left wave speed estimate
  double aR = vR + cR; // right wave speed estimate
  Flux_U    = ( rhoR * vR * ( aR - vR ) - rhoL * vL * ( aL - vL ) + pL - pR ) /
           ( rhoR * ( aR - vR ) - rhoL * ( aL - vL ) );
  Flux_P = rhoL * ( vL - aL ) * ( vL - Flux_U ) + pL;
}

// Compute Auxilliary

/**
 * Compute the fluid timestep.
 **/
double ComputeTimestep_Fluid( Kokkos::View<double***> U, GridStructure& Grid,
                              const double CFL )
{

  const double MIN_DT = 0.000000005;
  const double MAX_DT = 1.0;
  double dt_old       = 10000.0;

  const unsigned int ilo = Grid.Get_ilo( );
  const unsigned int ihi = Grid.Get_ihi( );

  double dt = 0.0;

  Kokkos::parallel_reduce(
      "Timestep", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( const int& iX, double& lmin ) {
        // --- Compute Cell Averages ---
        double tau_x  = U( 0, iX, 0 );
        double vel_x  = U( 1, iX, 0 );
        double eint_x = U( 2, iX, 0 );

        double dr = Grid.Get_Widths( iX );

        double Cs =
            ComputeSoundSpeedFromConserved_IDEAL( tau_x, vel_x, eint_x );
        double eigval = Cs;

        double dt_old = std::abs( dr ) / std::abs( eigval );

        if ( dt_old < lmin ) lmin = dt_old;
      },
      Kokkos::Min<double>( dt ) );

  dt = std::max( CFL * dt, MIN_DT );
  dt = std::min( dt, MAX_DT );

  // Triggers on NaN
  if ( dt != dt )
  {
    throw Error( "nan encountered in ComputeTimestep.\n" );
  }

  return dt;
}
