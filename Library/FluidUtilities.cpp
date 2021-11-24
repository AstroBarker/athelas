/**
 * File     :  FluidUtilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for fluid fields. Includes Riemann solvers.
**/ 

#include <iostream>
#include <vector>
#include <cstdlib>     /* abs */
#include <algorithm>    // std::min, std::max

#include "Error.h"
#include "DataStructures.h"
#include "Grid.h"
#include "EquationOfStateLibrary_IDEAL.h"


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
  else{ // Error case. Shouldn't ever trigger.
    throw Error("Please input a valid iCF! (0,1,2). ");
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
    return - V;
  }
  else if ( iCF == 1 )
  {
    return + P;
  }
  else if ( iCF == 2 )
  {
    return P * V;
  }
  else{ // Error case. Shouldn't ever trigger.
    throw Error("Please input a valid iCF! (0,1,2). ");
    return -1; // just a formality.
  }
}


/**
 * Gudonov style numerical flux. Constucts v* and p* states.
**/
void NumericalFlux_Gudonov( double vL, double vR, double pL, double pR, 
     double zL, double zR, double& Flux_U, double& Flux_P  )
{
  Flux_U = ( pL - pR + zR*vR + zL*vL ) / ( zR + zL );
  Flux_P = ( zR*pL + zL*pR + zL*zR * (vL - vR) ) / ( zR + zL );
}

void NumericalFlux_HLL( double tauL, double tauR, double vL, double vR, 
  double eL, double eR, double pL, double pR, double zL, double zR, 
  int iCF, double& out )
{

  double uL = Fluid( tauL, vL, eL, iCF );
  double uR = Fluid( tauR, vR, eR, iCF );

    // zL += vL / tauL;
    // zR += vR / tauR;

    double am = std::max( std::max( 0.0, - zL ), - zR );
    double ap = std::max( std::max( 0.0, + zL ), + zR );

    // f = (zR * Flux_Fluid( vL, pL) + zL * Flux_Fluid( vR, pR) - zL*zR \
    //   * ( uR - uL ) ) / (zL + zR)

    out = (ap * Flux_Fluid( vL, pL, iCF ) + am * Flux_Fluid( vR, pR, iCF ) - am*ap * (uR-uL) ) / ( am + ap );
}
// ComputePrimitive

//Compute Auxilliary

double ComputeTimestep_Fluid( DataStructure3D& U, 
       GridStructure& Grid, const double CFL )
{

  const double MIN_DT = 0.000000005;
  const double MAX_DT = 1.0;
  double dt_old = 10000.0;

  const unsigned int ilo    = Grid.Get_ilo();
  const unsigned int ihi    = Grid.Get_ihi();
  const unsigned int nNodes = Grid.Get_nNodes();

  // Store Weights - for cell averages
  std::vector<double> Weights(nNodes);
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    Weights[iN] = Grid.Get_Weights( iN );
  }

  double Cs     = 0.0;
  double eigval = 0.0;

  // hold cell averages
  double tau_x  = 0.0;
  double vel_x  = 0.0;
  double eint_x = 0.0;

  // Hold cell centers (not updated with grid)
  double r_np1 = 0.0; // r_{n+1}
  double r_n   = 0.0; // r_{n}

  double dt1 = 0.0;
  double dt2 = 0.0;
  double dt  = 0.0;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  {

    // --- Compute Cell Averages ---
    tau_x  = U( 0, iX, 0 );
    vel_x  = U( 1, iX, 0 );
    eint_x = U( 2, iX, 0 );

    r_n   = Grid.Get_Centers( iX );
    r_np1 = Grid.Get_Centers( iX + 1 );

    Cs     = ComputeSoundSpeedFromConserved_IDEAL( tau_x, vel_x, eint_x );
    eigval = Cs;

    dt1 = std::abs( r_np1 - r_n ) / std::abs( eigval - vel_x );
    dt2 = std::abs( r_np1 - r_n ) / std::abs( eigval + vel_x );

    dt = std::min( dt1, dt2 );

    dt     = std::min( dt, dt_old );
    dt_old = dt;
  }

  dt = std::max( CFL * dt, MIN_DT );
  dt = std::min( dt, MAX_DT );

  return dt;

}