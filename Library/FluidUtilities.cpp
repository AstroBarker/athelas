/**
 * Utility routines for fluid fields.
 * Riemann Solvers are here.
**/

#include <iostream>
#include <vector>
#include <cstdlib>     /* abs */
#include <algorithm>    // std::min, std::max

#include "DataStructures.h"
#include "Grid.h"
#include "EquationOfStateLibrary_IDEAL.h"

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
    std::cout << "Please input a valid iCF! (0,1,2). " << std::endl;
    std::cerr << "Please input a valid iCF! (0,1,2). " << std::endl;
    exit(-1);
    return -1;
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

// HLL

// ComputePrimitive

//Compute Auxilliary

double ComputeTimestep_Fluid( DataStructure3D& U, 
       GridStructure& Grid, const double CFL )
{

  const double MIN_DT = 0.000000001;
  double dt_old = 10000.0;

  const int ilo    = Grid.Get_ilo();
  const int ihi    = Grid.Get_ihi();
  const int nNodes = Grid.Get_nNodes();

  // Store Weights - for cell averages
  std::vector<double> Weights(nNodes);
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    Weights[iN] = Grid.Get_Weights( iN );
  }

  double Cs = 0.0;
  double eigval = 0.0;

  // hold cell averages
  double tau_x  = 0.0;
  double vel_x  = 0.0;
  double eint_x = 0.0;

  double dt = 0.0;

  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {

    tau_x  = U.CellAverage( 0, iX, nNodes, Weights );
    vel_x  = U.CellAverage( 1, iX, nNodes, Weights );
    eint_x = U.CellAverage( 2, iX, nNodes, Weights );

    Cs     = ComputeSoundSpeedFromConserved_IDEAL( tau_x, vel_x, eint_x );
    eigval = Cs;

    // put (eigval - 0*U[iNode,iX,1]) in denom
    dt = CFL * std::abs( Grid.Get_Centers(iX+1) - Grid.Get_Centers(iX) ) / ( eigval );

    dt     = std::min( dt, dt_old );
    dt_old = dt;
  }

  dt = std::max( dt, MIN_DT );

  return dt;

}