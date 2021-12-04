/**
 * File    :  Driver.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Main driver routine
**/  

#include <iostream>
#include <vector>
#include <string>

#include "DataStructures.h"
#include "Grid.h"
#include "BoundaryConditionsLibrary.h"
#include "SlopeLimiter.h"
#include "Initialization.h"
#include "IOLibrary.h"
#include "Fluid_Discretization.h"
#include "FluidUtilities.h"
#include "Timestepper.h"
#include "Error.h"
#include "Driver.h"


int main( int argc, char* argv[] )
{
  // --- Problem Parameters ---
  const std::string ProblemName = "Sod";

  const unsigned int nX            = 512;
  const unsigned int order         = 3;
  const unsigned int nNodes        = NumNodes( order );
  const unsigned int nStages = 3;

  const unsigned int nGuard = 1;

  const double xL = 0.0;
  const double xR = 1.0;

  double t           = 0.0;
  double dt          = 0.0;
  const double t_end = 0.2;

  const double CFL = 0.3 / ( 1.0 * ( 2.0 * ( order ) - 1.0 ) );

  // --- Create the Grid object ---
  GridStructure Grid( nNodes, nX, nGuard, xL, xR );

  // --- Create the data structures ---
  DataStructure3D uCF( 3, nX + 2*nGuard, order );
  DataStructure3D uPF( 3, nX + 2*nGuard, order );
  DataStructure3D uAF( 3, nX + 2*nGuard, order );

  // --- Initialize fields ---
  InitializeFields( uCF, uPF, Grid, ProblemName );
  // WriteState( uCF, uPF, uAF, Grid, ProblemName );

  ApplyBC_Fluid( uCF, Grid, order, "Homogenous" );

  // --- Compute grid quantities ---
  // TODO: Bundle this in an InitializeGrid?
  Grid.ComputeVolume( );
  Grid.ComputeMass( uPF );
  Grid.ComputeCenterOfMass( uPF );

  // --- Datastructure for modal basis ---
  ModalBasis Basis( uPF, Grid, order, nNodes, nX, nGuard );

  // --- Initialize timestepper ---
  TimeStepper SSPRK( nStages, nStages, order, Grid, "Homogenous" );
  
  // --- Initialize Slope Limiter ---
  const double Beta_TVD = 1.0;
  const double Beta_TVB = 0.0;
  const double SlopeLimiter_Threshold = 5e-6;
  const double TCI_Threshold = 0.1;
  const bool CharacteristicLimiting_Option = true;
  const bool TCI_Option = true;
  
  SlopeLimiter S_Limiter( Grid, nNodes, SlopeLimiter_Threshold, Beta_TVD, Beta_TVB,
    CharacteristicLimiting_Option, TCI_Option, TCI_Threshold );

  // --- Limit the initial conditions ---
  S_Limiter.ApplySlopeLimiter( uCF, Grid, Basis );

  // --- Evolution loop ---
  unsigned int iStep = 0;
  std::cout << "Step\tt\tdt" << std::endl;
  while( t < t_end && iStep >= 0 )
  {

    dt = ComputeTimestep_Fluid( uCF, Grid, CFL );
    // dt = 0.000005;

    if ( t + dt > t_end )
    {
      dt = t_end - t;
    }

    std::printf( "%d \t %.5e \t %.5e\n", iStep, t, dt );

    SSPRK.UpdateFluid( Compute_Increment_Explicit, dt, 
      uCF, Grid, Basis, S_Limiter );

    t += dt;

    iStep++;
  }

  WriteState( uCF, uPF, uAF, Grid, ProblemName );

}


/**
 * Pick number of quadrature points in order to evaluate polynomial of 
 * at least order^2. 
 * ! Broken for nNodes > order !
**/
int NumNodes( unsigned int order )
{
  if ( order <= 4 )
  {
    return order;
  }
  else
  {
    return 2 * order;
  }
}

/**
 * Return the cell average of a field iCF on cell iX.
**/
double CellAverage( DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
  unsigned int iCF, unsigned int iX )
{
  const unsigned int nNodes = Grid.Get_nNodes();

  double avg = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    avg += Grid.Get_Weights(iN) * Basis.BasisEval( U, iX, iCF, iN+1 );
  }

  return avg;
}