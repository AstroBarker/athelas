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
  /* --- Problem Parameters --- */
  const std::string ProblemName = "Sod";

  const unsigned int nX      = 512;
  const unsigned int order   = 3;
  const unsigned int nNodes  = NumNodes( order ) + 0;
  const unsigned int nStages = 5;
  const unsigned int tOrder  = 3;

  const unsigned int nGuard  = 1;

  const double xL = 0.0;
  const double xR = 1.0;

  double t           = 0.0;
  double dt          = 0.0;
  const double t_end = 0.2;

  bool Geometry  = true; /* false: Cartesian, true: Spherical */
  std::string BC = "Reflecting";

  const double CFL = ComputeCFL( 0.5, order, nStages, tOrder );

  // --- Create the Grid object ---
  GridStructure Grid( nNodes, nX, nGuard, xL, xR, Geometry );

  // --- Create the data structures ---
  DataStructure3D uCF( 3, nX + 2*nGuard, order );
  DataStructure3D uPF( 3, nX + 2*nGuard, nNodes );

  DataStructure3D uAF( 3, nX + 2*nGuard, order );

  // --- Initialize fields ---
  InitializeFields( uCF, uPF, Grid, order, ProblemName );
  // WriteState( uCF, uPF, uAF, Grid, ProblemName );

  ApplyBC_Fluid( uCF, Grid, order, BC );

  // --- Datastructure for modal basis ---
  ModalBasis Basis( uPF, Grid, order, nNodes, nX, nGuard );

  WriteBasis( Basis, nGuard, Grid.Get_ihi(), nNodes, order, ProblemName );

  // --- Initialize timestepper ---
  TimeStepper SSPRK( nStages, tOrder, order, Grid, Geometry, BC );
  
  // --- Initialize Slope Limiter ---
  const double alpha    = 0.85;
  const double SlopeLimiter_Threshold = 5e-6;
  const double TCI_Threshold = 0.65;
  const bool CharacteristicLimiting_Option = true;
  const bool TCI_Option = true;
  
  SlopeLimiter S_Limiter( Grid, nNodes, SlopeLimiter_Threshold,
    alpha, CharacteristicLimiting_Option, TCI_Option, TCI_Threshold );

  // --- Limit the initial conditions ---
  S_Limiter.ApplySlopeLimiter( uCF, Grid, Basis );

  // -- print run parameters ---
  PrintSimulationParameters( Grid, order, tOrder, nStages, CFL, 
    alpha, TCI_Threshold, CharacteristicLimiting_Option, TCI_Option, 
    ProblemName );

  // --- Evolution loop ---
  unsigned int iStep   = 0;
  unsigned int i_write = 10;
  std::cout << "Step\tt\tdt" << std::endl;
  while( t < t_end && iStep >= 0 )
  {

    dt = ComputeTimestep_Fluid( uCF, Grid, CFL );

    if ( t + dt > t_end )
    {
      dt = t_end - t;
    }

    if ( iStep % i_write == 0 )
    {
      std::printf( "%d \t %.5e \t %.5e\n", iStep, t, dt );
    }

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
    return order + 1;
  }
}


/**
 * Compute the CFL timestep restriction.
**/
double ComputeCFL( double CFL, unsigned int order, 
  unsigned int nStages, unsigned int tOrder )
{
  double c = 1.0;

  if ( nStages == tOrder ) c = 1.0;
  if ( nStages != tOrder )
  {
    if ( tOrder == 2 ) c = 4.0;
    if ( tOrder == 3 ) c = 2.65062919294483;
    if ( tOrder == 4 ) c = 1.50818004975927;
  }

  return c * CFL / ( ( 2.0 * ( order ) - 1.0 ) );
}
