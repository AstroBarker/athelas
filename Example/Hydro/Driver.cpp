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

  const unsigned int nX      = 256;
  const unsigned int order   = 2;
  const unsigned int nNodes  = NumNodes( order ) + 0;
  const unsigned int nStages = 5;
  const unsigned int tOrder  = 4;

  const unsigned int nGuard  = 1;

  const double xL = 0.0;
  const double xR = 1.0;

  double t           = 0.0;
  double dt          = 0.0;
  const double t_end = 0.2;

  bool Geometry = true; // Cartesian
  std::string BC = "Reflecting";

  const double CFL = ComputeCFL( 0.3, order, nStages, tOrder );

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
  ModalBasis Basis( uCF, Grid, order, nNodes, nX, nGuard );
  std::vector<double> Mass(nX+2, 0.0);
  for ( unsigned int iX = nGuard; iX < nX; iX++ )
  {
    Mass[iX] = Grid.Get_Mass(iX);
  }

  WriteBasis( Basis, nGuard, Grid.Get_ihi(), nNodes, order, ProblemName );

  // --- Initialize timestepper ---
  TimeStepper SSPRK( nStages, tOrder, order, Grid, Geometry, BC );
  
  // --- Initialize Slope Limiter ---
  const double Beta_TVD = 1.0;
  const double Beta_TVB = 50.0;
  const double SlopeLimiter_Threshold = 5e-4;
  const double TCI_Threshold = 0.5; //0.65
  const bool CharacteristicLimiting_Option = true;
  const bool TCI_Option = true;
  
  SlopeLimiter S_Limiter( Grid, nNodes, SlopeLimiter_Threshold, Beta_TVD, Beta_TVB,
    CharacteristicLimiting_Option, TCI_Option, TCI_Threshold );

  // --- Limit the initial conditions ---
  S_Limiter.ApplySlopeLimiter( uCF, Grid, Basis );

  // -- print run parameters ---
  PrintSimulationParameters( Grid, order, tOrder, nStages, CFL, Beta_TVD, 
    Beta_TVB, TCI_Threshold, CharacteristicLimiting_Option, TCI_Option, 
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

  double avg = 0.0;
  for ( unsigned int iCF = 0; iCF < 3; iCF++ )
  for ( unsigned int iX = 1; iX < nX; iX++ )
  {
    avg = CellAverage( uCF, Grid, Basis, iCF, iX );
    // std::printf("%f %f %f %.3e\n",Grid.Get_Centers(iX), avg, uCF(iCF,iX,0), (avg - uCF(iCF,iX,0))/avg  );
    std::printf("%f %f %.3e \n",avg, Mass[iX], (avg-Mass[iX])/avg );
  }

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
 * Return the cell average of a field iCF on cell iX.
**/
double CellAverage( DataStructure3D& U, GridStructure& Grid, ModalBasis& Basis,
  unsigned int iCF, unsigned int iX )
{
  const unsigned int nNodes = Grid.Get_nNodes();

  double avg  = 0.0;
  double mass = 0.0;
  double X   = 0.0;

  for ( unsigned int iN = 0; iN < nNodes; iN++ )
  {
    X = Grid.NodeCoordinate(iX,iN);
    mass += Grid.Get_Weights(iN) * Grid.Get_SqrtGm(X) * Grid.Get_Widths(iX) / Basis.BasisEval( U, iX, 0, iN+1 );/// U(0,iX,0);
    avg += Grid.Get_Weights(iN) * Basis.BasisEval( U, iX, iCF, iN+1 ) 
        * Grid.Get_SqrtGm(X) * Grid.Get_Widths(iX) / Basis.BasisEval( U, iX, 0, iN+1 ) ;/// U(0,iX,0);
  }
  
  return mass;
  // return avg / mass;
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