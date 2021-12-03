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
  const unsigned short int nStages = 3;

  const unsigned int nGuard = 1;

  const double xL = 0.0;
  const double xR = 1.0;

  double t           = 0.0;
  double dt          = 0.0;
  const double t_end = 0.2;

  const double CFL = 0.3 / ( 1.0 * ( 2.0 * ( order+1 ) - 1.0 ) );

  // --- Create the Grid object ---
  GridStructure Grid( nNodes, nX, nStages, nGuard, xL, xR );

  // --- Create the data structures ---
  DataStructure3D uCF( 3, nX + 2*nGuard, order );
  DataStructure3D uPF( 3, nX + 2*nGuard, order );
  DataStructure3D uAF( 3, nX + 2*nGuard, order );
  // DataStructure3D D( 2, nX + 2*nGuard, order );

  // --- Data Structures needed for update step ---

  DataStructure3D dU( 3, nX + 2*nGuard, order );
  DataStructure3D Flux_q( 3, nX + 2*nGuard + 1, nNodes );

  DataStructure2D dFlux_num( 3, nX + 2*nGuard + 1 );
  DataStructure2D uCF_F_L( 3, nX + 2*nGuard );
  DataStructure2D uCF_F_R( 3, nX + 2*nGuard );

  std::vector<std::vector<double>> Flux_U(nStages + 1, 
    std::vector<double>(nX + 2*nGuard + 1,0.0));
  std::vector<double> Flux_P(nX + 2*nGuard + 1, 0.0);
  std::vector<double> uCF_L(3, 0.0);
  std::vector<double> uCF_R(3, 0.0);

  // We may need more allocations later. Put them here.

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
  DataStructure2D a_jk(nStages, nStages);
  DataStructure2D b_jk(nStages, nStages);

  // Inter-stage data structures
  DataStructure3D SumVar( 3, nX + 2*nGuard + 1, order ); // Used in Integrator...
  std::vector<DataStructure3D> U_s(nStages+1, DataStructure3D( 3, nX + 2*nGuard, order ));
  std::vector<GridStructure> Grid_s(nStages+1, GridStructure( nNodes, nX, nStages, nGuard, xL, xR ));
  std::vector<DataStructure3D> dU_s(nStages, DataStructure3D( 3, nX + 2*nGuard, order ));

  InitializeTimestepper( nStages, a_jk, b_jk );
  
  // Slope limiter things
  const double Beta_TVD = 1.0;
  const double Beta_TVB = 0.0;
  const double SlopeLimiter_Threshold = 1e-6;
  const double TCI_Threshold = 0.05;
  const bool CharacteristicLimiting_Option = true;
  const bool TCI_Option = false;
  // --- Initialize Slope Limiter ---
  
  SlopeLimiter S_Limiter( Grid, nNodes, SlopeLimiter_Threshold, Beta_TVD, Beta_TVB,
    CharacteristicLimiting_Option, TCI_Option, TCI_Threshold );

  // Limit the initial conditions
  S_Limiter.ApplySlopeLimiter( uCF, Grid );

  // --- Evolution loop ---
  unsigned int iStep = 0;
  std::cout << "Step\tt\tdt" << std::endl;
  while( t < t_end && iStep < 1245 )
  {

    dt = ComputeTimestep_Fluid( uCF, Grid, CFL ); // Next: ComputeTimestep
    // dt = 0.000005;

    if ( t + dt > t_end )
    {
      dt = t_end - t;
    }

    std::printf( "%d \t %.5e \t %.5e\n", iStep, t, dt );

    UpdateFluid( Compute_Increment_Explicit, dt, uCF,  Grid, Basis, a_jk,  b_jk,
                 U_s,  dU_s, Grid_s, dU,  SumVar, Flux_q,  dFlux_num, uCF_F_L,  uCF_F_R,  
                 Flux_U, Flux_P,  uCF_L,  uCF_R, nStages, S_Limiter, "Homogenous" );

    t += dt;

    iStep++;
  }

  WriteState( uCF, uPF, uAF, Grid, ProblemName );

  std::printf("testing cell avg\n");
  for ( unsigned int iCF = 0; iCF < 3; iCF++)
  for ( unsigned int iX = 1; iX < nX-1; iX++ )
  {
    std::printf("%d %d %f\n ",iCF, iX, uCF(iCF,iX,0) - CellAverage( uCF, Grid, Basis, iCF, iX ) );
  }
  std::printf("\n");

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