/**
 * File    :  Driver.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Primary driver routine
**/  

#include <iostream>
#include <vector>
#include <string>

#include "DataStructures.h"
#include "Grid.h"
#include "Initialization.h"
#include "IOLibrary.h"
#include "Fluid_Discretization.h"
#include "FluidUtilities.h"
#include "Timestepper.h"

int main( int argc, char* argv[] )
{
  // Problem Parameters
  const std::string ProblemName = "Sod";

  const unsigned int nX     = 1000;
  const unsigned int nNodes = 2;
  const unsigned short int nStages = 2;

  const unsigned int nGuard = 1;

  const double xL = 0.0;
  const double xR = 1.0;

  double t           = 0.0;
  double dt          = -1.0;
  const double t_end = 0.075;

  const double CFL = 0.15 / ( 1.0 * ( 2.0 * ( nNodes ) - 1.0 ) );

  // --- Create the Grid object ---
  GridStructure Grid( nNodes, nX, nGuard, xL, xR );

  // --- Create the data structures ---
  DataStructure3D uCF( 3, nX + 2*nGuard, nNodes );
  DataStructure3D uPF( 3, nX + 2*nGuard, nNodes );
  DataStructure3D uAF( 3, nX + 2*nGuard, nNodes );

  // --- Data Structures needed for update step ---

  DataStructure3D dU( 3, nX + 2*nGuard, nNodes );
  DataStructure3D Flux_q( 3, nX + 2*nGuard + 1, nNodes );

  DataStructure2D dFlux_num( 3, nX + 2*nGuard + 1 );
  DataStructure2D uCF_F_L( 3, nX + 2*nGuard );
  DataStructure2D uCF_F_R( 3, nX + 2*nGuard );

  std::vector<double> Flux_U(nX + 2*nGuard + 1, 0.0);
  std::vector<double> Flux_P(nX + 2*nGuard + 1, 0.0);
  std::vector<double> uCF_L(3, 0.0);
  std::vector<double> uCF_R(3, 0.0);

  // We may need more allocations later. Put them here.

  // --- Initialize fields ---
  InitializeFields( uCF, uPF, Grid, ProblemName );
  // WriteState( uCF, uPF, uAF, Grid, ProblemName );

  // --- Initialize timestepper ---
  DataStructure2D a_jk(nStages, nStages);
  DataStructure2D b_jk(nStages, nStages);

  // Inter-stage data structures
  std::vector<DataStructure3D> U_s(nStages+1, DataStructure3D( 3, nX + 2*nGuard, nNodes ));
  std::vector<DataStructure3D> dU_s(nStages, DataStructure3D( 3, nX + 2*nGuard, nNodes ));

  InitializeTimestepper( nStages, nX + 2*nGuard, nNodes, 
                         a_jk, b_jk, U_s, dU_s );
  
  // Slope limiter things
  const double Beta_TVD = 1.0;
  const double Beta_TVB = 0.0;
  // --- Initialize Slope Limiter ---

  // Limit the initial conditions
  // ApplySlopeLimiter( Mesh, uCF, D, SL )

  unsigned int iStep = 0;
  // Evolution loop
  std::cout << "Step\tt\tdt" << std::endl;
  while( t < t_end )
  {

    dt = ComputeTimestep_Fluid( uCF, Grid, CFL ); // Next: ComputeTimestep
    std::cout << iStep << "\t" << t << "\t" << dt << std::endl;

    if ( t + dt > t_end )
    {
      dt = t_end - t;
    }

    // Compute_Increment_Explicit( uCF, Grid, dU, Flux_q, dFlux_num, uCF_F_L, uCF_F_R, Flux_U, Flux_P, uCF_L, uCF_R, "Homogenous" );
    UpdateFluid( Compute_Increment_Explicit, dt, uCF,  Grid, a_jk,  b_jk,
                 U_s,  dU_s, dU,  Flux_q,  dFlux_num, uCF_F_L,  uCF_F_R,  
                 Flux_U, Flux_P,  uCF_L,  uCF_R, nStages,  "Homogenous" );

    // dU.mult(dt);
    // uCF.add(dU);

    t += dt;

    iStep++;
  }

  WriteState( uCF, uPF, uAF, Grid, ProblemName );

}