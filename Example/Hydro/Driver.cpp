// Driver routine

#include <iostream>
#include <string>

#include "DataStructures.h"
#include "Grid.h"
#include "Initialization.h"
#include "IOLibrary.h"

int main( int argc, char* argv[] )
{
  // Problem Parameters
  const std::string ProblemName = "Sod";

  const unsigned int nX     = 100;
  const unsigned int nNodes = 2;

  const unsigned int nGuard = 1;

  const double xL = 0.0;
  const double xR = 1.0;

  // Create the Grid object
  GridStructure Grid( nNodes, nX, nGuard, xL, xR );

  // Create the data structures
  DataStructure3D uCF( nNodes, nX + 2*nGuard, 3 );
  DataStructure3D uPF( nNodes, nX + 2*nGuard, 3 );
  DataStructure3D uAF( nNodes, nX + 2*nGuard, 3 );

  // We may need more allocations later. Put them here.

  // Initialize fields
  InitializeFields( uCF, uPF, Grid, ProblemName );
  // WriteState( uCF, uPF, uAF, Grid, ProblemName );

  // Slope limiter things
  double Beta_TVD = 1.0;
  double Beta_TVB = 0.0;
  // Initialize Slope Limiter

  // Limit the initial conditions
  // ApplySlopeLimiter( Mesh, uCF, D, SL )

  unsigned int iStep = 0;
  // Evolution loop

}