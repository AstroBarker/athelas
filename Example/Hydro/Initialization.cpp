/**
 * Setup for initial conditions.
 **/

#include <iostream>
#include <string>

#include "DataStructures.cpp"
#include "Grid.cpp"
#include "Initialization.h"

int main( int argc, char* argv[] )
{
  // Make datta structures (uCF, uPF, Grid)
  // WRite InitializeFields to initialize them

  // Problem Parameters
  const std::string ProblemName = "Sod";

  const unsigned int nX     = 100;
  const unsigned int nNodes = 3;

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
  InitializeFields( uCF, ProblemName );
}

void InitializeFields( DataStructure3D& uCF, const std::string ProblemName )
{
  std::cout << ProblemName << std::endl;

  if ( ProblemName == "Sod" )
  {
    std::cout << "sod :) " << std::endl;
  }
  else{
    std::cerr << "Please choose a valid ProblemName";
    std::cout << "Please choose a valid ProblemName";
  }
}
