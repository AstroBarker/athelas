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
}

void InitializeFields( DataStructure3D& uCF, DataStructure3D& uPF, GridStructure& Grid, const std::string ProblemName )
{
  std::cout << " === Initializing: " << ProblemName << std::endl;

  const unsigned int ilo    = Grid.Get_ilo();
  const unsigned int ihi    = Grid.Get_ihi();
  const unsigned int nNodes = Grid.Get_nNodes();

  const unsigned int iCF_Tau = 0;
  const unsigned int iCF_V   = 1;
  const unsigned int iCF_E   = 2;

  const unsigned int iPF_D = 0;

  if ( ProblemName == "Sod" )
  {
    const double V0  = 0.0;
    const double D_L = 1.0;
    const double D_R = 0.125;
    const double P_L = 1.0;
    const double P_R = 0.1;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++  )
    {
      X1 = Grid.NodeCoordinate( iX, iNodeX );

      if ( X1 <= 0.5 )
      {
        uCF(iNodeX, iX, iCF_Tau) = 1.0 / D_L;
        uCF(iNodeX, iX, iCF_V)   = V0;
        uCF(iNodeX, iX, iCF_E)   = (P_L / 0.4) * uCF(iNodeX, iX, iCF_Tau);
        uPF(iNodeX, iX, iPF_D)   = D_L;
      }else{
        uCF(iNodeX, iX, iCF_Tau) = 1.0 / D_R;
        uCF(iNodeX, iX, iCF_V)   = V0;
        uCF(iNodeX, iX, iCF_E)   = (P_R / 0.4) * uCF(iNodeX, iX, iCF_Tau);
        uPF(iNodeX, iX, iPF_D)   = D_R;
      }
    }
  }
  else{
    std::cerr << "Please choose a valid ProblemName";
    std::cout << "Please choose a valid ProblemName";
  }
}
