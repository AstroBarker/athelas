/**
 * File    :  Initialization.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Initialize conserved fields for given problem
**/  

#include <iostream>
#include <string>
#include <math.h>       /* sin */

#include "Error.h"
#include "DataStructures.h"
#include "Grid.h"
#include "Constants.h"

/**
 * Initialize the conserved Fields for various problems
 **/
void InitializeFields( DataStructure3D& uCF, DataStructure3D& uPF, GridStructure& Grid, const std::string ProblemName )
{
  std::cout << " --- Initializing: " << ProblemName << " ---" << std::endl;

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
        uCF(iCF_Tau, iX, iNodeX) = 1.0 / D_L;
        uCF(iCF_V, iX, iNodeX)   = V0;
        uCF(iCF_E, iX, iNodeX)   = (P_L / 0.4) * uCF(iCF_Tau, iX, iNodeX);
        uPF(iPF_D, iX, iNodeX)   = D_L;
      }else{
        uCF(iCF_Tau, iX, iNodeX) = 1.0 / D_R;
        uCF(iCF_V, iX, iNodeX)   = V0;
        uCF(iCF_E, iX, iNodeX)   = (P_R / 0.4) * uCF(iCF_Tau, iX, iNodeX);
        uPF(iPF_D, iX, iNodeX)   = D_R;
      }
    }
  }
  else if ( ProblemName == "MovingContact" )
  {
    // Moving Contact problem.
    const double V0  = 0.1;
    const double D_L = 1.4;
    const double D_R = 1.0;
    const double P_L = 1.0;
    const double P_R = 1.0;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++  )
    {
      X1 = Grid.NodeCoordinate( iX, iNodeX );

      if ( X1 <= 0.5 )
      {
        uCF(iCF_Tau, iX, iNodeX) = 1.0 / D_L;
        uCF(iCF_V, iX, iNodeX)   = V0;
        uCF(iCF_E, iX, iNodeX)   = (P_L / 0.4) * uCF(iCF_Tau, iX, iNodeX) + 0.5 * V0*V0;
        uPF(iPF_D, iX, iNodeX)   = D_L;
      }else{
        uCF(iCF_Tau, iX, iNodeX) = 1.0 / D_R;
        uCF(iCF_V, iX, iNodeX)   = V0;
        uCF(iCF_E, iX, iNodeX)   = (P_R / 0.4) * uCF(iCF_Tau, iX, iNodeX) + 0.5 * V0*V0;
        uPF(iPF_D, iX, iNodeX)   = D_R;
      }
    }    

  }
  else if ( ProblemName == "SmoothAdvection" )
  {
    // Smooth advection problem
    const double V0 = 1.0;
    const double P0 = 0.01;
    const double Amp = 1.0;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++  )
    {
      X1 = Grid.NodeCoordinate( iX, iNodeX );

      uCF(iCF_Tau, iX, iNodeX) = 1.0 / (2.0 + Amp * sin( 2.0 * PI * X1 ));
      uCF(iCF_V, iX, iNodeX)   = V0;
      uCF(iCF_E, iX, iNodeX)   = (P0 / 0.4) * uCF(iCF_Tau, iX, iNodeX) + 0.5 * V0*V0;
      // uPF(iPF_D, iX, iNodeX)   = D_L;

    }    
  }
  else
  {
    throw Error("Please choose a valid ProblemName");
  }
}
