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
#include "Initialization.h"

/**
 * Initialize the conserved Fields for various problems.
 * TODO: For now I initialize constant on each cell. Is there a better way?
 * TODO: iNodeX and order separation
 **/
void InitializeFields( DataStructure3D& uCF, DataStructure3D& uPF, 
  GridStructure& Grid, const unsigned int pOrder, 
  const std::string ProblemName )
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
    for ( unsigned int k = 0; k < pOrder; k++ )
    for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++  )
    {
      X1 = Grid.NodeCoordinate( iX, iNodeX );
      uCF(iCF_Tau, iX, k) = 0.0;
      uCF(iCF_V, iX, k)   = 0.0;
      uCF(iCF_E, iX, k)   = 0.0;
      
      if ( X1 <= 0.5 )
      {
        if ( k == 0 )
        {
          uCF(iCF_Tau, iX, 0) = 1.0 / D_L;
          uCF(iCF_V, iX, 0)   = V0;
          uCF(iCF_E, iX, 0)   = (P_L / 0.4) * uCF(iCF_Tau, iX, 0);
        }

        uPF(iPF_D, iX, iNodeX) = D_L;
      }else{ // right domain
        if ( k == 0 )
        {
          uCF(iCF_Tau, iX, 0) = 1.0 / D_R;
          uCF(iCF_V, iX, 0)   = V0;
          uCF(iCF_E, iX, 0)   = (P_R / 0.4) * uCF(iCF_Tau, iX, 0);
        }

        uPF(iPF_D, iX, iNodeX) = D_R;
      }
    }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int iN = 0; iN < nNodes; iN++  )
    {
      uPF(0, ilo-1-iX, iN) = uPF(0, ilo+iX, nNodes-iN-1);
      uPF(0, ihi+1+iX, iN) = uPF(0, ihi-iX, nNodes-iN-1);
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
    for ( unsigned int k = 0; k < pOrder; k++ )
    for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++  )
    {
      X1 = Grid.NodeCoordinate( iX, iNodeX );
      uCF(iCF_Tau, iX, k) = 0.0;
      uCF(iCF_V, iX, k)   = 0.0;
      uCF(iCF_E, iX, k)   = 0.0;

      if ( X1 <= 0.5 )
      {
        if ( k == 0 )
        {
          uCF(iCF_Tau, iX, 0) = 1.0 / D_L;
          uCF(iCF_V, iX, 0)   = V0;
          uCF(iCF_E, iX, 0)   = (P_L / 0.4) * uCF(iCF_Tau, iX, 0) + 0.5 * V0*V0;
        }

        uPF(iPF_D, iX, iNodeX)   = D_L;
      }else{
        if ( k == 0 )
        {
          uCF(iCF_Tau, iX, k) = 1.0 / D_R;
          uCF(iCF_V, iX, k)   = V0;
          uCF(iCF_E, iX, k)   = (P_R / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0*V0;
        }

        uPF(iPF_D, iX, iNodeX)   = D_R;
      }
    }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int iN = 0; iN < nNodes; iN++  )
    {
      uPF(0, ilo-1-iX, iN) = uPF(0, ilo+iX, nNodes-iN-1);
      uPF(0, ihi+1+iX, iN) = uPF(0, ihi-iX, nNodes-iN-1);
    }    

  }
  else if ( ProblemName == "SmoothAdvection" )
  {
    // Smooth advection problem
    const double V0  = 1.0;
    const double P0  = 0.01;
    const double Amp = 1.0;

    double X1 = 0.0;
    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int k = 0; k < pOrder; k++ )
    for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++  )
    {
      X1 = Grid.NodeCoordinate( iX, iNodeX );

      if ( k != 0 )
      {
        uCF(iCF_Tau, iX, k) = 0.0;
        uCF(iCF_V, iX, k)   = 0.0;
        uCF(iCF_E, iX, k)   = 0.0;
      }
      else
      {
        uCF(iCF_Tau, iX, k) = 1.0 / (2.0 + Amp * sin( 2.0 * pi() * X1 ));
        uCF(iCF_V, iX, k)   = V0;
        uCF(iCF_E, iX, k)   = (P0 / 0.4) * uCF(iCF_Tau, iX, k) + 0.5 * V0*V0;
      }
      uPF(iPF_D, iX, iNodeX) = (2.0 + Amp * sin( 2.0 * PI() * X1 ));

    }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int iN = 0; iN < nNodes; iN++  )
    {
      uPF(0, ilo-1-iX, iN) = uPF(0, ilo+iX, nNodes-iN-1);
      uPF(0, ihi+1+iX, iN) = uPF(0, ihi-iX, nNodes-iN-1);
    }     
  }
  else
  {
    throw Error("Please choose a valid ProblemName");
  }
}
