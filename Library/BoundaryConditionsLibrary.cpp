/**
 * File    :  BoundaryConditions.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Apply boundary conditions
**/  

#include <iostream>
#include <string>

#include "DataStructures.h"
#include "Grid.h"
#include "BoundaryConditionsLibrary.h"

// Apply Boundary Conditions to fluid fields
void ApplyBC_Fluid( DataStructure3D& uCF, GridStructure& Grid, 
  unsigned int order, const std::string BC )
{

  const unsigned int ilo = Grid.Get_ilo();
  const unsigned int ihi = Grid.Get_ihi();

  const unsigned int nNodes = Grid.Get_nNodes();
  const unsigned int nX     = Grid.Get_nElements();
  const unsigned int nG     = Grid.Get_Guard();

  unsigned int j = 0;

  if ( BC == "Reflecting" )
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      // Inner Boudnary
      for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        j = nNodes - iN - 1;
        if (iCF != 1 ){
          uCF(iCF, iX, iN) = uCF(iCF, ilo, j);
        }else{
          uCF(iCF, iX, iN) = - uCF(1, ilo, j);
        }
      }

      // Outer Boundary
      for ( unsigned int iX = ihi+1; iX < nX+2*nG; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        j = nNodes - iN - 1;
        if (iCF != 1 ){
          uCF(iCF, iX, iN) = uCF(iCF, ihi, j);
        }else{
          uCF(iCF, iX, iN) = - uCF(1, ihi, j);
        }
      }

    }

  }
  else if ( BC == "Periodic" )
  {
    // TODO: Implement periodic BCs
  }
  else
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    for ( unsigned int iX = 0; iX < ilo; iX ++ )
    for ( unsigned int k = 0; k < order; k++ )
    {
      uCF(iCF, ilo-1-iX, k) = uCF(iCF, ilo+iX, k);
      uCF(iCF, ihi+1+iX, k) = uCF(iCF, ihi-iX, k);
    }

  }

}
