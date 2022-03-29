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

  const unsigned int nX     = Grid.Get_nElements();
  const unsigned int nG     = Grid.Get_Guard();

  // ! ? How to correctly implement reflecting BC ? !
  if ( BC == "Reflecting" )
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      // Inner Boudnary
      for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int k = 0; k < order; k++ )
      {
        if (iCF != 1 )
        {
          if ( k == 0 ) uCF(iCF, iX, k) = + uCF(iCF, ilo, k);
          if ( k != 0 ) uCF(iCF, iX, k) = - uCF(iCF, ilo, k);
        }
        else
        {
          if ( k == 0 ) uCF(1, iX, 0) = - uCF(1, ilo, 0);
          if ( k != 0 ) uCF(1, iX, k) = + uCF(1, ilo, k);
        }
      }

      // Outer Boundary
      for ( unsigned int iX = ihi+1; iX < nX+2*nG; iX++ )
      for ( unsigned int k = 0; k < order; k++ )
      {
        if (iCF != 1 )
        {
          uCF(iCF, iX, k) = uCF(iCF, ihi, k);
        }
        else
        {
          uCF(iCF, iX, k) = - uCF(1, ihi, k);
        }
      }

    }

  }
  else if ( BC == "Periodic" )
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    for ( unsigned int iX = 0; iX < ilo; iX ++ )
    for ( unsigned int k = 0; k < order; k++ )
    {
      uCF(iCF, ilo-1-iX, k) = uCF(iCF, ihi-iX, k);
      uCF(iCF, ihi+1+iX, k) = uCF(iCF, ilo+iX, k);
    }
  }
  else if ( BC == "ShocklessNoh" ) /* Special case for ShocklessNoh test */
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    for ( unsigned int iX = 0; iX < ilo; iX ++ )
    for ( unsigned int k = 0; k < order; k++ )
    {
      if ( k == 0 )
      {
        if (iCF == 0 )
        {
          uCF(iCF, ilo-1-iX, k) = uCF(iCF, ilo+iX, k);
          uCF(iCF, ihi+1+iX, k) = uCF(iCF, ihi-iX, k);
        }
        else if ( iCF == 1 )
        {
          uCF(iCF, ilo-1-iX, k) = -uCF(iCF, ilo+iX, k);
          uCF(iCF, ihi+1+iX, k) = uCF(iCF, ihi-iX, k) + (uCF(iCF, ihi-iX-1, k) - uCF(iCF, ihi-iX-2, k));
        }
        else
        {
          // Have to keep internal energy consistent with new velocities
          uCF(iCF, ilo-1-iX, k) = uCF(iCF, ilo+iX, k) - 0.5*uCF(1, ilo+iX, k)*uCF(1, ilo+iX, k) + 0.5*uCF(1, ilo-1-iX, k)*uCF(1, ilo-1-iX, k);
          uCF(iCF, ihi+1+iX, k) = uCF(iCF, ihi-iX, k) - 0.5*uCF(1, ihi-iX, k)*uCF(1, ihi-iX, k) + 0.5*uCF(1, ihi+1+iX, k)*uCF(1, ihi+1+iX, k);
        }
      }
      else
      {
        uCF(0, ilo-1-iX, k) = -uCF(0, ilo+iX, k);
        uCF(0, ihi+1+iX, k) = uCF(0, ihi-iX, k);

        uCF(1, ilo-1-iX, k) = uCF(1, ilo+iX, k);
        uCF(1, ihi+1+iX, k) = uCF(1, ihi-iX, k);

        uCF(2, ilo-1-iX, k) = - uCF(1, ilo+iX, 0) * uCF(1, ilo+iX, 1);
        uCF(2, ihi+1+iX, k) = uCF(1, ihi+1+iX, 0) * uCF(1, ihi+1+iX, 1);
      }
    }
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
