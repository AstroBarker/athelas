/**
 * File    :  BoundaryConditions.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Apply boundary conditions
 **/

#include <iostream>
#include <string>

#include "Kokkos_Core.hpp"

#include "Grid.h"
#include "BoundaryConditionsLibrary.h"

// Apply Boundary Conditions to fluid fields
// TODO: Optimize the loops.
void ApplyBC_Fluid( Kokkos::View<double***> uCF, const GridStructure& Grid,
                    const unsigned int order, const std::string BC )
{

  const unsigned int ilo = Grid.Get_ilo( );
  const unsigned int ihi = Grid.Get_ihi( );

  const unsigned int nX = Grid.Get_nElements( );
  const unsigned int nG = Grid.Get_Guard( );

  // ! ? How to correctly implement reflecting BC ? !
  if ( BC == "Reflecting" )
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    {
      // Inner Boudnary
      for ( unsigned int iX = 0; iX < ilo; iX++ )
        for ( unsigned int k = 0; k < order; k++ )
        {
          if ( iCF != 1 )
          {
            if ( k == 0 ) uCF( iX, k, iCF ) = +uCF( ilo, k, iCF );
            if ( k != 0 ) uCF( iX, k, iCF ) = -uCF( ilo, k, iCF );
          }
          else
          {
            if ( k == 0 ) uCF( iX, 0, 1 ) = -uCF( ilo, 0, 1 );
            if ( k != 0 ) uCF( iX, k, 1 ) = +uCF( ilo, k, 1 );
          }
        }

      // Outer Boundary
      for ( unsigned int iX = ihi + 1; iX < nX + 2 * nG; iX++ )
        for ( unsigned int k = 0; k < order; k++ )
        {
          if ( iCF != 1 )
          {
            uCF( iX, k, iCF ) = uCF( ihi, k, iCF );
          }
          else
          {
            uCF( iX, k, iCF ) = -uCF( ihi, k, 1 );
          }
        }
    }
  }
  else if ( BC == "Periodic" )
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      for ( unsigned int iX = 0; iX < ilo; iX++ )
        for ( unsigned int k = 0; k < order; k++ )
        {
          uCF( ilo - 1 - iX, k, iCF ) = uCF( ihi - iX, k, iCF );
          uCF( ihi + 1 + iX, k, iCF ) = uCF( ilo + iX, k, iCF );
        }
  }
  else if ( BC == "ShocklessNoh" ) /* Special case for ShocklessNoh test */
  {
    // for ( unsigned int iCF = 0; iCF < 3; iCF++ )
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int k = 0; k < order; k++ )
      // for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      {
        if ( k == 0 )
        {

          uCF( ilo - 1 - iX, k, 0 ) = uCF( ilo + iX, k, 0 );
          uCF( ihi + 1 + iX, k, 0 ) = uCF( ihi - iX, k, 0 );

          uCF( ilo - 1 - iX, k, 1 ) = -uCF( ilo + iX, k, 1 );
          uCF( ihi + 1 + iX, k, 1 ) =
              uCF( ihi - iX, k, 1 ) +
              ( uCF( ihi - iX - 1, k, 1 ) - uCF( ihi - iX - 2, k, 1 ) );

          // Have to keep internal energy consistent with new velocities
          uCF( ilo - 1 - iX, k, 2 ) =
              uCF( ilo + iX, k, 2 ) -
              0.5 * uCF( ilo + iX, k, 1 ) * uCF( ilo + iX, k, 1 ) +
              0.5 * uCF( ilo - 1 - iX, k, 1 ) * uCF( ilo - 1 - iX, k, 1 );
          uCF( ihi + 1 + iX, k, 2 ) =
              uCF( ihi - iX, k, 2 ) -
              0.5 * uCF( ihi - iX, k, 1 ) * uCF( ihi - iX, k, 1 ) +
              0.5 * uCF( ihi + 1 + iX, k, 1 ) * uCF( ihi + 1 + iX, k, 1 );
        }
        else if ( k == 1 )
        {
          uCF( ilo - 1 - iX, k, 0 ) = 0.0;
          uCF( ihi + 1 + iX, k, 0 ) = 0.0;

          uCF( ilo - 1 - iX, k, 1 ) = uCF( ilo + iX, k, 1 );
          uCF( ihi + 1 + iX, k, 1 ) = uCF( ihi - iX, k, 1 );

          uCF( ilo - 1 - iX, k, 2 ) =
              -uCF( ilo + iX, 0, 1 ) * uCF( ilo + iX, 1, 1 );
          uCF( ihi + 1 + iX, k, 2 ) =
              uCF( ihi + 1 + iX, 0, 1 ) * uCF( ihi + 1 + iX, 1, 1 );
        }
        else if ( k == 2 )
        {

          // uCF( iCF, ilo - 1 - iX, k ) = 0.0;
          // uCF( iCF, ihi + 1 + iX, k ) = 0.0;
          uCF( ilo - 1 - iX, k, 0 ) = uCF( ilo + iX, k, 0 );
          uCF( ihi + 1 + iX, k, 0 ) = uCF( ihi - iX, k, 0 );

          // uCF( iCF, ilo - 1 - iX, k ) = 0.0;
          // uCF( iCF, ihi + 1 + iX, k ) = 0.0;
          uCF( ilo - 1 - iX, k, 1 ) = uCF( ilo + iX, k, 1 );
          uCF( ihi + 1 + iX, k, 1 ) = uCF( ihi - iX, k, 1 );

          // Have to keep internal energy consistent with new velocities
          uCF( ilo - 1 - iX, k, 2 ) =
              uCF( ilo + iX, 2, 2 ); // * uCF( 1, ilo + iX, 1 );
          uCF( ihi + 1 + iX, k, 2 ) =
              uCF( ihi - iX, 2, 2 ); //* uCF( 1, ihi - iX, 1 );
        }
        else
        {
          uCF( ilo - 1 - iX, k, 0 ) = 0.0; // uCF( 0, ilo + iX, k );
          uCF( ihi + 1 + iX, k, 0 ) = 0.0; // uCF( 0, ihi - iX, k );

          uCF( ilo - 1 - iX, k, 1 ) = uCF( ilo + iX, k, 1 );
          uCF( ihi + 1 + iX, k, 1 ) = uCF( ihi - iX, k, 1 );

          uCF( ilo - 1 - iX, k, 2 ) = -uCF( ilo + iX, k, 2 );
          uCF( ihi + 1 + iX, k, 2 ) = uCF( ihi - iX, k, 2 );
        }
      }
  }
  else
  {
    for ( unsigned int iCF = 0; iCF < 3; iCF++ )
      for ( unsigned int iX = 0; iX < ilo; iX++ )
        for ( unsigned int k = 0; k < order; k++ )
        {
          uCF( ilo - 1 - iX, k, iCF ) = uCF( ilo + iX, k, iCF );
          uCF( ihi + 1 + iX, k, iCF ) = uCF( ihi - iX, k, iCF );
        }
  }
}
