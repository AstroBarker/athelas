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

#include "BoundaryConditionsLibrary.hpp"
#include "Grid.hpp"

// Apply Boundary Conditions to fluid fields
void ApplyBC( View3D U, GridStructure *Grid, const UInt order,
              const std::string BC ) {

  const UInt ilo  = Grid->Get_ilo( );
  const UInt ihi  = Grid->Get_ihi( );
  const int nvars = U.extent( 0 );

  const UInt nX = Grid->Get_nElements( );
  const UInt nG = Grid->Get_Guard( );

  // ! ? How to correctly implement reflecting BC ? !
  if ( BC == "Reflecting" ) {
    for ( UInt iCF = 0; iCF < nvars; iCF++ ) {
      // Inner Boudnary
      for ( UInt iX = 0; iX < ilo; iX++ )
        for ( UInt k = 0; k < order; k++ ) {
          if ( iCF != 1 ) {
            if ( k == 0 ) U( iCF, iX, k ) = +U( iCF, ilo, k );
            if ( k != 0 ) U( iCF, iX, k ) = -U( iCF, ilo, k );
          } else {
            if ( k == 0 ) U( 1, iX, 0 ) = -U( 1, ilo, 0 );
            if ( k != 0 ) U( 1, iX, k ) = +U( 1, ilo, k );
          }
        }

      // Outer Boundary
      for ( UInt iX = ihi + 1; iX < nX + 2 * nG; iX++ )
        for ( UInt k = 0; k < order; k++ ) {
          if ( iCF != 1 ) {
            U( iCF, iX, k ) = U( iCF, ihi, k );
          } else {
            U( iCF, iX, k ) = -U( 1, ihi, k );
          }
        }
    }
  } else if ( BC == "Periodic" ) {
    for ( UInt iCF = 0; iCF < nvars; iCF++ )
      for ( UInt iX = 0; iX < ilo; iX++ )
        for ( UInt k = 0; k < order; k++ ) {
          U( iCF, ilo - 1 - iX, k ) = U( iCF, ihi - iX, k );
          U( iCF, ihi + 1 + iX, k ) = U( iCF, ilo + iX, k );
        }
  } else if ( BC == "ShocklessNoh" ) /* Special case for ShocklessNoh test */
  {
    // for ( UInt iCF = 0; iCF < 3; iCF++ )
    for ( UInt iX = 0; iX < ilo; iX++ )
      for ( UInt k = 0; k < order; k++ )
      // for ( UInt iCF = 0; iCF < 3; iCF++ )
      {
        if ( k == 0 ) {

          U( 0, ilo - 1 - iX, k ) = U( 0, ilo + iX, k );
          U( 0, ihi + 1 + iX, k ) = U( 0, ihi - iX, k );

          U( 1, ilo - 1 - iX, k ) = -U( 1, ilo + iX, k );
          U( 1, ihi + 1 + iX, k ) =
              U( 1, ihi - iX, k ) +
              ( U( 1, ihi - iX - 1, k ) - U( 1, ihi - iX - 2, k ) );

          // Have to keep internal energy consistent with new velocities
          U( 2, ilo - 1 - iX, k ) =
              U( 2, ilo + iX, k ) -
              0.5 * U( 1, ilo + iX, k ) * U( 1, ilo + iX, k ) +
              0.5 * U( 1, ilo - 1 - iX, k ) * U( 1, ilo - 1 - iX, k );
          U( 2, ihi + 1 + iX, k ) =
              U( 2, ihi - iX, k ) -
              0.5 * U( 1, ihi - iX, k ) * U( 1, ihi - iX, k ) +
              0.5 * U( 1, ihi + 1 + iX, k ) * U( 1, ihi + 1 + iX, k );
        } else if ( k == 1 ) {
          U( 0, ilo - 1 - iX, k ) = 0.0;
          U( 0, ihi + 1 + iX, k ) = 0.0;

          U( 1, ilo - 1 - iX, k ) = U( 1, ilo + iX, k );
          U( 1, ihi + 1 + iX, k ) = U( 1, ihi - iX, k );

          U( 2, ilo - 1 - iX, k ) = -U( 1, ilo + iX, 0 ) * U( 1, ilo + iX, 1 );
          U( 2, ihi + 1 + iX, k ) =
              U( 1, ihi + 1 + iX, 0 ) * U( 1, ihi + 1 + iX, 1 );
        } else if ( k == 2 ) {

          // U( iCF, ilo - 1 - iX, k ) = 0.0;
          // U( iCF, ihi + 1 + iX, k ) = 0.0;
          U( 0, ilo - 1 - iX, k ) = U( 0, ilo + iX, k );
          U( 0, ihi + 1 + iX, k ) = U( 0, ihi - iX, k );

          // U( iCF, ilo - 1 - iX, k ) = 0.0;
          // U( iCF, ihi + 1 + iX, k ) = 0.0;
          U( 1, ilo - 1 - iX, k ) = U( 1, ilo + iX, k );
          U( 1, ihi + 1 + iX, k ) = U( 1, ihi - iX, k );

          // Have to keep internal energy consistent with new velocities
          U( 2, ilo - 1 - iX, k ) =
              U( 2, ilo + iX, 2 ); // * U( 1, ilo + iX, 1 );
          U( 2, ihi + 1 + iX, k ) =
              U( 2, ihi - iX, 2 ); //* U( 1, ihi - iX, 1 );
        } else {
          U( 0, ilo - 1 - iX, k ) = 0.0; // U( 0, ilo + iX, k );
          U( 0, ihi + 1 + iX, k ) = 0.0; // U( 0, ihi - iX, k );

          U( 1, ilo - 1 - iX, k ) = U( 1, ilo + iX, k );
          U( 1, ihi + 1 + iX, k ) = U( 1, ihi - iX, k );

          U( 2, ilo - 1 - iX, k ) = -U( 2, ilo + iX, k );
          U( 2, ihi + 1 + iX, k ) = U( 2, ihi - iX, k );
        }
      }
  } else {
    for ( UInt iCF = 0; iCF < nvars; iCF++ )
      for ( UInt iX = 0; iX < ilo; iX++ )
        for ( UInt k = 0; k < order; k++ ) {
          U( iCF, ilo - 1 - iX, k ) = U( iCF, ilo + iX, k );
          U( iCF, ihi + 1 + iX, k ) = U( iCF, ihi - iX, k );
        }
  }
}
