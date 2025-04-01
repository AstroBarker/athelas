#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

#define GAMMA 1.4

/**
 * Initialize radiation advection test
 **/
void RadAdvectionInit( View3D<Real> uCF, View3D<Real> uPF, View3D<Real> uCR,
                       GridStructure *Grid, const int pOrder ) {

  const int ilo    = Grid->Get_ilo( );
  const int ihi    = Grid->Get_ihi( );
  const int nNodes = Grid->Get_nNodes( );

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const int iCR_E = 0;

  const Real V0                   = 1.0;
  const Real D                    = 1.0;
  constexpr static Real E_rad_amp = 1.0;

  constexpr static Real width = 0.05;

  for ( int iX = 0; iX <= ihi + 1; iX++ )
    for ( int k = 0; k < pOrder; k++ )
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        Real X1               = Grid->Get_Centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k )       = 0.0;
        uCR( 1, iX, k )       = 0.0;

        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
          uCF( iCF_V, iX, 0 )   = V0;
          uCF( iCF_E, iX, 0 )   = 1.0;
          uCR( iCR_E, iX, k ) =
              E_rad_amp *
              std::max(
                  std::exp( -std::pow( ( X1 - 0.5 ) / width, 2.0 ) / 2.0 ),
                  1.0e-8 );
        }

        uPF( iPF_D, iX, iNodeX ) = D;
      }
  // Fill density in guard cells
  for ( int iX = 0; iX < ilo; iX++ )
    for ( int iN = 0; iN < nNodes; iN++ ) {
      uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
      uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
    }
}
