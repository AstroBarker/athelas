#include "Abstractions.hpp"

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "Constants.hpp"
#include "Error.hpp"
#include "Grid.hpp"
#include "Initialization.hpp"

#define GAMMA 1.4

/**
 * Initialize equilibrium rad test
 **/
void RadEquilibriumInit( View3D uCF, View3D uPF, View3D uCR,
                         GridStructure *Grid, const UInt pOrder ) {

  const unsigned int ilo    = Grid->Get_ilo( );
  const unsigned int ihi    = Grid->Get_ihi( );
  const unsigned int nNodes = Grid->Get_nNodes( );

  const unsigned int iCF_Tau = 0;
  const unsigned int iCF_V   = 1;
  const unsigned int iCF_E   = 2;

  const unsigned int iPF_D = 0;

  const unsigned int iCR_E = 0;
  const unsigned int iCR_F = 1;

  const Real V0 = 0.0;
  const Real D  = std::pow( 10.0, -7.0 );
  const Real P  = 1.0;

  Real X1 = 0.0;
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int k = 0; k < pOrder; k++ )
      for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        X1                    = Grid->Get_Centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k )       = 0.0;
        uCR( 1, iX, k )       = 0.0;

        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
          uCF( iCF_V, iX, 0 )   = V0;
          uCF( iCF_E, iX, 0 ) = std::pow( 10.0, 10.0 ) * uCF( iCF_Tau, iX, 0 );

          uCR( iCR_E, iX, 0 ) = std::pow( 10.0, 12.0 ) * uCF( iCF_Tau, iX, 0 );
        }

        uPF( iPF_D, iX, iNodeX ) = D;
      }
  // Fill density in guard cells
  for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int iN = 0; iN < nNodes; iN++ ) {
      uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
      uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
    }
}
