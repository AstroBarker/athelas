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
 * Initialize the conserved Fields for various problems.
 * TODO: For now I initialize constant on each cell. Is there a better way?
 * TODO: iNodeX and order separation
 **/
void LightbulbInit( View3D uCF, View3D uPF, View3D uCR,
                       GridStructure *Grid, const UInt pOrder )
{

  const unsigned int ilo    = Grid->Get_ilo( );
  const unsigned int ihi    = Grid->Get_ihi( );
  const unsigned int nNodes = Grid->Get_nNodes( );

  const unsigned int iCF_Tau = 0;
  const unsigned int iCF_V   = 1;
  const unsigned int iCF_E   = 2;

  const unsigned int iPF_D = 0;

  const unsigned int iCR_E = 0;
  const unsigned int iCR_F = 1;
  const Real V0  = 0.0;
  const Real D_L = 1.0;
  const Real D_R = 0.125;
  const Real P_L = 1.0;
  const Real P_R = 0.1;

  Real X1 = 0.0;
  for ( unsigned int iX = ilo; iX <= ihi; iX++ )
    for ( unsigned int k = 0; k < pOrder; k++ )
      for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
      {
        X1                    = Grid->Get_Centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k ) = 0.0;
        uCR( 1, iX, k ) = 0.0;

        if ( X1 <= 0.5 )
        {
          if ( k == 0 )
          {
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
            uCF( iCF_V, iX, 0 )   = V0;
            uCF( iCF_E, iX, 0 )   = ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 );
          }

          uPF( iPF_D, iX, iNodeX ) = D_L;
        }
        else
        { // right domain
          if ( k == 0 )
          {
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D_R;
            uCF( iCF_V, iX, 0 )   = V0;
            uCF( iCF_E, iX, 0 )   = ( P_R / 0.4 ) * uCF( iCF_Tau, iX, 0 );
          }

          uPF( iPF_D, iX, iNodeX ) = D_R;
        }
      }
  // Fill density in guard cells
  for ( unsigned int iX = 0; iX < ilo; iX++ )
    for ( unsigned int iN = 0; iN < nNodes; iN++ )
    {
      uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
      uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
    }

}
