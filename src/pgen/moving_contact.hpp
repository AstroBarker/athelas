#ifndef MOVING_CONTACT_HPP_
#define MOVING_CONTACT_HPP_
/**
 * @file moving_contact.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Moving contact wave test
 */

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * @brief Initialize moving contact discontinuity test
 **/
void moving_contact_init( State *state, GridStructure *Grid,
                          const ProblemIn *pin ) {

  View3D<Real> uCF = state->Get_uCF( );
  View3D<Real> uPF = state->Get_uPF( );
  const int pOrder = state->Get_pOrder( );

  const int ilo    = Grid->Get_ilo( );
  const int ihi    = Grid->Get_ihi( );
  const int nNodes = Grid->Get_nNodes( );

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const Real V0  = pin->in_table["problem"]["params"]["v0"].value_or( 0.1 );
  const Real D_L = pin->in_table["problem"]["params"]["rhoL"].value_or( 1.4 );
  const Real D_R = pin->in_table["problem"]["params"]["rhoR"].value_or( 1.0 );
  const Real P_L = pin->in_table["problem"]["params"]["pL"].value_or( 1.0 );
  const Real P_R = pin->in_table["problem"]["params"]["pR"].value_or( 1.0 );

  Real X1 = 0.0;
  for ( int iX = ilo; iX <= ihi; iX++ )
    for ( int k = 0; k < pOrder; k++ )
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        X1                    = Grid->Get_Centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;

        if ( X1 <= 0.5 ) {
          if ( k == 0 ) {
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
            uCF( iCF_V, iX, 0 )   = V0;
            uCF( iCF_E, iX, 0 ) =
                ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 ) + 0.5 * V0 * V0;
          }

          uPF( iPF_D, iX, iNodeX ) = D_L;
        } else {
          if ( k == 0 ) {
            uCF( iCF_Tau, iX, k ) = 1.0 / D_R;
            uCF( iCF_V, iX, k )   = V0;
            uCF( iCF_E, iX, k ) =
                ( P_R / 0.4 ) * uCF( iCF_Tau, iX, k ) + 0.5 * V0 * V0;
          }

          uPF( iPF_D, iX, iNodeX ) = D_R;
        }
      }
  // Fill density in guard cells
  for ( int iX = 0; iX < ilo; iX++ )
    for ( int iN = 0; iN < nNodes; iN++ ) {
      uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
      uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
    }
}
#endif // MOVING_CONTACT_HPP_
