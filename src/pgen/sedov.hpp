#ifndef SEDOV_HPP_
#define SEDOV_HPP_

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * Initialize sedov blast wave
 **/
void sedov_init( State *state, GridStructure *Grid, const ProblemIn *pin ) {

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

  const Real V0 = pin->in_table["problem"]["params"]["v0"].value_or( 0.0 );
  const Real D0 = pin->in_table["problem"]["params"]["rho0"].value_or( 1.0 );
  const Real E0 = pin->in_table["problem"]["params"]["E0"].value_or( 0.3 );

  const int origin = 1;

  // TODO: geometry aware volume for energy
  const Real volume = ( 4.0 * M_PI / 3.0 ) *
                      std::pow( Grid->Get_LeftInterface( origin + 1 ), 3.0 );
  const Real P0 = ( 5.0 / 3.0 - 1.0 ) * E0 / volume;

  for ( int iX = ilo; iX <= ihi; iX++ )
    for ( int k = 0; k < pOrder; k++ )
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {

        if ( k != 0 ) {
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;
        } else {
          uCF( iCF_Tau, iX, k ) = 1.0 / D0;
          uCF( iCF_V, iX, k )   = V0;
          if ( iX == origin - 0 || iX == origin ) {
            uCF( iCF_E, iX, k ) =
                ( P0 / ( 5.0 / 3.0 - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                0.5 * V0 * V0;
          } else {
            uCF( iCF_E, iX, k ) =
                ( 1.0e-6 / ( 5.0 / 3.0 - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                0.5 * V0 * V0;
          }
        }
        uPF( iPF_D, iX, iNodeX ) = D0;
      }
  // Fill density in guard cells
  for ( int iX = 0; iX < ilo; iX++ )
    for ( int iN = 0; iN < nNodes; iN++ ) {
      uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
      uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
    }
}
#endif // SEDOV_HPP_
