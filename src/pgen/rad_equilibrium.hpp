#ifndef RAD_EQUILIBRIUM_HPP_
#define RAD_EQUILIBRIUM_HPP_

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

#define GAMMA 1.4

/**
 * Initialize equilibrium rad test
 **/
void rad_equilibrium_init( State *state, GridStructure *Grid,
                           const ProblemIn *pin ) {
  View3D<Real> uCF = state->Get_uCF( );
  View3D<Real> uPF = state->Get_uPF( );
  View3D<Real> uCR = state->Get_uCR( );
  const int pOrder = state->Get_pOrder( );

  const int ilo    = Grid->Get_ilo( );
  const int ihi    = Grid->Get_ihi( );
  const int nNodes = Grid->Get_nNodes( );

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const int iCR_E = 0;

  const Real V0 = pin->in_table["problem"]["params"]["v0"].value_or( 0.0 );
  const Real logD =
      pin->in_table["problem"]["params"]["logrho"].value_or( -7.0 );
  const Real logE_gas = pin->in_table["problem"]["params"]["logE_gas"].value_or(
      10.0 ); // erg / cm^3
  const Real logE_rad = pin->in_table["problem"]["params"]["logE_rad"].value_or(
      12.0 ); // erg / cm^3
  const Real D      = std::pow( 10.0, logD );
  const Real Ev_gas = std::pow( 10.0, logE_gas );
  const Real Ev_rad = std::pow( 10.0, logE_rad );

  for ( int iX = 0; iX <= ihi + 1; iX++ )
    for ( int k = 0; k < pOrder; k++ )
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k )       = 0.0;
        uCR( 1, iX, k )       = 0.0;

        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
          uCF( iCF_V, iX, 0 )   = V0;
          uCF( iCF_E, iX, 0 )   = Ev_gas * uCF( iCF_Tau, iX, 0 );

          uCR( iCR_E, iX, 0 ) = Ev_rad * uCF( iCF_Tau, iX, 0 );
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
#endif // RAD_EQUILIBRIUM_HPP_
