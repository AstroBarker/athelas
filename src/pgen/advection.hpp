#ifndef ADVECTION_HPP_
#define ADVECTION_HPP_
/**
 * @file advection.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Fluid advection test
 */

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * Initialize advection test
 **/
void advection_init( State* state, GridStructure* Grid, const ProblemIn* pin ) {
  // Smooth advection problem
  View3D<Real> uCF = state->Get_uCF( );
  View3D<Real> uPF = state->Get_uPF( );
  const int pOrder = state->Get_pOrder( );

  const int ilo    = Grid->Get_ilo( );
  const int ihi    = Grid->Get_ihi( );
  const int nNodes = Grid->Get_nNodes( );

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const Real V0  = pin->in_table["problem"]["params"]["v0"].value_or( -1.0 );
  const Real P0  = pin->in_table["problem"]["params"]["p0"].value_or( 0.01 );
  const Real Amp = pin->in_table["problem"]["params"]["amp"].value_or( 1.0 );

  Real X1 = 0.0;
  for ( int iX = ilo; iX <= ihi; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        X1 = Grid->Get_Centers( iX );

        if ( k != 0 ) {
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;
        } else {
          uCF( iCF_Tau, iX, k ) =
              1.0 / ( 2.0 + Amp * sin( 2.0 * constants::PI( ) * X1 ) );
          uCF( iCF_V, iX, k ) = V0;
          uCF( iCF_E, iX, k ) =
              ( P0 / 0.4 ) * uCF( iCF_Tau, iX, k ) + 0.5 * V0 * V0;
        }
        uPF( iPF_D, iX, iNodeX ) =
            ( 2.0 + Amp * sin( 2.0 * constants::PI( ) * X1 ) );
      }
    }
  }
  // Fill density in guard cells
  for ( int iX = 0; iX < ilo; iX++ ) {
    for ( int iN = 0; iN < nNodes; iN++ ) {
      uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
      uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
    }
  }
}
#endif // ADVECTION_HPP_
