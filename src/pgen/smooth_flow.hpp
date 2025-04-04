#ifndef SMOOTH_FLOW_HPP_
#define SMOOTH_FLOW_HPP_

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * Initialize smooth flow test problem
 **/
void smooth_flow_init( State *state, GridStructure *Grid,
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

  const Real amp = pin->in_table["problem"]["params"]["amp"].value_or(
      0.9999999999999999999 );

  Real X1 = 0.0;
  for ( int iX = ilo; iX <= ihi; iX++ )
    for ( int k = 0; k < pOrder; k++ )
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        X1                    = Grid->Get_Centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;

        if ( k == 0 ) {
          Real D                = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
          uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
          uCF( iCF_V, iX, 0 )   = 0.0;
          uCF( iCF_E, iX, 0 )   = ( D * D * D / 2.0 ) * uCF( iCF_Tau, iX, 0 );
        } else if ( k == 1 ) {
          Real D  = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
          Real dD = ( amp * constants::PI( ) * cos( constants::PI( ) * X1 ) );
          uCF( iCF_Tau, iX, k ) =
              ( -1 / ( D * D ) ) * dD * Grid->Get_Widths( iX );
          uCF( iCF_V, iX, k ) = 0.0;
          uCF( iCF_E, iX, k ) =
              ( ( 2.0 / 2.0 ) * D ) * dD * Grid->Get_Widths( iX );
        } else if ( k == 2 ) {
          Real D   = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
          Real ddD = -( amp * constants::PI( ) * constants::PI( ) ) *
                     sin( constants::PI( ) * X1 );
          uCF( iCF_Tau, iX, k ) = ( 2.0 / ( D * D * D ) ) * ddD *
                                  Grid->Get_Widths( iX ) *
                                  Grid->Get_Widths( iX );
          uCF( iCF_V, iX, k ) = 0.0;
          uCF( iCF_E, iX, k ) = ( 2.0 / 2.0 ) * ddD * Grid->Get_Widths( iX ) *
                                Grid->Get_Widths( iX );
        } else if ( k == 3 ) {
          Real D    = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
          Real dddD = -( amp * constants::PI( ) * constants::PI( ) *
                         constants::PI( ) ) *
                      cos( constants::PI( ) * X1 );
          uCF( iCF_Tau, iX, k ) =
              ( -6.0 / ( D * D * D * D ) ) * dddD * Grid->Get_Widths( iX ) *
              Grid->Get_Widths( iX ) * Grid->Get_Widths( iX );
          uCF( iCF_V, iX, k ) = 0.0;
          uCF( iCF_E, iX, k ) = 0.0;
        } else {
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;
        }

        uPF( iPF_D, iX, iNodeX ) = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
      }
  // Fill density in guard cells
  for ( int iX = 0; iX < ilo; iX++ )
    for ( int iN = 0; iN < nNodes; iN++ ) {
      uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
      uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
    }
}
#endif // SMOOTH_FLOW_HPP_
