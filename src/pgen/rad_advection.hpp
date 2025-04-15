/**
 * @file rad_advection.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation advection test
 */

#ifndef RAD_ADVECTION_HPP_
#define RAD_ADVECTION_HPP_

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * @brief Initialize radiation advection test
 * @note EXPERIMENTAL
 **/
void rad_advection_init( State* state, GridStructure* grid,
                         const ProblemIn* pin ) {
  View3D<Real> uCF = state->get_u_cf( );
  View3D<Real> uPF = state->get_u_pf( );
  View3D<Real> uCR = state->get_u_cr( );
  const int pOrder = state->get_p_order( );

  const int ilo    = grid->get_ilo( );
  const int ihi    = grid->get_ihi( );
  const int nNodes = grid->get_n_nodes( );

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const int iCR_E = 0;

  const Real V0  = pin->in_table["problem"]["params"]["v0"].value_or( 0.5 );
  const Real D   = pin->in_table["problem"]["params"]["rho"].value_or( 1.0 );
  const Real amp = pin->in_table["problem"]["params"]["amp"].value_or( 1.0 );
  const Real width =
      pin->in_table["problem"]["params"]["width"].value_or( 0.05 );

  for ( int iX = 0; iX <= ihi + 1; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        Real const X1         = grid->get_centers( iX );
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
              amp * std::max( std::exp( -std::pow( ( X1 - 0.5 ) / width, 2.0 ) /
                                        2.0 ),
                              1.0e-8 );
        }

        uPF( iPF_D, iX, iNodeX ) = D;
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
#endif // RAD_ADVECTION_HPP_
