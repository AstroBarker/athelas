#pragma once
/**
 * @file noh.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Noh test
 */

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * @brief Initialize Noh problem
 **/
void noh_init( State* state, GridStructure* grid, const ProblemIn* pin ) {

  View3D<Real> uCF = state->get_u_cf( );
  View3D<Real> uPF = state->get_u_pf( );
  const int pOrder = state->get_p_order( );

  const int ilo    = grid->get_ilo( );
  const int ihi    = grid->get_ihi( );
  const int nNodes = grid->get_n_nodes( );

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const Real V0 = pin->in_table["problem"]["params"]["v0"].value_or( -1.0 );
  const Real D0 = pin->in_table["problem"]["params"]["rho0"].value_or( 1.0 );
  const Real P0 =
      pin->in_table["problem"]["params"]["p0"].value_or( 0.000001 );

  const Real GAMMA = 5.0 / 3.0;

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;

          if ( k == 0 ) {
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D0;
            uCF( iCF_V, iX, 0 )   = V0;
            uCF( iCF_E, iX, 0 ) =
                ( P0 / ( GAMMA - 1.0 ) ) * uCF( iCF_Tau, iX, 0 ) +
                0.5 * V0 * V0;
          }

          uPF( iPF_D, iX, iNodeX ) = D0;
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
