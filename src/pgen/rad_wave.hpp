#pragma once
/**
 * @file rad_shock.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation shock test
 */

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * @brief Initialize radiating shock
 **/
void rad_shock_init( State* state, GridStructure* grid, const ProblemIn* pin ) {
  View3D<Real> uCF = state->get_u_cf( );
  View3D<Real> uPF = state->get_u_pf( );
  View3D<Real> uCR = state->get_u_cr( );
  const int pOrder = state->get_p_order( );

  const int ilo    = grid->get_ilo( );
  const int ihi    = grid->get_ihi( );
  const int nNodes = grid->get_n_nodes( );

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 0;

  const Real lambda =
      pin->in_table["problem"]["params"]["lambda"].value_or( 0.1 );
  const Real kappa =
      pin->in_table["problem"]["params"]["kappa"].value_or( 1.0 );
  const Real epsilon =
      pin->in_table["problem"]["params"]["epsilon"].value_or( 1.0e-6 );
  const Real rho0 = pin->in_table["problem"]["params"]["rho0"].value_or( 1.0 );
  const Real P0 =
      pin->in_table["problem"]["params"]["p0"].value_or( 1.0e-6 ); // K

  // TODO(astrobarker): thread through
  const Real gamma = 5.0 / 3.0;
  const Real gm1   = gamma - 1.0;

  for ( int iX = 0; iX <= ihi + 1; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        Real X1               = grid->get_centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k )       = 0.0;
        uCR( 1, iX, k )       = 0.0;

        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / rho0;
          uCF( iCF_V, iX, 0 )   = 0.0;
          uCF( iCF_E, iX, 0 )   = em_gas_R + 0.5 * V_R * V_R;

          uCR( iCR_E, iX, 0 ) = em_rad_R;
        }
        uPF( iPF_D, iX, iNodeX ) = rhoR;
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
