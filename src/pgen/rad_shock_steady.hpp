#pragma once
/**
 * @file rad_equilibrium.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation fluid equilibriation test
 */

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "abstractions.hpp"
#include "constants.hpp"
#include "error.hpp"
#include "grid.hpp"

/**
 * @brief Initialize steady radiating shock
 * 
 * tTwo different cases: Mach 2 and Mach 5.
 * 
 * Mach 2 Case:
 * - Left side (pre-shock):
 *   - Density: 1.0 g/cm^3
 *   - Internal energy: 1.60217663e-10 erg (100eV)
 * - Right side (post-shock):
 *   - Density: 2.286 g/cm^3
 *   - Internal energy: 3.3286e-10 ergs (207.756 eV)
 * 
 * Mach 5 Case:
 * - Left side (pre-shock):
 *   - Density: 1.0 g/cm^3
 *   - Internal energy: 1.60217663e-10 erg (100 eV)
 * - Right side (post-shock):
 *   - Density: 3.598 g/cm^3
 *   - Internal energy: 1.37101e-9 ergs (855.720 eV)
 **/
void rad_shock_steady_init( State* state, GridStructure* grid,
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

  const Real V0 = pin->in_table["problem"]["params"]["v0"].value_or( 0.0 );
  const Real rhoL =
      pin->in_table["problem"]["params"]["rhoL"].value_or( 1.0 );
  const Real rhoR =
      pin->in_table["problem"]["params"]["rhoR"].value_or( 2.286 );
  const Real eL = pin->in_table["problem"]["params"]["eL"].value_or(
      1.60217663e-10 ); // erg
  const Real eR = pin->in_table["problem"]["params"]["eR"].value_or(
      3.3286-10 ); // erg

  for ( int iX = 0; iX <= ihi + 1; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        Real X1                    = grid->get_centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k )       = 0.0;
        uCR( 1, iX, k )       = 0.0;

        if ( X1 <= 0.0 ) {
        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / rhoL;
          uCF( iCF_V, iX, 0 )   = V0;
          uCF( iCF_E, iX, 0 )   = eL * uCF( iCF_Tau, iX, 0 );

          uCR( iCR_E, iX, 0 ) = eL * uCF( iCF_Tau, iX, 0 );
        }
        uPF( iPF_D, iX, iNodeX ) = rhoL;
        } else {
        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / rhoR;
          uCF( iCF_V, iX, 0 )   = V0;
          uCF( iCF_E, iX, 0 )   = eR * uCF( iCF_Tau, iX, 0 );

          uCR( iCR_E, iX, 0 ) = eR * uCF( iCF_Tau, iX, 0 );
        }
        uPF( iPF_D, iX, iNodeX ) = rhoR;

      }
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
