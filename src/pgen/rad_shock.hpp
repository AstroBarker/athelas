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
void rad_shock_init( State* state, GridStructure* grid,
                           const ProblemIn* pin ) {
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

  const Real V_L = pin->in_table["problem"]["params"]["vL"].value_or( 5.19e7 );
  const Real V_R = pin->in_table["problem"]["params"]["vR"].value_or( 1.73e7 );
  const Real rhoL =
      pin->in_table["problem"]["params"]["rhoL"].value_or( 5.69 );
  const Real rhoR =
      pin->in_table["problem"]["params"]["rhoR"].value_or( 17.1 );
  const Real T_L = pin->in_table["problem"]["params"]["T_L"].value_or(
      2.18e6 ); // K
  const Real T_R = pin->in_table["problem"]["params"]["T_R"].value_or(
       7.98e6); // K
  const Real x_d = pin->in_table["problem"]["params"]["x_d"].value_or(0.013);

  // TODO(astrobarker): thread through
  const Real mu = 1.0 + constants::m_e / constants::m_p;
  const Real gamma = 5.0 / 3.0;
  const Real gm1 = gamma - 1.0;
  const Real em_gas_L = constants::k_B * T_L / (gm1 * mu * constants::m_p);
  const Real em_gas_R = constants::k_B * T_R / (gm1 * mu * constants::m_p);
  const Real em_rad_L = constants::a * std::pow(T_L, 4.0) / rhoL;
  const Real em_rad_R = constants::a * std::pow(T_R, 4.0) / rhoR;

  for ( int iX = 0; iX <= ihi + 1; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        Real X1                    = grid->get_centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k )       = 0.0;
        uCR( 1, iX, k )       = 0.0;

        if ( X1 <= x_d ) {
        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / rhoL;
          uCF( iCF_V, iX, 0 )   = V_L;
          uCF( iCF_E, iX, 0 )   = em_gas_L + 0.5 * V_L * V_L;

          uCR( iCR_E, iX, 0 ) = em_rad_L;
        }
        uPF( iPF_D, iX, iNodeX ) = rhoL;
        } else {
        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / rhoR;
          uCF( iCF_V, iX, 0 )   = V_R;
          uCF( iCF_E, iX, 0 )   = em_gas_R + 0.5 * V_R * V_R;

          uCR( iCR_E, iX, 0 ) = em_rad_R;
        }
        uPF( iPF_D, iX, iNodeX ) = rhoR;

      }
        //std::println("erad {:10e} ", uCR(0,iX, 0));
        std::println("egas {:10e} ", uCF(2,iX, 0));
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
