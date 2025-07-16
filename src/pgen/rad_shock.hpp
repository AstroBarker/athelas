#pragma once
/**
 * @file rad_shock.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation shock test
 */

#include <cmath> /* sin */

#include "abstractions.hpp"
#include "constants.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize radiating shock
 **/
void rad_shock_init( State* state, GridStructure* grid, const ProblemIn* pin ) {
  View3D<double> uCF = state->get_u_cf( );
  View3D<double> uPF = state->get_u_pf( );
  View3D<double> uCR = state->get_u_cr( );
  const int pOrder   = state->get_p_order( );

  const int ilo    = grid->get_ilo( );
  const int ihi    = grid->get_ihi( );
  const int nNodes = grid->get_n_nodes( );

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 0;

  const double V_L =
      pin->in_table["problem"]["params"]["vL"].value_or( 5.19e7 );
  const double V_R =
      pin->in_table["problem"]["params"]["vR"].value_or( 1.73e7 );
  const double rhoL =
      pin->in_table["problem"]["params"]["rhoL"].value_or( 5.69 );
  const double rhoR =
      pin->in_table["problem"]["params"]["rhoR"].value_or( 17.1 );
  const double T_L =
      pin->in_table["problem"]["params"]["T_L"].value_or( 2.18e6 ); // K
  const double T_R =
      pin->in_table["problem"]["params"]["T_R"].value_or( 7.98e6 ); // K
  const double x_d =
      pin->in_table["problem"]["params"]["x_d"].value_or( 0.013 );

  // TODO(astrobarker): thread through
  const double mu       = 1.0 + constants::m_e / constants::m_p;
  const double gamma    = 5.0 / 3.0;
  const double gm1      = gamma - 1.0;
  const double em_gas_L = constants::k_B * T_L / ( gm1 * mu * constants::m_p );
  const double em_gas_R = constants::k_B * T_R / ( gm1 * mu * constants::m_p );
  const double e_rad_L  = constants::a * std::pow( T_L, 4.0 );
  const double e_rad_R  = constants::a * std::pow( T_R, 4.0 );

  for ( int iX = 0; iX <= ihi + 1; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        double X1             = grid->get_centers( iX );
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

            uCR( iCR_E, iX, 0 ) = e_rad_L;
          }
          uPF( iPF_D, iX, iNodeX ) = rhoL;
        } else {
          if ( k == 0 ) {
            uCF( iCF_Tau, iX, 0 ) = 1.0 / rhoR;
            uCF( iCF_V, iX, 0 )   = V_R;
            uCF( iCF_E, iX, 0 )   = em_gas_R + 0.5 * V_R * V_R;

            uCR( iCR_E, iX, 0 ) = e_rad_R;
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
