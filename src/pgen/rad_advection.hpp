#pragma once
/**
 * @file rad_advection.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation advection test
 */

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
  View3D<double> uCF = state->get_u_cf( );
  View3D<double> uPF = state->get_u_pf( );
  View3D<double> uCR = state->get_u_cr( );
  const int pOrder = state->get_p_order( );

  const int ilo    = grid->get_ilo( );
  const int ihi    = grid->get_ihi( );
  const int nNodes = grid->get_n_nodes( );

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;

  const int iPF_D = 0;

  const int iCR_E = 0;
  const int iCR_F = 1;

  const double V0  = pin->in_table["problem"]["params"]["v0"].value_or( 0.1 );
  const double p0  = pin->in_table["problem"]["params"]["p0"].value_or( 1.0e-6 );
  const double D   = pin->in_table["problem"]["params"]["rho"].value_or( 1.0 );
  const double amp = pin->in_table["problem"]["params"]["amp"].value_or( 1.0 );
  const double width =
      pin->in_table["problem"]["params"]["width"].value_or( 0.05 );

  const double mu = 1.0;// + constants::m_e / constants::m_p;
  const double gamma = 5.0 / 3.0;
  const double gm1 = gamma - 1.0;

  for ( int iX = 0; iX <= ihi + 1; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        double const X1         = grid->get_centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( 0, iX, k )       = 0.0;
        uCR( 1, iX, k )       = 0.0;

        if ( k == 0 ) {
          uCR( iCR_E, iX, k ) =
              amp * std::max( std::exp( -std::pow( ( X1 - 0.5 ) / width, 2.0 ) /
                                        2.0 ), 1.0e-8 );
              //amp * std::exp( -std::pow( ( X1 - 0.5 ) / width, 2.0 ) /
              //                          2.0 );
          uCR( iCR_F, iX, k ) = 1.0 * constants::c_cgs * uCR( iCR_E, iX, k );

          const double Trad = std::pow( uCR(iCR_E, iX, k) / constants::a, 0.25 );
          const double sie_fluid = 0.1;//constants::k_B * Trad / ( gm1 * mu * constants::m_p );
          std::println("T erad egad {} {} {}", Trad, uCR(iCR_E, iX, k), sie_fluid);
          uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
          uCF( iCF_V, iX, 0 )   = V0;
          uCF( iCF_E, iX, 0 )   = sie_fluid + 0.5 * V0 * V0;//p0 / (gamma - 1.0) / D + 0.5 * V0 * V0;
          //uCR( iCR_F, iX, k ) = -1.0e-12;
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
