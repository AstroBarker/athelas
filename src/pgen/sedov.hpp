#pragma once
/**
 * @file sedov.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Sedov blast wave
 */

#include <cmath>

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize sedov blast wave
 **/
void sedov_init( State* state, GridStructure* grid, const ProblemIn* pin ) {

  View3D<double> uCF = state->get_u_cf( );
  View3D<double> uPF = state->get_u_pf( );
  const int pOrder   = state->get_p_order( );

  const int ilo    = grid->get_ilo( );
  const int ihi    = grid->get_ihi( );
  const int nNodes = grid->get_n_nodes( );

  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  const double V0 = pin->in_table["problem"]["params"]["v0"].value_or( 0.0 );
  const double D0 = pin->in_table["problem"]["params"]["rho0"].value_or( 1.0 );
  const double E0 = pin->in_table["problem"]["params"]["E0"].value_or( 0.3 );

  const int origin = 1;

  // TODO(astrobarker): geometry aware volume for energy
  const double volume = ( 4.0 * M_PI / 3.0 ) *
                        std::pow( grid->get_left_interface( origin + 1 ), 3.0 );
  const double gamma = 5.0 / 3.0;
  const double P0    = ( gamma - 1.0 ) * E0 / volume;

  for ( int iX = ilo; iX <= ihi; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {

        if ( k != 0 ) {
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;
        } else {
          uCF( iCF_Tau, iX, k ) = 1.0 / D0;
          uCF( iCF_V, iX, k )   = V0;
          if ( iX == origin - 1 || iX == origin ) {
            uCF( iCF_E, iX, k ) =
                ( P0 / ( gamma - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                0.5 * V0 * V0;
          } else {
            uCF( iCF_E, iX, k ) =
                ( 1.0e-6 / ( gamma - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                0.5 * V0 * V0;
          }
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
