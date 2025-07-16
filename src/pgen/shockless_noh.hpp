#pragma once
/**
 * @file shockless_noh.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Shockless Noh collapse
 */

#include "abstractions.hpp"
#include "grid.hpp"
#include "state.hpp"

/**
 * @brief Initialize shockless Noh problem
 **/
void shockless_noh_init( State* state, GridStructure* grid,
                         const ProblemIn* pin ) {

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

  const double D = pin->in_table["problem"]["params"]["rho"].value_or( 1.0 );
  const double E_M =
      pin->in_table["problem"]["params"]["specific_energy"].value_or( 1.0 );

  double X1 = 0.0;
  for ( int iX = ilo; iX <= ihi; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        X1                    = grid->get_centers( iX );
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;

        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
          uCF( iCF_V, iX, 0 )   = -X1;
          uCF( iCF_E, iX, 0 ) =
              E_M + 0.5 * uCF( iCF_V, iX, 0 ) * uCF( iCF_V, iX, 0 );
        } else if ( k == 1 ) {
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = -grid->get_widths( iX );
          uCF( iCF_E, iX, k )   = ( -X1 ) * ( -grid->get_widths( iX ) );
        } else if ( k == 2 ) {
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = uCF( iCF_V, iX, 1 ) * uCF( iCF_V, iX, 1 );
        } else {
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;
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
