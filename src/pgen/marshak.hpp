#pragma once
/**
 * @file marshak.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Radiation marshak wave test
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
void marshak_init( State* state, GridStructure* grid, const ProblemIn* pin ) {
  View3D<Real> uCF = state->get_u_cf( );
  View3D<Real> uPF = state->get_u_pf( );
  View3D<Real> uCR = state->get_u_cr( );
  const int pOrder = state->get_p_order( );

  const int ilo    = grid->get_ilo( );
  const int ihi    = grid->get_ihi( );
  const int nNodes = grid->get_n_nodes( );

  // TODO(astrobarker) move these to a namespace like constants
  constexpr static int iCF_Tau = 0;
  constexpr static int iCF_V   = 1;
  constexpr static int iCF_E   = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 0;
  constexpr static int iCR_F = 1;

  auto su_olson_energy = [&]( const Real alpha, const Real T ) {
    return ( alpha / 4.0 ) * std::pow( T, 4.0 );
  };

  const Real V0 = pin->in_table["problem"]["params"]["vL"].value_or( 0.0 );
  const Real epsilon =
      pin->in_table["problem"]["params"]["epsilon"].value_or( 1.0 );
  const Real rho0 = pin->in_table["problem"]["params"]["rho0"].value_or( 10.0 );
  const Real T0 =
      pin->in_table["problem"]["params"]["T0"].value_or( 1.0e4 ); // K

  const Real alpha  = 4.0 * constants::a / epsilon;
  const Real em_gas = su_olson_energy( alpha, T0 ) / rho0;

  // TODO(astrobarker): thread through
  const Real em_rad = constants::a * std::pow( T0, 4.0 ) / rho0;

  for ( int iX = 0; iX <= ihi + 1; iX++ ) {
    for ( int k = 0; k < pOrder; k++ ) {
      for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
        uCF( iCF_Tau, iX, k ) = 0.0;
        uCF( iCF_V, iX, k )   = 0.0;
        uCF( iCF_E, iX, k )   = 0.0;
        uCR( iCR_E, iX, k )   = 0.0;
        uCR( iCR_F, iX, k )   = 0.0;

        if ( k == 0 ) {
          uCF( iCF_Tau, iX, 0 ) = 1.0 / rho0;
          uCF( iCF_V, iX, 0 )   = V0;
          uCF( iCF_E, iX, 0 )   = em_gas + 0.5 * V0 * V0;

          uCR( iCR_E, iX, 0 ) = em_rad;
        }
        uPF( iPF_D, iX, iNodeX ) = rho0;
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
