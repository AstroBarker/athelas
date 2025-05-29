#pragma once
/**
 * @file characteristic_decomposition.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Hydro characteristic decomposition
 *
 * @details Implements a characteristic decomposition of the hydro variables.
 *          Currently this is only implemented ofr an ideal EOS.
 *          TODO(astrobarker): Write down for radiation.
 */

#include <iostream>
#include <math.h> /* sqrt */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "error.hpp"

template <class T1, class T2, class EOS>
void compute_characteristic_decomposition( T1 U, T2 R, T2 R_inv, EOS eos ) {

  const double tau  = U( 0 );
  const double V    = U( 1 );
  const double Em_T = U( 2 );

  // const double P  = ComputePressureFromConserved_IDEAL( tau, V, Em_T );
  // const double Cs = ComputeSoundSpeedFromConserved_IDEAL( tau, V, Em_T );

  const double Em = Em_T - ( 0.5 * V * V );

  const double GAMMA = eos->get_gamma( );

  const double k      = std::sqrt( GAMMA * ( GAMMA - 1.0 ) );
  const double sqrt_e = std::sqrt( Em );
  const double InvTau = 1.0 / tau;

  /* --- Thermodynamic Derivatives of Pressure --- */

  // const double P_Tau = - (GAMMA - 1.0) * Em / ( tau * tau );
  // const double P_Em  = + (GAMMA - 1.0) / tau;
  // const double P_T_E = - Em / tau; // ratio of derivatives

  // Eigenvalues are rho * Cs...
  // const double lam = std::sqrt( P * P_Em - P_Tau );

  /*  --- Compute Matrix Elements --- */

  for ( int i = 0; i < 3; i++ ) {
    R( 0, i ) = 1.0;
  }

  R( 1, 0 ) = +sqrt_e * k * InvTau;
  R( 1, 1 ) = +0.0;
  R( 1, 2 ) = -sqrt_e * k * InvTau;

  R( 2, 0 ) = ( Em + sqrt_e * k * V - Em * GAMMA ) * InvTau;
  R( 2, 1 ) = ( Em * InvTau );
  R( 2, 2 ) = ( Em - sqrt_e * k * V - Em * GAMMA ) * InvTau;

  R_inv( 0, 0 ) = 0.5;
  R_inv( 0, 1 ) = tau * ( k * V + sqrt_e * GAMMA ) / ( 2.0 * Em * k );
  R_inv( 0, 2 ) = -tau / ( 2.0 * Em );

  R_inv( 1, 0 ) = GAMMA - 1.0;
  R_inv( 1, 1 ) = -V * tau / Em;
  R_inv( 1, 2 ) = tau / Em;

  R_inv( 2, 0 ) = 0.5;
  R_inv( 2, 1 ) = tau * ( k * V - sqrt_e * GAMMA ) / ( 2.0 * Em * k );
  R_inv( 2, 2 ) = -tau / ( 2.0 * Em );

  for ( int i = 0; i < 3; i++ ) {
    for ( int j = 0; j < 3; j++ ) {
      R_inv( i, j ) /= GAMMA;
    }
  }
}
