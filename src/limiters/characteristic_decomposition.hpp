#ifndef CHARACTERISTIC_DECOMPOSITION_HPP_
#define CHARACTERISTIC_DECOMPOSITION_HPP_
/**
 * @file characteristic_decomposition.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Hydro characteristic decomposition
 *
 * @details Implements a characteristic decomposition of the hydro variables.
 *          Currently this is only implemented ofr an ideal EOS.
 *          TODO: Template on EOS? Write down for radiation.
 */

#include <iostream>
#include <math.h> /* sqrt */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "error.hpp"

template <class T1, class T2>
void compute_characteristic_decomposition( T1 U, T2 R, T2 R_inv ) {

  const Real tau  = U( 0 );
  const Real V    = U( 1 );
  const Real Em_T = U( 2 );

  // const Real P  = ComputePressureFromConserved_IDEAL( tau, V, Em_T );
  // const Real Cs = ComputeSoundSpeedFromConserved_IDEAL( tau, V, Em_T );

  const Real Em = Em_T - ( 0.5 * V * V );

  const float GAMMA = 1.4;

  const Real k      = std::sqrt( GAMMA * ( GAMMA - 1.0 ) );
  const Real sqrt_e = std::sqrt( Em );
  const Real InvTau = 1.0 / tau;

  /* --- Thermodynamic Derivatives of Pressure --- */

  // const Real P_Tau = - (GAMMA - 1.0) * Em / ( tau * tau );
  // const Real P_Em  = + (GAMMA - 1.0) / tau;
  // const Real P_T_E = - Em / tau; // ratio of derivatives

  // Eigenvalues are rho * Cs...
  // const Real lam = std::sqrt( P * P_Em - P_Tau );

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

#endif // CHARACTERISTIC_DECOMPOSITION_HPP_
