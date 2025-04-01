/**
 * File     :  rad_utilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for radiation fields. Includes Riemann solvers.
 **/

#include <algorithm> // std::min, std::max
#include <cmath> // pow, abs, sqrt
#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "constants.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"
#include "riemann.hpp"
#include "utilities.hpp"

/**
 * radiation flux factor
 **/
Real FluxFactor( const Real E, const Real F ) {
  assert( E > 0.0 &&
          "Radiation :: FluxFactor :: non positive definite energy density." );
  constexpr static Real c = constants::c_cgs;
  return ( F ) / ( c * E );
}

/**
 * The radiation fluxes
 * Here E and F are per unit volume
 **/
Real Flux_Rad( Real E, Real F, Real P, Real V, int iCR ) {
  assert( ( iCR == 0 || iCR == 1 ) && "Radiation :: FluxFactor :: bad iCR." );
  assert( E > 0.0 &&
          "Radiation :: FluxFactor :: non positive definite energy density." );

  constexpr static Real c = constants::c_cgs;
  return ( iCR == 0 ) ? 1.0 * F - E * V : c * c * P - F * V;
}

/**
 * Radiation 4 force for rad-matter interactions
 * D : Density
 * V : Velocity
 * T : Temperature
 * kappa : kappa
 * E : radiation energy density
 * F : radiation momentum density
 * Pr : radiation momentum closure
 * TODO: total opacity X?
 **/
void RadiationFourForce( Real D, Real V, Real T, Real kappa, Real E, Real F,
                         Real Pr, Real &G0, Real &G ) {
  assert(
      D >= 0.0 &&
      "Radiation :: RadiationFourFource :: Non positive definite density." );
  assert( T > 0.0 &&
          "Radiation :: RadiationFourFource :: Non positive temperature." );
  assert( kappa > 0.0 &&
          "Radiation :: RadiationFourFource :: Non positive opacity." );
  assert( E > 0.0 && "Radiation :: RadiationFourFource :: Non positive "
                     "definite radiation energy density." );

  constexpr Real a = constants::a;
  constexpr Real c = constants::c_cgs;

  const Real b     = V / c;
  const Real term1 = E - a * T * T * T * T;
  F /= c;

  // O(b^2) ala Fuksman
  G0 = D * kappa * ( term1 - b * F - b * b * E - b * b * Pr );
  G  = D * kappa * ( b * ( term1 - 2.0 * b * F ) + ( F - b * E - b * Pr ) );

  // ala Skinner & Ostriker, simpler.
  // G0 = D * kappa * ( term1 - b * F );
  // G  = D * kappa * ( F - b * E + b * Pr );
}

/**
 * source terms for radiation
 * TODO: total opacity X
 **/
Real Source_Rad( Real D, Real V, Real T, Real X, Real kappa, Real E, Real F,
                 Real Pr, int iCR ) {
  assert( ( iCR == 0 || iCR == 1 ) && "Radiation :: source_rad :: bad iCR." );
  assert( D >= 0.0 &&
          "Radiation :: source_rad :: Non positive definite density." );
  assert( T > 0.0 && "Radiation :: source_rad :: Non positive temperature." );
  assert( E > 0.0 && "Radiation :: source_rad :: Non positive "
                     "definite radiation energy density." );

  const Real c = constants::c_cgs;

  Real G0, G;
  RadiationFourForce( D, V, T, kappa, E, F, Pr, G0, G );

  return ( iCR == 0 ) ? -c * G0 : -c * c * G;
}

/**
 * Emissivity
 * TODO: actually implement this
 **/
Real ComputeEmissivity( const Real D, const Real V, const Real Em ) {
  return ComputeOpacity( D, V, Em );
}

/**
 * Opacity kappa
 * TODO: actually implement this
 **/
Real ComputeOpacity( const Real D, const Real V, const Real Em ) {
  return 4.0 * std::pow( 10.0, -8.0 ) * D;
}

/* pressure tensor closure */
// TODO: check Closure
Real ComputeClosure( const Real E, const Real F ) {
  assert( E > 0.0 && "Radiation :: ComputeClosure :: Non positive definite "
                     "radiation energy density." );
  const Real f = FluxFactor( E, F );
  const Real chi =
      ( 3.0 + 4.0 * f * f ) / ( 5.0 + 2.0 * std::sqrt( 4.0 - 3.0 * f * f ) );
  const Real T = ( 1.0 - chi ) / 2.0 + ( 3.0 * chi - 1.0 ) *
                                           1.0 / // utilities::sgn(F)
                                           2.0; // TODO: Is this right?
  return E * T;
}

void llf_flux( const Real Fp, const Real Fm, const Real Up, const Real Um,
               const Real alpha, Real &out ) {
  out = 0.5 * ( Fp - alpha * Up + Fm + alpha * Um );
}

/**
 * eigenvalues of JAcobian for radiation solve
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 41a,b
 * and references therein
 **/
Real Lambda_HLL( const Real f, const int sign ) {
  constexpr Real c        = constants::c_cgs;
  constexpr Real twothird = 2.0 / 3.0;

  const Real f2       = f * f;
  const Real sqrtterm = std::sqrt( 4.0 - 3.0 * f2 );
  return c *
         ( f + sign * std::sqrt( twothird * ( 4.0 - 3.0 * f2 - sqrtterm ) +
                                 2.0 * ( 2.0 - f2 - sqrtterm ) ) ) /
         sqrtterm;
}

/**
 * HLL Riemann solver for radiation
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 39
 * and references & discussion therein
 *
 * Note: pass in Eulerian varaibles ( _ / cm^3 )
 **/
void numerical_flux_hll_rad( const Real E_L, const Real E_R, const Real F_L,
                             const Real F_R, const Real P_L, const Real P_R,
                             Real &Flux_E, Real &Flux_F ) {
  // flux factors
  const Real f_L = FluxFactor( E_L, F_L );
  const Real f_R = FluxFactor( E_R, F_R );

  // eigenvalues
  const Real s_r_p = std::max(
      std::max( Lambda_HLL( f_L, 1.0 ), Lambda_HLL( f_R, 1.0 ) ), 0.0 );
  const Real s_l_m = std::min(
      std::min( Lambda_HLL( f_L, -1.0 ), Lambda_HLL( f_R, -1.0 ) ), 0.0 );

  Flux_E = hll( E_L, E_R, F_L, F_R, s_l_m, s_r_p );
  Flux_F = hll( F_L, F_R, P_L, P_R, s_l_m, s_r_p );
}
