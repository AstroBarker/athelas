/**
 * File     :  RadUtilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for radiation fields. Includes Riemann solvers.
 **/

#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "EoS.hpp"
#include "Constants.hpp"
#include "Error.hpp"
#include "PolynomialBasis.hpp"
#include "RadUtilities.hpp"
#include "Utilities.hpp"

/** 
 * radiation flux factor
 **/
Real FluxFactor( const Real E, const Real F ) {
  const Real c = constants::c_cgs;
  return std::fabs( F ) / ( c * E );
}

/**
 * The radiation fluxes
 * Here E and F are per unit mass
 **/
Real Flux_Rad( Real E, Real F, Real P, Real V, UInt iRF ) {
  assert ( iRF == 0 || iRF == 1 );

  if ( iRF == 0 ) {
    return F - E * V;
  } else {
    return P - F * V;
  }
}

/**
 * source terms for radiation
 **/
Real Source_Rad( Real D, Real V, Real T, Real X, Real kappa, 
                 Real E, Real F, Real Pr, UInt iRF ) {
  assert ( iRF == 0 || iRF == 1 );

  const Real a = constants::a;
  const Real c = constants::c_cgs;

  const Real b = V / c;
  const Real term1 = E - a * T*T*T*T - 2.0 * b * F;
  const Real term2 = F - E * b - b * Pr;

  if ( iRF == 0 ) {
    return - ( D * kappa * term1 + D * X * b * term2 );
  } else {
    return - ( D * kappa * term1 * b + D * X * term2 );
  }
}

/**
 * Emissivity chi
 * TODO: actually implement this
 **/
Real ComputeEmissivity( const Real D, const Real V, const Real Em ) {
  return 1.0;
}

/**
 * Opacity kappa
 * TODO: actually implement this
 **/
Real ComputeOpacity( const Real D, const Real V, const Real Em ) {
  return 1.0;
}

/* pressure tensor closure */
// TODO: check Closure
Real ComputeClosure( const Real E, const Real F ) {
  if (E == 0.0) return 0.0; // This is a hack
  const Real f = FluxFactor( E, F );
  const Real chi = ( 3.0 + 4.0 * f * f ) 
    / ( 5.0 + 2.0 * std::sqrt( 4.0 - 3.0 * f * f ) );
  const Real T = ( 1.0 - chi ) / 2.0 + ( 3.0 * chi - 1.0) * sgn( F ) / 2.0; // TODO: Is this right?
  return E * T;
}

void llf_flux( const Real Fp, const Real Fm, const Real Up, const Real Um, const Real alpha, Real &out ) {
  out = 0.5 * ( Fp - alpha * Up + Fm + alpha * Um );
}

/**
 * eigenvalues of JAcobian for radiation solve
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 41a,b
 * and references therein
 **/
Real Lambda_HLL( const Real f, const int sign ) { 
  const Real twothird = 2.0 / 3.0;
  const Real f2 = f * f;
  const Real sqrtterm = std::sqrt( 4.0 - 3.0 * f2 );
  return ( f + sign * std::sqrt(twothird * (4.0 - 3.0 * f2 - sqrtterm) 
         + 2.0 * ( 2.0 - f2 - sqrtterm )) ) / sqrtterm;
}

/* HLL Riemann solver for radiation */
void NumericalFlux_HLL_Rad( const Real E_L, const Real E_R, const Real F_L,
                            const Real F_R, const Real P_L, const Real P_R, 
                            const Real V_L, const Real V_R, 
                            Real &Flux_E, Real &Flux_F ) {
  // flux factors
  const Real f_L = FluxFactor( E_L, F_L );
  const Real f_R = FluxFactor( E_R, F_R );

  // eigenvalues
  const Real S_r_p = std::max( std::max( Lambda_HLL ( f_L, 1.0 ), 
        Lambda_HLL( f_R, 1.0 ) ), 0.0);
  const Real S_l_m = std::min( std::min( Lambda_HLL ( f_L, - 1.0 ), 
        Lambda_HLL( f_R, - 1.0 ) ), 0.0);
  
  // TODO: what flux to use in the Riemann solver? "Normal," or "Lagrangian"?
  // For now: using mine.
  const Real flux_E_L = 0.0;
  const Real flux_E_R = 0.0;
  const Real flux_F_L = 0.0;
  const Real flux_F_R = 0.0;

  Flux_E = 0.0;
  Flux_F = 0.0;
}
