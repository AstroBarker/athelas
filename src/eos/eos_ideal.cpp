/**
 * File     :  eos_ideal.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : ideal equation of state routines
 **/

#include <math.h> /* sqrt */

#include "abstractions.hpp"
#include "constants.hpp"
#include "eos.hpp"
#include "polynomial_basis.hpp"

Real IdealGas::PressureFromConserved( const Real Tau, const Real V,
                                      const Real EmT, Real *lambda ) const {
  const Real Em = EmT - 0.5 * V * V;
  const Real Ev = Em / Tau;
  return ( gamma - 1.0 ) * Ev;
}

Real IdealGas::SoundSpeedFromConserved( const Real Tau, const Real V,
                                        const Real EmT, Real *lambda ) const {
  const Real Em = EmT - 0.5 * V * V;
  return std::sqrt( gamma * ( gamma - 1.0 ) * Em );
}

Real IdealGas::TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                               const Real Abar,
                                               Real *lambda ) const {
  return ( P * Abar * Tau ) / ( constants::N_A * constants::k_B );
}

Real IdealGas::TemperatureFromTauPressure( const Real Tau, const Real P,
                                           Real *lambda ) const {
  const Real Abar = 1.0;
  return TemperatureFromTauPressureAbar( Tau, P, Abar, lambda );
}

Real IdealGas::RadiationPressure( const Real T, Real *lambda ) const {
  return constants::a * T * T * T * T;
}
