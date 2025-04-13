/**
 * @file eos_ideal.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Ideal gas equation of state
 *
 * @details A standard ideal gas EOS
 *            P = (\gamma - 1) u
 */

#include <cmath> /* sqrt */

#include "abstractions.hpp"
#include "constants.hpp"
#include "eos.hpp"
#include "polynomial_basis.hpp"

auto IdealGas::PressureFromConserved( const Real Tau, const Real V,
                                      const Real EmT, Real* /*lambda*/ ) const
    -> Real {
  const Real Em = EmT - ( 0.5 * V * V );
  const Real Ev = Em / Tau;
  return ( gamma - 1.0 ) * Ev;
}

auto IdealGas::SoundSpeedFromConserved( const Real /*Tau*/, const Real V,
                                        const Real EmT, Real* /*lambda*/ ) const
    -> Real {
  const Real Em = EmT - ( 0.5 * V * V );
  return std::sqrt( gamma * ( gamma - 1.0 ) * Em );
}

Real IdealGas::TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                               const Real Abar,
                                               Real* /*lambda*/ ) {
  return ( P * Abar * Tau ) / ( constants::N_A * constants::k_B );
}

auto IdealGas::TemperatureFromTauPressure( const Real Tau, const Real P,
                                           Real* lambda ) const -> Real {
  const Real Abar = 0.6;
  return TemperatureFromTauPressureAbar( Tau, P, Abar, lambda );
}

Real IdealGas::RadiationPressure( const Real T, Real* /*lambda*/ ) {
  return constants::a * T * T * T * T;
}
