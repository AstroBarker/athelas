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

auto IdealGas::pressure_from_conserved( const Real tau, const Real V,
                                        const Real EmT, Real* /*lambda*/ ) const
    -> Real {
  const Real Em = EmT - ( 0.5 * V * V );
  const Real Ev = Em / tau;
  return ( gamma_ - 1.0 ) * Ev;
}

auto IdealGas::sound_speed_from_conserved( const Real /*tau*/, const Real V,
                                           const Real EmT,
                                           Real* /*lambda*/ ) const -> Real {
  const Real Em = EmT - ( 0.5 * V * V );
  return std::sqrt( gamma_ * ( gamma_ - 1.0 ) * Em );
}

auto IdealGas::temperature_from_tau_pressure_abar( const Real tau, const Real P,
                                                   const Real Abar,
                                                   Real* /*lambda*/ ) const -> Real {
  return ( P * Abar * tau ) / ( constants::N_A * constants::k_B );
}

auto IdealGas::temperature_from_tau_pressure( const Real tau, const Real P,
                                              Real* lambda ) const -> Real {
  const Real Abar = 0.6;
  return temperature_from_tau_pressure_abar( tau, P, Abar, lambda );
}

auto IdealGas::radiation_pressure( const Real T, Real* /*lambda*/ ) -> Real {
  return constants::a * T * T * T * T;
}

auto IdealGas::get_gamma() const noexcept -> Real {return gamma_;}
