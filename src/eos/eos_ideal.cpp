/**
 * @file eos_ideal.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Ideal gas equation of state
 *
 * @details A standard ideal gas EOS
 *            P = (\gamma - 1) u
 **/

#include <cmath> /* sqrt */

#include "abstractions.hpp"
#include "constants.hpp"
#include "eos.hpp"
#include "polynomial_basis.hpp"

[[nodiscard]] auto
IdealGas::pressure_from_conserved( const double tau, const double V, const double EmT,
                                   double* /*lambda*/ ) const -> double {
  const double Em = EmT - ( 0.5 * V * V );
  const double Ev = Em / tau;
  return ( gamma_ - 1.0 ) * Ev;
}

[[nodiscard]] auto
IdealGas::sound_speed_from_conserved( const double /*tau*/, const double V,
                                      const double EmT, double* /*lambda*/ ) const
    -> double {
  const double Em = EmT - ( 0.5 * V * V );
  return std::sqrt( gamma_ * ( gamma_ - 1.0 ) * Em );
}

[[nodiscard]] auto IdealGas::temperature_from_conserved( double tau, double V,
                                                         double E,
                                                         double* lambda ) const
    -> double {
  const double sie = E - 0.5 * V * V;
  const double mu = 
      1.0 + constants::m_e / constants::m_p; // TODO(astrobarker) generalize
  return ( gamma_ - 1.0 ) * sie * mu * constants::m_p / constants::k_B;
  //const double ev = sie / tau;
  //return std::pow(ev / constants::a, 0.25);
}

[[nodiscard]] auto
IdealGas::temperature_from_tau_pressure_abar( const double tau, const double P,
                                              const double Abar,
                                              double* /*lambda*/ ) const -> double {
  return ( P * Abar * tau ) / ( constants::N_A * constants::k_B );
}

[[nodiscard]] auto IdealGas::temperature_from_tau_pressure( const double tau,
                                                            const double P,
                                                            double* lambda ) const
    -> double {
  const double Abar = 1.0;
  return temperature_from_tau_pressure_abar( tau, P, Abar, lambda );
}

[[nodiscard]] auto IdealGas::radiation_pressure( const double T,
                                                 double* /*lambda*/ ) -> double {
  return constants::a * T * T * T * T;
}

[[nodiscard]] auto IdealGas::get_gamma( ) const noexcept -> double {
  return gamma_;
}
