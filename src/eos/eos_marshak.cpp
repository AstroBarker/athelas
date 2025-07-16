/**
 * @file eos_marshak.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Marshak equation of state
 *
 * @details A standard ideal gas EOS with a weird temperature
 *            P = (\gamma - 1) u
 *            T = (E_r/a)^(1/4)
 **/

#include <cmath>

#include "constants.hpp"
#include "eos.hpp"

[[nodiscard]] auto
Marshak::pressure_from_conserved( const double tau, const double V,
                                  const double EmT, double* /*lambda*/ ) const
    -> double {
  const double Em = EmT - ( 0.5 * V * V );
  const double Ev = Em / tau;
  return ( gamma_ - 1.0 ) * Ev;
}

[[nodiscard]] auto
Marshak::sound_speed_from_conserved( const double /*tau*/, const double V,
                                     const double EmT,
                                     double* /*lambda*/ ) const -> double {
  const double Em = EmT - ( 0.5 * V * V );
  return std::sqrt( gamma_ * ( gamma_ - 1.0 ) * Em );
}

[[nodiscard]] auto Marshak::temperature_from_conserved( double tau, double V,
                                                        double E,
                                                        double* lambda ) const
    -> double {
  const double sie = E - 0.5 * V * V;
  const double ev  = sie / tau;
  return std::pow( ev / constants::a, 0.25 );
}

[[nodiscard]] auto Marshak::get_gamma( ) const noexcept -> double {
  return gamma_;
}
