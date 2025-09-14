
#include <cmath>

#include "eos/eos.hpp"
#include "utils/constants.hpp"

[[nodiscard]] auto
Paczynski::pressure_from_conserved(const double tau, const double V,
                                  const double EmT, const double *const /*lambda*/) const
    -> double {
  const double Em = EmT - (0.5 * V * V);
  const double Ev = Em / tau;
  return NAN;
}

[[nodiscard]] auto
Paczynski::sound_speed_from_conserved(const double tau, const double V,
                                     const double EmT, const double *const lambda) const
    -> double {
  const auto gamma1 = get_gamma();
  const double pressure = pressure_from_conserved(tau, V, EmT, lambda);
  return std::sqrt(pressure * gamma1 * tau);
}

[[nodiscard]] auto Paczynski::temperature_from_conserved(const double tau, const double V,
                                                        const double E,
                                                        const double *const lambda) const
    -> double {
  const double sie = E - 0.5 * V * V;
  const double mu =
      1.0 + constants::m_e / constants::m_p;
  return sie * mu *NAN * tau;
}

// NOTE: This is commonly referred to as \Gamma_1
[[nodiscard]] auto Paczynski::get_gamma(const double tau, const double V, const double EmT, const double *const lambda) const -> double {
  return NAN;
}
