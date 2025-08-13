
#include <cmath>

#include "constants.hpp"
#include "eos.hpp"

[[nodiscard]] auto
Polytropic::pressure_from_conserved(const double tau, const double /*V*/,
                                  const double /*EmT*/, double* /*lambda*/) const
    -> double {
  return k_ * std::pow((1.0 / tau), 1.0 + 1.0 / n_);
}

[[nodiscard]] auto
Polytropic::sound_speed_from_conserved(const double tau, const double /*V*/,
                                     const double /*EmT*/, double* /*lambda*/) const
    -> double {
  return std::sqrt((1.0 + 1.0 / n_) * k_ * std::pow((1.0 / tau), 1.0 / n_));
}

// Assuming this polytrope is in an ideal gas!
[[nodiscard]] auto Polytropic::temperature_from_conserved(double tau, double V,
                                                        double E,
                                                        double* lambda) const
    -> double {
  const double p = pressure_from_conserved(tau, V, E, lambda);
  const double mu =
      1.0 + constants::m_e / constants::m_p; // TODO(astrobarker) generalize
  return tau * p * mu * constants::m_p / constants::k_B;
}

[[nodiscard]] auto Polytropic::get_gamma() const noexcept -> double {
  return 1.0 + 1.0 / n_; // Gamma_2
}
