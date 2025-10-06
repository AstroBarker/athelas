#include <cmath>

#include "eos/eos.hpp"
#include "utils/constants.hpp"

namespace athelas::eos {

[[nodiscard]] auto IdealGas::pressure_from_conserved(
    const double tau, const double V, const double EmT,
    const double *const /*lambda*/) const -> double {
  const double Em = EmT - (0.5 * V * V);
  const double Ev = Em / tau;
  return (gamma_ - 1.0) * Ev;
}

[[nodiscard]] auto IdealGas::sound_speed_from_conserved(
    const double /*tau*/, const double V, const double EmT,
    const double *const /*lambda*/) const -> double {
  const double Em = EmT - (0.5 * V * V);
  return std::sqrt(gamma_ * (gamma_ - 1.0) * Em);
}

[[nodiscard]] auto IdealGas::temperature_from_conserved(
    const double /*tau*/, const double V, const double E,
    const double *const /*lambda*/) const -> double {
  const double sie = E - 0.5 * V * V;
  const double mu =
      1.0 + constants::m_e / constants::m_p; // TODO(astrobarker) generalize
  return (gamma_ - 1.0) * sie * mu * constants::m_p / constants::k_B;
}

[[nodiscard]] auto
IdealGas::get_gamma(const double /*tau*/, const double /*V*/,
                    const double /*EmT*/,
                    const double *const /*lambda*/) const noexcept -> double {
  return get_gamma();
}

[[nodiscard]] auto IdealGas::get_gamma() const noexcept -> double {
  return gamma_;
}

} // namespace athelas::eos
