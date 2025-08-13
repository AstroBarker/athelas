#pragma once

/**
 * @class EosBase
 * @brief Base class for equations of state using the Curiously Recurring
 *Template Pattern (CRTP)
 *
 * @details This header defines the EosBase template class that serves as the
 * foundation for all equation of state implementations in the codebase. It uses
 * the CRTP to provide a common interface while allowing derived classes to
 * implement specific EOS behaviors.
 *
 *          The class provides the following:
 *          - pressure_from_conserved
 *          - sound_speed_from_conserved
 *          - temperature_from_conserved
 *
 *          These interfaces are implemented for all EOS
 */
template <class EOS>
class EosBase {
 public:
  auto pressure_from_conserved(const double tau, const double V,
                               const double EmT, double* lambda) const
      -> double {
    return static_cast<EOS const*>(this)->pressure_from_conserved(tau, V, EmT,
                                                                  lambda);
  }
  auto sound_speed_from_conserved(const double tau, const double V,
                                  const double EmT, double* lambda) const
      -> double {
    return static_cast<EOS const*>(this)->sound_speed_from_conserved(
        tau, V, EmT, lambda);
  }
  auto temperature_from_conserved(const double tau, const double V,
                                  const double EmT, double* lambda) const
      -> double {
    return static_cast<EOS const*>(this)->temperature_from_conserved(
        tau, V, EmT, lambda);
  }
  auto get_gamma() -> double {
    return static_cast<EOS const*>(this)->get_gamma();
  }
};
