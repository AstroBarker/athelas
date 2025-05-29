#pragma once
/**
 * @file eos_base.hpp
 * --------------
 *
 * @author Brandon L. Barker
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
 *          - temperature_from_tau_pressure_abar
 *          - temperature_from_tau_pressure
 *          - radiation_pressure
 *
 *          These interfaces are implemented for all EOS
 **/

#include "abstractions.hpp"
#include "polynomial_basis.hpp"

template <class EOS>
class EosBase {
 public:
  auto pressure_from_conserved( const double tau, const double V, const double EmT,
                                double* lambda ) const -> double {
    return static_cast<EOS const*>( this )->pressure_from_conserved(
        tau, V, EmT, lambda );
  }
  auto sound_speed_from_conserved( const double tau, const double V, const double EmT,
                                   double* lambda ) const -> double {
    return static_cast<EOS const*>( this )->sound_speed_from_conserved(
        tau, V, EmT, lambda );
  }
  auto temperature_from_conserved( const double tau, const double V, const double EmT,
                                   double* lambda ) const -> double {
    return static_cast<EOS const*>( this )->temperature_from_conserved(
        tau, V, EmT, lambda );
  }
  auto temperature_from_tau_pressure_abar( const double tau, const double P,
                                           const double Abar, double* lambda ) const
      -> double {
    return static_cast<EOS const*>( this )->temperature_from_tau_pressure_abar(
        tau, P, Abar, lambda );
  }
  auto temperature_from_tau_pressure( const double tau, const double P,
                                      double* lambda ) const -> double {
    return static_cast<EOS const*>( this )->temperature_from_tau_pressure(
        tau, P, lambda );
  }
  auto radiation_pressure( const double T, double* lambda ) const -> double {
    return static_cast<EOS const*>( this )->radiation_pressure( T, lambda );
  }
};
