#ifndef EOS_HPP_
#define EOS_HPP_
/**
 * @file eos.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Declares equation of state classes that implement the EosBase
 * interface
 *
 * @details Defines specific equation of state implementations that inherit
 *          from the EosBase template class. It serves as the central
 * declaration point for all EOS classes in the codebase, with their
 * implementations provided in separate .cpp files.
 *
 *          We support the following equations of state:
 *          - IdealGas (default): ideal gas EOS
 *          - Stellar: currently an unused placceholder
 *
 *          Note: implementations are supplied in eos-specific cpp files.
 */

#include <variant>

#include "abstractions.hpp"
#include "eos_base.hpp"
#include "error.hpp"

class IdealGas : public EosBase<IdealGas> {
 public:
  IdealGas( ) = default;
  explicit IdealGas( double gm ) : gamma_( gm ) {
    if ( gamma_ <= 0.0 ) {
      THROW_ATHELAS_ERROR( " ! IdealGas :: Adiabatic gamma <= 0.0!" );
    }
  }

  auto pressure_from_conserved( Real tau, Real V, Real EmT, Real* lambda ) const
      -> Real;
  auto sound_speed_from_conserved( Real tau, Real V, Real EmT,
                                   Real* lambda ) const -> Real;
  auto temperature_from_tau_pressure_abar( Real tau, Real P, Real Abar,
                                                  Real* lambda ) const -> Real;
  auto temperature_from_tau_pressure( Real tau, Real P, Real* lambda ) const
      -> Real;
  static auto radiation_pressure( Real T, Real* lambda ) -> Real;
  auto get_gamma( ) const noexcept -> Real;

 private:
  Real gamma_{ };
};

/* placeholder */
class Stellar : public EosBase<Stellar> {
 public:
  Stellar( ) = default;

  auto pressure_from_conserved( Real tau, Real V, Real EmT, Real* lambda ) const
      -> Real;
  auto sound_speed_from_conserved( Real tau, Real V, Real EmT,
                                   Real* lambda ) const -> Real;
  auto temperature_from_tau_pressure_abar( Real tau, Real P, Real Abar,
                                           Real* lambda ) const -> Real;
  auto temperature_from_tau_pressure( Real tau, Real P, Real* lambda ) const
      -> Real;
  auto radiation_pressure( Real T, Real* lambda ) const -> Real;

 private:
  Real gamma_{ };
};

// TODO(astrobarker): adjust when we support more than one EOS
using EOS = IdealGas;

#endif // EOS_HPP_
