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
  explicit IdealGas( double gm ) : gamma( gm ) {
    if ( gamma <= 0.0 ) {
      THROW_ATHELAS_ERROR( " ! Adiabatic gamma <= 0.0!" );
    }
  }

  auto PressureFromConserved( Real Tau, Real V, Real EmT, Real* lambda ) const
      -> Real;
  auto SoundSpeedFromConserved( Real Tau, Real V, Real EmT, Real* lambda ) const
      -> Real;
  static auto TemperatureFromTauPressureAbar( Real Tau, Real P, Real Abar,
                                              Real* lambda ) -> Real;
  auto TemperatureFromTauPressure( Real Tau, Real P, Real* lambda ) const
      -> Real;
  static auto RadiationPressure( Real T, Real* lambda ) -> Real;

 private:
  Real gamma{ };
};

/* placeholder */
class Stellar : public EosBase<Stellar> {
 public:
  Stellar( ) = default;

  auto PressureFromConserved( Real Tau, Real V, Real EmT, Real* lambda ) const
      -> Real;
  auto SoundSpeedFromConserved( Real Tau, Real V, Real EmT, Real* lambda ) const
      -> Real;
  auto TemperatureFromTauPressureAbar( Real Tau, Real P, Real Abar,
                                       Real* lambda ) const -> Real;
  auto TemperatureFromTauPressure( Real Tau, Real P, Real* lambda ) const
      -> Real;
  auto RadiationPressure( Real T, Real* lambda ) const -> Real;

 private:
  Real gamma{ };
};

// TODO(astrobarker): adjust when we support more than one EOS
using EOS = IdealGas;

#endif // EOS_HPP_
