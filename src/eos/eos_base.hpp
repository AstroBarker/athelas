#ifndef EOS_BASE_HPP_
#define EOS_BASE_HPP_
/**
 * @file eos_base.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Base class for equations of state using the Curiously Recurring
 *Template Pattern (CRTP)
 *
 * @details This header defines the EosBase template class that serves as the
 *foundation for all equation of state implementations in the codebase. It uses
 *the CRTP to provide a common interface while allowing derived classes to
 *implement specific EOS behaviors.
 *
 *          The class provides the following:
 *          - PressureFromConserved
 *          - SoundSpeedFromConserved
 *          - TemperatureFromTauPressureAbar
 *          - TemperatureFromTauPressure
 *          - RadiationPressure
 *
 *          These interfaces are implemented for all EOS
 *
 *          Each method is implemented as a non-virtual interface that delegates
 *to the derived class implementation through static_cast. This pattern allows
 *          for compile-time polymorphism with minimal runtime overhead.
 *
 **/

#include "abstractions.hpp"
#include "polynomial_basis.hpp"

template <class EOS>
class EosBase {
 public:
  Real PressureFromConserved( const Real Tau, const Real V, const Real EmT,
                              Real *lambda ) const {
    return static_cast<EOS const *>( this )->PressureFromConserved( Tau, V, EmT,
                                                                    lambda );
  }
  Real SoundSpeedFromConserved( const Real Tau, const Real V, const Real EmT,
                                Real *lambda ) const {
    return static_cast<EOS const *>( this )->SoundSpeedFromConserved(
        Tau, V, EmT, lambda );
  }
  Real TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                       const Real Abar, Real *lambda ) const {
    return static_cast<EOS const *>( this )->TemperatureFromTauPressureAbar(
        Tau, P, Abar, lambda );
  }
  Real TemperatureFromTauPressure( const Real Tau, const Real P,
                                   Real *lambda ) const {
    return static_cast<EOS const *>( this )->TemperatureFromTauPressure(
        Tau, P, lambda );
  }
  Real RadiationPressure( const Real T, Real *lambda ) const {
    return static_cast<EOS const *>( this )->RadiationPressure( T, lambda );
  }
};

#endif // EOS_BASE_HPP_
