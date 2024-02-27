#ifndef _EOS_BASE_HPP_
#define _EOS_BASE_HPP_

/**
 * define a base class using curiously recurring template pattern
 **/
#include "Abstractions.hpp"
#include "PolynomialBasis.hpp"

template <class EOS>
class EosBase {
 public:
  Real PressureFromConserved( const Real Tau, const Real V, const Real EmT ) const {
    return static_cast<EOS const *>( this )->PressureFromConserved( Tau, V, EmT );
  }
  Real SoundSpeedFromConserved( const Real Tau, const Real V, const Real EmT ) const {
    return static_cast<EOS const *>( this )->SoundSpeedFromConserved( Tau, V, EmT );
  }
  Real TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                       const Real Abar ) const {
    return static_cast<EOS const *>( this )->TemperatureFromTauPressureAbar( Tau, P,
                                                                      Abar );
  }
  Real TemperatureFromTauPressure( const Real Tau, const Real P ) const {
    return static_cast<EOS const *>( this )->TemperatureFromTauPressure( Tau, P );
  }
  Real RadiationPressure( const Real T ) const {
    return static_cast<EOS const *>( this )->RadiationPressure( T );
  }
  Real ComputeInternalEnergy( const View3D U, const ModalBasis *Basis,
                              const int iX, const int iN ) const {
    return static_cast<EOS const *>( this )->ComputeInternalEnergy( U, Basis,
                                                                    iX, iN );
  }
  Real ComputeInternalEnergy( const View3D U, const int iX ) const {
    return static_cast<EOS const *>( this )->ComputeInternalEnergy( U, iX );
  }
};

#endif // _EOS_BASE_HPP_
