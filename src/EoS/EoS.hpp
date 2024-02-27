#ifndef _EOS_HPP_
#define _EOS_HPP_

/**
 * Specific EoS classes here
 **/

#include <variant>

#include "Abstractions.hpp"
#include "EoS_Base.hpp"

class IdealGas : public EosBase<IdealGas> {
 public:
  IdealGas( ) = default;
  IdealGas( double gm ) : gamma( gm ) {}

  Real PressureFromConserved( const Real Tau, const Real V, const Real EmT ) const;
  Real SoundSpeedFromConserved( const Real Tau, const Real V, const Real EmT ) const;
  Real TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                       const Real Abar ) const;
  Real TemperatureFromTauPressure( const Real Tau, const Real P ) const;
  Real RadiationPressure( const Real T ) const;
  Real ComputeInternalEnergy( const View3D U, const ModalBasis *Basis,
                              const int iX, const int iN ) const;
  Real ComputeInternalEnergy( const View3D U, const int iX ) const;

 private:
  Real gamma;
};

/* placeholder */
class Stellar : public EosBase<Stellar> {
 public:
  Stellar( ) = default;

  Real PressureFromConserved( const Real Tau, const Real V, const Real EmT ) const;
  Real SoundSpeedFromConserved( const Real Tau, const Real V, const Real EmT ) const;
  Real TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                       const Real Abar ) const;
  Real TemperatureFromTauPressure( const Real Tau, const Real P ) const;
  Real RadiationPressure( const Real T ) const;
  Real ComputeInternalEnergy( const View3D U, const ModalBasis *Basis,
                              const int iX, const int iN ) const;
  Real ComputeInternalEnergy( const View3D U, const int iX ) const;

 private:
  Real gamma;
};

// TODO: adjust when we support more than one EOS
using EOS = IdealGas;

#endif // _EOS_HPP_
