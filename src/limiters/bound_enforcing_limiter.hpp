#ifndef BOUND_ENFORCING_LIMITER_HPP_
#define BOUND_ENFORCING_LIMITER_HPP_

#include "abstractions.hpp"
#include "eos.hpp"
#include "polynomial_basis.hpp"

void LimitDensity( View3D<Real> U, const ModalBasis *Basis );
void LimitInternalEnergy( View3D<Real> U, const ModalBasis *Basis,
                          const EOS *eos );
void LimitRadMomentum( View3D<Real> U, const ModalBasis *Basis, const EOS *eos );
void ApplyBoundEnforcingLimiter( View3D<Real> U, const ModalBasis *Basis,
                                 const EOS *eos );
void ApplyBoundEnforcingLimiterRad( View3D<Real> U, const ModalBasis *Basis, const EOS *eos );
Real ComputeThetaState( const View3D<Real> U, const ModalBasis *Basis,
                        const EOS *eos, const Real theta, const int iCF,
                        const int iX, const int iN );
Real TargetFunc( const View3D<Real> U, const ModalBasis *Basis, const EOS *eos,
                 const Real theta, const int iX, const int iN );
Real TargetFuncRad( const View3D<Real> U, const ModalBasis *Basis, const EOS *eos,
                 const Real theta, const int iX, const int iN );
Real Bisection( const View3D<Real> U, ModalBasis *Basis, EOS *eos, const int iX,
                const int iN );
//Real Backtrace( const View3D<Real> U, const ModalBasis *Basis, const EOS *eos,
//                const int iX, const int iN );
template <typename F>
Real Backtrace( const View3D<Real> U, F target, const ModalBasis *Basis, const EOS *eos,
                const int iX, const int iN ) {
  Real theta = 1.0;
  Real nodal = -1.0;

  while ( theta >= 0.01 && nodal < 0.0 && nodal > U(0,iX,0) ) {
    nodal = target( U, Basis, eos, theta, iX, iN );

    theta -= 0.05;
  }

  return theta;
}
#endif // BOUND_ENFORCING_LIMITER_HPP_
