#ifndef BOUNDENFORCINGLIMITER_H
#define BOUNDENFORCINGLIMITER_H

#include "Abstractions.hpp"

void LimitDensity( Kokkos::View<Real***> U, const ModalBasis& Basis );
void LimitInternalEnergy( Kokkos::View<Real***> U, const ModalBasis& Basis );
void ApplyBoundEnforcingLimiter( Kokkos::View<Real***> U,
                                 const ModalBasis& Basis );
Real ComputeThetaState( const Kokkos::View<Real***> U,
                          const ModalBasis& Basis, const Real theta,
                          const unsigned int iCF, const unsigned int iX,
                          const unsigned int iN );
Real TargetFunc( const Kokkos::View<Real***> U, const ModalBasis& Basis,
                   const Real theta, const unsigned int iX,
                   const unsigned int iN );
Real Bisection( const Kokkos::View<Real***> U, const ModalBasis& Basis,
                  const unsigned int iX, const unsigned int iN );
Real Backtrace( const Kokkos::View<Real***> U, const ModalBasis& Basis,
                  const unsigned int iX, const unsigned int iN );

#endif
