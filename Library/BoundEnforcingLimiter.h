#ifndef BOUNDENFORCINGLIMITER_H
#define BOUNDENFORCINGLIMITER_H

#include "Abstractions.hpp"

void LimitDensity( Kokkos::View<Real ***> U, ModalBasis *Basis );
void LimitInternalEnergy( Kokkos::View<Real ***> U, ModalBasis *Basis );
void ApplyBoundEnforcingLimiter( Kokkos::View<Real ***> U, ModalBasis *Basis );
Real ComputeThetaState( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                        const Real theta, const UInt iCF, const UInt iX,
                        const UInt iN );
Real TargetFunc( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                 const Real theta, const UInt iX, const UInt iN );
Real Bisection( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                const UInt iX, const UInt iN );
Real Backtrace( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                const UInt iX, const UInt iN );

#endif
