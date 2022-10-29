#ifndef EQUATIONOFSTATELIBRARY_IDEAL_H
#define EQUATIONOFSTATELIBRARY_IDEAL_H

#include "Abstractions.hpp"

Real ComputePressureFromPrimitive_IDEAL( const Real Ev,
                                         const Real GAMMA = 1.4 );
Real ComputePressureFromConserved_IDEAL( const Real Tau, const Real V,
                                         const Real Em_T,
                                         const Real GAMMA = 1.4 );
Real ComputeSoundSpeedFromConserved_IDEAL( const Real Tau, const Real V,
                                           const Real Em_T,
                                           const Real GAMMA = 1.4 );
Real ComputeInternalEnergy( const Kokkos::View<Real***> U,
                            ModalBasis *Basis, const unsigned int iX,
                            const unsigned int iN );
Real ComputeInternalEnergy( const Kokkos::View<Real***> U,
                            const unsigned int iX );

#endif
