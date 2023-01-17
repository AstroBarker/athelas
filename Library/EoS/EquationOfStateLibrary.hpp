#ifndef _EQUATIONOFSTATELIBRARY_HPP_
#define _EQUATIONOFSTATELIBRARY_HPP_

#include "Abstractions.hpp"

Real ComputePressureFromPrimitive_IDEAL( const Real Ev );
Real ComputePressureFromConserved_IDEAL( const Real Tau, const Real V,
                                         const Real Em_T );
Real ComputeSoundSpeedFromConserved_IDEAL( const Real Tau, const Real V,
                                           const Real Em_T );
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                            const UInt iX, const UInt iN );
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, const UInt iX );

#endif
