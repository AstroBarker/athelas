#ifndef EQUATIONOFSTATELIBRARY_IDEAL_H
#define EQUATIONOFSTATELIBRARY_IDEAL_H

#include "Abstractions.hpp"

Real ComputePressureFromPrimitive_IDEAL( const Real Ev );
Real ComputePressureFromConserved_IDEAL( const Real Tau, const Real V,
                                         const Real Em_T );
Real ComputeSoundSpeedFromConserved_IDEAL( const Real Tau, const Real V,
                                           const Real Em_T );
Real ComputeTemperature( const Real Tau, const Real P );
Real ComputeTemperature( const Real Tau, const Real P, const Real A );
Real RadiationPressure( const Real T ); 
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                            const UInt iX, const UInt iN );
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, const UInt iX );

#endif
