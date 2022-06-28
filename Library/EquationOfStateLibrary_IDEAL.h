#ifndef EQUATIONOFSTATELIBRARY_IDEAL_H
#define EQUATIONOFSTATELIBRARY_IDEAL_H

double ComputePressureFromPrimitive_IDEAL( const double Ev,
                                           const double GAMMA = 1.4 );
double ComputePressureFromConserved_IDEAL( const double Tau, const double V,
                                           const double Em_T,
                                           const double GAMMA = 1.4 );
double ComputeSoundSpeedFromConserved_IDEAL( const double Tau, const double V,
                                             const double Em_T,
                                             const double GAMMA = 1.4 );
double ComputeInternalEnergy( const Kokkos::View<double***> U,
                              const ModalBasis& Basis, const unsigned int iX,
                              const unsigned int iN );
double ComputeInternalEnergy( const Kokkos::View<double***> U,
                              const unsigned int iX );

#endif