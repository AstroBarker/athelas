#ifndef EQUATIONOFSTATELIBRARY_IDEAL_H
#define EQUATIONOFSTATELIBRARY_IDEAL_H

double ComputePressureFromPrimitive_IDEAL( double Ev,
                                           double GAMMA = 1.4 );
double ComputePressureFromConserved_IDEAL( double Tau, double V, double Em_T,
                                           double GAMMA = 1.4 );
double ComputeSoundSpeedFromConserved_IDEAL( double Tau, double V, double Em_T,
                                             double GAMMA = 1.4 );
double ComputeInternalEnergy( Kokkos::View<double***> U, const ModalBasis& Basis, 
                              const unsigned int iX, const unsigned int iN );
double ComputeInternalEnergy( Kokkos::View<double***> U, const unsigned int iX );

#endif