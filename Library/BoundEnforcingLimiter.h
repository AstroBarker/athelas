#ifndef BOUNDENFORCINGLIMITER_H
#define BOUNDENFORCINGLIMITER_H

void LimitDensity( Kokkos::View<double***> U, const ModalBasis& Basis );
void LimitInternalEnergy( Kokkos::View<double***> U, const ModalBasis& Basis );
void ApplyBoundEnforcingLimiter( Kokkos::View<double***> U,
                                 const ModalBasis& Basis );
double ComputeThetaState( const Kokkos::View<double***> U,
                          const ModalBasis& Basis, const double theta,
                          const unsigned int iCF, const unsigned int iX,
                          const unsigned int iN );
double TargetFunc( const Kokkos::View<double***> U, const ModalBasis& Basis,
                   const double theta, const unsigned int iX,
                   const unsigned int iN );
double Bisection( const Kokkos::View<double***> U, const ModalBasis& Basis,
                  const unsigned int iX, const unsigned int iN );

#endif