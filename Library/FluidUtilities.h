#ifndef FLUIDUTILITIES_H
#define FLUIDUTILITIES_H

#include "Kokkos_Core.hpp"

void ComputePrimitiveFromConserved( Kokkos::View<double***> uCF,
                                    Kokkos::View<double***> uPF,
                                    ModalBasis& Basis, GridStructure& Grid );
double Flux_Fluid( double V, double P, unsigned int iCF );
double Fluid( double Tau, double V, double Em_T, int iCF );
void NumericalFlux_Gudonov( double vL, double vR, double pL, double pR,
                            double zL, double zR, double& Flux_U,
                            double& Flux_P );
void NumericalFlux_HLLC( double vL, double vR, double pL, double pR, double cL,
                         double cR, double rhoL, double rhoR, double& Flux_U,
                         double& Flux_P );
double ComputeTimestep_Fluid( Kokkos::View<double***> U, GridStructure& Grid,
                              const double CFL );

#endif
