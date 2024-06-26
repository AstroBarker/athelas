#ifndef _FLUID_UTILITIES_HPP_
#define _FLUID_UTILITIES_HPP_

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "eos.hpp"

void ComputePrimitiveFromConserved( View3D<Real> uCF, View3D<Real> uPF,
                                    ModalBasis *Basis, GridStructure *Grid );
Real Flux_Fluid( const Real V, const Real P, const int iCF );
Real Source_Fluid_Rad( Real D, Real V, Real T, Real X, Real kappa, Real E,
                       Real F, Real Pr, int iCF );
void NumericalFlux_Gudonov( const Real vL, const Real vR, const Real pL,
                            const Real pR, const Real zL, const Real zR,
                            Real &Flux_U, Real &Flux_P );
void NumericalFlux_HLLC( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
                         Real rhoL, Real rhoR, Real &Flux_U, Real &Flux_P );
Real ComputeTimestep_Fluid( const View3D<Real> U, const GridStructure *Grid,
                            EOS *eos, const Real CFL );

#endif // _FLUID_UTILITIES_HPP_
