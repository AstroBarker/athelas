#ifndef FLUID_UTILITIES_HPP_
#define FLUID_UTILITIES_HPP_
/**
 * @file fluid_utilities.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Utilities for fluid evolution
 *
 * @details Contains functions necessary for fluid evolution:
 *          - Flux_Fluid
 *          - Source_Fluid_Rad
 *          - NumericalFlux_Gudonov
 *          - NumericalFlux_HLLC
 *          - ComputeTimestep_Fluid
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "eos.hpp"

namespace fluid {

auto Flux_Fluid( Real V, Real P, int iCF ) -> Real;
auto Source_Fluid_Rad( Real D, Real V, Real T, Real kappa_r, Real kappa_p,
                       Real E, Real F, Real Pr, int iCF ) -> Real;
void NumericalFlux_Gudonov( Real vL, Real vR, Real pL, Real pR, Real zL,
                            Real zR, Real& Flux_U, Real& Flux_P );
void NumericalFlux_HLLC( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
                         Real rhoL, Real rhoR, Real& Flux_U, Real& Flux_P );
auto ComputeTimestep_Fluid( View3D<Real> U, const GridStructure* Grid, EOS* eos,
                            Real CFL ) -> Real;

} // namespace fluid
#endif // FLUID_UTILITIES_HPP_
