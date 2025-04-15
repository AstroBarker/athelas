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
 *          - flux_fluid
 *          - source_fluid_rad
 *          - numerical_flux_gudonov
 *          - numerical_flux_hllc
 *          - compute_timestep_fluid
 */

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"
#include "eos.hpp"

namespace fluid {

auto flux_fluid( Real V, Real P, int iCF ) -> Real;
auto source_fluid_rad( Real D, Real V, Real T, Real kappa_r, Real kappa_p,
                       Real E, Real F, Real Pr, int iCF ) -> Real;
void numerical_flux_gudonov( Real vL, Real vR, Real pL, Real pR, Real zL,
                             Real zR, Real& Flux_U, Real& Flux_P );
void numerical_flux_hllc( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
                          Real rhoL, Real rhoR, Real& Flux_U, Real& Flux_P );
auto compute_timestep_fluid( View3D<Real> U, const GridStructure* grid,
                             EOS* eos, Real CFL ) -> Real;

} // namespace fluid
#endif // FLUID_UTILITIES_HPP_
