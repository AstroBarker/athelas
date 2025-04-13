#ifndef RAD_UTILITIES_HPP_
#define RAD_UTILITIES_HPP_
/**
 * @file rad_utilities.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Functions for radiation evolution.
 *
 * @details Key functions for radiation udates:
 *          - FluxFactor
 *          - Flux_Rad
 *          - RadiationFourForce
 *          - Source_Rad
 *          - Compute_Closure
 *          - Lambda_HLL
 *          - numerical_flux_hll_rad
 *          - computeTimestep_Rad
 */

#include <tuple>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"

namespace radiation {

auto FluxFactor( Real E, Real F ) -> Real;
auto Flux_Rad( Real E, Real F, Real P, Real V, int iCR ) -> Real;
auto RadiationFourForce( Real D, Real V, Real T, Real kappa_r, Real kappa_p,
                         Real E, Real F, Real Pr ) -> std::tuple<Real, Real>;
auto Source_Rad( Real D, Real V, Real T, Real kappa_r, Real kappa_p, Real E,
                 Real F, Real Pr, int iCR ) -> Real;
auto ComputeClosure( Real E, Real F ) -> Real;
auto Lambda_HLL( Real f, int sign ) -> Real;
void llf_flux( Real Fp, Real Fm, Real Up, Real Um, Real alpha, Real& out );
std::tuple<Real, Real> numerical_flux_hll_rad( Real E_L, Real E_R, Real F_L,
                                               Real F_R, Real P_L, Real P_R );
auto ComputeTimestep_Rad( const GridStructure* Grid, Real CFL ) -> Real;

} // namespace radiation
#endif // RAD_UTILITIES_HPP_
