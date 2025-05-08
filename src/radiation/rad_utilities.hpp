#pragma once
/**
 * @file rad_utilities.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Functions for radiation evolution.
 *
 * @details Key functions for radiation udates:
 *          - flux_factor
 *          - flux_rad
 *          - radiation_four_force
 *          - source_rad
 *          - Compute_Closure
 *          - lambda_hll
 *          - numerical_flux_hll_rad
 *          - computeTimestep_Rad
 */

#include <tuple>

#include "Kokkos_Core.hpp"

#include "abstractions.hpp"

namespace radiation {

auto flux_factor( Real E, Real F ) -> Real;
auto flux_rad( Real E, Real F, Real P, Real V, int iCR ) -> Real;
auto radiation_four_force( Real D, Real V, Real T, Real kappa_r, Real kappa_p,
                           Real E, Real F, Real Pr ) -> std::tuple<Real, Real>;
auto source_rad( Real D, Real V, Real T, Real kappa_r, Real kappa_p, Real E,
                 Real F, Real Pr, int iCR ) -> Real;
auto compute_closure( Real E, Real F ) -> Real;
auto lambda_hll( Real f, int sign ) -> Real;
auto llf_flux( Real Fp, Real Fm, Real Up, Real Um, Real alpha ) -> Real;
auto numerical_flux_hll_rad( Real E_L, Real E_R, Real F_L, Real F_R, Real P_L,
                             Real P_R, Real vstar ) -> std::tuple<Real, Real>;
auto compute_timestep_rad( const GridStructure* grid, Real CFL ) -> Real;

} // namespace radiation
