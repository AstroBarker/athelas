#pragma once
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

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "utils/abstractions.hpp"

namespace athelas::fluid {

auto flux_fluid(const double V, const double P)
    -> std::tuple<double, double, double>;
auto numerical_flux_gudonov(double vL, double vR, double pL, double pR,
                            double zL, double zR) -> std::tuple<double, double>;
auto numerical_flux_gudonov_positivity(double tauL, double tauR, double vL,
                                       double vR, double pL, double pR,
                                       double csL, double csR)
    -> std::tuple<double, double>;
void numerical_flux_hllc(double vL, double vR, double pL, double pR, double cL,
                         double cR, double rhoL, double rhoR, double &Flux_U,
                         double &Flux_P);
} // namespace athelas::fluid
