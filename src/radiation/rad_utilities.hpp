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

#include "grid.hpp"

namespace radiation {

auto flux_factor(double E, double F) -> double;
auto flux_rad(double E, double F, double P, double vstar, int iCR) -> double;
auto flux_rad(double E, double F, double P, double V)
    -> std::tuple<double, double>;
auto radiation_four_force(double D, double V, double T, double kappa_r,
                          double kappa_p, double E, double F, double Pr)
    -> std::tuple<double, double>;
auto source_factor_rad() -> std::tuple<double, double>;
auto compute_closure(double E, double F) -> double;
auto lambda_hll(double f, int sign) -> double;
auto llf_flux(double Fp, double Fm, double Up, double Um, double alpha)
    -> double;
auto numerical_flux_hll_rad(double E_L, double E_R, double F_L, double F_R,
                            double P_L, double P_R, double vstar)
    -> std::tuple<double, double>;
auto compute_timestep_rad(const GridStructure* grid, double CFL) -> double;

} // namespace radiation
