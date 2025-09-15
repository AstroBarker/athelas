/**
 * @file opac_powerlaw_rho.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Density power law opacity model
 */

#include <cmath>

#include "opac.hpp"

auto PowerlawRho::planck_mean(const double rho, const double /*T*/,
                              const double /*X*/, const double /*Y*/,
                              const double /*Z*/, double * /*lambda*/) const
    -> double {
  return kP_ * std::pow(rho, exp_);
}

auto PowerlawRho::rosseland_mean(const double rho, const double /*T*/,
                                 const double /*X*/, const double /*Y*/,
                                 const double /*Z*/, double * /*lambda*/) const
    -> double {
  return kR_ * std::pow(rho, exp_);
}
