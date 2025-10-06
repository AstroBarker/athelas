/**
 * @file opac_powerlaw_rho.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Density power law opacity model
 * TODO(astrobarker): generalize to rho,T powerlaw
 */

#include <cmath>

#include "opacity/opac.hpp"

namespace athelas {

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

} // namespace athelas
