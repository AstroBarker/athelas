/**
 * @file opac_constant.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Constant opacity model
 */

#include "opac.hpp"

auto Constant::planck_mean(const double /*rho*/, const double /*T*/,
                           const double /*X*/, const double /*Y*/,
                           const double /*Z*/, double* /*lambda*/) const
    -> double {
  return kP_;
}

auto Constant::rosseland_mean(const double /*rho*/, const double /*T*/,
                              const double /*X*/, const double /*Y*/,
                              const double /*Z*/, double* /*lambda*/) const
    -> double {
  return kR_;
}
