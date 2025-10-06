/**
 * @file opac_constant.cpp
 * --------------
 *
 * @brief Constant opacity model
 */

#include "opacity/opac.hpp"

namespace athelas {

auto Constant::planck_mean(const double /*rho*/, const double /*T*/,
                           const double /*X*/, const double /*Y*/,
                           const double /*Z*/, double * /*lambda*/) const
    -> double {
  return kP_;
}

auto Constant::rosseland_mean(const double /*rho*/, const double /*T*/,
                              const double /*X*/, const double /*Y*/,
                              const double /*Z*/, double * /*lambda*/) const
    -> double {
  return kR_;
}

} // namespace athelas
