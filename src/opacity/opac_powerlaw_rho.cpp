/**
 * @file opac_powerlaw_rho.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Density power law opacity model
 */

#include <cmath>

#include "abstractions.hpp"
#include "constants.hpp"
#include "opac.hpp"
#include "opac_base.hpp"

auto PowerlawRho::planck_mean( const Real rho, const Real /*T*/,
                               const Real /*X*/, const Real /*Y*/,
                               const Real /*Z*/, Real* /*lambda*/ ) const
    -> Real {
  return kP_ * std::pow( rho, exp_ );
}

auto PowerlawRho::rosseland_mean( const Real rho, const Real /*T*/,
                                  const Real /*X*/, const Real /*Y*/,
                                  const Real /*Z*/, Real* /*lambda*/ ) const
    -> Real {
  return kR_ * std::pow( rho, exp_ );
}
