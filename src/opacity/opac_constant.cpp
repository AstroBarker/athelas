/**
 * @file opac_constant.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Constant opacity model
 */

#include <cmath>

#include "abstractions.hpp"
#include "constants.hpp"
#include "opac.hpp"
#include "opac_base.hpp"
#include "opac_variant.hpp"

auto Constant::PlanckMean( const Real /*rho*/, const Real /*T*/,
                           const Real /*X*/, const Real /*Y*/, const Real /*Z*/,
                           Real* /*lambda*/ ) const -> Real {
  return k;
}

auto Constant::RosselandMean( const Real /*rho*/, const Real /*T*/,
                              const Real /*X*/, const Real /*Y*/,
                              const Real /*Z*/, Real* /*lambda*/ ) const
    -> Real {
  return k;
}
