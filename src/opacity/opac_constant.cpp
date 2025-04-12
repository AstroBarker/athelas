/**
 * @file opac_constant.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Constant opacity model
 */

#include <math.h>

#include "abstractions.hpp"
#include "constants.hpp"
#include "opac.hpp"
#include "opac_base.hpp"
#include "opac_variant.hpp"

Real Constant::PlanckMean( const Real rho, const Real T, const Real X,
                           const Real Y, const Real Z, Real *lambda ) const {
  return k;
}

Real Constant::RosselandMean( const Real rho, const Real T, const Real X,
                              const Real Y, const Real Z, Real *lambda ) const {
  return k;
}
