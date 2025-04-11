/**
 * File     :  opac_constant.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : opacity = constant
 * Rosseland and Plank are equal
 **/

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
