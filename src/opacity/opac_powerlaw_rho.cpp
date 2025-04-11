/**
 * File     :  opac_powerlaw_rho.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : opacity = c * rho^a
 * Rosseland and Plank are equal
 **/

#include <math.h>

#include "abstractions.hpp"
#include "constants.hpp"
#include "opac.hpp"
#include "opac_base.hpp"

Real PowerlawRho::PlanckMean( const Real rho, const Real T, const Real X,
                              const Real Y, const Real Z, Real *lambda ) const {
  return k * std::pow( rho, exp );
}

Real PowerlawRho::RosselandMean( const Real rho, const Real T, const Real X,
                                 const Real Y, const Real Z,
                                 Real *lambda ) const {
  return k * std::pow( rho, exp );
}
