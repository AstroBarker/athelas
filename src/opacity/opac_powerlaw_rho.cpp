/**
 * @file opac_powerlaw_rho.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Density power law opacity model
 */

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
