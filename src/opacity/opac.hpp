#ifndef OPAC_HPP_
#define OPAC_HPP_

/**
 * Specific opacity classes here
 **/

#include "abstractions.hpp"
#include "error.hpp"
#include "opac_base.hpp"

class Constant : public OpacBase<Constant> {
 public:
  Constant( ) = default;
  Constant( double k_ ) : k( k_ ) {}

  Real PlanckMean( const Real rho, const Real T, const Real X, const Real Y,
                   const Real Z, Real *lambda ) const;

  Real RosselandMean( const Real rho, const Real T, const Real X, const Real Y,
                      const Real Z, Real *lambda ) const;

 private:
  Real k;
};

class PowerlawRho : public OpacBase<PowerlawRho> {
 public:
  PowerlawRho( ) = default;
  PowerlawRho( double k_, double exp_ ) : k( k_ ), exp( exp_ ) {}

  Real PlanckMean( const Real rho, const Real T, const Real X, const Real Y,
                   const Real Z, Real *lambda ) const;

  Real RosselandMean( const Real rho, const Real T, const Real X, const Real Y,
                      const Real Z, Real *lambda ) const;

 private:
  Real k;
  Real exp;
};

#endif // OPAC_HPP_
