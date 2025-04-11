#ifndef OPAC_BASE_HPP_
#define OPAC_BASE_HPP_

/**
 * define a base class using curiously recurring template pattern
 **/
#include "abstractions.hpp"
#include "opac.hpp"
#include "opac_base.hpp"
#include "polynomial_basis.hpp"

template <class OPAC>
class OpacBase {
 public:
  Real PlanckMean( const Real rho, const Real T, const Real X, const Real Y,
                   const Real Z, Real *lambda ) const {
    return static_cast<OPAC const *>( this )->PlanckMean( rho, T, X, Y, Z,
                                                          lambda );
  }

  Real RosselandMean( const Real rho, const Real T, const Real X, const Real Y,
                      const Real Z, Real *lambda ) const {
    return static_cast<OPAC const *>( this )->RosselandMean( rho, T, X, Y, Z,
                                                             lambda );
  }
};

#endif // OPAC_BASE_HPP_
