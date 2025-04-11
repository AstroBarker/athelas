#ifndef OPAC_VARIANT_HPP_
#define OPAC_VARIANT_HPP_

/**
 * Specific EoS classes here
 **/

#include <variant>

#include "abstractions.hpp"
#include "error.hpp"
#include "opac.hpp"
#include "opac_base.hpp"

using Opacity = std::variant<Constant, PowerlawRho>;

KOKKOS_INLINE_FUNCTION Real PlanckMean( const Opacity *opac, const Real rho,
                                        const Real T, const Real X,
                                        const Real Y, const Real Z,
                                        Real *lambda ) {
  return std::visit(
      [&rho, &T, &X, &Y, &Z, &lambda]( auto &opac ) {
        return opac.PlanckMean( rho, T, X, Y, Z, lambda );
      },
      *opac );
}

KOKKOS_INLINE_FUNCTION Real RosselandMean( const Opacity *opac, const Real rho,
                                           const Real T, const Real X,
                                           const Real Y, const Real Z,
                                           Real *lambda ) {
  return std::visit(
      [&rho, &T, &X, &Y, &Z, &lambda]( auto &opac ) {
        return opac.RosselandMean( rho, T, X, Y, Z, lambda );
      },
      *opac );
}

// put init function here..

KOKKOS_INLINE_FUNCTION Opacity InitializeOpacity( const ProblemIn *pin ) {
  Opacity opac;
  if ( pin->opac_type == "constant" ) {
    std::printf( "const\n" );
    opac = Constant( pin->in_table["opacity"]["k"].value_or( 1.0 ) );
  } else { // powerlaw rho
    std::printf( "powerlaw\n" );
    opac = PowerlawRho(
        pin->in_table["opacity"]["k"].value_or( 4.0 * std::pow( 10.0, -8.0 ) ),
        pin->in_table["opacity"]["exp"].value_or( 1.0 ) );
  }
  return opac;
}

#endif // OPAC_VARIANT_HPP_
