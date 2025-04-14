#ifndef OPAC_BASE_HPP_
#define OPAC_BASE_HPP_
/**
 * @file opac_base.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Base class for opacity models.
 *
 * @details Defines the OpacBase template class.
 *
 *          The class provides two interface methods:
 *          - planck_mean
 *          - rosseland_mean
 *
 *          The interface methods take density, temperature, and composition
 *          parameters to compute the appropriate mean opacity values.
 */

#include "abstractions.hpp"
#include "opac.hpp"
#include "opac_base.hpp"
#include "polynomial_basis.hpp"

template <class OPAC>
class OpacBase {
 public:
  auto planck_mean( const Real rho, const Real T, const Real X, const Real Y,
                    const Real Z, Real* lambda ) const -> Real {
    return static_cast<OPAC const*>( this )->planck_mean( rho, T, X, Y, Z,
                                                          lambda );
  }

  auto rosseland_mean( const Real rho, const Real T, const Real X, const Real Y,
                       const Real Z, Real* lambda ) const -> Real {
    return static_cast<OPAC const*>( this )->rosseland_mean( rho, T, X, Y, Z,
                                                             lambda );
  }
};

#endif // OPAC_BASE_HPP_
