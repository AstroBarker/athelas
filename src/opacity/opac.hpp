#ifndef OPAC_HPP_
#define OPAC_HPP_
/**
 * @file opac.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Declares concrete opacity model classes that implement the OpacBase
 *        interface
 *
 * @details This header defines specific opacity model implementations that
 *          inherit from the OpacBase template class. It serves as the central
 *          declaration point for all opacity model classes in the codebase.
 *
 *          We provide the following opacity models:
 *          - Constant: A simple model with constant opacity value
 *          - PowerlawRho: \kappa = k rho^exp
 *
 */

#include "abstractions.hpp"
#include "error.hpp"
#include "opac_base.hpp"

class Constant : public OpacBase<Constant> {
 public:
  Constant( ) = default;
  explicit Constant( double k ) : k_( k ) {}

  auto planck_mean( Real rho, Real T, Real X, Real Y, Real Z,
                    Real* lambda ) const -> Real;

  auto rosseland_mean( Real rho, Real T, Real X, Real Y, Real Z,
                       Real* lambda ) const -> Real;

 private:
  Real k_{ };
};

class PowerlawRho : public OpacBase<PowerlawRho> {
 public:
  PowerlawRho( ) = default;
  PowerlawRho( double k, double exp ) : k_( k ), exp_( exp ) {}

  auto planck_mean( Real rho, Real T, Real X, Real Y, Real Z,
                    Real* lambda ) const -> Real;

  auto rosseland_mean( Real rho, Real T, Real X, Real Y, Real Z,
                       Real* lambda ) const -> Real;

 private:
  Real k_{ };
  Real exp_{ };
};

#endif // OPAC_HPP_
