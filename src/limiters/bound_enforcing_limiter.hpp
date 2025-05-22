#pragma once
/**
 * @file bound_enforcing_limiter.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Implementation of bound enforcing limiters for enforcing physicality.
 *
 * @details This file implements a suite of bound enforcing limiters based on
 *          K. Schaal et al 2015 (ADS: 10.1093/mnras/stv1859). These limiters
 *          ensure physicality of the solution by preventing negative values of
 *          key physical quantities:
 *
 *          - limit_density: Prevents negative density by scaling slope
 *            coefficients
 *          - limit_internal_energy: Maintains positive internal energy using
 *            root-finding algorithms
 *          - limit_rad_momentum: Ensures physical radiation momentum values
 *
 *          Multiple root finders for the internal energy solve are implemented
 *          and an Anderson accelerated fixed point iteration is the default.
 *          point iteration being the default choice.
 */

#include <print>

#include "abstractions.hpp"
#include "eos.hpp"
#include "polynomial_basis.hpp"
#include "utils/utilities.hpp"

namespace bel {

void limit_density( View3D<Real> U, const ModalBasis* basis );
void limit_internal_energy( View3D<Real> U, const ModalBasis* basis,
                            const EOS* eos );
void limit_rad_energy( View3D<Real> U, const ModalBasis* basis,
                       const EOS* eos );
void limit_rad_momentum( View3D<Real> U, const ModalBasis* basis,
                         const EOS* eos );
void apply_bound_enforcing_limiter( View3D<Real> U, const ModalBasis* basis,
                                    const EOS* eos );
void apply_bound_enforcing_limiter_rad( View3D<Real> U, const ModalBasis* basis,
                                        const EOS* eos );
auto compute_theta_state( View3D<Real> U, const ModalBasis* basis,
                          const EOS* eos, Real theta, int iCF, int iX, int iN )
    -> Real;
auto target_func( Real theta, View3D<Real> U, const ModalBasis* basis,
                  const EOS* eos, int iX, int iN ) -> Real;
auto target_func_rad_flux( Real theta, View3D<Real> U, const ModalBasis* basis,
                           const EOS* eos, int iX, int iN ) -> Real;
auto target_func_rad_energy( Real theta, View3D<Real> U,
                             const ModalBasis* basis, const EOS* eos, int iX,
                             int iN ) -> Real;

template <typename F>
auto bisection( const View3D<Real> U, F target, const ModalBasis* basis,
                const EOS* eos, const int iX, const int iN ) -> Real {
  constexpr static Real TOL      = 1e-10;
  constexpr static int MAX_ITERS = 100;
  constexpr static Real delta    = 1.0e-3; // reduce root by delta

  // bisection bounds on theta
  Real a = 0.0;
  Real b = 1.0;
  Real c = 0.5;

  Real fa = 0.0; // f(a) etc
  Real fc = 0.0;

  int n = 0;
  while ( n <= MAX_ITERS ) {
    c = ( a + b ) / 2.0;

    fa = target( a, U, basis, eos, iX, iN );
    fc = target( c, U, basis, eos, iX, iN );

    if ( std::abs( fc ) <= TOL || ( b - a ) / 2.0 < TOL ) {
      return c - delta;
    }

    // new interval
    if ( utilities::SGN( fc ) == utilities::SGN( fa ) ) {
      a = c;
    } else {
      b = c;
    }

    n++;
  }

  std::println( "Max Iters Reach In bisection" );
  return c - delta;
}

template <typename F>
auto backtrace( const View3D<Real> U, F target, const ModalBasis* basis,
                const EOS* eos, const int iX, const int iN ) -> Real {
  constexpr static Real EPSILON = 1.0e-10; // maybe make this smarter
  Real theta                    = 1.0;
  Real nodal                    = -1.0;

  while ( theta >= 0.01 && nodal < EPSILON ) {
    nodal = target( theta, U, basis, eos, iX, iN );

    theta -= 0.05;
  }

  return theta;
}

} // namespace bel
