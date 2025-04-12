#ifndef BOUND_ENFORCING_LIMITER_HPP_
#define BOUND_ENFORCING_LIMITER_HPP_
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
 *          - LimitDensity: Prevents negative density by scaling slope
 *            coefficients
 *          - LimitInternalEnergy: Maintains positive internal energy using
 *            root-finding algorithms
 *          - LimitRadMomentum: Ensures physical radiation momentum values
 *
 *          Multiple root finders for the internal energy solve are implemented
 *          and an Anderson accelerated fixed point iteration is the default.
 *          point iteration being the default choice.
 */

#include "abstractions.hpp"
#include "eos.hpp"
#include "polynomial_basis.hpp"
#include "utils/utilities.hpp"

void LimitDensity( View3D<Real> U, const ModalBasis *Basis );
void LimitInternalEnergy( View3D<Real> U, const ModalBasis *Basis,
                          const EOS *eos );
void LimitRadMomentum( View3D<Real> U, const ModalBasis *Basis,
                       const EOS *eos );
void ApplyBoundEnforcingLimiter( View3D<Real> U, const ModalBasis *Basis,
                                 const EOS *eos );
void ApplyBoundEnforcingLimiterRad( View3D<Real> U, const ModalBasis *Basis,
                                    const EOS *eos );
Real ComputeThetaState( const View3D<Real> U, const ModalBasis *Basis,
                        const EOS *eos, const Real theta, const int iCF,
                        const int iX, const int iN );
Real TargetFunc( const Real theta, const View3D<Real> U,
                 const ModalBasis *Basis, const EOS *eos, const int iX,
                 const int iN );
Real TargetFuncRad( const Real theta, const View3D<Real> U,
                    const ModalBasis *Basis, const EOS *eos, const int iX,
                    const int iN );

template <typename F>
Real Bisection( const View3D<Real> U, F target, const ModalBasis *Basis,
                const EOS *eos, const int iX, const int iN ) {
  const Real TOL      = 1e-10;
  const int MAX_ITERS = 100;
  const Real delta    = 1.0e-3; // reduce root by delta

  // bisection bounds on theta
  Real a = 0.0;
  Real b = 1.0;
  Real c = 0.5;

  Real fa = 0.0; // f(a) etc
  Real fc = 0.0;

  int n = 0;
  while ( n <= MAX_ITERS ) {
    c = ( a + b ) / 2.0;

    fa = target( a, U, Basis, eos, iX, iN );
    fc = target( c, U, Basis, eos, iX, iN );

    if ( std::abs( fc ) <= TOL || ( b - a ) / 2.0 < TOL ) {
      return c - delta;
    }

    // new interval
    if ( utilities::sgn( fc ) == utilities::sgn( fa ) ) {
      a = c;
    } else {
      b = c;
    }

    n++;
  }

  std::printf( "Max Iters Reach In Bisection\n" );
  return c - delta;
}

template <typename F>
Real Backtrace( const View3D<Real> U, F target, const ModalBasis *Basis,
                const EOS *eos, const int iX, const int iN ) {
  constexpr static Real EPSILON = 1.0e-10; // maybe make this smarter
  Real theta                    = 1.0;
  Real nodal                    = -1.0;

  while ( theta >= 0.01 && nodal < EPSILON ) {
    nodal = target( theta, U, Basis, eos, iX, iN );

    theta -= 0.05;
  }

  return theta;
}
#endif // BOUND_ENFORCING_LIMITER_HPP_
