/**
 * File     :  bound_enforcing_limiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Limit solution to maintain physicality
 * TODO: Need to give BEL much more thought
 * TODO: Can some functions here be simplified?
 *   ? If I pass U, iX, iCF, iN... why not just the value
 **/

#include <algorithm> // std::min, std::max
#include <cstdlib> /* abs */
#include <iostream>

#include "Kokkos_Core.hpp"

#include "bound_enforcing_limiter.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "polynomial_basis.hpp"
#include "solvers/root_finders.hpp"
#include "utilities.hpp"

/**
 * Limit density following K. Schaal et al 2015
 * ADS: 10.1093/mnras/stv1859
 *
 * Find theta = min((rho_avg - eps)/(rho_nodal - rho_avg) ,1)
 * Scale slope, ... by theta
 **/
void LimitDensity( View3D<Real> U, const ModalBasis *Basis ) {
  constexpr static Real EPSILON = 1.0e-10; // maybe make this smarter
  const int order               = Basis->Get_Order( );

  if ( order == 1 ) return;

  Kokkos::parallel_for(
      "BEF::Limit Density", Kokkos::RangePolicy<>( 1, U.extent( 1 ) - 1 ),
      KOKKOS_LAMBDA( const int iX ) {
        Real theta1 = 100000.0; // big
        Real nodal  = 0.0;
        Real frac   = 0.0;
        Real avg    = U( 0, iX, 0 );

        for ( int iN = 0; iN <= order; iN++ ) {
          nodal = Basis->basis_eval( U, iX, 0, iN );
          if ( std::isnan( nodal ) ) {
            theta1 = 0.0;
            break;
          }
          frac   = std::abs( ( avg - EPSILON ) / ( avg - nodal ) );
          theta1 = std::min( theta1, std::min( 1.0, frac ) );
        }

        for ( int k = 1; k < order; k++ ) {
          U( 0, iX, k ) *= theta1;
        }
      } );
}

/**
 * Limit other variables based on K. Schaal et al 2015
 * in order to preserve positivity of internal energy
 * ADS: 10.1093/mnras/stv1859
 *
 *
 * Find a theta s.t. (1 - theta) U_bar + theta U_q
 * is positive for U being the specific internal energy.
 *
 * There are three root finders implemented:
 *  - A Bisection
 *  - An Anderson accelerated fixed point iteration
 *  - A simple "back tracing" algorithm (step back from theta = 1)
 *
 * The fixed point iteration is default. All yield the same
 * result and seem stable on difficult problems, but if
 * stability issues arise, consider switching the solver to
 * Bisection below.
 * Note that in the bisection and fixed point solvers
 * a small delta = 1.0e-3 is subtracted from the root
 * to ensure positivity. The backtrace algorithm overshoots
 * the root and so this is not necessary.
 *
 * The simulation time is insensitive to solver choice.
 **/
void LimitInternalEnergy( View3D<Real> U, const ModalBasis *Basis,
                          const EOS *eos ) {
  const int order = Basis->Get_Order( );

  if ( order == 1 ) return;

  Kokkos::parallel_for(
      "BEF::Limit Internal Energy",
      Kokkos::RangePolicy<>( 1, U.extent( 1 ) - 1 ),
      KOKKOS_LAMBDA( const int iX ) {
        constexpr static Real EPSILON = 1.0e-10; // maybe make this smarter
        Real theta2                   = 10000000.0;
        Real nodal                    = 0.0;
        Real temp                     = 0.0;

        for ( int iN = 0; iN <= order + 1; iN++ ) {
          nodal = utilities::ComputeInternalEnergy( U, Basis, iX, iN );

          if ( nodal > EPSILON ) {
            temp = 1.0;
          } else {
            // temp = Backtrace( U, TargetFunc, Basis, eos, iX, iN );
            // const Real theta_guess = 0.9; // needed for fixed point
            // temp = root_finders::fixed_point_aa_root(TargetFunc, theta_guess,
            // U, Basis, eos, iX, iN) - 1.0e-3;
            temp = Bisection( U, TargetFunc, Basis, eos, iX, iN );
          }
          theta2 = std::min( theta2, temp );
        }

        for ( int k = 1; k < order; k++ ) {
          U( 0, iX, k ) *= theta2;
          U( 1, iX, k ) *= theta2;
          U( 2, iX, k ) *= theta2;
        }
      } );
}

void ApplyBoundEnforcingLimiter( View3D<Real> U, const ModalBasis *Basis,
                                 const EOS *eos )

{
  LimitDensity( U, Basis );
  LimitInternalEnergy( U, Basis, eos );
}

// TODO: much more here.
void ApplyBoundEnforcingLimiterRad( View3D<Real> U, const ModalBasis *Basis,
                                    const EOS *eos ) {
  if ( Basis->Get_Order( ) == 1 ) return;
  LimitRadMomentum( U, Basis, eos );
}
void LimitRadMomentum( View3D<Real> U, const ModalBasis *Basis,
                       const EOS *eos ) {
  const int order = Basis->Get_Order( );

  Kokkos::parallel_for(
      "BEF::Limit Rad Momentum", Kokkos::RangePolicy<>( 1, U.extent( 1 ) - 1 ),
      KOKKOS_LAMBDA( const int iX ) {
        Real theta2 = 10000000.0;
        Real nodal  = 0.0;
        Real temp   = 0.0;

        for ( int iN = 0; iN <= order + 1; iN++ ) {
          nodal = Basis->basis_eval( U, iX, 1, iN );

          if ( nodal >= 0.0 && nodal <= U( 0, iX, 0 ) ) {
            temp = 1.0;
          } else {
            // TODO: Backtracing may be working okay...
            const Real theta_guess = 0.9;
            // temp = Backtrace( TargetFuncRad, theta_guess, U, Basis, eos, iX,
            // iN );
            temp = Bisection( U, TargetFuncRad, Basis, eos, iX, iN );
          }
          theta2 = std::min( theta2, temp );
        }

        for ( int k = 1; k < order; k++ ) {
          U( 1, iX, k ) *= theta2;
        }
      } );
}

/* --- Utility Functions --- */

// ( 1 - theta ) U_bar + theta U_q
Real ComputeThetaState( const View3D<Real> U, const ModalBasis *Basis,
                        const Real theta, const int iCF, const int iX,
                        const int iN ) {
  Real result = Basis->basis_eval( U, iX, iCF, iN );
  result -= U( iCF, iX, 0 );
  result *= theta;
  result += U( iCF, iX, 0 );
  return result;
}

Real TargetFunc( const Real theta, const View3D<Real> U,
                 const ModalBasis *Basis, const EOS *eos, const int iX,
                 const int iN ) {
  const Real w = std::min( 1.0e-10, utilities::ComputeInternalEnergy( U, iX ) );
  const Real s1 = ComputeThetaState( U, Basis, theta, 1, iX, iN );
  const Real s2 = ComputeThetaState( U, Basis, theta, 2, iX, iN );

  Real e = s2 - 0.5 * s1 * s1;

  return e - w;
}

Real TargetFuncRad( const Real theta, const View3D<Real> U,
                    const ModalBasis *Basis, const EOS *eos, const int iX,
                    const int iN ) {
  const Real w  = std::min( 1.0e-13, U( 1, iX, 0 ) );
  const Real s1 = ComputeThetaState( U, Basis, theta, 1, iX, iN );

  const Real e = s1;

  return e - w;
}
