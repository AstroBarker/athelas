/**
 * File     :  BoundEnforcingLimiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Limit solution to maintain physicality
 * TODO: Bisection / process not getting sutiable theta
 * TODO: Can some functions here be simplified?
 *   ? If I pass U, iX, iCF, iN... why not just the value
 **/

#include <iostream>
#include <algorithm> // std::min, std::max
#include <cstdlib>   /* abs */

#include "Kokkos_Core.hpp"

#include "Error.h"
#include "Utilities.h"
#include "PolynomialBasis.h"
#include "EquationOfStateLibrary_IDEAL.h"
#include "BoundEnforcingLimiter.h"

void LimitDensity( Kokkos::View<double***> U, const ModalBasis& Basis )
{
  const double EPSILON     = 1.0e-13; // maybe make this smarter
  const unsigned int order = Basis.Get_Order( );

  if ( order == 1 ) return;

  Kokkos::parallel_for(
      "Limit Density", Kokkos::RangePolicy<>( 1, U.extent( 1 ) ),
      KOKKOS_LAMBDA( unsigned int iX ) {
        double theta1 = 100000.0;
        double nodal  = 0.0;
        double frac   = 0.0;
        double avg    = U( 0, iX, 0 );

        for ( unsigned int iN = 0; iN <= order; iN++ )
        {
          nodal  = Basis.BasisEval( U, iX, 0, iN, false );
          frac   = std::abs( ( avg - EPSILON ) / ( avg - nodal ) );
          theta1 = std::min( theta1, std::min( 1.0, frac ) );
        }

        for ( unsigned int k = 1; k < order; k++ )
        {
          U( 0, iX, k ) *= theta1;
        }
      } );
}

void LimitInternalEnergy( Kokkos::View<double***> U, const ModalBasis& Basis )
{
  const unsigned int order = Basis.Get_Order( );

  if ( order == 1 ) return;

  Kokkos::parallel_for(
      "Limit Internal Energy", Kokkos::RangePolicy<>( 1, U.extent( 1 ) ),
      KOKKOS_LAMBDA( unsigned int iX ) {
        double theta2 = 10000000.0;
        double nodal  = 0.0;
        double temp   = 0.0;
        double avg    = ComputeInternalEnergy( U, iX );

        for ( unsigned int iN = 0; iN <= order + 1; iN++ )
        {
          nodal = ComputeInternalEnergy( U, Basis, iX, iN );

          if ( nodal >= 0.0 )
          {
            temp = 1.0;
          }
          else
          {
            // TODO: Backtracing may be working okay...
            temp = Backtrace( U, Basis, iX, iN );
            std::printf("%f\n", temp);
            // TODO: This is hacked and Does Not Really Work
            // temp = Bisection( U, Basis, iX, iN ) / 2.0;
          }
          theta2 = std::min( theta2, temp );
        }

        for ( unsigned int k = 1; k < order; k++ )
        {
          U( 0, iX, k ) *= theta2;
          U( 1, iX, k ) *= theta2;
          U( 2, iX, k ) *= theta2;
        }
      } );
}

void ApplyBoundEnforcingLimiter( Kokkos::View<double***> U,
                                 const ModalBasis& Basis )

{
  LimitDensity( U, Basis );
  LimitInternalEnergy( U, Basis );
}

/* --- Utility Functions --- */

// ( 1 - theta ) U_bar + theta U_q
double ComputeThetaState( const Kokkos::View<double***> U,
                          const ModalBasis& Basis, const double theta,
                          const unsigned int iCF, const unsigned int iX,
                          const unsigned int iN )
{
  double result = Basis.BasisEval( U, iX, iCF, iN, false );
  result -= U( iCF, iX, 0 );
  result *= theta;
  result += U( iCF, iX, 0 );
  return result;
}

double TargetFunc( const Kokkos::View<double***> U, const ModalBasis& Basis,
                   const double theta, const unsigned int iX,
                   const unsigned int iN )
{
  const double w  = std::min( 1.0e-13, ComputeInternalEnergy( U, iX ) );
  const double s1 = ComputeThetaState( U, Basis, theta, 1, iX, iN );
  const double s2 = ComputeThetaState( U, Basis, theta, 2, iX, iN );

  double e = s2 - 0.5 * s1 * s1;

  return e - w;
}


double Bisection( const Kokkos::View<double***> U, const ModalBasis& Basis,
                  const unsigned int iX, const unsigned int iN )
{
  const double TOL             = 1e-10;
  const unsigned int MAX_ITERS = 100;

  // bisection bounds on theta
  double a = 0.0;
  double b = 1.0;
  double c = 0.5;

  double fa = 0.0; // f(a) etc
  double fb = 0.0;
  double fc = 0.0;

  unsigned int n = 0;
  while ( n <= MAX_ITERS )
  {
    c = ( a + b ) / 2.0;

    fa = TargetFunc( U, Basis, a, iX, iN );
    fb = TargetFunc( U, Basis, b, iX, iN );
    fc = TargetFunc( U, Basis, c, iX, iN );

    if ( std::abs( fc <= TOL / 10.0 ) || ( b - a ) / 2.0 < TOL )
    {
      return c;
    }

    // new interval
    if ( sgn( fc ) == sgn( fa ) )
    {
      a = c;
    }
    else
    {
      b = c;
    }

    n++;
  }

  std::printf( "Max Iters Reach In Bisection\n" );
  return c;
}


double Backtrace( const Kokkos::View<double***> U, const ModalBasis& Basis,
                  const unsigned int iX, const unsigned int iN )
{
  double theta = 1.0;
  double nodal = - 1.0;

  while ( theta >= 0.01 && nodal < 0.0 )
  {
    nodal = TargetFunc( U, Basis, theta, iX, iN );

    theta -= 0.05;
  }

  return theta;
}