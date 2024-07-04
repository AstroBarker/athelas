/**
 * File    : root_finders.hpp
 * Author  : Brandon Barker
 * Purpose : Root finders
 **/

#ifndef ROOT_FINDERS_HPP_
#define ROOT_FINDERS_HPP_

#include <cmath>
#include <cstdio>

#include "error.hpp"
#include "root_finder_opts.hpp"

/* templated residual function */
template <typename T, typename F, typename... Args>
T Residual( F g, T x0, Args... args ) {
  return g( x0, args... ) - x0;
}

/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 **/
template <typename T, typename F, typename... Args>
T FixedPointAA( F target, T x0, Args... args ) {

  // puts f(x) = 0 into fixed point form
  auto f = [&]( const Real x, Args... args ) {
    return target( x, args... ) + x;
  };

  unsigned int n = 0;
  T error        = 1.0;
  T xkm1, xk, xkp1;
  xk   = target( x0, args... ); // one fixed point step
  xkm1 = x0;
  if ( std::abs( xk - x0 ) <= Root_Finder_Opts::FPTOL ) return xk;
  while ( n <= Root_Finder_Opts::MAX_ITERS &&
          error >= Root_Finder_Opts::FPTOL ) {
    /* Anderson acceleration step */
    T alpha =
        -Residual( target, xk, args... ) /
        ( Residual( target, xkm1, args... ) - Residual( target, xk, args... ) );

    T xkp1 = alpha * target( xkm1, args... ) +
             ( 1.0 - alpha ) * target( xk, args... );
    error = std::abs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    n += 1;

#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xk, error );
#endif
    if ( n == Root_Finder_Opts::MAX_ITERS ) {
      throw Error( " ! Root Finder :: Anderson Accelerated Fixed Point "
                   "Iteration Failed To "
                   "Converge ! \n" );
    }
  }

  return xk;
}

/* Fixed point solver templated on type, function, and args for func */
template <typename T, typename F, typename... Args>
T FixedPointSolve( F target, T x0, Args... args ) {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= Root_Finder_Opts::MAX_ITERS &&
          error >= Root_Finder_Opts::FPTOL ) {
    T x1  = target( x0, args... );
    error = std::abs( Residual( target, x0, args... ) );
#ifdef ATHELAS_DEBUG
    std::printf( " %d %f %f \n", n, x1, error );
#endif
    x0 = x1;
    n += 1;

    if ( n == Root_Finder_Opts::MAX_ITERS ) {
      throw Error(
          " ! Root Finder :: Fixed Point Iteration Failed To Converge ! \n" );
    }
  }

  return x0;
}

template <typename T, typename F>
T FixedPointSolve( F target, T x0 ) {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= Root_Finder_Opts::MAX_ITERS &&
          error >= Root_Finder_Opts::FPTOL ) {
    T x1  = target( x0 );
    error = std::abs( Residual( target, x0 ) );
#ifdef ATHELAS_DEBUG
    std::printf( " %d %f %f \n", n, x1, error );
#endif
    x0 = x1;
    n += 1;

    if ( n == Root_Finder_Opts::MAX_ITERS ) {
      throw Error(
          " ! Root Finder :: Fixed Point Iteration Failed To Converge ! \n" );
    }
  }

  return x0;
}

/* Newton iteration templated on type, function, args */
template <typename T, typename F, typename... Args>
T Newton( F target, F dTarget, T x0, Args... args ) {

  unsigned int n = 0;
  T h            = target( x0, args... ) / dTarget( x0, args... );
  T error        = 1.0;
  while ( n <= Root_Finder_Opts::MAX_ITERS &&
          error >= Root_Finder_Opts::FPTOL ) {
    T xn  = x0;
    T h   = target( xn, args... ) / dTarget( xn, args... );
    x0    = xn - h;
    error = std::abs( xn - x0 );
    n += 1;
#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xn, error );
#endif
    if ( n == Root_Finder_Opts::MAX_ITERS ) {
      throw Error(
          " ! Root Finder :: Newton Iteration Failed To Converge ! \n" );
    }
  }
  return x0;
}

/* Anderson Accelerated Newton iteration templated on type, function */
template <typename T, typename F, typename... Args>
T AANewton( F target, F dTarget, T x0, Args... args ) {

  unsigned int n = 0;
  T h            = target( x0, args... ) / dTarget( x0, args... );
  T error        = 1.0;
  T xkm1, xk, xkp1;
  xk   = std::min( x0 - h, Root_Finder_Opts::FPTOL ); // keep positive definite
  xkm1 = x0;
  T ans;
  if ( std::abs( xk - x0 ) <= Root_Finder_Opts::FPTOL ) return xk;
  while ( n <= Root_Finder_Opts::MAX_ITERS &&
          error >= Root_Finder_Opts::FPTOL ) {
    T hp1 = target( xk, args... ) / dTarget( xk, args... );
    T h   = target( xkm1, args... ) / dTarget( xkm1, args... );
    /* Anderson acceleration step */
    T gamma = hp1 / ( hp1 - h );

    xkp1  = xk - hp1 - gamma * ( xk - xkm1 - hp1 + h );
    error = std::abs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    n += 1;
#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xk, error );
#endif
    if ( n == Root_Finder_Opts::MAX_ITERS ) {
      throw Error( " ! Root Finder :: Anderson Accelerated Newton Iteration "
                   "Failed To Converge ! \n" );
    }
    ans = xk;
  }
  return ans;
}

#endif // ROOT_FINDERS_HPP_
