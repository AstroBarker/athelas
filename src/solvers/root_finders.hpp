#ifndef ROOT_FINDERS_HPP_
#define ROOT_FINDERS_HPP_

/**
 * File    : root_finders.hpp
 * Author  : Brandon Barker
 * Purpose : Implement various root finders
 **/

#include <algorithm>
#include <cstdio>

#include "abstractions.hpp"
#include "error.hpp"

namespace Solver_Opts {

constexpr Real FPTOL             = 1.0e-11;
constexpr unsigned int MAX_ITERS = 50;

} // namespace Solver_Opts

/* templated residual function */
template <typename T, typename F, typename... Args>
T residual( F g, T x0, Args... args ) {
  return g( x0, args... ) - x0;
}

template <typename T, typename F>
T residual( F g, T x0 ) {
  return g( x0 ) - x0;
}

/* returns true if bracket contains a root (ya * yb < 0) */
bool check_bracket( Real ya, Real yb ) { return ( ya * yb <= 0.0 ); }

/**
 * Fixed point solver templated on type, function
 **/
template <typename T, typename F>
T fixed_point( F target, T x0 ) {

  unsigned int n = 0;

  T error = 1.0;
  T ans;
  while ( n <= Solver_Opts::MAX_ITERS && error >= Solver_Opts::FPTOL ) {
    T x1  = target( x0 );
    error = std::fabs( x1 - x0 );
    x0    = x1;

    n++;

#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, x1, error );
#endif
    ans = x1;
  }
  if ( n == Solver_Opts::MAX_ITERS ) {
    throw Error( " ! Fixed Point :: Failed To Converge ! \n" );
  }

  return ans;
}

/**
 * Anderson accelerated fixed point solver templated on type, function
 **/
template <typename T, typename F>
T fixed_pointAA( F target, T x0 ) {

  unsigned int n = 0;

  T error = 1.0;
  T xkm1, xk, xkp1;
  xk   = target( x0 ); // one fixed point step
  xkm1 = x0;
  while ( n <= Solver_Opts::MAX_ITERS && error >= Solver_Opts::FPTOL ) {
    /* Anderson acceleration step */
    T alpha = -residual( xk ) / ( residual( xkm1 ) - residual( xk ) );

    T xkp1 = alpha * target( xkm1 ) + ( 1.0 - alpha ) * target( xk );
    error  = std::fabs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    n++;

#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xk, error );
#endif
  }
  if ( n == Solver_Opts::MAX_ITERS ) {
    throw Error(
        " ! Anderson Accelerated Fixed Point :: Failed To Converge ! \n" );
  }

  return xk;
}

/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 * Assumes target is of type f(x) = x
 * TODO: update style?
 **/
template <typename T, typename F, typename... Args>
T fixed_point_aa( F target, T x0, Args... args ) {

  unsigned int n = 0;

  T error = 1.0;
  T xkm1, xk, xkp1;
  xk   = target( x0, args... ); // one fixed point step
  xkm1 = x0;
  while ( n <= Solver_Opts::MAX_ITERS && error >= Solver_Opts::FPTOL ) {
    /* Anderson acceleration step */
    T alpha =
        -residual( target, xk, args... ) /
        ( residual( target, xkm1, args... ) - residual( target, xk, args... ) );

    T xkp1 = alpha * target( xkm1, args... ) +
             ( 1.0 - alpha ) * target( xk, args... );
    error = std::fabs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    n++;

#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xk, error );
#endif
  }
  if ( n == Solver_Opts::MAX_ITERS ) {
    throw Error(
        " ! Anderson Accelerated Fixed Point :: Failed To Converge ! \n" );
  }

  return xk;
}

/* Newton iteration templated on type, function */
template <typename T, typename F>
T newton( F target, F dTarget, T x0 ) {

  unsigned int n = 0;

  T h     = target( x0 ) / dTarget( x0 );
  T error = 1.0;
  while ( n <= Solver_Opts::MAX_ITERS && error >= Solver_Opts::FPTOL ) {
    T xn  = x0;
    T h   = target( xn ) / dTarget( xn );
    x0    = xn - h;
    error = std::fabs( xn - x0 );

    n++;

#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xn, error );
#endif
  }
  if ( n == Solver_Opts::MAX_ITERS ) {
    throw Error( " ! Newton :: Failed To Converge ! \n" );
  }
  return x0;
}

/* Anderson Accelerated Newton iteration templated on type, function */
template <typename T, typename F>
T newton_aa( F target, F dTarget, T x0 ) {

  unsigned int n = 0;

  T h     = target( x0 ) / dTarget( x0 );
  T error = 1.0;
  T xkm1, xk, xkp1;
  xk   = x0 - h;
  xkm1 = x0;
  T ans;
  while ( n <= Solver_Opts::MAX_ITERS && error >= Solver_Opts::FPTOL ) {
    T hp1 = target( xk ) / dTarget( xk );
    T h   = target( xkm1 ) / dTarget( xkm1 );
    /* Anderson acceleration step */
    T gamma = hp1 / ( hp1 - h );

    xkp1  = xk - hp1 - gamma * ( xk - xkm1 - hp1 + h );
    error = std::fabs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    n++;

#ifdef ATHELAS_DEBUG
    printf( " %d %e %e \n", n, xk, error );
#endif
    ans = xk;
  }
  if ( n == Solver_Opts::MAX_ITERS ) {
    throw Error( " ! Anderson Accelerated Newton :: Failed To Converge ! \n" );
  }

  return ans;
}

/* Anderson Accelerated Newton iteration templated on type, function, args */
template <typename T, typename F, typename... Args>
T newton_aa( F target, F dTarget, T x0, Args... args ) {

  unsigned int n = 0;

  T h     = target( x0, args... ) / dTarget( x0, args... );
  T error = 1.0;
  T xkm1, xk, xkp1;
  xk   = x0 - h;
  xkm1 = x0;
  T ans;
  while ( n <= Solver_Opts::MAX_ITERS && error >= Solver_Opts::FPTOL ) {
    T hp1 = target( xk, args... ) / dTarget( xk, args... );
    T h   = target( xkm1, args... ) / dTarget( xkm1, args... );
    /* Anderson acceleration step */
    T gamma = hp1 / ( hp1 - h );

    xkp1  = xk - hp1 - gamma * ( xk - xkm1 - hp1 + h );
    error = std::fabs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    n++;

#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xk, error );
#endif
    ans = xk;
  }
  if ( n == Solver_Opts::MAX_ITERS ) {
    throw Error( " ! Anderson Accelerated Newton :: Failed To Converge ! \n" );
  }

  return ans;
}

#endif // ROOT_FINDERS_HPP_
