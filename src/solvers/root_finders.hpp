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

namespace root_finders {

/* templated residual function */
// template <typename T, typename F, typename... Args>
// Real Residual( F g, T x0, Args... args ) {
//   return g( x0, args... ) - x0;
// }

template <typename T, typename F, typename... Args>
Real Residual( F g, T x0, const int k, const int iC, Args... args ) {
  return g( x0, k, iC, args... ) - x0(iC, k);
}

/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 **/
// template <typename T, typename F, typename... Args>
// T fixed_point_aa( F target, T x0, Args... args ) {
//
//   // puts f(x) = 0 into fixed point form
//   auto f = [&]( const Real x, Args... args ) {
//     return target( x, args... ) + x;
//   };
//
//   unsigned int n = 0;
//   T error        = 1.0;
//   T xkm1, xk, xkp1;
//   xk   = target( x0, args... ); // one fixed point step
//   xkm1 = x0;
//   if ( std::abs( xk - x0 ) <= root_finders::FPTOL ) return xk;
//   while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
//     /* Anderson acceleration step */
//     T alpha =
//         -Residual( target, xk, args... ) /
//         ( Residual( target, xkm1, args... ) - Residual( target, xk, args... )
//         );
//
//     T xkp1 = alpha * target( xkm1, args... ) +
//              ( 1.0 - alpha ) * target( xk, args... );
//     error = std::abs( xk - xkp1 );
//
//     xkm1 = xk;
//     xk   = xkp1;
//
//     n += 1;
//
// #ifdef ATHELAS_DEBUG
//     std::printf( " %d %e %e \n", n, xk, error );
// #endif
//     if ( n == root_finders::MAX_ITERS ) {
//       throw Error( " ! Root Finder :: Anderson Accelerated Fixed Point "
//                    "Iteration Failed To "
//                    "Converge ! \n" );
//     }
//   }
//
//   return xk;
// }

/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 **/
//TODO: NOTE: What if the scratch array contains u_
// TODO: the problem here is that the source term functions 
// want a full basis function U(k), but we are iterating 
// for a single k, and the source term functions return a Real
// NOTE: Potential issue: reference symantics for the views here.
//   Might need to use vectors throughout
template <typename F, typename T, typename... Args>
Real fixed_point_aa( F target, const int k, T scratch, const int iC,
                     Args... args ) {

  // auto x0 = scratch[0];
  unsigned int n = 0;
  Real error     = 1.0;
  T xkm1, xk, xkp1;
  xk = scratch;
  std::printf("FILE :: Line = %s %d\n", __FILE__, __LINE__);
  xk(iC, k) = target( scratch, k, iC, args... ); // one fixed point step
  //xkm1 = scratch( k );
  xkm1 = scratch;
  xkp1 = xk;
  if ( std::abs( xk(iC, k)- xkm1(iC, k) ) <= root_finders::FPTOL ) return xk(iC, k);
  std::printf("FILE :: Line = %s %d\n", __FILE__, __LINE__);
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    /* Anderson acceleration step */
    // UPDATED TODO: 
    // Plan: get number back from xkp1, update scratch
    // I think the move is to pass in View1D u_h, u_r, construct source term for k
    Real alpha = -Residual( target, xk, k, iC, args... ) /
                 ( Residual( target, xkm1, k, iC, args... ) -
                   Residual( target, xk, k, iC, args... ) );
    std::printf("FILE :: Line = %s %d\n", __FILE__, __LINE__);

    xkp1(iC, k) = alpha * target( xkm1, k, iC, args... ) +
                ( 1.0 - alpha ) * target( xk, k, iC, args... );
    
    std::printf("FILE :: Line = %s %d\n", __FILE__, __LINE__);
    error = std::abs( xk(iC, k) - xkp1(iC, k));

    xkm1 = xk;
    xk   = xkp1;

    n += 1;
    std::printf(" ! DEBUG: xk, xkm1, xkp1 (at k) = (%f %f %f)", xk(iC, k), xkm1(iC, k), xkp1(iC, k));
//#ifdef ATHELAS_DEBUG
//    std::printf( " %d %e %e \n", n, xk, error );
//#endif
    if ( n == root_finders::MAX_ITERS ) {
      throw Error( " ! Root Finder :: Anderson Accelerated Fixed Point "
                   "Iteration Failed To "
                   "Converge ! \n" );
    }
  }

  return xk(iC, k);
}

/* Fixed point solver templated on type, function, and args for func */
template <typename T, typename F, typename... Args>
T fixed_point( F target, T x0, Args... args ) {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    T x1  = target( x0, args... );
    error = std::abs( Residual( target, x0, args... ) );
#ifdef ATHELAS_DEBUG
    std::printf( " %d %f %f \n", n, x1, error );
#endif
    x0 = x1;
    n += 1;

    if ( n == root_finders::MAX_ITERS ) {
      throw Error(
          " ! Root Finder :: Fixed Point Iteration Failed To Converge ! \n" );
    }
  }

  return x0;
}

template <typename T, typename F>
T fixed_point( F target, T x0 ) {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    T x1  = target( x0 );
    error = std::abs( Residual( target, x0 ) );
#ifdef ATHELAS_DEBUG
    std::printf( " %d %f %f \n", n, x1, error );
#endif
    x0 = x1;
    n += 1;

    if ( n == root_finders::MAX_ITERS ) {
      throw Error(
          " ! Root Finder :: Fixed Point Iteration Failed To Converge ! \n" );
    }
  }

  return x0;
}

/* Newton iteration templated on type, function, args */
template <typename T, typename F, typename... Args>
T newton( F target, F dTarget, T x0, Args... args ) {

  unsigned int n = 0;
  T h            = target( x0, args... ) / dTarget( x0, args... );
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    T xn  = x0;
    T h   = target( xn, args... ) / dTarget( xn, args... );
    x0    = xn - h;
    error = std::abs( xn - x0 );
    n += 1;
#ifdef ATHELAS_DEBUG
    std::printf( " %d %e %e \n", n, xn, error );
#endif
    if ( n == root_finders::MAX_ITERS ) {
      throw Error(
          " ! Root Finder :: Newton Iteration Failed To Converge ! \n" );
    }
  }
  return x0;
}

/* Anderson Accelerated newton iteration templated on type, function */
template <typename T, typename F, typename... Args>
T AAnewton( F target, F dTarget, T x0, Args... args ) {

  unsigned int n = 0;
  T h            = target( x0, args... ) / dTarget( x0, args... );
  T error        = 1.0;
  T xkm1, xk, xkp1;
  xk   = std::min( x0 - h, root_finders::FPTOL ); // keep positive definite
  xkm1 = x0;
  T ans;
  if ( std::abs( xk - x0 ) <= root_finders::FPTOL ) return xk;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
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
    if ( n == root_finders::MAX_ITERS ) {
      throw Error( " ! Root Finder :: Anderson Accelerated Newton Iteration "
                   "Failed To Converge ! \n" );
    }
    ans = xk;
  }
  return ans;
}

} // namespace root_finders

#endif // ROOT_FINDERS_HPP_
