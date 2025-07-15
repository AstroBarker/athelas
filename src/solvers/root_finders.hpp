#pragma once
/**
 * @file root_finders.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Root finders
 *
 * @details Root finders provided for various needs:
 *          - fixed_point
 *          - newton
 *
 *          Both are implemented with an Anderson acceleration scheme.
 *          The contents here are in a state of mess..
 */

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "concepts/arithmetic.hpp"
#include "error.hpp"
#include "rad_discretization.hpp"
#include "radiation/rad_discretization.hpp"
#include "root_finder_opts.hpp"
#include "utils/utilities.hpp"

namespace root_finders {

using utilities::l2_norm;
using utilities::l2_norm_diff;
using utilities::ratio;

/* templated residual function */
// template <typename T, typename F, typename... Args>
// double residual( F g, T x0, Args... args ) {
//   return g( x0, args... ) - x0;
// }

template <typename T, typename F, typename... Args>
auto residual( F g, T x0, const int k, const int iC, Args... args ) -> double {
  return g( x0, k, iC, args... ) - x0( iC, k );
}

template <Subtractable T>
KOKKOS_INLINE_FUNCTION
auto residual( const T f, const T x ) -> T {
  return f - x;
}

KOKKOS_INLINE_FUNCTION
auto alpha_aa( const double r_n, const double r_nm1 ) -> double {
  //return (r_n == r_nm1) * utilities::ratio(r_n, (r_n - r_nm1));
  //return utilities::ratio(r_n, (r_n - r_nm1));
  return utilities::make_bounded(utilities::ratio(r_n, (r_n - r_nm1)), 0.0, 1.0);
}


/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 * Assumes target is in f(x) = x form
 **/
/*
template <typename T, typename F, typename... Args>
auto fixed_point_aa( F target, T x0, Args... args ) -> T {

  unsigned int n = 0;

  T error = 1.0;
  T xkm1  = 0.0;
  T xk    = 0.0;
  T xkp   = 0.01;
  xk      = target( x0, args... ); // one fixed point step
  xkm1    = x0;
  if ( std::abs( xk - x0 ) <= root_finders::FPTOL ) {
    return xk;
  }
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    //--- Anderson acceleration step --- //
    T alpha =
        -residual( target, xk, args... ) /
        ( residual( target, xkm1, args... ) - residual( target, xk, args... ) );

    T xkp1 = ( alpha * target( xkm1, args... ) ) +
             ( ( 1.0 - alpha ) * target( xk, args... ) );
    error = std::abs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    ++n;

#ifdef ATHELAS_DEBUG
    // std::println( " {} {:e}, {:e}" n, xk, error );
#endif
  }

  return xk;
}
*/
/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 * Assumes target is in f(x) = 0 form and transforms
 **/
template <typename T, typename F, typename... Args>
auto fixed_point_aa_root( F target, T x0, Args... args ) -> T {

  // puts f(x) = 0 into fixed point form
  auto f = [&]( const double x, Args... args ) {
    return target( x, args... ) + x;
  };
  // residual function, used in AA algorithm
  auto g = [&]( const double x, Args... args ) { return f( x, args... ) - x; };

  unsigned int n = 0;

  T error = 1.0;
  T xkm1  = 0.0;
  T xk    = 0.0;
  T xkp1  = 0.0;
  xk      = f( x0, args... ); // one fixed point step
  xkm1    = x0;
  if ( std::abs( xk - x0 ) <= root_finders::FPTOL ) {
    return xk;
  }
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    /* Anderson acceleration step */
    T alpha = -g( xk, args... ) / ( g( xkm1, args... ) - g( xk, args... ) );

    T xkp1 =
        ( alpha * f( xkm1, args... ) ) + ( ( 1.0 - alpha ) * f( xk, args... ) );
    error = std::abs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    ++n;

#ifdef ATHELAS_DEBUG
    // std::println( " {} {:e} {:e}", n, xk, error );
#endif
  }

  return xk;
}

/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 * Note that this is only used for the rad-hydro implicit update
 * It is void and the result is stored in T scratch_nm1
 * TODO(astrobarker) remove input int k
 * TODO(astrobarker) change scratch_n to proper scratch
 **/
template <typename F, typename T, typename G, typename... Args>
KOKKOS_INLINE_FUNCTION
void fixed_point_aa( F target, const int dummy, T scratch_n, G scratch_nm1, G scratch, const int iC,
                     Args... args ) {

  static_assert(T::rank == 2, "fixed_point_aa expects rank-2 views.");
  const int num_modes = scratch_n.extent(1);


  double error = 1.0;
  // --- first fixed point iteration ---
  for (int k = 0; k < num_modes; ++k) {
    scratch_nm1(iC, k) = scratch_n(iC, k); // set to initial guess
    scratch_n(iC, k) = target( scratch_n, k, iC, args... );
  }

  error = l2_norm_diff( scratch_n, scratch_nm1, iC ) / (l2_norm( scratch_n, iC ) + 1.0e-20);
  //error = ratio(l2_norm_diff( scratch_n, scratch_nm1, iC ), l2_norm( scratch_n, iC ) + 1.0e-20);

  // --- check convergence before iterating ---
  if ( error <= root_finders::RELTOL ) {
    return;
  }

  unsigned int n = 1;
  // TODO(astrobarker): can likely optimize by precomputing target calls
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::RELTOL ) {
    for (int k = 0; k < num_modes; ++k) {
      /* Anderson acceleration step */
      const double alpha = -residual( target, scratch_n, k, iC, args... ) /
                   ( residual( target, scratch_nm1, k, iC, args... ) -
                     residual( target, scratch_n, k, iC, args... ) );
      if (std::isnan(alpha)) {
        scratch(iC, k) = scratch_n(iC, k);
        continue;
      }

      const double xkp1_k = alpha * target( scratch_nm1, k, iC, args... ) +
             ( 1.0 - alpha ) * target( scratch_n, k, iC, args... );
      scratch(iC, k) = xkp1_k; // store new solution in scrach
    }
    // --- update --- // TODO move up
    for (int k = 0; k < num_modes; ++k) {
      scratch_nm1(iC, k) = scratch_n(iC, k);
      scratch_n(iC, k) = scratch(iC, k);
    }


    //error = std::abs( xk - xkp1 ) / std::abs( xk );
    error = l2_norm_diff( scratch_n, scratch_nm1, iC );// / l2_norm( scratch_n, iC );

    // #ifdef ATHELAS_DEBUG
    if ( std::isnan(scratch(iC, 1)) ) {
      THROW_ATHELAS_ERROR("nan in implicit solve!");
    }
    // #endif

    // TODO(astrobarker): handle convergence failures?
    // if ( n == root_finders::MAX_ITERS ) {
    //  std::printf("FPAA convergence failure! Error: %e\n", error);
    //}
    ++n;
  }

  return;
}

template <typename T, typename ... Args>
KOKKOS_INLINE_FUNCTION
void fixed_point_radhydro(T R, double dt_a_ii, T scratch_n, T scratch_nm1, T scratch, Args... args) {
  static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  static constexpr double c = constants::c_cgs;
  static constexpr double min_error = 1.0e-15;
  constexpr static int nvars = 5;

  const int num_modes = scratch_n.extent(1);

  auto target = [&]( T u, const int k ) {
      const auto [s_1_k, s_2_k, s_3_k, s_4_k] = radiation::compute_increment_radhydro_source(u, k, args...);
      return std::make_tuple(R(1, k) + dt_a_ii * s_1_k, 
      R(2, k) + dt_a_ii * s_2_k, 
      R(3, k) + dt_a_ii * s_3_k, 
      R(4, k) + dt_a_ii * s_4_k);
  };

  auto error_func = [&]( T u_n, T u_nm1, const int q ) {
    double e = 0.0;
    for (int k = 0; k < num_modes; ++k) {
      e += std::pow((std::abs(u_n(q, k) - u_nm1(q, k))) / std::max({std::abs(u_n(q,k)), std::abs(u_nm1(q, k)), min_error}), 2.0);
    }
    return std::sqrt(e);
  };


  double error = 1.0;
  for (int iC = 0; iC < nvars; ++iC) {
    for (int k = 0; k < num_modes; ++k) {
      scratch_nm1(iC, k) = scratch_n(iC, k); // set to initial guess
    }
  }

  unsigned int n = 0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::RELTOL ) {
    for (int k = 0; k < num_modes; ++k) {
      const auto [xkp1_1_k, xkp1_2_k, xkp1_3_k, xkp1_4_k] = target(scratch_n, k);
      scratch(1, k) = xkp1_1_k; // fluid vel
      scratch(2, k) = xkp1_2_k; // fluid energy
      scratch(3, k) = xkp1_3_k; // rad energy
      scratch(4, k) = xkp1_4_k; // rad flux
    }

    // --- update --- // TODO move up
  for (int iC = 1; iC < nvars; ++iC) {
    for (int k = 0; k < num_modes; ++k) {
      scratch_nm1(iC, k) = scratch_n(iC, k);
      scratch_n(iC, k) = scratch(iC, k);
    }
  }

  error = std::max({
      error_func(scratch_n, scratch_nm1, 1), 
      error_func(scratch_n, scratch_nm1, 2), 
      error_func(scratch_n, scratch_nm1, 3), 
      error_func(scratch_n, scratch_nm1, 4)
      });
  //std::println("n, e_nm1, e_n, error {} {:.13e} {:.13e} {:.5e}", n, scratch_nm1(3, 1), scratch_n(3, 1), error);

    if ( n == root_finders::MAX_ITERS ) {
    std::println( "energy avg :: n xk error {} {:e} {:e}", n, scratch_n(3, 0), error );
    }
  
  ++n;
  } // while not converged
}

template <typename T, typename ... Args>
KOKKOS_INLINE_FUNCTION
void fixed_point_radhydro_aa(T R, double dt_a_ii, T scratch_n, T scratch_nm1, T scratch, Args... args) {
  static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  static constexpr double c = constants::c_cgs;
  static constexpr double min_error = 1.0e-15;
  constexpr static int nvars = 5;

  const int num_modes = scratch_n.extent(1);

  auto target = [&]( T u, const int k ) {
      const auto [s_1_k, s_2_k, s_3_k, s_4_k] = radiation::compute_increment_radhydro_source(u, k, args...);
      return std::make_tuple(R(1, k) + dt_a_ii * s_1_k, 
      R(2, k) + dt_a_ii * s_2_k, 
      R(3, k) + dt_a_ii * s_3_k, 
      R(4, k) + dt_a_ii * s_4_k);
  };

  auto error_func = [&]( T u_n, T u_nm1, const int q ) {
    double e = 0.0;
    for (int k = 0; k < num_modes; ++k) {
      e += std::pow((std::abs(u_n(q, k) - u_nm1(q, k))) / std::max({std::abs(u_n(q,k)), std::abs(u_nm1(q, k)), min_error}), 2.0);
    }
    return std::sqrt(e);
  };

  // --- first fixed point iteration ---
  for (int k = 0; k < num_modes; ++k) {
    const auto [xnp1_1_k, xnp1_2_k, xnp1_3_k, xnp1_4_k] = target(scratch_n, k);
    scratch(1, k) = xnp1_1_k;
    scratch(2, k) = xnp1_2_k;
    scratch(3, k) = xnp1_3_k;
    scratch(4, k) = xnp1_4_k;
  }
  for (int iC = 1; iC < nvars; ++iC) {
    for (int k = 0; k < num_modes; ++k) {
      scratch_nm1(iC, k) = scratch_n(iC, k);
      scratch_n(iC, k) = scratch(iC, k);
    }
  }

  double error = std::max({error_func(scratch_n, scratch_nm1, 1), 
      error_func(scratch_n, scratch_nm1, 2), 
      error_func(scratch_n, scratch_nm1, 3), 
      error_func(scratch_n, scratch_nm1, 4) / c});

  if (error <= root_finders::RELTOL) {
    return;
  }

  unsigned int n = 1;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::RELTOL ) {
    for (int k = 0; k < num_modes; ++k) {
      const auto [s_1_n, s_2_n, s_3_n, s_4_n] = target(scratch_n, k);
      const auto [s_1_nm1, s_2_nm1, s_3_nm1, s_4_nm1] = target(scratch_nm1, k);

      // residuals
      const auto r_1_n = residual(s_1_n, scratch_n(1,k));
      const auto r_2_n = residual(s_2_n, scratch_n(2,k));
      const auto r_3_n = residual(s_3_n, scratch_n(3,k));
      const auto r_4_n = residual(s_4_n, scratch_n(4,k));
      const auto r_1_nm1 = residual(s_1_nm1, scratch_nm1(1,k));
      const auto r_2_nm1 = residual(s_2_nm1, scratch_nm1(2,k));
      const auto r_3_nm1 = residual(s_3_nm1, scratch_nm1(3,k));
      const auto r_4_nm1 = residual(s_4_nm1, scratch_nm1(4,k));

      // Anderson acceleration alpha
      const auto a_1 = alpha_aa(r_1_n, r_1_nm1);
      const auto a_2 = alpha_aa(r_2_n, r_2_nm1);
      const auto a_3 = alpha_aa(r_3_n, r_3_nm1);
      const auto a_4 = alpha_aa(r_4_n, r_4_nm1);
      //std::println("rn {} {} {} {}", r_1_n, r_2_n, r_3_n, r_4_n);
      //std::println("rnm1 {} {} {} {}", r_1_nm1, r_2_nm1, r_3_nm1, r_4_nm1);
      //std::println("alpha {} {} {} {}", a_1, a_2, a_2, a_4);

      // Anderson acceleration update
      const auto xnp1_1_k = a_1 * s_1_nm1 + (1.0 - a_1) * s_1_n;
      const auto xnp1_2_k = a_2 * s_2_nm1 + (1.0 - a_2) * s_2_n;
      const auto xnp1_3_k = a_3 * s_3_nm1 + (1.0 - a_3) * s_3_n;
      const auto xnp1_4_k = a_4 * s_4_nm1 + (1.0 - a_4) * s_4_n;

      scratch(1, k) = xnp1_1_k; // fluid vel
      scratch(2, k) = xnp1_2_k; // fluid energy
      scratch(3, k) = xnp1_3_k; // rad energy
      scratch(4, k) = xnp1_4_k; // rad flux
    }

    // --- update --- // TODO move up
  for (int iC = 1; iC < nvars; ++iC) {
    for (int k = 0; k < num_modes; ++k) {
      scratch_nm1(iC, k) = scratch_n(iC, k);
      scratch_n(iC, k) = scratch(iC, k);
    }
  }

  error = std::max({error_func(scratch_n, scratch_nm1, 1), 
      error_func(scratch_n, scratch_nm1, 2), 
      error_func(scratch_n, scratch_nm1, 3), 
      error_func(scratch_n, scratch_nm1, 4) / c});

  //std::println("n, e_nm1, e_n, error {} {:.13e} {:.13e} {:.5e}", n, scratch_nm1(3, 1), scratch_n(3, 1), error);

    if ( n == root_finders::MAX_ITERS ) {
    std::println( "energy avg :: n xk error {} {:e} {:e}", n, scratch_n(3, 0), error );
    }
  
  ++n;
  } // while not converged
}

// unused standard fixed point iteration
template <typename F, typename T, typename G, typename... Args>
KOKKOS_INLINE_FUNCTION
void fixed_point_implicit( F target, const int dummy, T scratch_n, G scratch_nm1, G scratch, const int iC,
                     Args... args ) {

  static_assert(T::rank == 2, "fixed_point_aa expects rank-2 views.");
  const int num_modes = scratch_n.extent(1);


  double error = 1.0;
  // --- first fixed point iteration ---
  for (int k = 0; k < num_modes; ++k) {
    scratch_nm1(iC, k) = scratch_n(iC, k); // set to initial guess
    scratch_n(iC, k) = target( scratch_n, k, iC, args... );
  }

  unsigned int n = 0;
  // TODO(astrobarker): can likely optimize by precomputing target calls
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::RELTOL ) {
    for (int k = 0; k < num_modes; ++k) {
      const double xkp1_k = target( scratch_n, k, iC, args... );
      scratch(iC, k) = xkp1_k; // store new solution in scrach
    }
    // --- update --- // TODO move up
    for (int k = 0; k < num_modes; ++k) {
      scratch_nm1(iC, k) = scratch_n(iC, k);
      scratch_n(iC, k) = scratch(iC, k);
    }


    //error = std::abs( xk - xkp1 ) / std::abs( xk );
    error = l2_norm_diff( scratch_n, scratch_nm1, iC ) / l2_norm( scratch_n, iC );

    // #ifdef ATHELAS_DEBUG
//    std::println( "n iC, k xk error {} {} {} {:e} {:e}", n, iC, 1, scratch_n(iC, 1), error );
    // #endif

    // TODO(astrobarker): handle convergence failures?
    if ( n == root_finders::MAX_ITERS ) {
      std::printf("FPAA convergence failure! Error: %e\n", error);
    std::println( "n iC, k xk error {} {} {} {:e} {:e}", n, iC, 0, scratch_n(iC, 0), error );
    }
    ++n;
  }

  return;
}

/* Fixed point solver templated on type, function, and args for func */
template <typename T, typename F, typename... Args>
auto fixed_point( F target, T x0, Args... args ) -> T {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    T x1  = target( x0, args... );
    error = std::abs( residual( target, x0, args... ) );
#ifdef ATHELAS_DEBUG
    // std::println( " {} {:e} {:e}", n, xk, error );
#endif
    x0 = x1;
    n += 1;

    if ( n == root_finders::MAX_ITERS ) {
      THROW_ATHELAS_ERROR(
          " ! Root Finder :: Fixed Point Iteration Failed To Converge ! \n" );
    }
  }

  return x0;
}

template <typename T, typename F>
auto fixed_point( F target, T x0 ) -> T {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::FPTOL ) {
    T x1  = target( x0 );
    error = std::abs( residual( target, x0 ) );
#ifdef ATHELAS_DEBUG
    // std::println( " {} {:e} {:e}", n, xk, error );
#endif
    x0 = x1;
    n += 1;

    if ( n == root_finders::MAX_ITERS ) {
      THROW_ATHELAS_ERROR(
          " ! Root Finder :: Fixed Point Iteration Failed To Converge ! \n" );
    }
  }

  return x0;
}

/* Newton iteration templated on type, function, args */
template <typename T, typename F, typename... Args>
auto newton( F target, F dTarget, T x0, Args... args ) -> T {

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
    // std::println( " {} {:e} {:e}", n, xk, error );
#endif
    if ( n == root_finders::MAX_ITERS ) {
      THROW_ATHELAS_ERROR(
          " ! Root Finder :: Newton Iteration Failed To Converge ! \n" );
    }
  }
  return x0;
}

/* Anderson Accelerated newton iteration templated on type, function */
template <typename T, typename F, typename... Args>
auto newton_aa( F target, F dTarget, T x0, Args... args ) -> T {

  unsigned int n = 0;

  T h     = target( x0, args... ) / dTarget( x0, args... );
  T error = 1.0;
  T xkm1  = 0.0;
  T xk    = 0.0;
  T xkp1  = 0.0;
  xk      = std::min( x0 - h, root_finders::FPTOL ); // keep positive definite
  xkm1    = x0;
  T result;
  if ( std::abs( xk - x0 ) <= root_finders::FPTOL ) {
    return xk;
  }
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
    // std::println( " {} {:e} {:e}", n, xk, error );
#endif
    if ( n == root_finders::MAX_ITERS ) {
      THROW_ATHELAS_ERROR(
          " ! Root Finder :: Anderson Accelerated Newton Iteration "
          "Failed To Converge ! \n" );
    }
    result = xk;
  }
  return result;
}

} // namespace root_finders
