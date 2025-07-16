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

#include "concepts/arithmetic.hpp"
#include "radiation/rad_discretization.hpp"
#include "root_finder_opts.hpp"
#include "utils/utilities.hpp"

namespace root_finders {

using utilities::ratio;

template <typename T, typename F, typename... Args>
auto residual( F g, T x0, const int k, const int iC, Args... args ) -> double {
  return g( x0, k, iC, args... ) - x0( iC, k );
}

template <Subtractable T>
KOKKOS_INLINE_FUNCTION auto residual( const T f, const T x ) -> T {
  return f - x;
}

KOKKOS_INLINE_FUNCTION
auto alpha_aa( const double r_n, const double r_nm1 ) -> double {
  return utilities::make_bounded( utilities::ratio( r_n, ( r_n - r_nm1 ) ), 0.0,
                                  1.0 );
}

// physical scales for normalization
// TODO(BLB): This can be simplified into an array or such.
struct PhysicalScales {
  double velocity_scale;
  double energy_scale;
  double rad_energy_scale;
  double rad_flux_scale;
};

// robust convergence metric for modal dg radiation hydrodynamics
template <typename t>
class RadHydroConvergence {
 private:
  PhysicalScales scales_;
  double abs_tol_;
  double rel_tol_;
  int num_modes_;

  std::vector<double> mode_weights_;

 public:
  explicit RadHydroConvergence( const PhysicalScales& scales,
                                double abs_tol = 1e-10, double rel_tol = 1e-8,
                                int num_modes = 1 )
      : scales_( scales ), abs_tol_( abs_tol ), rel_tol_( rel_tol ),
        num_modes_( num_modes ) {

    mode_weights_.resize( num_modes );
    for ( int k = 0; k < num_modes; ++k ) {
      // exponential decay for higher modes
      mode_weights_[k] = std::exp( -0.5 * k );
    }
  }

  // separate error metrics for different variable types
  // TODO(astrobarker) combine the following
  auto fluid_velocity_error( const t u_n, const t u_nm1, const int q )
      -> double {
    double max_error = 0.0;
    for ( int k = 0; k < num_modes_; ++k ) {
      const double abs_err = std::abs( u_n( q, k ) - u_nm1( q, k ) );
      const double scale =
          std::max( { scales_.velocity_scale, std::abs( u_n( q, k ) ),
                      std::abs( u_nm1( q, k ) ) } );
      const double normalized_err = abs_err / scale;
      const double weighted_err   = normalized_err * mode_weights_[k];
      max_error                   = std::max( max_error, weighted_err );
    }
    return max_error;
  }

  auto fluid_energy_error( const t u_n, const t u_nm1, const int q ) -> double {
    double max_error = 0.0;
    for ( int k = 0; k < num_modes_; ++k ) {
      const double abs_err = std::abs( u_n( q, k ) - u_nm1( q, k ) );
      const double scale =
          std::max( { scales_.energy_scale, std::abs( u_n( q, k ) ),
                      std::abs( u_nm1( q, k ) ) } );
      const double normalized_err = abs_err / scale;
      const double weighted_err   = normalized_err * mode_weights_[k];
      max_error                   = std::max( max_error, weighted_err );
    }
    return max_error;
  }

  auto radiation_energy_error( const t u_n, const t u_nm1, const int q )
      -> double {
    double max_error = 0.0;
    for ( int k = 0; k < num_modes_; ++k ) {
      const double abs_err = std::abs( u_n( q, k ) - u_nm1( q, k ) );
      const double scale =
          std::max( { scales_.rad_energy_scale, std::abs( u_n( q, k ) ),
                      std::abs( u_nm1( q, k ) ) } );
      const double normalized_err = abs_err / scale;
      const double weighted_err   = normalized_err * mode_weights_[k];
      max_error                   = std::max( max_error, weighted_err );
    }
    return max_error;
  }

  auto radiation_flux_error( const t u_n, const t u_nm1, const int q )
      -> double {
    double max_error = 0.0;
    for ( int k = 0; k < num_modes_; ++k ) {
      const double abs_err = std::abs( u_n( q, k ) - u_nm1( q, k ) );
      const double scale =
          std::max( { scales_.rad_flux_scale, std::abs( u_n( q, k ) ),
                      std::abs( u_nm1( q, k ) ) } );
      const double normalized_err = abs_err / scale;
      const double weighted_err   = normalized_err * mode_weights_[k];
      max_error                   = std::max( max_error, weighted_err );
    }
    return max_error;
  }

  // combined convergence check
  template <typename T>
  auto check_convergence( const T state_n, const T state_nm1 ) -> bool {

    double max_velocity_error   = 0.0;
    double max_energy_error     = 0.0;
    double max_rad_energy_error = 0.0;
    double max_rad_flux_error   = 0.0;

    max_velocity_error = std::max(
        max_velocity_error, fluid_velocity_error( state_n, state_nm1, 0 ) );
    max_energy_error     = std::max( max_energy_error,
                                     fluid_energy_error( state_n, state_nm1, 2 ) );
    max_rad_energy_error = std::max(
        max_rad_energy_error, radiation_energy_error( state_n, state_nm1, 3 ) );
    max_rad_flux_error = std::max(
        max_rad_flux_error, radiation_flux_error( state_n, state_nm1, 4 ) );

    // all variables must converge
    bool velocity_converged   = max_velocity_error < rel_tol_;
    bool energy_converged     = max_energy_error < rel_tol_;
    bool rad_energy_converged = max_rad_energy_error < rel_tol_;
    bool rad_flux_converged   = max_rad_flux_error < rel_tol_;

    return velocity_converged && energy_converged && rad_energy_converged &&
           rad_flux_converged;
  }
};

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
  if ( std::abs( xk - x0 ) <= root_finders::ABSTOL ) {
    return xk;
  }
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL ) {
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
  }

  return xk;
}
*/

template <typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro( T R, double dt_a_ii,
                                                  T scratch_n, T scratch_nm1,
                                                  T scratch, Args... args ) {
  static_assert( T::rank == 2, "fixed_point_radhydro expects rank-2 views." );
  constexpr static int nvars = 5;

  const int num_modes = scratch_n.extent( 1 );

  auto target = [&]( T u, const int k ) {
    const auto [s_1_k, s_2_k, s_3_k, s_4_k] =
        radiation::compute_increment_radhydro_source( u, k, args... );
    return std::make_tuple(
        R( 1, k ) + dt_a_ii * s_1_k, R( 2, k ) + dt_a_ii * s_2_k,
        R( 3, k ) + dt_a_ii * s_3_k, R( 4, k ) + dt_a_ii * s_4_k );
  };

  for ( int iC = 0; iC < nvars; ++iC ) {
    for ( int k = 0; k < num_modes; ++k ) {
      scratch_nm1( iC, k ) = scratch_n( iC, k ); // set to initial guess
    }
  }

  // Set up physical scales based on your problem
  PhysicalScales scales{ };
  scales.velocity_scale   = 1e7; // Typical velocity (cm/s)
  scales.energy_scale     = 1e12; // Typical energy density
  scales.rad_energy_scale = 1e12; // Typical radiation energy density
  scales.rad_flux_scale   = 1e20; // Typical radiation flux

  static RadHydroConvergence<View2D<double>> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, num_modes );

  unsigned int n = 0;
  bool converged = false;
  while ( n <= root_finders::MAX_ITERS && !converged ) {
    for ( int k = 0; k < num_modes; ++k ) {
      const auto [xkp1_1_k, xkp1_2_k, xkp1_3_k, xkp1_4_k] =
          target( scratch_n, k );
      scratch( 1, k ) = xkp1_1_k; // fluid vel
      scratch( 2, k ) = xkp1_2_k; // fluid energy
      scratch( 3, k ) = xkp1_3_k; // rad energy
      scratch( 4, k ) = xkp1_4_k; // rad flux

      // --- update ---
      for ( int iC = 1; iC < nvars; ++iC ) {
        scratch_nm1( iC, k ) = scratch_n( iC, k );
        scratch_n( iC, k )   = scratch( iC, k );
      }
    }

    converged = convergence_checker.check_convergence( scratch_n, scratch_nm1 );
    ++n;
  } // while not converged
}

template <typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro_aa( T R, double dt_a_ii,
                                                     T scratch_n, T scratch_nm1,
                                                     T scratch, Args... args ) {
  static_assert( T::rank == 2, "fixed_point_radhydro expects rank-2 views." );
  constexpr static int nvars = 5;

  const int num_modes = scratch_n.extent( 1 );

  auto target = [&]( T u, const int k ) {
    const auto [s_1_k, s_2_k, s_3_k, s_4_k] =
        radiation::compute_increment_radhydro_source( u, k, args... );
    return std::make_tuple(
        R( 1, k ) + dt_a_ii * s_1_k, R( 2, k ) + dt_a_ii * s_2_k,
        R( 3, k ) + dt_a_ii * s_3_k, R( 4, k ) + dt_a_ii * s_4_k );
  };

  // --- first fixed point iteration ---
  for ( int k = 0; k < num_modes; ++k ) {
    const auto [xnp1_1_k, xnp1_2_k, xnp1_3_k, xnp1_4_k] =
        target( scratch_n, k );
    scratch( 1, k ) = xnp1_1_k;
    scratch( 2, k ) = xnp1_2_k;
    scratch( 3, k ) = xnp1_3_k;
    scratch( 4, k ) = xnp1_4_k;
  }
  for ( int iC = 1; iC < nvars; ++iC ) {
    for ( int k = 0; k < num_modes; ++k ) {
      scratch_nm1( iC, k ) = scratch_n( iC, k );
      scratch_n( iC, k )   = scratch( iC, k );
    }
  }

  // Set up physical scales based on your problem
  PhysicalScales scales{ };
  scales.velocity_scale   = 1e7; // Typical velocity (cm/s)
  scales.energy_scale     = 1e12; // Typical energy density
  scales.rad_energy_scale = 1e12; // Typical radiation energy density
  scales.rad_flux_scale   = 1e20; // Typical radiation flux

  static RadHydroConvergence<View2D<double>> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, num_modes );

  bool converged =
      convergence_checker.check_convergence( scratch_n, scratch_nm1 );

  if ( converged ) {
    return;
  }

  unsigned int n = 1;
  while ( n <= root_finders::MAX_ITERS && !converged ) {
    for ( int k = 0; k < num_modes; ++k ) {
      const auto [s_1_n, s_2_n, s_3_n, s_4_n] = target( scratch_n, k );
      const auto [s_1_nm1, s_2_nm1, s_3_nm1, s_4_nm1] =
          target( scratch_nm1, k );

      // residuals
      const auto r_1_n   = residual( s_1_n, scratch_n( 1, k ) );
      const auto r_2_n   = residual( s_2_n, scratch_n( 2, k ) );
      const auto r_3_n   = residual( s_3_n, scratch_n( 3, k ) );
      const auto r_4_n   = residual( s_4_n, scratch_n( 4, k ) );
      const auto r_1_nm1 = residual( s_1_nm1, scratch_nm1( 1, k ) );
      const auto r_2_nm1 = residual( s_2_nm1, scratch_nm1( 2, k ) );
      const auto r_3_nm1 = residual( s_3_nm1, scratch_nm1( 3, k ) );
      const auto r_4_nm1 = residual( s_4_nm1, scratch_nm1( 4, k ) );

      // Anderson acceleration alpha
      const auto a_1 = alpha_aa( r_1_n, r_1_nm1 );
      const auto a_2 = alpha_aa( r_2_n, r_2_nm1 );
      const auto a_3 = alpha_aa( r_3_n, r_3_nm1 );
      const auto a_4 = alpha_aa( r_4_n, r_4_nm1 );

      // Anderson acceleration update
      const auto xnp1_1_k = a_1 * s_1_nm1 + ( 1.0 - a_1 ) * s_1_n;
      const auto xnp1_2_k = a_2 * s_2_nm1 + ( 1.0 - a_2 ) * s_2_n;
      const auto xnp1_3_k = a_3 * s_3_nm1 + ( 1.0 - a_3 ) * s_3_n;
      const auto xnp1_4_k = a_4 * s_4_nm1 + ( 1.0 - a_4 ) * s_4_n;

      scratch( 1, k ) = xnp1_1_k; // fluid vel
      scratch( 2, k ) = xnp1_2_k; // fluid energy
      scratch( 3, k ) = xnp1_3_k; // rad energy
      scratch( 4, k ) = xnp1_4_k; // rad flux
    }

    // --- update --- // TODO move up
    for ( int iC = 1; iC < nvars; ++iC ) {
      for ( int k = 0; k < num_modes; ++k ) {
        scratch_nm1( iC, k ) = scratch_n( iC, k );
        scratch_n( iC, k )   = scratch( iC, k );
      }
    }

    bool converged =
        convergence_checker.check_convergence( scratch_n, scratch_nm1 );

    ++n;
  } // while not converged
}

/* Fixed point solver templated on type, function, and args for func */
template <typename T, typename F, typename... Args>
auto fixed_point( F target, T x0, Args... args ) -> T {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL ) {
    T x1  = target( x0, args... );
    error = std::abs( residual( target, x0, args... ) );
    x0    = x1;
    ++n;
  }

  return x0;
}

template <typename T, typename F>
auto fixed_point( F target, T x0 ) -> T {

  unsigned int n = 0;
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL ) {
    T x1  = target( x0 );
    error = std::abs( residual( target, x0 ) );
    x0    = x1;
    ++n;
  }

  return x0;
}

/* Newton iteration templated on type, function, args */
template <typename T, typename F, typename... Args>
auto newton( F target, F dTarget, T x0, Args... args ) -> T {

  unsigned int n = 0;
  T h            = target( x0, args... ) / dTarget( x0, args... );
  T error        = 1.0;
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL ) {
    T xn  = x0;
    T h   = target( xn, args... ) / dTarget( xn, args... );
    x0    = xn - h;
    error = std::abs( xn - x0 );
    ++n;
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
  xk      = std::min( x0 - h, root_finders::ABSTOL ); // keep positive definite
  xkm1    = x0;
  T result;
  if ( std::abs( xk - x0 ) <= root_finders::ABSTOL ) {
    return xk;
  }
  while ( n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL ) {
    T hp1 = target( xk, args... ) / dTarget( xk, args... );
    T h   = target( xkm1, args... ) / dTarget( xkm1, args... );
    /* Anderson acceleration step */
    T gamma = hp1 / ( hp1 - h );

    xkp1  = xk - hp1 - gamma * ( xk - xkm1 - hp1 + h );
    error = std::abs( xk - xkp1 );

    xkm1 = xk;
    xk   = xkp1;

    ++n;
    result = xk;
  }
  return result;
}

} // namespace root_finders
