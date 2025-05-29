/**
 * @file rad_utilities.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Functions for radiation evolution.
 *
 * @details Key functions for radiation udates:
 *          - flux_factor
 *          - flux_rad
 *          - radiation_four_force
 *          - source_rad
 *          - Compute_Closure
 *          - lambda_hll
 *          - numerical_flux_hll_rad
 *          - computeTimestep_Rad
 */

#include <algorithm> // std::min, std::max
#include <cmath> // pow, abs, sqrt
#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "constants.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"
#include "riemann.hpp"
#include "utilities.hpp"

using namespace riemann;

namespace radiation {

/**
 * radiation flux factor
 **/
auto flux_factor( const double E, const double F ) -> double {
  assert( E > 0.0 &&
          "Radiation :: flux_factor :: non positive definite energy density." );
  constexpr static double c = constants::c_cgs;
  return std::abs( F ) / ( c * E );
}

/**
 * The radiation fluxes
 * Here E and F are per unit volume
 **/
auto flux_rad( const double E, const double F, const double P, const double V, const int iCR ) -> double {
  assert( ( iCR == 0 || iCR == 1 ) && "Radiation :: flux_factor :: bad iCR." );
  assert( E > 0.0 &&
          "Radiation :: flux_rad :: non positive definite energy density." );

  constexpr static double c = constants::c_cgs;
  return ( iCR == 0 ) ? F - V * E : c * c * P - V * F;
}

/**
 * Radiation 4 force for rad-matter interactions
 * Assumes kappa_e ~ kappa_p, kappa_F ~ kappa_r
 * D : Density
 * V : Velocity
 * T : Temperature
 * kappa_r : rosseland kappa
 * kappa_p : planck kappa
 * E : radiation energy density
 * F : radiation momentum density
 * Pr : radiation momentum closure
 **/
[[nodiscard]] auto radiation_four_force( const double D, const double V,
                                         const double T, const double kappa_r,
                                         const double kappa_p, const double E,
                                         const double F, const double Pr )
    -> std::tuple<double, double> {
  assert(
      D >= 0.0 &&
      "Radiation :: RadiationFourFource :: Non positive definite density." );
  assert( T > 0.0 &&
          "Radiation :: RadiationFourFource :: Non positive temperature." );
  assert( E > 0.0 && "Radiation :: RadiationFourFource :: Non positive "
                     "definite radiation energy density." );

  constexpr static double a = constants::a;
  constexpr static double c = constants::c_cgs;

  const double b     = V / c;
  const double term1 = E - ( a * T * T * T * T );
  const double Fc    = F / c;

  // O(b^2) ala Fuksman
  /*
  const double kappa = kappa_r;
  const double G0 = D * kappa * ( term1 - b * Fc - b * b * E - b * b * Pr );
  const double G  = D * kappa * ( b * ( term1 - 2.0 * b * Fc ) + ( Fc - b * E -
  b * Pr ) );
  */

  // Krumholz et al. 2007 O(b^2)
  const double G0 =
      D *
      ( kappa_p * term1 + ( kappa_r - 2.0 * kappa_p ) * b * Fc +
        0.5 * ( 2.0 * ( kappa_p - kappa_r ) * E + kappa_p * term1 ) * b * b +
        ( kappa_p - kappa_r ) * b * b * Pr );

  const double G = D * ( kappa_r * Fc + kappa_p * term1 * b -
                       kappa_r * b * ( E + Pr ) + 0.5 * kappa_r * Fc * b * b +
                       2.0 * ( kappa_r - kappa_p ) * b * b * Fc );

  // ala Skinner & Ostriker, simpler.
  /*
  const double kappa = kappa_r;
  const double G0 = D * kappa * ( term1 - b * Fc );
  const double G  = D * kappa * ( Fc - b * E + b * Pr );
  */
  return { G0, G };
}

/**
 * source terms for radiation
 * TODO: total opacity X
 **/
[[nodiscard]] auto source_rad( const double D, const double V, const double T,
                               const double kappa_r, const double kappa_p,
                               const double E, const double F, const double Pr,
                               const int iCR ) -> double {
  assert( ( iCR == 0 || iCR == 1 ) && "Radiation :: source_rad :: bad iCR." );
  assert( D >= 0.0 &&
          "Radiation :: source_rad :: Non positive definite density." );
  assert( T > 0.0 && "Radiation :: source_rad :: Non positive temperature." );
  assert( E > 0.0 && "Radiation :: source_rad :: Non positive "
                     "definite radiation energy density." );

  const double c = constants::c_cgs;

  auto [G0, G] = radiation_four_force( D, V, T, kappa_r, kappa_p, E, F, Pr );

  return ( iCR == 0 ) ? -c * G0 : -c * c * G;
}

/* pressure tensor closure */
// TODO(astrobarker): check Closure
[[nodiscard]] auto compute_closure( const double E, const double F ) -> double {
  assert( E > 0.0 && "Radiation :: compute_closure :: Non positive definite "
                     "radiation energy density." );
  constexpr static double one_third = 1.0 / 3.0;
  const double f   = utilities::make_bounded( flux_factor( E, F ), 0.0, 1.0 );
  const double chi = ( 3.0 + 4.0 * f * f ) /
                   ( 5.0 + 2.0 * std::sqrt( 4.0 - ( 3.0 * f * f ) ) );
  const double T = utilities::make_bounded(
      ( ( 1.0 - chi ) / 2.0 ) +
          ( ( 3.0 * chi - 1.0 ) * 1.0 / // utilities::SGN(F)
            2.0 ),
      one_third, 1.0 ); // TODO(astrobarker): Is this right?
  return E * T;
}

auto llf_flux( const double Fp, const double Fm, const double Up, const double Um,
               const double alpha ) -> double {
  return 0.5 * ( Fp - alpha * Up + Fm + alpha * Um );
}

/**
 * eigenvalues of JAcobian for radiation solve
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 41a,b
 * and references therein
 **/
auto lambda_hll( const double f, const int sign ) -> double {
  constexpr static double c        = constants::c_cgs;
  constexpr static double twothird = 2.0 / 3.0;

  const double f2       = f * f;
  const double sqrtterm = std::sqrt( 4.0 - ( 3.0 * f2 ) );
  return c *
         ( f + sign * std::sqrt( ( twothird * ( 4.0 - 3.0 * f2 - sqrtterm ) ) +
                                 ( 2.0 * ( 2.0 - f2 - sqrtterm ) ) ) ) /
         sqrtterm;
}

/**
 * HLL Riemann solver for radiation
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 39
 * and references & discussion therein
 *
 * Note: pass in Eulerian variables ( _ / cm^3 )
 **/
auto numerical_flux_hll_rad( const double E_L, const double E_R, const double F_L,
                             const double F_R, const double P_L, const double P_R,
                             const double vstar ) -> std::tuple<double, double> {
  // flux factors
  const double f_L = flux_factor( E_L, F_L );
  const double f_R = flux_factor( E_R, F_R );

  // TODO(astrobarker) - vstar?
  constexpr static double c2 = constants::c_cgs * constants::c_cgs;
  const double lambda1_L     = lambda_hll( f_L, -1.0 ) - vstar;
  const double lambda1_R     = lambda_hll( f_R, -1.0 ) - vstar;
  const double lambda3_L     = lambda_hll( f_L, 1.0 ) - vstar;
  const double lambda3_R     = lambda_hll( f_R, 1.0 ) - vstar;
  const double lambda_min_L  = std::min( lambda1_L, lambda3_L );
  const double lambda_min_R  = std::min( lambda1_R, lambda3_R );
  const double lambda_max_L  = std::max( lambda1_L, lambda3_L );
  const double lambda_max_R  = std::max( lambda1_R, lambda3_R );

  const double s_r = std::max( lambda_max_L, lambda_max_R );
  const double s_l = std::min( lambda_min_L, lambda_min_R );

  const double s_r_p = std::max( s_r, 0.0 );
  const double s_l_m = std::min( s_l, 0.0 );

  /*
  // eigenvalues
  const double s_r_p = std::max(
      std::max( lambda_hll( f_L, 1.0 ), lambda_hll( f_R, 1.0 ) ), 0.0 );
  const double s_l_m = std::min(
      std::min( lambda_hll( f_L, -1.0 ), lambda_hll( f_R, -1.0 ) ), 0.0 );
  */
  const double Flux_E = hll( E_L, E_R, F_L - vstar * E_L, F_R - vstar * E_R, s_l_m, s_r_p );
  const double Flux_F = hll( F_L, F_R, c2 * P_L - vstar * F_L, c2 * P_R - vstar * F_R, s_l_m, s_r_p );
  return { Flux_E, Flux_F };
}

/**
 * Compute the rad timestep.
 **/
auto compute_timestep_rad( const GridStructure* grid, const double CFL ) -> double {

  const double MIN_DT = 1.0e-18;
  const double MAX_DT = 100.0;

  const int& ilo = grid->get_ilo( );
  const int& ihi = grid->get_ihi( );

  double dt = 0.0;
  Kokkos::parallel_reduce(
      "Compute Timestep", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( const int iX, double& lmin ) {
        const double dr = grid->get_widths( iX );

        const double eigval = constants::c_cgs;

        const double dt_old = std::abs( dr ) / std::abs( eigval );

        if ( dt_old < lmin ) lmin = dt_old;
      },
      Kokkos::Min<double>( dt ) );

  dt = std::max( CFL * dt, MIN_DT );
  dt = std::min( dt, MAX_DT );

  assert( !std::isnan( dt ) && "NaN encountered in compute_timestep_rad.\n" );

  return dt;
}

} // namespace radiation
