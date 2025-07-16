/**
 * @file fluid_utilities.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Utilities for fluid evolution
 *
 * @details Contains functions necessary for fluid evolution
 */

#include <algorithm> // std::min, std::max
#include <cstdlib> /* abs */

#include "constants.hpp"
#include "eos_variant.hpp"
#include "error.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "rad_utilities.hpp"
#include "utils/utilities.hpp"

using utilities::pos_part;

namespace fluid {
/**
 * Return a component iCF of the flux vector.
 **/
auto flux_fluid( const double V, const double P, const int iCF ) -> double {
  assert( iCF == 0 || iCF == 1 || iCF == 2 );
  assert( P > 0.0 && "Flux_Flux :: negative pressure" );
  assert( iCF >= 0 && iCF <= 2 );
  assert( P > 0.0 && "Flux_Flux :: negative pressure" );

  switch ( iCF ) {
  case 0:
    return -V;
  case 1:
    return P;
  case 2:
    return P * V;
  default:
    // Should never reach here due to assert, but added for completeness
    THROW_ATHELAS_ERROR( "! Please input a valid iCF! (0,1,2)." );
    return AthelasExitCodes::FAILURE;
  }
}

/**
 * Fluid radiation sources. Kind of redundant with Rad_sources.
 **/
auto source_fluid_rad( const double D, const double V, const double T,
                       const double kappa_r, const double kappa_p,
                       const double E, const double F, const double Pr,
                       const int iCF ) -> double {
  assert( iCF == 1 || iCF == 2 );

  constexpr static double c = constants::c_cgs;

  auto [G0, G] =
      radiation::radiation_four_force( D, V, T, kappa_r, kappa_p, E, F, Pr );

  return ( iCF == 1 ) ? G : c * G0;
}

/**
 * Positivity preserving numerical flux. Constructs v* and p* states.
 * TODO(astrobarker): do I need tau_r_star if I construct p* with left?
 **/
auto numerical_flux_gudonov_positivity( const double tauL, const double tauR,
                                        const double vL, const double vR,
                                        const double pL, const double pR,
                                        const double csL, const double csR )
    -> std::tuple<double, double> {
  assert( pL > 0.0 && pR > 0.0 &&
          "numerical_flux_gudonov :: negative pressure" );
  const double pRmL = pR - pL; // [[p]]
  const double vRmL = vR - vL; // [[v]]
  /*
  const double zL   = std::max(
      std::max( std::sqrt( pos_part( pRmL ) / tauL ), -( vRmL ) / tauL ),
      csL / tauL );
  const double zR = std::max(
      std::max( std::sqrt( pos_part( -pR + pL ) / tauR ), -( vRmL ) / tauR ),
      csR / tauR );
  */
  const double zL    = csL / tauL;
  const double zR    = csR / tauR;
  const double z_sum = zL + zR;

  // get tau star states
  const double term1_l    = tauL - ( pRmL ) / ( zL * zL );
  const double term2_l    = tauL + vRmL / zL;
  const double tau_l_star = ( zL * term1_l + zR * term2_l ) / z_sum;

  /*
  const double term1_r = tauR + vRmL / zR;
  const double term2_r = tauR + pRmL / (zR * zR);
  const double tau_r_star = (zL * term1_r + zR * term2_r) / z_sum;
  */

  // vstar, pstar
  const double Flux_U = ( -pRmL + zR * vR + zL * vL ) / ( z_sum );
  const double Flux_P = pL - ( zL * zL ) * ( tau_l_star - tauL );
  return { Flux_U, Flux_P };
}

/**
 * Gudonov style numerical flux. Constructs v* and p* states.
 **/
auto numerical_flux_gudonov( const double vL, const double vR, const double pL,
                             const double pR, const double zL, const double zR )
    -> std::tuple<double, double> {
  assert( pL > 0.0 && pR > 0.0 &&
          "numerical_flux_gudonov :: negative pressure" );
  const double Flux_U = ( pL - pR + zR * vR + zL * vL ) / ( zR + zL );
  const double Flux_P =
      ( zR * pL + zL * pR + zL * zR * ( vL - vR ) ) / ( zR + zL );
  return { Flux_U, Flux_P };
}

/**
 * Gudonov style numerical flux. Constructs v* and p* states.
 **/
void numerical_flux_hllc( double vL, double vR, double pL, double pR, double cL,
                          double cR, double rhoL, double rhoR, double& Flux_U,
                          double& Flux_P ) {
  double const aL = vL - cL; // left wave speed estimate
  double const aR = vR + cR; // right wave speed estimate
  Flux_U = ( rhoR * vR * ( aR - vR ) - rhoL * vL * ( aL - vL ) + pL - pR ) /
           ( rhoR * ( aR - vR ) - rhoL * ( aL - vL ) );
  Flux_P = rhoL * ( vL - aL ) * ( vL - Flux_U ) + pL;
}

// Compute Auxiliary

/**
 * Compute the fluid timestep.
 **/
auto compute_timestep_fluid( const View3D<double> U, const GridStructure* grid,
                             EOS* eos, const double CFL ) -> double {

  const double MIN_DT = 1.0e-14;
  const double MAX_DT = 100.0;

  const int& ilo = grid->get_ilo( );
  const int& ihi = grid->get_ihi( );

  double dt = 0.0;
  Kokkos::parallel_reduce(
      "Compute Timestep", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( const int iX, double& lmin ) {
        // --- Compute Cell Averages ---
        double tau_x  = U( 0, iX, 0 );
        double vel_x  = U( 1, iX, 0 );
        double eint_x = U( 2, iX, 0 );

        assert( tau_x > 0.0 && "Compute Timestep :: bad specific volume" );
        assert( eint_x > 0.0 && "Compute Timestep :: bad specific energy" );

        double dr = grid->get_widths( iX );

        auto lambda = nullptr;
        const double Cs =
            sound_speed_from_conserved( eos, tau_x, vel_x, eint_x, lambda );
        double eigval = Cs + std::abs( vel_x );

        double dt_old = std::abs( dr ) / std::abs( eigval );

        lmin = std::min( dt_old, lmin );
      },
      Kokkos::Min<double>( dt ) );

  dt = std::max( CFL * dt, MIN_DT );
  dt = std::min( dt, MAX_DT );

  assert( !std::isnan( dt ) && "NaN encountered in compute_timestep_fluid.\n" );

  return dt;
}
} // namespace fluid
